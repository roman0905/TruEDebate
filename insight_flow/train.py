"""
TruEDebate (TED) — 训练循环、验证与评估
包含梯度累积、混合精度训练、macF1/Acc/F1_real/F1_fake 评估指标。
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import config
from insight_flow.networks import TEDClassifier

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: TEDClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None = None,
    grad_accum_steps: int = config.GRAD_ACCUM_STEPS,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip_max_norm: float = config.GRAD_CLIP_MAX_NORM,
) -> float:
    """
    训练一个 epoch。

    Args:
        model: TEDClassifier 模型
        loader: 训练 DataLoader
        optimizer: 优化器
        criterion: 损失函数
        device: 训练设备
        scaler: 混合精度缩放器 (None 表示不使用 AMP)
        grad_accum_steps: 梯度累积步数
        scheduler: 学习率调度器
        grad_clip_max_norm: 梯度裁剪阈值，<=0 表示禁用

    Returns:
        平均训练损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch_data in enumerate(loader):
        batch_data = batch_data.to(device)

        # 提取字段
        node_input_ids = batch_data.node_input_ids
        node_attention_mask = batch_data.node_attention_mask
        role_ids = batch_data.role_ids
        numeric_features = getattr(batch_data, "numeric_features", None)
        edge_index = batch_data.edge_index
        edge_type = getattr(batch_data, "edge_type", None)
        batch_vec = batch_data.batch
        news_input_ids = batch_data.news_input_ids
        news_attention_mask = batch_data.news_attention_mask
        labels = batch_data.y

        # 修正 news tensor 维度:
        # PyG batching 会把 [1, seq_len] 拼成 [batch_size, seq_len]
        # 但如果是 [batch_size*1, seq_len]，需要确认
        if news_input_ids.dim() == 2 and news_input_ids.shape[0] != labels.shape[0]:
            # 如果 news 也被 PyG 当成节点维度拼接了，需要恢复
            bs = labels.shape[0]
            seq_len = news_input_ids.shape[-1]
            news_input_ids = news_input_ids.view(bs, seq_len)
            news_attention_mask = news_attention_mask.view(bs, seq_len)

        use_amp = scaler is not None

        # 前向传播
        if use_amp:
            with autocast(str(device).split(":")[0]):
                logits = model(
                    node_input_ids=node_input_ids,
                    node_attention_mask=node_attention_mask,
                    role_ids=role_ids,
                    numeric_features=numeric_features,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    batch=batch_vec,
                    news_input_ids=news_input_ids,
                    news_attention_mask=news_attention_mask,
                )
                loss = criterion(logits, labels)
                loss = loss / grad_accum_steps
        else:
            logits = model(
                node_input_ids=node_input_ids,
                node_attention_mask=node_attention_mask,
                role_ids=role_ids,
                numeric_features=numeric_features,
                edge_index=edge_index,
                edge_type=edge_type,
                batch=batch_vec,
                news_input_ids=news_input_ids,
                news_attention_mask=news_attention_mask,
            )
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps

        # 反向传播
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积: 每 grad_accum_steps 步更新一次参数
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                if grad_clip_max_norm and grad_clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_max_norm
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip_max_norm and grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_max_norm
                    )
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: TEDClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    在验证/测试集上评估模型。

    Args:
        model: TEDClassifier 模型
        loader: 验证/测试 DataLoader
        criterion: 损失函数
        device: 设备

    Returns:
        dict 包含: loss, accuracy, macro_f1, f1_real, f1_fake
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_data in loader:
        batch_data = batch_data.to(device)

        node_input_ids = batch_data.node_input_ids
        node_attention_mask = batch_data.node_attention_mask
        role_ids = batch_data.role_ids
        numeric_features = getattr(batch_data, "numeric_features", None)
        edge_index = batch_data.edge_index
        edge_type = getattr(batch_data, "edge_type", None)
        batch_vec = batch_data.batch
        news_input_ids = batch_data.news_input_ids
        news_attention_mask = batch_data.news_attention_mask
        labels = batch_data.y

        # 修正 news tensor 维度
        if news_input_ids.dim() == 2 and news_input_ids.shape[0] != labels.shape[0]:
            bs = labels.shape[0]
            seq_len = news_input_ids.shape[-1]
            news_input_ids = news_input_ids.view(bs, seq_len)
            news_attention_mask = news_attention_mask.view(bs, seq_len)

        logits = model(
            node_input_ids=node_input_ids,
            node_attention_mask=node_attention_mask,
            role_ids=role_ids,
            numeric_features=numeric_features,
            edge_index=edge_index,
            edge_type=edge_type,
            batch=batch_vec,
            news_input_ids=news_input_ids,
            news_attention_mask=news_attention_mask,
        )
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    num_batches = max(len(loader), 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_real = f1_score(all_labels, all_preds, average="binary", pos_label=0, zero_division=0)
    f1_fake = f1_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)
    fake_precision = precision_score(
        all_labels, all_preds, pos_label=1, zero_division=0
    )
    fake_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()

    metrics = {
        "loss": total_loss / num_batches,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_real": f1_real,
        "f1_fake": f1_fake,
        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "pred_fake": int((all_preds == 1).sum()),
        "true_fake": int((all_labels == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics


def _compute_class_weights(loader: DataLoader, device: torch.device) -> torch.Tensor:
    """根据训练集标签计算二分类均衡权重。"""
    labels = []
    dataset = loader.dataset
    file_paths = getattr(dataset, "file_paths", None)
    if file_paths is not None:
        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
            label = record["label"]
            if isinstance(label, str):
                label = config.LABEL_MAP.get(label, label)
            labels.append(int(label))
    else:
        for data in dataset:
            labels.append(int(data.y))

    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=2)
    total = max(int(counts.sum()), 1)
    weights = total / (2.0 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float, device=device)


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = config.WARMUP_RATIO,
    min_lr_ratio: float = config.MIN_LR_RATIO,
) -> torch.optim.lr_scheduler.LambdaLR:
    """构建 warmup + cosine decay 调度器。"""
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(warmup_steps, 1)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: TEDClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None = None,
    device: torch.device = torch.device("cpu"),
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    use_amp: bool = config.USE_AMP,
    checkpoint_dir: str | Path = config.CHECKPOINT_DIR,
    grad_accum_steps: int = config.GRAD_ACCUM_STEPS,
) -> dict:
    """
    完整训练流程。

    Args:
        model: TEDClassifier 模型
        train_loader: 训练 DataLoader
        val_loader: 验证 DataLoader
        test_loader: 测试 DataLoader (可选)
        device: 训练设备
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        use_amp: 是否使用混合精度
        checkpoint_dir: 模型保存目录
        grad_accum_steps: 梯度累积步数

    Returns:
        最佳验证指标 dict
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    class_weights = None
    if config.USE_CLASS_WEIGHT:
        class_weights = _compute_class_weights(train_loader, device)
        logger.info(
            f"类别权重启用: real={class_weights[0].item():.4f}, "
            f"fake={class_weights[1].item():.4f}"
        )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.LABEL_SMOOTHING,
    )

    # 区分 BERT 参数和其余参数，使用不同学习率
    bert_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bert" in name:
                bert_params.append(param)
            else:
                other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": lr * config.BERT_LR_FACTOR},
        {"params": other_params, "lr": lr},
    ], weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / max(grad_accum_steps, 1))
    total_steps = max(steps_per_epoch * epochs, 1)
    scheduler = _build_warmup_cosine_scheduler(optimizer, total_steps)

    scaler = GradScaler(str(device)) if (use_amp and device.type == "cuda") else None

    best_val_f1 = -1.0
    best_metrics = {}
    patience_counter = 0

    logger.info(f"开始训练: {epochs} epochs, device={device}, AMP={use_amp}")
    logger.info(f"BERT 可训练参数: {sum(p.numel() for p in bert_params):,}")
    logger.info(f"其他可训练参数: {sum(p.numel() for p in other_params):,}")
    logger.info(
        f"Scheduler: warmup+cosine | total_steps={total_steps}, "
        f"warmup_steps={int(total_steps * config.WARMUP_RATIO)}, "
        f"min_lr_ratio={config.MIN_LR_RATIO}"
    )
    logger.info(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    logger.info(f"Label smoothing: {config.LABEL_SMOOTHING}")
    logger.info(f"Grad clip max norm: {config.GRAD_CLIP_MAX_NORM}")

    for epoch in range(1, epochs + 1):
        # ── 训练 ──
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            grad_accum_steps=grad_accum_steps,
            scheduler=scheduler,
        )

        # ── 验证 ──
        val_metrics = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val macF1: {val_metrics['macro_f1']:.4f} | "
            f"Val F1_real: {val_metrics['f1_real']:.4f} | "
            f"Val F1_fake: {val_metrics['f1_fake']:.4f} | "
            f"Fake P/R: {val_metrics['fake_precision']:.4f}/"
            f"{val_metrics['fake_recall']:.4f} | "
            f"Pred fake: {val_metrics['pred_fake']}/{val_metrics['true_fake']}"
        )

        # 保存最佳模型 (基于 Val Macro F1)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics.copy()
            best_metrics["epoch"] = epoch

            save_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val_metrics,
            }, save_path)
            logger.info(f"  ★ 最佳模型已保存 (macF1={best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping 触发: 连续 {patience_counter} 轮无提升，"
                    f"停止于 epoch {epoch}"
                )
                break

    # ── 测试集评估 ──
    if test_loader is not None:
        # 加载最佳模型
        best_ckpt = torch.load(
            checkpoint_dir / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_ckpt["model_state_dict"])

        test_metrics = evaluate(model, test_loader, criterion, device)
        logger.info("=" * 60)
        logger.info("测试集最终结果:")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1:  {test_metrics['macro_f1']:.4f}")
        logger.info(f"  F1 (Real): {test_metrics['f1_real']:.4f}")
        logger.info(f"  F1 (Fake): {test_metrics['f1_fake']:.4f}")
        logger.info(
            f"  Fake P/R:  {test_metrics['fake_precision']:.4f}/"
            f"{test_metrics['fake_recall']:.4f}"
        )
        logger.info(
            f"  Confusion: TN={test_metrics['tn']} FP={test_metrics['fp']} "
            f"FN={test_metrics['fn']} TP={test_metrics['tp']}"
        )
        logger.info("=" * 60)
        best_metrics["test"] = test_metrics

    metrics_path = checkpoint_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {metrics_path}")

    return best_metrics
