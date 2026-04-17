"""
TruEDebate (TED) — 训练循环、验证与评估
包含梯度累积、混合精度训练、macF1/Acc/F1_real/F1_fake 评估指标。
"""

import logging
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score

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
        edge_index = batch_data.edge_index
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
                    edge_index=edge_index,
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
                edge_index=edge_index,
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
                scaler.unscale_(optimizer)
            if grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
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
    use_amp: bool = config.USE_AMP,
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

    amp_enabled = use_amp and device.type == "cuda"

    for batch_data in loader:
        batch_data = batch_data.to(device)

        node_input_ids = batch_data.node_input_ids
        node_attention_mask = batch_data.node_attention_mask
        role_ids = batch_data.role_ids
        edge_index = batch_data.edge_index
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

        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(
                node_input_ids=node_input_ids,
                node_attention_mask=node_attention_mask,
                role_ids=role_ids,
                edge_index=edge_index,
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

    metrics = {
        "loss": total_loss / num_batches,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_real": f1_real,
        "f1_fake": f1_fake,
    }

    return metrics


def _estimate_class_weights(dataset, device: torch.device) -> torch.Tensor | None:
    """
    从数据集 JSON 标签估计类别权重。

    采用总数/类别数 的逆频率权重，适配二分类 real(0)/fake(1)。
    """
    file_paths = getattr(dataset, "file_paths", None)
    if not file_paths:
        return None

    class_counts = torch.zeros(2, dtype=torch.float32)
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            record = json.load(f)
        label = record.get("label", 0)
        if isinstance(label, str):
            label = config.LABEL_MAP.get(label, label)
        label = int(label)
        if label in (0, 1):
            class_counts[label] += 1

    if torch.any(class_counts == 0):
        return None

    weights = class_counts.sum() / (2.0 * class_counts)
    return weights.to(device)


def _build_optimizer(model: TEDClassifier, lr: float, weight_decay: float, bert_lr_factor: float):
    """构建 AdamW 优化器，且对 bias/norm 参数关闭 weight decay。"""
    no_decay_names = ("bias", "LayerNorm.weight", "layernorm.weight", "norm.weight")

    bert_decay, bert_no_decay = [], []
    other_decay, other_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_bert = "bert" in name
        is_no_decay = any(token in name for token in no_decay_names)

        if is_bert and is_no_decay:
            bert_no_decay.append(param)
        elif is_bert:
            bert_decay.append(param)
        elif is_no_decay:
            other_no_decay.append(param)
        else:
            other_decay.append(param)

    param_groups = []
    if bert_decay:
        param_groups.append({"params": bert_decay, "lr": lr * bert_lr_factor, "weight_decay": weight_decay})
    if bert_no_decay:
        param_groups.append({"params": bert_no_decay, "lr": lr * bert_lr_factor, "weight_decay": 0.0})
    if other_decay:
        param_groups.append({"params": other_decay, "lr": lr, "weight_decay": weight_decay})
    if other_no_decay:
        param_groups.append({"params": other_no_decay, "lr": lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer, bert_decay + bert_no_decay, other_decay + other_no_decay


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_update_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
):
    """构建线性 warmup + cosine decay 的按 step 调度器。"""
    warmup_steps = int(total_update_steps * warmup_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, warmup_steps


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
    bert_lr_factor: float = config.BERT_LR_FACTOR,
    warmup_ratio: float = config.WARMUP_RATIO,
    min_lr_ratio: float = config.MIN_LR_RATIO,
    early_stopping_patience: int = config.EARLY_STOPPING_PATIENCE,
    label_smoothing: float = config.LABEL_SMOOTHING,
    use_class_weight: bool = config.USE_CLASS_WEIGHT,
    grad_clip_max_norm: float = config.GRAD_CLIP_MAX_NORM,
    checkpoint_dir: str | Path = config.CHECKPOINT_DIR,
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

    Returns:
        最佳验证指标 dict
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    class_weights = None
    if use_class_weight:
        class_weights = _estimate_class_weights(train_loader.dataset, device)
        if class_weights is not None:
            logger.info(
                f"类别权重启用: real={class_weights[0].item():.4f}, fake={class_weights[1].item():.4f}"
            )
        else:
            logger.warning("类别权重启用失败，回退到无权重损失。")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    optimizer, bert_params, other_params = _build_optimizer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        bert_lr_factor=bert_lr_factor,
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(config.GRAD_ACCUM_STEPS, 1))
    total_update_steps = max(steps_per_epoch * epochs, 1)
    scheduler, warmup_steps = _build_scheduler(
        optimizer=optimizer,
        total_update_steps=total_update_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )

    scaler = GradScaler(str(device)) if (use_amp and device.type == "cuda") else None

    best_val_f1 = -float("inf")
    best_metrics = {}
    no_improve_count = 0

    logger.info(f"开始训练: {epochs} epochs, device={device}, AMP={use_amp}")
    logger.info(f"BERT 可训练参数: {sum(p.numel() for p in bert_params):,}")
    logger.info(f"其他可训练参数: {sum(p.numel() for p in other_params):,}")
    logger.info(
        f"Scheduler: warmup+cosine | total_steps={total_update_steps}, warmup_steps={warmup_steps}, min_lr_ratio={min_lr_ratio}"
    )
    logger.info(f"Early stopping patience: {early_stopping_patience}")

    for epoch in range(1, epochs + 1):
        # ── 训练 ──
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            grad_accum_steps=config.GRAD_ACCUM_STEPS,
            scheduler=scheduler,
            grad_clip_max_norm=grad_clip_max_norm,
        )

        # ── 验证 ──
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val macF1: {val_metrics['macro_f1']:.4f} | "
            f"Val F1_real: {val_metrics['f1_real']:.4f} | "
            f"Val F1_fake: {val_metrics['f1_fake']:.4f}"
        )

        # 保存最佳模型 (基于 Val Macro F1)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics.copy()
            best_metrics["epoch"] = epoch
            no_improve_count = 0

            save_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, save_path)
            logger.info(f"  ★ 最佳模型已保存 (macF1={best_val_f1:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                logger.info(
                    f"Early stopping 触发: 连续 {early_stopping_patience} 轮无提升，停止于 epoch {epoch}"
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

        test_metrics = evaluate(model, test_loader, criterion, device, use_amp=use_amp)
        logger.info("=" * 60)
        logger.info("测试集最终结果:")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1:  {test_metrics['macro_f1']:.4f}")
        logger.info(f"  F1 (Real): {test_metrics['f1_real']:.4f}")
        logger.info(f"  F1 (Fake): {test_metrics['f1_fake']:.4f}")
        logger.info("=" * 60)
        best_metrics["test"] = test_metrics

    return best_metrics
