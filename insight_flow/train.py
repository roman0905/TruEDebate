"""
TruEDebate (TED/PAMD) — 训练循环、验证与评估（创新版本）

相对原始实现的核心改进：
1. 主损失支持 Focal Loss + 类别权重，针对 F1_fake 偏低。
2. Perspective 节点的辅助分类损失（multi-task）。
3. SWA：训练后期对模型权重做平均（torch.optim.swa_utils.AveragedModel）。
4. 验证集阈值调优：扫描 fake 类阈值最大化 macF1，应用到测试集。
5. 评估返回 probs/labels 供阈值搜索复用。
"""

import json
import logging
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config
from insight_flow.networks import TEDClassifier

logger = logging.getLogger(__name__)


# ──────────────────────────────── Focal Loss ────────────────────────────────


class FocalLoss(nn.Module):
    """支持类别权重的 Focal Loss。"""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        if self.label_smoothing > 0:
            num_classes = logits.shape[-1]
            with torch.no_grad():
                soft_target = torch.full_like(logits, self.label_smoothing / num_classes)
                soft_target.scatter_(
                    1,
                    target.unsqueeze(1),
                    1.0 - self.label_smoothing + self.label_smoothing / num_classes,
                )
            ce_per_example = -(soft_target * log_p).sum(dim=-1)
        else:
            ce_per_example = F.nll_loss(log_p, target, reduction="none")

        with torch.no_grad():
            pt = log_p.exp().gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_per_example

        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ──────────────────────────────── 前向 helper ────────────────────────────────


def _extract_batch(batch_data, labels_required: bool = True):
    """统一从 PyG batch 里取出 forward 所需字段。"""
    node_input_ids = batch_data.node_input_ids
    node_attention_mask = batch_data.node_attention_mask
    role_ids = batch_data.role_ids
    node_group_ids = batch_data.node_group_ids
    numeric_features = getattr(batch_data, "numeric_features", None)
    edge_index = batch_data.edge_index
    edge_type = getattr(batch_data, "edge_type", None)
    batch_vec = batch_data.batch
    news_input_ids = batch_data.news_input_ids
    news_attention_mask = batch_data.news_attention_mask
    labels = batch_data.y if labels_required else None

    # PyG 的 batching 可能把 [1, seq_len] news 拼成 [bs, seq_len]，这里做一致性矫正。
    if labels is not None and news_input_ids.dim() == 2 and news_input_ids.shape[0] != labels.shape[0]:
        bs = labels.shape[0]
        seq_len = news_input_ids.shape[-1]
        news_input_ids = news_input_ids.view(bs, seq_len)
        news_attention_mask = news_attention_mask.view(bs, seq_len)

    return {
        "node_input_ids": node_input_ids,
        "node_attention_mask": node_attention_mask,
        "role_ids": role_ids,
        "node_group_ids": node_group_ids,
        "numeric_features": numeric_features,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "batch": batch_vec,
        "news_input_ids": news_input_ids,
        "news_attention_mask": news_attention_mask,
        "labels": labels,
    }


def _call_model(model: nn.Module, fields: dict, return_aux: bool = False):
    aux_supported = isinstance(model, TEDClassifier) and getattr(model, "use_aux_loss", False)
    # SWA AveragedModel 把真正的模块包在 .module 里。
    real = model.module if hasattr(model, "module") else model
    aux_supported = aux_supported or (
        isinstance(real, TEDClassifier) and getattr(real, "use_aux_loss", False)
    )
    if return_aux and aux_supported:
        return model(
            node_input_ids=fields["node_input_ids"],
            node_attention_mask=fields["node_attention_mask"],
            role_ids=fields["role_ids"],
            node_group_ids=fields["node_group_ids"],
            numeric_features=fields["numeric_features"],
            edge_index=fields["edge_index"],
            edge_type=fields["edge_type"],
            batch=fields["batch"],
            news_input_ids=fields["news_input_ids"],
            news_attention_mask=fields["news_attention_mask"],
            return_aux=True,
        )
    return model(
        node_input_ids=fields["node_input_ids"],
        node_attention_mask=fields["node_attention_mask"],
        role_ids=fields["role_ids"],
        node_group_ids=fields["node_group_ids"],
        numeric_features=fields["numeric_features"],
        edge_index=fields["edge_index"],
        edge_type=fields["edge_type"],
        batch=fields["batch"],
        news_input_ids=fields["news_input_ids"],
        news_attention_mask=fields["news_attention_mask"],
    )


# ──────────────────────────────── 训练循环 ────────────────────────────────


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
    aux_loss_weight: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch_data in enumerate(loader):
        batch_data = batch_data.to(device)
        fields = _extract_batch(batch_data)
        labels = fields["labels"]

        use_amp = scaler is not None

        def _forward_and_loss():
            if aux_loss_weight > 0 and getattr(model, "use_aux_loss", False):
                logits, aux_logits, aux_batch_ids = _call_model(model, fields, return_aux=True)
                main_loss = criterion(logits, labels)
                if aux_logits.numel() > 0:
                    aux_labels = labels[aux_batch_ids]
                    aux_loss = criterion(aux_logits, aux_labels)
                    total = main_loss + aux_loss_weight * aux_loss
                else:
                    total = main_loss
                return total
            logits = _call_model(model, fields)
            return criterion(logits, labels)

        if use_amp:
            with autocast(str(device).split(":")[0]):
                loss = _forward_and_loss() / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            loss = _forward_and_loss() / grad_accum_steps
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                if grad_clip_max_norm and grad_clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip_max_norm and grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ──────────────────────────────── 评估 ────────────────────────────────


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """收集预测概率与 loss，方便阈值调优。"""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch_data in loader:
        batch_data = batch_data.to(device)
        fields = _extract_batch(batch_data)
        labels = fields["labels"]
        logits = _call_model(model, fields)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = F.softmax(logits.float(), dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return {
        "loss": total_loss / max(len(loader), 1),
        "probs": np.concatenate(all_probs) if all_probs else np.array([]),
        "labels": np.concatenate(all_labels) if all_labels else np.array([]),
    }


def metrics_from_probs(
    probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict:
    if probs.size == 0:
        return {
            "accuracy": 0.0, "macro_f1": 0.0,
            "f1_real": 0.0, "f1_fake": 0.0,
            "fake_precision": 0.0, "fake_recall": 0.0,
            "pred_fake": 0, "true_fake": 0,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
            "threshold": threshold,
        }
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    f1_real = f1_score(labels, preds, average="binary", pos_label=0, zero_division=0)
    f1_fake = f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)
    fake_p = precision_score(labels, preds, pos_label=1, zero_division=0)
    fake_r = recall_score(labels, preds, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "f1_real": float(f1_real),
        "f1_fake": float(f1_fake),
        "fake_precision": float(fake_p),
        "fake_recall": float(fake_r),
        "pred_fake": int((preds == 1).sum()),
        "true_fake": int((labels == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    pack = collect_predictions(model, loader, criterion, device)
    metrics = metrics_from_probs(pack["probs"], pack["labels"], threshold=threshold)
    metrics["loss"] = pack["loss"]
    return metrics


def search_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    lo: float = config.THRESHOLD_SEARCH_RANGE[0],
    hi: float = config.THRESHOLD_SEARCH_RANGE[1],
    step: float = config.THRESHOLD_SEARCH_STEP,
) -> tuple[float, float]:
    if probs.size == 0:
        return 0.5, 0.0
    best_th, best_f1 = 0.5, -1.0
    for th in np.arange(lo, hi + 1e-9, step):
        f1 = f1_score(
            labels, (probs >= th).astype(int), average="macro", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)
    return best_th, best_f1


# ──────────────────────────────── 主训练流程 ────────────────────────────────


def _compute_class_weights(loader: DataLoader, device: torch.device) -> torch.Tensor:
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
    warmup_steps = max(int(total_steps * warmup_ratio), 1)

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
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # 类别权重
    class_weights = None
    if config.USE_CLASS_WEIGHT:
        class_weights = _compute_class_weights(train_loader, device)
        logger.info(
            f"类别权重启用: real={class_weights[0].item():.4f}, "
            f"fake={class_weights[1].item():.4f}"
        )

    # 主损失
    if config.USE_FOCAL_LOSS:
        criterion = FocalLoss(
            gamma=config.FOCAL_LOSS_GAMMA,
            alpha=class_weights,
            label_smoothing=config.LABEL_SMOOTHING,
        )
        logger.info(
            f"主损失: FocalLoss(gamma={config.FOCAL_LOSS_GAMMA}, "
            f"label_smoothing={config.LABEL_SMOOTHING})"
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=config.LABEL_SMOOTHING,
        )
        logger.info(
            f"主损失: CrossEntropy(label_smoothing={config.LABEL_SMOOTHING})"
        )

    # 优化器
    bert_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bert" in name:
                bert_params.append(param)
            else:
                other_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": lr * config.BERT_LR_FACTOR},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(grad_accum_steps, 1))
    total_steps = max(steps_per_epoch * epochs, 1)
    scheduler = _build_warmup_cosine_scheduler(optimizer, total_steps)
    scaler = GradScaler(str(device)) if (use_amp and device.type == "cuda") else None

    # SWA
    swa_model = None
    swa_start_epoch = int(epochs * config.SWA_START_RATIO)

    best_val_f1 = -1.0
    best_metrics: dict = {}
    patience_counter = 0
    aux_weight = config.AUX_LOSS_WEIGHT if getattr(model, "use_aux_loss", False) else 0.0

    logger.info(f"开始训练: {epochs} epochs, device={device}, AMP={use_amp}")
    logger.info(f"BERT 可训练参数: {sum(p.numel() for p in bert_params):,}")
    logger.info(f"其他可训练参数: {sum(p.numel() for p in other_params):,}")
    logger.info(
        f"Scheduler: warmup+cosine | total_steps={total_steps}, "
        f"warmup_steps={int(total_steps * config.WARMUP_RATIO)}, "
        f"min_lr_ratio={config.MIN_LR_RATIO}"
    )
    logger.info(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    logger.info(f"Aux loss weight: {aux_weight}")
    logger.info(f"Node dropout: {config.NODE_DROPOUT_P}, Edge dropout: {config.EDGE_DROPOUT_P}")
    logger.info(
        f"SWA: enabled={config.USE_SWA}, start_epoch={swa_start_epoch}"
    )
    logger.info(
        f"Threshold tuning: enabled={config.USE_THRESHOLD_TUNING}"
    )
    logger.info(f"Grad clip max norm: {config.GRAD_CLIP_MAX_NORM}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            grad_accum_steps=grad_accum_steps, scheduler=scheduler,
            aux_loss_weight=aux_weight,
        )

        val_pack = collect_predictions(model, val_loader, criterion, device)
        val_metrics = metrics_from_probs(val_pack["probs"], val_pack["labels"], threshold=0.5)
        val_metrics["loss"] = val_pack["loss"]

        # SWA 累积
        if config.USE_SWA and epoch >= swa_start_epoch:
            if swa_model is None:
                swa_model = AveragedModel(model)
                logger.info(f"  ✦ SWA 累积开始 (epoch {epoch})")
            else:
                swa_model.update_parameters(model)

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

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics.copy()
            best_metrics["epoch"] = epoch
            best_metrics["val_probs"] = val_pack["probs"]
            best_metrics["val_labels"] = val_pack["labels"]

            save_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": {k: v for k, v in val_metrics.items() if not isinstance(v, np.ndarray)},
            }, save_path)
            logger.info(f"  ★ 最佳模型已保存 (macF1={best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping 触发: 连续 {patience_counter} 轮无提升，停止于 epoch {epoch}"
                )
                break

    # ── SWA 评估并与最优 best_model 比较 ──
    swa_better = False
    if swa_model is not None:
        logger.info("评估 SWA 模型…")
        swa_val_pack = collect_predictions(swa_model, val_loader, criterion, device)
        swa_val_metrics = metrics_from_probs(swa_val_pack["probs"], swa_val_pack["labels"], threshold=0.5)
        swa_val_metrics["loss"] = swa_val_pack["loss"]
        logger.info(
            f"SWA Val | Loss: {swa_val_metrics['loss']:.4f} | "
            f"Acc: {swa_val_metrics['accuracy']:.4f} | "
            f"macF1: {swa_val_metrics['macro_f1']:.4f} | "
            f"F1_fake: {swa_val_metrics['f1_fake']:.4f}"
        )
        if swa_val_metrics["macro_f1"] > best_val_f1:
            logger.info(
                f"  ✦ SWA 优于 best_model (Δ macF1 = {swa_val_metrics['macro_f1'] - best_val_f1:.4f})，"
                "将使用 SWA 模型作最终预测。"
            )
            best_val_f1 = swa_val_metrics["macro_f1"]
            best_metrics = swa_val_metrics.copy()
            best_metrics["epoch"] = "swa"
            best_metrics["val_probs"] = swa_val_pack["probs"]
            best_metrics["val_labels"] = swa_val_pack["labels"]
            swa_better = True
            torch.save({
                "epoch": "swa",
                "model_state_dict": swa_model.state_dict(),
                "val_metrics": {k: v for k, v in swa_val_metrics.items() if not isinstance(v, np.ndarray)},
            }, checkpoint_dir / "swa_model.pt")

    # ── 阈值调优 ──
    best_threshold = 0.5
    if config.USE_THRESHOLD_TUNING and "val_probs" in best_metrics:
        best_threshold, tuned_f1 = search_best_threshold(
            best_metrics["val_probs"], best_metrics["val_labels"]
        )
        tuned_val = metrics_from_probs(
            best_metrics["val_probs"], best_metrics["val_labels"], threshold=best_threshold
        )
        logger.info(
            f"阈值调优: best_threshold={best_threshold:.3f} | "
            f"tuned Val macF1: {tuned_val['macro_f1']:.4f} | "
            f"F1_fake: {tuned_val['f1_fake']:.4f} | "
            f"Fake P/R: {tuned_val['fake_precision']:.4f}/{tuned_val['fake_recall']:.4f}"
        )
        # 用调优后的指标覆盖 best_metrics（不影响 epoch 字段）
        for k, v in tuned_val.items():
            best_metrics[k] = v
        best_metrics["threshold"] = best_threshold

    # ── 测试集评估 ──
    if test_loader is not None:
        # 选用模型：SWA 或 最佳 epoch
        if swa_better and swa_model is not None:
            test_model = swa_model
            logger.info("使用 SWA 模型评估测试集。")
        else:
            best_ckpt = torch.load(
                checkpoint_dir / "best_model.pt",
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(best_ckpt["model_state_dict"])
            test_model = model
            logger.info(f"使用 best epoch={best_ckpt['epoch']} 模型评估测试集。")

        test_pack = collect_predictions(test_model, test_loader, criterion, device)
        test_metrics = metrics_from_probs(
            test_pack["probs"], test_pack["labels"], threshold=best_threshold
        )
        test_metrics["loss"] = test_pack["loss"]
        # 同时记录默认阈值 0.5 的结果
        test_metrics_05 = metrics_from_probs(
            test_pack["probs"], test_pack["labels"], threshold=0.5
        )

        logger.info("=" * 60)
        logger.info(f"测试集结果 (阈值={best_threshold:.3f}):")
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
        logger.info(
            f"  对照(阈值=0.5): macF1={test_metrics_05['macro_f1']:.4f} | "
            f"F1_fake={test_metrics_05['f1_fake']:.4f}"
        )
        logger.info("=" * 60)
        best_metrics["test"] = test_metrics
        best_metrics["test_at_0_5"] = test_metrics_05

    # 去掉不可序列化字段，再写出
    serializable = {}
    for k, v in best_metrics.items():
        if isinstance(v, np.ndarray):
            continue
        serializable[k] = v
    metrics_path = checkpoint_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {metrics_path}")

    return best_metrics
