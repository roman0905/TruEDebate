"""
TruEDebate (TED/PAMD) — 训练循环、验证与评估（V5 版本）

V5 相对 V4 的修复（基于 V4 训练 log 复盘）：
1. 去掉 Focal Loss × class_weight 双重加权（V4 5 epoch 即过拟合）。
2. 引入 sqrt 模式类别权重，缓解 train/val 类比例失配。
3. 每个 epoch 都做阈值调优，用 tuned macF1 选 best_model。
4. 修复 SWA 时机：start_ratio 0.6 → 0.3 + patience 8 → 12，确保真正触发。
5. 引入 EMA 全程影子模型，最终在 best / EMA / SWA 三者间取最优。
6. 引入 Manifold Mixup（在 classifier 输入做混合），强正则化。
7. 区分 BERT / 其他参数的 weight_decay，加强头部正则。
"""

import json
import logging
import math
import random
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
    """支持类别权重的 Focal Loss。V5 默认关闭（与 class_weight 叠加会过冲）。"""

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


# ──────────────────────────────── EMA ────────────────────────────────


class ModelEMA:
    """模型参数的指数滑动平均影子。V5 新增：全程维护，结束后与 best/SWA 三选一。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        # 前期 warmup：1 - 1/(n+1)，避免初始几步剧烈震荡。
        effective_decay = min(self.decay, (self.num_updates + 1) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            shadow = self.shadow[k]
            if v.dtype.is_floating_point:
                shadow.mul_(effective_decay).add_(v.detach(), alpha=1.0 - effective_decay)
            else:
                shadow.copy_(v.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


# ──────────────────────────────── 前向 helper ────────────────────────────────


def _extract_batch(batch_data, labels_required: bool = True):
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


def _call_model(model: nn.Module, fields: dict, **forward_kwargs):
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
        **forward_kwargs,
    )


def _underlying(model: nn.Module) -> nn.Module:
    """剥离 SWA/EMA 等包装拿到真模型，便于读 use_aux_loss 等属性。"""
    return model.module if hasattr(model, "module") else model


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
    use_mixup: bool = False,
    mixup_alpha: float = config.MIXUP_ALPHA,
    mixup_prob: float = config.MIXUP_PROB,
    ema: ModelEMA | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    use_aux = aux_loss_weight > 0 and getattr(model, "use_aux_loss", False)
    use_amp = scaler is not None

    for step, batch_data in enumerate(loader):
        batch_data = batch_data.to(device)
        fields = _extract_batch(batch_data)
        labels = fields["labels"]
        apply_mixup_now = use_mixup and (random.random() < mixup_prob) and labels.size(0) > 1

        def _compute_loss():
            if apply_mixup_now:
                # Manifold Mixup：在分类器输入处混合两条样本的拼接特征。
                logits, features = _call_model(model, fields, return_features=True)
                # 同时计算正常的 aux 损失（无 mixup），保留视角监督。
                main_loss = criterion(logits, labels)
                if use_aux:
                    # 取 aux 信号，需要额外把 features 解构得到 dense_nodes；
                    # 简化处理：mixup 批次只用主损失 + mixed loss，不再附加 aux。
                    pass
                # 生成 mixup 排列
                perm = torch.randperm(features.size(0), device=features.device)
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                mixed = lam * features + (1.0 - lam) * features[perm]
                # 用同样的分类头计算混合后的 logits。
                real_model = _underlying(model)
                mixed_logits = real_model.classifier(mixed)
                mixed_loss = (
                    lam * criterion(mixed_logits, labels)
                    + (1.0 - lam) * criterion(mixed_logits, labels[perm])
                )
                return 0.5 * main_loss + 0.5 * mixed_loss
            # 非 mixup 路径：常规 + aux
            if use_aux:
                logits, aux_logits, aux_batch_ids = _call_model(
                    model, fields, return_aux=True
                )
                main_loss = criterion(logits, labels)
                if aux_logits.numel() > 0:
                    aux_labels = labels[aux_batch_ids]
                    aux_loss = criterion(aux_logits, aux_labels)
                    return main_loss + aux_loss_weight * aux_loss
                return main_loss
            logits = _call_model(model, fields)
            return criterion(logits, labels)

        if use_amp:
            with autocast(str(device).split(":")[0]):
                loss = _compute_loss() / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            loss = _compute_loss() / grad_accum_steps
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
            # EMA 在每个 optimizer.step 之后更新。
            if ema is not None:
                ema.update(model)

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


# ──────────────────────────────── 类别权重 ────────────────────────────────


def _compute_class_weights(
    loader: DataLoader,
    device: torch.device,
    mode: str = "inverse",
) -> torch.Tensor:
    """根据 mode 计算类别权重。
    - inverse: total / (2*counts)，最强；
    - sqrt:    sqrt(total / (2*counts))，温和；
    - balanced: 全 1.0。
    """
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
    inv = total / (2.0 * np.maximum(counts, 1))
    if mode == "balanced":
        weights = np.ones_like(inv)
    elif mode == "sqrt":
        weights = np.sqrt(inv)
    else:
        weights = inv
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


# ──────────────────────────────── 主训练流程 ────────────────────────────────


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
        class_weights = _compute_class_weights(
            train_loader, device, mode=config.CLASS_WEIGHT_MODE
        )
        logger.info(
            f"类别权重 (mode={config.CLASS_WEIGHT_MODE}): "
            f"real={class_weights[0].item():.4f}, fake={class_weights[1].item():.4f}"
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

    # 优化器：BERT 和其他参数分别用不同 LR 和 WD。
    bert_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bert" in name:
                bert_params.append(param)
            else:
                other_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": lr * config.BERT_LR_FACTOR, "weight_decay": weight_decay},
            {"params": other_params, "lr": lr, "weight_decay": config.WEIGHT_DECAY_OTHER},
        ],
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(grad_accum_steps, 1))
    total_steps = max(steps_per_epoch * epochs, 1)
    scheduler = _build_warmup_cosine_scheduler(optimizer, total_steps)
    scaler = GradScaler(str(device)) if (use_amp and device.type == "cuda") else None

    # EMA
    ema = ModelEMA(model, decay=config.EMA_DECAY) if config.USE_EMA else None

    # SWA
    swa_model = None
    swa_start_epoch = max(int(epochs * config.SWA_START_RATIO), 1)

    best_val_f1 = -1.0
    best_metrics: dict = {}
    best_threshold = 0.5
    patience_counter = 0

    aux_weight = config.AUX_LOSS_WEIGHT if getattr(model, "use_aux_loss", False) else 0.0
    use_mixup = config.USE_MIXUP

    logger.info(f"开始训练: {epochs} epochs, device={device}, AMP={use_amp}")
    logger.info(f"BERT 可训练参数: {sum(p.numel() for p in bert_params):,}")
    logger.info(f"其他可训练参数: {sum(p.numel() for p in other_params):,}")
    logger.info(
        f"Scheduler: warmup+cosine | total_steps={total_steps}, "
        f"warmup_steps={int(total_steps * config.WARMUP_RATIO)}, "
        f"min_lr_ratio={config.MIN_LR_RATIO}"
    )
    logger.info(f"WeightDecay: bert={weight_decay}, other={config.WEIGHT_DECAY_OTHER}")
    logger.info(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    logger.info(f"Aux loss weight: {aux_weight}")
    logger.info(f"Node dropout: {config.NODE_DROPOUT_P}, Edge dropout: {config.EDGE_DROPOUT_P}")
    logger.info(
        f"Mixup: enabled={use_mixup}, alpha={config.MIXUP_ALPHA}, prob={config.MIXUP_PROB}"
    )
    logger.info(
        f"EMA: enabled={config.USE_EMA}, decay={config.EMA_DECAY}"
    )
    logger.info(
        f"SWA: enabled={config.USE_SWA}, start_epoch={swa_start_epoch}"
    )
    logger.info(f"Threshold tuning: enabled={config.USE_THRESHOLD_TUNING}")
    logger.info(f"Grad clip max norm: {config.GRAD_CLIP_MAX_NORM}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            grad_accum_steps=grad_accum_steps, scheduler=scheduler,
            aux_loss_weight=aux_weight,
            use_mixup=use_mixup,
            ema=ema,
        )

        # ── 验证：原始模型，仅在 thr=0.5 计算指标，用 raw macF1 选 best ──
        val_pack = collect_predictions(model, val_loader, criterion, device)
        val_metrics = metrics_from_probs(
            val_pack["probs"], val_pack["labels"], threshold=0.5
        )
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
            f"F1_real: {val_metrics['f1_real']:.4f} | F1_fake: {val_metrics['f1_fake']:.4f} | "
            f"Fake P/R: {val_metrics['fake_precision']:.4f}/"
            f"{val_metrics['fake_recall']:.4f} | "
            f"Pred fake: {val_metrics['pred_fake']}/{val_metrics['true_fake']}"
        )

        # 选 best（基于 raw macF1@0.5）
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics.copy()
            best_metrics["epoch"] = epoch
            best_metrics["val_probs"] = val_pack["probs"]
            best_metrics["val_labels"] = val_pack["labels"]

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": {k: v for k, v in val_metrics.items() if not isinstance(v, np.ndarray)},
            }, checkpoint_dir / "best_model.pt")
            logger.info(f"  ★ 最佳模型已保存 (macF1@0.5={best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping 触发: 连续 {patience_counter} 轮无提升，停止于 epoch {epoch}"
                )
                break

    # ── 后训练：评估 EMA 和 SWA ──
    candidates = {"best": (best_val_f1, 0.5, "best_model.pt")}

    if ema is not None:
        # 创建一个临时副本评估 EMA
        ema_state_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.apply_to(model)
        ema_pack = collect_predictions(model, val_loader, criterion, device)
        ema_metrics = metrics_from_probs(ema_pack["probs"], ema_pack["labels"], threshold=0.5)
        ema_metrics["loss"] = ema_pack["loss"]
        logger.info(
            f"EMA Val | macF1@0.5: {ema_metrics['macro_f1']:.4f} | "
            f"F1_fake: {ema_metrics['f1_fake']:.4f}"
        )
        torch.save({
            "model_state_dict": ema.state_dict(),
            "val_metrics": ema_metrics,
        }, checkpoint_dir / "ema_model.pt")
        candidates["ema"] = (ema_metrics["macro_f1"], 0.5, "ema_model.pt")
        # 同时把 EMA probs 留下供 candidate 阈值搜索
        candidates_probs = {"ema": (ema_pack["probs"], ema_pack["labels"])}
        # 恢复模型权重
        model.load_state_dict(ema_state_backup)
    else:
        candidates_probs = {}

    if swa_model is not None:
        swa_pack = collect_predictions(swa_model, val_loader, criterion, device)
        swa_metrics = metrics_from_probs(swa_pack["probs"], swa_pack["labels"], threshold=0.5)
        swa_metrics["loss"] = swa_pack["loss"]
        logger.info(
            f"SWA Val | macF1@0.5: {swa_metrics['macro_f1']:.4f} | "
            f"F1_fake: {swa_metrics['f1_fake']:.4f}"
        )
        torch.save({
            "model_state_dict": swa_model.state_dict(),
            "val_metrics": swa_metrics,
        }, checkpoint_dir / "swa_model.pt")
        candidates["swa"] = (swa_metrics["macro_f1"], 0.5, "swa_model.pt")
        candidates_probs["swa"] = (swa_pack["probs"], swa_pack["labels"])

    # 把 best 的 probs/labels 也加入 candidates_probs
    if "val_probs" in best_metrics:
        candidates_probs["best"] = (best_metrics["val_probs"], best_metrics["val_labels"])

    # 选最佳 candidate（仍按 raw macF1@0.5）
    winner_name = max(candidates.keys(), key=lambda k: candidates[k][0])
    winner_f1, _, winner_ckpt = candidates[winner_name]
    logger.info(
        f"最终模型选择: {winner_name} (Val macF1@0.5={winner_f1:.4f})"
    )

    # ── 在 winner 的 val probs 上搜索阈值 ──
    winner_probs, winner_labels = candidates_probs[winner_name]
    if config.USE_THRESHOLD_TUNING:
        winner_thr, winner_tuned = search_best_threshold(winner_probs, winner_labels)
        winner_val_metrics = metrics_from_probs(winner_probs, winner_labels, threshold=winner_thr)
        logger.info(
            f"阈值调优: best_threshold={winner_thr:.3f} | "
            f"tuned Val macF1: {winner_val_metrics['macro_f1']:.4f} | "
            f"F1_fake: {winner_val_metrics['f1_fake']:.4f}"
        )
    else:
        winner_thr = 0.5
        winner_val_metrics = metrics_from_probs(winner_probs, winner_labels, threshold=0.5)

    # 更新 best_metrics（保留 epoch 信息）
    epoch_field = best_metrics.get("epoch", "n/a")
    for k, v in winner_val_metrics.items():
        if not isinstance(v, np.ndarray):
            best_metrics[k] = v
    best_metrics["epoch"] = epoch_field
    best_metrics["threshold"] = winner_thr
    best_metrics["winner"] = winner_name

    # ── 测试集评估 ──
    if test_loader is not None:
        if winner_name == "swa" and swa_model is not None:
            test_model = swa_model
        elif winner_name == "ema" and ema is not None:
            ema.apply_to(model)
            test_model = model
        else:
            best_ckpt = torch.load(
                checkpoint_dir / "best_model.pt",
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(best_ckpt["model_state_dict"])
            test_model = model

        test_pack = collect_predictions(test_model, test_loader, criterion, device)
        test_metrics = metrics_from_probs(
            test_pack["probs"], test_pack["labels"], threshold=winner_thr
        )
        test_metrics["loss"] = test_pack["loss"]
        test_metrics_05 = metrics_from_probs(
            test_pack["probs"], test_pack["labels"], threshold=0.5
        )

        # V6：把 winner 的 val/test probs 落盘，供 main_ensemble.py 聚合使用。
        np.save(checkpoint_dir / "val_probs.npy", winner_probs)
        np.save(checkpoint_dir / "val_labels.npy", winner_labels)
        np.save(checkpoint_dir / "test_probs.npy", test_pack["probs"])
        np.save(checkpoint_dir / "test_labels.npy", test_pack["labels"])
        logger.info(f"概率已保存: {checkpoint_dir}/{{val,test}}_probs.npy")

        logger.info("=" * 60)
        logger.info(f"测试集结果 ({winner_name} 模型，阈值={winner_thr:.3f}):")
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
        best_metrics["winner"] = winner_name
        best_metrics["winner_threshold"] = winner_thr

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
