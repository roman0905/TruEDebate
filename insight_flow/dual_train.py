"""
路 C：双流融合模型的训练/评估循环。

复用 train.py 里的所有损失函数（FocalLoss / supcon_loss / xpcr_loss /
datr_aggregate / kdpe_loss）和工具（_compute_class_weights / 调度器 /
metrics_from_probs / search_best_threshold / ModelEMA）。

唯一差异：forward 接收两个 batch (ted_batch, pamd_batch)，
DataLoader.collate_fn 用 paired_collate。
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader as TorchDataLoader

import config
from insight_flow.dual_networks import DualStreamClassifier
from insight_flow.train import (
    FocalLoss,
    ModelEMA,
    _compute_class_weights,
    _build_warmup_cosine_scheduler,
    _extract_batch,
    datr_aggregate,
    kdpe_loss,
    metrics_from_probs,
    search_best_threshold,
    supcon_loss,
    xpcr_loss,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────── 双流前向 ────────────────────────────────


def _call_dual_model(model, ted_fields: dict, pamd_fields: dict, **kwargs):
    return model(ted_fields, pamd_fields, **kwargs)


def _underlying(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


# ──────────────────────────────── 训练单个 epoch ────────────────────────────────


def train_dual_one_epoch(
    model: DualStreamClassifier,
    loader: TorchDataLoader,
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
    supcon_weight: float = 0.0,
    supcon_temperature: float = 0.07,
    xpcr_weight: float = 0.0,
    kdpe_weight: float = 0.0,
    kdpe_temperature: float = 4.0,
    teacher_probs: torch.Tensor | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    use_aux = aux_loss_weight > 0
    use_supcon = supcon_weight > 0
    use_xpcr = xpcr_weight > 0
    use_kdpe = kdpe_weight > 0 and teacher_probs is not None
    need_aux_output = use_aux or use_xpcr
    use_amp = scaler is not None

    for step, paired in enumerate(loader):
        ted_batch = paired["ted"].to(device)
        pamd_batch = paired["pamd"].to(device)
        ted_fields = _extract_batch(ted_batch)
        pamd_fields = _extract_batch(pamd_batch)

        labels = pamd_fields["labels"]
        sample_idx = pamd_fields["sample_idx"]
        bs = labels.size(0)
        apply_mixup_now = use_mixup and (random.random() < mixup_prob) and bs > 1

        def _compute_loss():
            if apply_mixup_now:
                logits, features = _call_dual_model(
                    model, ted_fields, pamd_fields, return_features=True
                )
                main_loss = criterion(logits, labels)
                perm = torch.randperm(features.size(0), device=features.device)
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                mixed = lam * features + (1.0 - lam) * features[perm]
                real_model = _underlying(model)
                mixed_logits = real_model.classifier(mixed)
                mixed_loss = (
                    lam * criterion(mixed_logits, labels)
                    + (1.0 - lam) * criterion(mixed_logits, labels[perm])
                )
                total = 0.5 * main_loss + 0.5 * mixed_loss
                if use_supcon and bs >= 2:
                    total = total + supcon_weight * supcon_loss(
                        features, labels, supcon_temperature
                    )
                return total

            if need_aux_output or use_supcon:
                logits, aux_logits, aux_batch_ids, features = _call_dual_model(
                    model, ted_fields, pamd_fields,
                    return_aux=True, return_features=True,
                )
            else:
                logits = _call_dual_model(model, ted_fields, pamd_fields)
                aux_logits = aux_batch_ids = features = None

            main_loss = criterion(logits, labels)
            total = main_loss

            if use_aux and aux_logits is not None and aux_logits.numel() > 0:
                aux_labels = labels[aux_batch_ids]
                total = total + aux_loss_weight * criterion(aux_logits, aux_labels)

            if use_xpcr and aux_logits is not None and aux_logits.numel() > 0:
                total = total + xpcr_weight * xpcr_loss(aux_logits, aux_batch_ids, bs)

            if use_supcon and features is not None and bs >= 2:
                total = total + supcon_weight * supcon_loss(
                    features, labels, supcon_temperature
                )

            if use_kdpe and sample_idx is not None:
                idx = sample_idx.view(-1).long().cpu()
                t_probs = teacher_probs[idx].to(device=logits.device)
                total = total + kdpe_weight * kdpe_loss(
                    logits, t_probs, kdpe_temperature
                )

            return total

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
            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ──────────────────────────────── 评估 ────────────────────────────────


@torch.no_grad()
def collect_dual_predictions(
    model: nn.Module,
    loader: TorchDataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_datr: bool = False,
    datr_alpha: float = 0.3,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for paired in loader:
        ted_batch = paired["ted"].to(device)
        pamd_batch = paired["pamd"].to(device)
        ted_fields = _extract_batch(ted_batch)
        pamd_fields = _extract_batch(pamd_batch)
        labels = pamd_fields["labels"]

        if use_datr:
            logits, aux_logits, aux_batch_ids, _ = _call_dual_model(
                model, ted_fields, pamd_fields,
                return_aux=True, return_features=True,
            )
            logits = datr_aggregate(
                logits, aux_logits, aux_batch_ids,
                batch_size=labels.size(0),
                alpha=datr_alpha,
            )
        else:
            logits = _call_dual_model(model, ted_fields, pamd_fields)

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


# ──────────────────────────────── 主训练流程 ────────────────────────────────


def train_dual(
    model: DualStreamClassifier,
    train_loader: TorchDataLoader,
    val_loader: TorchDataLoader,
    test_loader: TorchDataLoader | None = None,
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

    # 类别权重计算需要 file_paths 接口；用 PairedDataset 内的 pamd_dataset。
    class_weights = None
    if config.USE_CLASS_WEIGHT:
        # 用 pamd 子集（与 ted 同样标签）计算
        pamd_loader_proxy = type("L", (), {
            "dataset": train_loader.dataset.pamd_dataset,
        })()
        class_weights = _compute_class_weights(
            pamd_loader_proxy, device, mode=config.CLASS_WEIGHT_MODE
        )
        logger.info(
            f"类别权重 (mode={config.CLASS_WEIGHT_MODE}): "
            f"real={class_weights[0].item():.4f}, fake={class_weights[1].item():.4f}"
        )

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
        logger.info(f"主损失: CrossEntropy(label_smoothing={config.LABEL_SMOOTHING})")

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

    ema = ModelEMA(model, decay=config.EMA_DECAY) if config.USE_EMA else None

    swa_model = None
    swa_start_epoch = max(int(epochs * config.SWA_START_RATIO), 1)

    best_val_f1 = -1.0
    best_metrics: dict = {}
    patience_counter = 0

    aux_weight = config.AUX_LOSS_WEIGHT  # PAMD 分支自带 aux head
    use_mixup = config.USE_MIXUP
    supcon_w = config.SCRA_WEIGHT if getattr(config, "USE_SCRA", False) else 0.0
    supcon_t = getattr(config, "SCRA_TEMPERATURE", 0.07)
    xpcr_w = config.XPCR_WEIGHT if getattr(config, "USE_XPCR", False) else 0.0
    use_datr = getattr(config, "USE_DATR", False)
    datr_alpha = getattr(config, "DATR_ALPHA", 0.3)
    kdpe_w = config.KDPE_WEIGHT if getattr(config, "USE_KDPE", False) else 0.0
    kdpe_t = getattr(config, "KDPE_TEMPERATURE", 4.0)

    teacher_probs = None
    if kdpe_w > 0:
        teacher_path = Path(getattr(config, "KDPE_TEACHER_PROBS_PATH", ""))
        if teacher_path.exists():
            arr = np.load(teacher_path)
            teacher_probs = torch.from_numpy(arr).float()
            if teacher_probs.dim() == 1:
                teacher_probs = torch.stack(
                    [1.0 - teacher_probs, teacher_probs], dim=-1
                )
            logger.info(
                f"KDPE 启用，teacher probs 形状 {tuple(teacher_probs.shape)} 来自 {teacher_path}"
            )
        else:
            logger.warning(
                f"KDPE 启用但 teacher probs 文件不存在: {teacher_path}，自动禁用 KDPE"
            )
            kdpe_w = 0.0

    logger.info(f"[DualStream] 开始训练: {epochs} epochs, device={device}, AMP={use_amp}")
    logger.info(f"BERT 可训练参数: {sum(p.numel() for p in bert_params):,}")
    logger.info(f"其他可训练参数: {sum(p.numel() for p in other_params):,}")
    logger.info(
        f"Scheduler: warmup+cosine | total_steps={total_steps}, "
        f"warmup_steps={int(total_steps * config.WARMUP_RATIO)}, "
        f"min_lr_ratio={config.MIN_LR_RATIO}"
    )
    logger.info(f"WeightDecay: bert={weight_decay}, other={config.WEIGHT_DECAY_OTHER}")
    logger.info(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    logger.info(f"Aux loss weight (PAMD branch): {aux_weight}")
    logger.info(f"Node dropout: {config.NODE_DROPOUT_P}, Edge dropout: {config.EDGE_DROPOUT_P}")
    logger.info(f"Mixup: {use_mixup}, EMA: {config.USE_EMA}, SWA: {config.USE_SWA}")
    logger.info(
        f"V8 modules: XPCR={xpcr_w>0} (w={xpcr_w}), DATR={use_datr} (a={datr_alpha}), "
        f"KDPE={kdpe_w>0} (w={kdpe_w}, T={kdpe_t})"
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_dual_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            grad_accum_steps=grad_accum_steps, scheduler=scheduler,
            aux_loss_weight=aux_weight,
            use_mixup=use_mixup,
            ema=ema,
            supcon_weight=supcon_w,
            supcon_temperature=supcon_t,
            xpcr_weight=xpcr_w,
            kdpe_weight=kdpe_w,
            kdpe_temperature=kdpe_t,
            teacher_probs=teacher_probs,
        )

        val_pack = collect_dual_predictions(
            model, val_loader, criterion, device,
            use_datr=use_datr, datr_alpha=datr_alpha,
        )
        val_metrics = metrics_from_probs(val_pack["probs"], val_pack["labels"], threshold=0.5)
        val_metrics["loss"] = val_pack["loss"]

        if config.USE_SWA and epoch >= swa_start_epoch:
            if swa_model is None:
                swa_model = AveragedModel(model)
                logger.info(f"  ✦ SWA 累积开始 (epoch {epoch})")
            else:
                swa_model.update_parameters(model)

        logger.info(
            f"[Dual] Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val macF1: {val_metrics['macro_f1']:.4f} | "
            f"F1_real: {val_metrics['f1_real']:.4f} | F1_fake: {val_metrics['f1_fake']:.4f} | "
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
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": {
                    k: v for k, v in val_metrics.items() if not isinstance(v, np.ndarray)
                },
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

    # ── EMA / SWA 候选 ──
    candidates = {"best": (best_val_f1, 0.5)}
    candidates_probs = {"best": (best_metrics.get("val_probs"), best_metrics.get("val_labels"))}

    if ema is not None:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.apply_to(model)
        ema_pack = collect_dual_predictions(
            model, val_loader, criterion, device,
            use_datr=use_datr, datr_alpha=datr_alpha,
        )
        ema_metrics = metrics_from_probs(ema_pack["probs"], ema_pack["labels"], threshold=0.5)
        logger.info(
            f"EMA Val | macF1@0.5: {ema_metrics['macro_f1']:.4f} | "
            f"F1_fake: {ema_metrics['f1_fake']:.4f}"
        )
        torch.save({
            "model_state_dict": ema.state_dict(),
            "val_metrics": ema_metrics,
        }, checkpoint_dir / "ema_model.pt")
        candidates["ema"] = (ema_metrics["macro_f1"], 0.5)
        candidates_probs["ema"] = (ema_pack["probs"], ema_pack["labels"])
        model.load_state_dict(backup)

    if swa_model is not None:
        swa_pack = collect_dual_predictions(
            swa_model, val_loader, criterion, device,
            use_datr=use_datr, datr_alpha=datr_alpha,
        )
        swa_metrics = metrics_from_probs(swa_pack["probs"], swa_pack["labels"], threshold=0.5)
        logger.info(
            f"SWA Val | macF1@0.5: {swa_metrics['macro_f1']:.4f} | "
            f"F1_fake: {swa_metrics['f1_fake']:.4f}"
        )
        torch.save({
            "model_state_dict": swa_model.state_dict(),
            "val_metrics": swa_metrics,
        }, checkpoint_dir / "swa_model.pt")
        candidates["swa"] = (swa_metrics["macro_f1"], 0.5)
        candidates_probs["swa"] = (swa_pack["probs"], swa_pack["labels"])

    winner_name = max(candidates.keys(), key=lambda k: candidates[k][0])
    winner_f1, _ = candidates[winner_name]
    logger.info(f"最终模型选择: {winner_name} (Val macF1@0.5={winner_f1:.4f})")

    winner_probs, winner_labels = candidates_probs[winner_name]
    if config.USE_THRESHOLD_TUNING and winner_probs is not None:
        winner_thr, _ = search_best_threshold(winner_probs, winner_labels)
        tuned = metrics_from_probs(winner_probs, winner_labels, threshold=winner_thr)
        logger.info(
            f"阈值调优: best_threshold={winner_thr:.3f} | tuned Val macF1: {tuned['macro_f1']:.4f}"
        )
    else:
        winner_thr = 0.5
        tuned = metrics_from_probs(winner_probs, winner_labels, threshold=0.5)

    for k, v in tuned.items():
        if not isinstance(v, np.ndarray):
            best_metrics[k] = v
    best_metrics["threshold"] = winner_thr
    best_metrics["winner"] = winner_name

    if test_loader is not None:
        if winner_name == "swa" and swa_model is not None:
            test_model = swa_model
        elif winner_name == "ema" and ema is not None:
            ema.apply_to(model)
            test_model = model
        else:
            best_ckpt = torch.load(
                checkpoint_dir / "best_model.pt", map_location=device, weights_only=False,
            )
            model.load_state_dict(best_ckpt["model_state_dict"])
            test_model = model

        test_pack = collect_dual_predictions(
            test_model, test_loader, criterion, device,
            use_datr=use_datr, datr_alpha=datr_alpha,
        )
        test_metrics = metrics_from_probs(
            test_pack["probs"], test_pack["labels"], threshold=winner_thr
        )
        test_metrics_05 = metrics_from_probs(
            test_pack["probs"], test_pack["labels"], threshold=0.5
        )
        test_metrics["loss"] = test_pack["loss"]

        np.save(checkpoint_dir / "val_probs.npy", winner_probs)
        np.save(checkpoint_dir / "val_labels.npy", winner_labels)
        np.save(checkpoint_dir / "test_probs.npy", test_pack["probs"])
        np.save(checkpoint_dir / "test_labels.npy", test_pack["labels"])

        logger.info("=" * 60)
        logger.info(f"[Dual] 测试集结果 ({winner_name} 模型，阈值={winner_thr:.3f}):")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1:  {test_metrics['macro_f1']:.4f}")
        logger.info(f"  F1 (Real): {test_metrics['f1_real']:.4f}")
        logger.info(f"  F1 (Fake): {test_metrics['f1_fake']:.4f}")
        logger.info(
            f"  Fake P/R:  {test_metrics['fake_precision']:.4f}/{test_metrics['fake_recall']:.4f}"
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

    serializable = {
        k: v for k, v in best_metrics.items() if not isinstance(v, np.ndarray)
    }
    with open(checkpoint_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存: {checkpoint_dir / 'metrics.json'}")

    return best_metrics
