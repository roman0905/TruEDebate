"""
R-TED — 训练循环、验证与评估
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader

import config
from insight_flow.networks import FocalLoss, TEDClassifier

logger = logging.getLogger(__name__)


def _compute_binary_metrics(all_labels: np.ndarray, all_preds: np.ndarray) -> dict[str, float]:
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_real = f1_score(all_labels, all_preds, average="binary", pos_label=0, zero_division=0)
    f1_fake = f1_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_real": f1_real,
        "f1_fake": f1_fake,
    }


def _search_best_threshold(all_labels: np.ndarray, fake_probs: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = _compute_binary_metrics(all_labels, (fake_probs >= 0.5).astype(int))
    for threshold in np.linspace(0.2, 0.8, 121):
        preds = (fake_probs >= threshold).astype(int)
        metrics = _compute_binary_metrics(all_labels, preds)
        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
        elif metrics["macro_f1"] == best_metrics["macro_f1"]:
            if metrics["f1_fake"] > best_metrics["f1_fake"]:
                best_threshold = float(threshold)
                best_metrics = metrics
            elif metrics["f1_fake"] == best_metrics["f1_fake"] and metrics["accuracy"] > best_metrics["accuracy"]:
                best_threshold = float(threshold)
                best_metrics = metrics
    return best_threshold, best_metrics


def _compute_reliability_loss(alpha: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    valid_mask = (targets >= 0.0) & (targets <= 1.0)
    if not torch.any(valid_mask):
        return alpha.new_zeros(())
    device_type = alpha.device.type if alpha.device.type in {"cuda", "cpu"} else "cpu"
    with autocast(device_type=device_type, enabled=False):
        return F.binary_cross_entropy(alpha[valid_mask].float(), targets[valid_mask].float())


def _compute_consistency_loss(
    logits: torch.Tensor,
    td_pred: torch.Tensor,
    cs_pred: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for weak_pred in (td_pred, cs_pred):
        valid_mask = (weak_pred == 0.0) | (weak_pred == 1.0)
        if torch.any(valid_mask):
            losses.append(
                F.cross_entropy(
                    logits[valid_mask],
                    weak_pred[valid_mask].long(),
                )
            )
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def _compute_per_sample_cls_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    if isinstance(criterion, FocalLoss):
        return criterion.per_sample_loss(logits, labels)

    if isinstance(criterion, nn.CrossEntropyLoss):
        return F.cross_entropy(
            logits,
            labels,
            weight=criterion.weight,
            label_smoothing=criterion.label_smoothing,
            reduction="none",
        )

    loss = criterion(logits, labels)
    if loss.dim() == 0:
        return loss.expand(labels.shape[0])
    return loss


def _safe_probability(value: float) -> float:
    return max(1e-4, min(float(value), 1.0 - 1e-4))


def _compute_causal_debias_terms(
    logits: torch.Tensor,
    labels: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    use_causal_debias: bool,
    causal_prior: float,
    causal_conf_weight: float,
    causal_struct_weight: float,
    causal_min_weight: float,
    causal_max_weight: float,
) -> dict[str, torch.Tensor]:
    batch_size = labels.shape[0]
    ones = logits.new_ones(batch_size)
    zeros = logits.new_zeros(())
    prior_value = _safe_probability(causal_prior)
    prior_vec = logits.new_full((batch_size,), prior_value)
    default_stats = {
        "sample_weights": ones,
        "causal_kl_loss": zeros,
        "causal_posterior": prior_vec,
        "cls_confidence": prior_vec,
        "structure_likelihood": prior_vec,
    }
    if not use_causal_debias:
        return default_stats

    probs = logits.softmax(dim=-1)
    cls_confidence = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-4, 1.0 - 1e-4)

    structure_likelihood = outputs.get("structure_likelihood")
    if structure_likelihood is None:
        structure_likelihood = logits.new_full((batch_size,), prior_value)
    else:
        structure_likelihood = structure_likelihood.view(-1)
        if structure_likelihood.shape[0] != batch_size:
            fixed = logits.new_full((batch_size,), prior_value)
            fixed[: min(batch_size, structure_likelihood.shape[0])] = structure_likelihood[:batch_size]
            structure_likelihood = fixed
        structure_likelihood = structure_likelihood.clamp(1e-4, 1.0 - 1e-4)

    prior = logits.new_tensor(prior_value)
    prior_logit = torch.log(prior / (1.0 - prior))
    posterior_logit = (
        prior_logit
        + causal_conf_weight * torch.logit(cls_confidence)
        + causal_struct_weight * torch.logit(structure_likelihood)
    )
    posterior = torch.sigmoid(posterior_logit).clamp(1e-4, 1.0 - 1e-4)

    sample_weights = posterior.detach()
    sample_weights = sample_weights / sample_weights.mean().clamp(min=1e-6)
    sample_weights = sample_weights.clamp(min=causal_min_weight, max=causal_max_weight)
    sample_weights = sample_weights / sample_weights.mean().clamp(min=1e-6)

    causal_kl_loss = (
        posterior * (posterior.log() - prior.log())
        + (1.0 - posterior) * ((1.0 - posterior).log() - (1.0 - prior).log())
    ).mean()

    return {
        "sample_weights": sample_weights,
        "causal_kl_loss": causal_kl_loss,
        "causal_posterior": posterior,
        "cls_confidence": cls_confidence,
        "structure_likelihood": structure_likelihood,
    }


def _prepare_model_inputs(batch_data) -> dict[str, torch.Tensor]:
    return {
        "node_input_ids": batch_data.node_input_ids,
        "node_attention_mask": batch_data.node_attention_mask,
        "news_input_ids": batch_data.news_input_ids.view(batch_data.y.shape[0], -1),
        "news_attention_mask": batch_data.news_attention_mask.view(batch_data.y.shape[0], -1),
        "node_type_ids": batch_data.node_type_ids,
        "speaker_role_ids": batch_data.speaker_role_ids,
        "node_reliability": batch_data.node_reliability,
        "edge_index": batch_data.edge_index,
        "edge_type": batch_data.edge_type,
        "source_bucket": batch_data.source_bucket.view(batch_data.y.shape[0]),
        "time_features": batch_data.time_features.view(batch_data.y.shape[0], -1),
        "teacher_features": batch_data.teacher_features.view(batch_data.y.shape[0], -1),
        "batch": batch_data.batch,
    }


def _compute_loss_terms(
    outputs: dict[str, torch.Tensor],
    batch_data,
    labels: torch.Tensor,
    criterion: nn.Module,
    rationale_loss_weight: float,
    consistency_loss_weight: float,
    structure_loss_weight: float,
    use_causal_debias: bool,
    causal_kl_weight: float,
    causal_prior: float,
    causal_conf_weight: float,
    causal_struct_weight: float,
    causal_min_weight: float,
    causal_max_weight: float,
) -> dict[str, torch.Tensor]:
    logits = outputs["logits"]
    per_sample_cls_loss = _compute_per_sample_cls_loss(logits, labels, criterion)
    causal_terms = _compute_causal_debias_terms(
        logits=logits,
        labels=labels,
        outputs=outputs,
        use_causal_debias=use_causal_debias,
        causal_prior=causal_prior,
        causal_conf_weight=causal_conf_weight,
        causal_struct_weight=causal_struct_weight,
        causal_min_weight=causal_min_weight,
        causal_max_weight=causal_max_weight,
    )
    cls_loss = (per_sample_cls_loss * causal_terms["sample_weights"]).mean()

    rel_loss = logits.new_zeros(())
    if rationale_loss_weight > 0:
        rel_td = _compute_reliability_loss(outputs["alpha_td"], batch_data.td_acc)
        rel_cs = _compute_reliability_loss(outputs["alpha_cs"], batch_data.cs_acc)
        rel_loss = rel_td + rel_cs

    consistency_loss = logits.new_zeros(())
    if consistency_loss_weight > 0:
        consistency_loss = _compute_consistency_loss(
            logits,
            batch_data.td_pred,
            batch_data.cs_pred,
        )

    structure_loss = outputs.get("structure_loss", logits.new_zeros(()))

    total_loss = (
        cls_loss
        + rationale_loss_weight * rel_loss
        + consistency_loss_weight * consistency_loss
        + structure_loss_weight * structure_loss
        + causal_kl_weight * causal_terms["causal_kl_loss"]
    )
    return {
        "logits": logits,
        "loss": total_loss,
        "cls_loss": cls_loss,
        "cls_raw_loss": per_sample_cls_loss.mean(),
        "rel_loss": rel_loss,
        "consistency_loss": consistency_loss,
        "structure_loss": structure_loss,
        "causal_kl_loss": causal_terms["causal_kl_loss"],
        "sample_weight_mean": causal_terms["sample_weights"].mean(),
        "causal_posterior_mean": causal_terms["causal_posterior"].mean(),
        "cls_confidence_mean": causal_terms["cls_confidence"].mean(),
        "structure_likelihood_mean": causal_terms["structure_likelihood"].mean(),
    }


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
    rationale_loss_weight: float = config.RTED_RELIABILITY_LOSS_WEIGHT,
    consistency_loss_weight: float = config.RTED_CONSISTENCY_LOSS_WEIGHT,
    structure_loss_weight: float = config.EVITED_STRUCTURE_LOSS_WEIGHT,
    use_causal_debias: bool = config.EVITED_USE_CAUSAL_DEBIAS,
    causal_kl_weight: float = config.EVITED_CAUSAL_KL_WEIGHT,
    causal_prior: float = config.EVITED_CAUSAL_PRIOR,
    causal_conf_weight: float = config.EVITED_CAUSAL_CONF_WEIGHT,
    causal_struct_weight: float = config.EVITED_CAUSAL_STRUCT_WEIGHT,
    causal_min_weight: float = config.EVITED_CAUSAL_MIN_WEIGHT,
    causal_max_weight: float = config.EVITED_CAUSAL_MAX_WEIGHT,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_rel_loss = 0.0
    total_consistency_loss = 0.0
    total_structure_loss = 0.0
    total_causal_kl_loss = 0.0
    total_sample_weight = 0.0
    total_causal_posterior = 0.0
    total_cls_confidence = 0.0
    total_structure_likelihood = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch_data in enumerate(loader):
        batch_data = batch_data.to(device)
        labels = batch_data.y
        model_inputs = _prepare_model_inputs(batch_data)
        use_amp = scaler is not None

        if use_amp:
            with autocast(str(device).split(":")[0]):
                outputs = model(**model_inputs)
                loss_terms = _compute_loss_terms(
                    outputs=outputs,
                    batch_data=batch_data,
                    labels=labels,
                    criterion=criterion,
                    rationale_loss_weight=rationale_loss_weight,
                    consistency_loss_weight=consistency_loss_weight,
                    structure_loss_weight=structure_loss_weight,
                    use_causal_debias=use_causal_debias,
                    causal_kl_weight=causal_kl_weight,
                    causal_prior=causal_prior,
                    causal_conf_weight=causal_conf_weight,
                    causal_struct_weight=causal_struct_weight,
                    causal_min_weight=causal_min_weight,
                    causal_max_weight=causal_max_weight,
                )
                loss = loss_terms["loss"] / grad_accum_steps
        else:
            outputs = model(**model_inputs)
            loss_terms = _compute_loss_terms(
                outputs=outputs,
                batch_data=batch_data,
                labels=labels,
                criterion=criterion,
                rationale_loss_weight=rationale_loss_weight,
                consistency_loss_weight=consistency_loss_weight,
                structure_loss_weight=structure_loss_weight,
                use_causal_debias=use_causal_debias,
                causal_kl_weight=causal_kl_weight,
                causal_prior=causal_prior,
                causal_conf_weight=causal_conf_weight,
                causal_struct_weight=causal_struct_weight,
                causal_min_weight=causal_min_weight,
                causal_max_weight=causal_max_weight,
            )
            loss = loss_terms["loss"] / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

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
        total_cls_loss += loss_terms["cls_loss"].item()
        total_rel_loss += loss_terms["rel_loss"].item()
        total_consistency_loss += loss_terms["consistency_loss"].item()
        total_structure_loss += loss_terms["structure_loss"].item()
        total_causal_kl_loss += loss_terms["causal_kl_loss"].item()
        total_sample_weight += loss_terms["sample_weight_mean"].item()
        total_causal_posterior += loss_terms["causal_posterior_mean"].item()
        total_cls_confidence += loss_terms["cls_confidence_mean"].item()
        total_structure_likelihood += loss_terms["structure_likelihood_mean"].item()
        num_batches += 1

    denom = max(num_batches, 1)
    return {
        "loss": total_loss / denom,
        "cls_loss": total_cls_loss / denom,
        "rel_loss": total_rel_loss / denom,
        "consistency_loss": total_consistency_loss / denom,
        "structure_loss": total_structure_loss / denom,
        "causal_kl_loss": total_causal_kl_loss / denom,
        "sample_weight_mean": total_sample_weight / denom,
        "causal_posterior": total_causal_posterior / denom,
        "cls_confidence": total_cls_confidence / denom,
        "structure_likelihood": total_structure_likelihood / denom,
    }


@torch.no_grad()
def evaluate(
    model: TEDClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = config.USE_AMP,
    rationale_loss_weight: float = config.RTED_RELIABILITY_LOSS_WEIGHT,
    consistency_loss_weight: float = config.RTED_CONSISTENCY_LOSS_WEIGHT,
    structure_loss_weight: float = config.EVITED_STRUCTURE_LOSS_WEIGHT,
    use_causal_debias: bool = config.EVITED_USE_CAUSAL_DEBIAS,
    causal_kl_weight: float = config.EVITED_CAUSAL_KL_WEIGHT,
    causal_prior: float = config.EVITED_CAUSAL_PRIOR,
    causal_conf_weight: float = config.EVITED_CAUSAL_CONF_WEIGHT,
    causal_struct_weight: float = config.EVITED_CAUSAL_STRUCT_WEIGHT,
    causal_min_weight: float = config.EVITED_CAUSAL_MIN_WEIGHT,
    causal_max_weight: float = config.EVITED_CAUSAL_MAX_WEIGHT,
    tune_threshold: bool = False,
    decision_threshold: float | None = None,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_rel_loss = 0.0
    total_consistency_loss = 0.0
    total_structure_loss = 0.0
    total_causal_kl_loss = 0.0
    total_sample_weight = 0.0
    total_causal_posterior = 0.0
    total_cls_confidence = 0.0
    total_structure_likelihood = 0.0
    all_preds = []
    all_labels = []
    all_fake_probs = []

    amp_enabled = use_amp and device.type == "cuda"

    for batch_data in loader:
        batch_data = batch_data.to(device)
        labels = batch_data.y
        model_inputs = _prepare_model_inputs(batch_data)

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(**model_inputs)
            loss_terms = _compute_loss_terms(
                outputs=outputs,
                batch_data=batch_data,
                labels=labels,
                criterion=criterion,
                rationale_loss_weight=rationale_loss_weight,
                consistency_loss_weight=consistency_loss_weight,
                structure_loss_weight=structure_loss_weight,
                use_causal_debias=use_causal_debias,
                causal_kl_weight=causal_kl_weight,
                causal_prior=causal_prior,
                causal_conf_weight=causal_conf_weight,
                causal_struct_weight=causal_struct_weight,
                causal_min_weight=causal_min_weight,
                causal_max_weight=causal_max_weight,
            )

        logits = loss_terms["logits"]
        total_loss += loss_terms["loss"].item()
        total_cls_loss += loss_terms["cls_loss"].item()
        total_rel_loss += loss_terms["rel_loss"].item()
        total_consistency_loss += loss_terms["consistency_loss"].item()
        total_structure_loss += loss_terms["structure_loss"].item()
        total_causal_kl_loss += loss_terms["causal_kl_loss"].item()
        total_sample_weight += loss_terms["sample_weight_mean"].item()
        total_causal_posterior += loss_terms["causal_posterior_mean"].item()
        total_cls_confidence += loss_terms["cls_confidence_mean"].item()
        total_structure_likelihood += loss_terms["structure_likelihood_mean"].item()

        preds = logits.argmax(dim=-1).cpu().numpy()
        fake_probs = logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_fake_probs.extend(fake_probs.tolist())

    num_batches = max(len(loader), 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_fake_probs = np.array(all_fake_probs)

    raw_metrics = _compute_binary_metrics(all_labels, all_preds)
    chosen_threshold = decision_threshold
    tuned_metrics = raw_metrics
    if tune_threshold:
        chosen_threshold, tuned_metrics = _search_best_threshold(all_labels, all_fake_probs)
    elif decision_threshold is not None:
        tuned_metrics = _compute_binary_metrics(
            all_labels,
            (all_fake_probs >= decision_threshold).astype(int),
        )

    return {
        "loss": total_loss / num_batches,
        "cls_loss": total_cls_loss / num_batches,
        "rel_loss": total_rel_loss / num_batches,
        "consistency_loss": total_consistency_loss / num_batches,
        "structure_loss": total_structure_loss / num_batches,
        "causal_kl_loss": total_causal_kl_loss / num_batches,
        "sample_weight_mean": total_sample_weight / num_batches,
        "causal_posterior": total_causal_posterior / num_batches,
        "cls_confidence": total_cls_confidence / num_batches,
        "structure_likelihood": total_structure_likelihood / num_batches,
        "accuracy": tuned_metrics["accuracy"],
        "macro_f1": tuned_metrics["macro_f1"],
        "f1_real": tuned_metrics["f1_real"],
        "f1_fake": tuned_metrics["f1_fake"],
        "accuracy_raw": raw_metrics["accuracy"],
        "macro_f1_raw": raw_metrics["macro_f1"],
        "f1_real_raw": raw_metrics["f1_real"],
        "f1_fake_raw": raw_metrics["f1_fake"],
        "decision_threshold": chosen_threshold if chosen_threshold is not None else 0.5,
    }


def _estimate_class_weights(dataset, device: torch.device) -> torch.Tensor | None:
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
    rationale_loss_weight: float = config.RTED_RELIABILITY_LOSS_WEIGHT,
    consistency_loss_weight: float = config.RTED_CONSISTENCY_LOSS_WEIGHT,
    structure_loss_weight: float = config.EVITED_STRUCTURE_LOSS_WEIGHT,
    use_causal_debias: bool = config.EVITED_USE_CAUSAL_DEBIAS,
    causal_kl_weight: float = config.EVITED_CAUSAL_KL_WEIGHT,
    causal_prior: float = config.EVITED_CAUSAL_PRIOR,
    causal_conf_weight: float = config.EVITED_CAUSAL_CONF_WEIGHT,
    causal_struct_weight: float = config.EVITED_CAUSAL_STRUCT_WEIGHT,
    causal_min_weight: float = config.EVITED_CAUSAL_MIN_WEIGHT,
    causal_max_weight: float = config.EVITED_CAUSAL_MAX_WEIGHT,
    checkpoint_dir: str | Path = config.CHECKPOINT_DIR,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    class_weights = None
    if use_class_weight and hasattr(train_loader, "dataset"):
        class_weights = _estimate_class_weights(train_loader.dataset, device)

    if config.USE_FOCAL_LOSS:
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=config.FOCAL_GAMMA,
            label_smoothing=label_smoothing,
        )
        logger.info(
            "Using FocalLoss(gamma=%s, label_smoothing=%s)",
            config.FOCAL_GAMMA,
            label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        logger.info(
            "Using CrossEntropyLoss(label_smoothing=%s, class_weight=%s)",
            label_smoothing,
            class_weights.tolist() if class_weights is not None else None,
        )

    optimizer, bert_params, other_params = _build_optimizer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        bert_lr_factor=bert_lr_factor,
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, config.GRAD_ACCUM_STEPS))
    total_update_steps = steps_per_epoch * epochs
    scheduler, warmup_steps = _build_scheduler(
        optimizer,
        total_update_steps=total_update_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )

    scaler = GradScaler("cuda") if use_amp and device.type == "cuda" else None
    logger.info(
        "BERT params: %s | Other params: %s",
        f"{sum(p.numel() for p in bert_params):,}",
        f"{sum(p.numel() for p in other_params):,}",
    )
    logger.info(
        "Scheduler: warmup+cosine | total_steps=%s, warmup_steps=%s, min_lr_ratio=%s",
        total_update_steps,
        warmup_steps,
        min_lr_ratio,
    )
    logger.info("Early stopping patience: %s", early_stopping_patience)
    logger.info("R-TED reliability loss weight: %s", rationale_loss_weight)
    logger.info("R-TED consistency loss weight: %s", consistency_loss_weight)
    logger.info("EviTED structure loss weight: %s", structure_loss_weight)
    logger.info(
        "EviTED causal debias: enabled=%s, lambda_kl=%s, prior=%s, conf_w=%s, struct_w=%s, weight_clip=[%s,%s]",
        use_causal_debias,
        causal_kl_weight,
        causal_prior,
        causal_conf_weight,
        causal_struct_weight,
        causal_min_weight,
        causal_max_weight,
    )

    best_metrics = {}
    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_ckpt_path = checkpoint_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            grad_accum_steps=config.GRAD_ACCUM_STEPS,
            scheduler=scheduler,
            grad_clip_max_norm=grad_clip_max_norm,
            rationale_loss_weight=rationale_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            structure_loss_weight=structure_loss_weight,
            use_causal_debias=use_causal_debias,
            causal_kl_weight=causal_kl_weight,
            causal_prior=causal_prior,
            causal_conf_weight=causal_conf_weight,
            causal_struct_weight=causal_struct_weight,
            causal_min_weight=causal_min_weight,
            causal_max_weight=causal_max_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            rationale_loss_weight=rationale_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            structure_loss_weight=structure_loss_weight,
            use_causal_debias=use_causal_debias,
            causal_kl_weight=causal_kl_weight,
            causal_prior=causal_prior,
            causal_conf_weight=causal_conf_weight,
            causal_struct_weight=causal_struct_weight,
            causal_min_weight=causal_min_weight,
            causal_max_weight=causal_max_weight,
            tune_threshold=True,
        )

        logger.info(
            "Epoch %3d/%d | Train Loss: %.4f | Train CE: %.4f | Train Rel: %.4f | Train Cons: %.4f | Train Struct: %.4f | "
            "Train KL: %.4f | Train q: %.4f | Train S: %.4f | "
            "Val Loss: %.4f | Val CE: %.4f | Val Rel: %.4f | Val Cons: %.4f | Val Struct: %.4f | "
            "Val KL: %.4f | Val q: %.4f | Val S: %.4f | "
            "Val Acc(raw/cal): %.4f/%.4f | Val macF1(raw/cal): %.4f/%.4f | "
            "Val F1_real(raw/cal): %.4f/%.4f | Val F1_fake(raw/cal): %.4f/%.4f | Th=%.3f",
            epoch,
            epochs,
            train_stats["loss"],
            train_stats["cls_loss"],
            train_stats["rel_loss"],
            train_stats["consistency_loss"],
            train_stats["structure_loss"],
            train_stats["causal_kl_loss"],
            train_stats["causal_posterior"],
            train_stats["structure_likelihood"],
            val_metrics["loss"],
            val_metrics["cls_loss"],
            val_metrics["rel_loss"],
            val_metrics["consistency_loss"],
            val_metrics["structure_loss"],
            val_metrics["causal_kl_loss"],
            val_metrics["causal_posterior"],
            val_metrics["structure_likelihood"],
            val_metrics["accuracy_raw"],
            val_metrics["accuracy"],
            val_metrics["macro_f1_raw"],
            val_metrics["macro_f1"],
            val_metrics["f1_real_raw"],
            val_metrics["f1_real"],
            val_metrics["f1_fake_raw"],
            val_metrics["f1_fake"],
            val_metrics["decision_threshold"],
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            best_metrics = {"epoch": epoch, **val_metrics}
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "label_smoothing": label_smoothing,
                    "use_focal": config.USE_FOCAL_LOSS,
                    "lambda_rel": rationale_loss_weight,
                    "lambda_cons": consistency_loss_weight,
                    "lambda_struct": structure_loss_weight,
                    "use_causal_debias": use_causal_debias,
                    "lambda_kl": causal_kl_weight,
                    "causal_prior": causal_prior,
                    "causal_conf_weight": causal_conf_weight,
                    "causal_struct_weight": causal_struct_weight,
                    "causal_min_weight": causal_min_weight,
                    "causal_max_weight": causal_max_weight,
                    "decision_threshold": val_metrics["decision_threshold"],
                },
            }
            torch.save(ckpt, best_ckpt_path)
            logger.info(
                "  -> 保存最佳模型到 %s (Val macF1=%.4f, threshold=%.3f)",
                best_ckpt_path,
                best_val_f1,
                val_metrics["decision_threshold"],
            )
        else:
            patience_counter += 1
            logger.info(
                "  -> 验证集未提升，patience: %s/%s",
                patience_counter,
                early_stopping_patience,
            )

        if patience_counter >= early_stopping_patience:
            logger.info(
                "Early stopping at epoch %s. Best epoch: %s, best val macF1=%.4f",
                epoch,
                best_epoch,
                best_val_f1,
            )
            break

    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        logger.info(
            "加载最佳模型 (epoch=%s, val macF1=%.4f)",
            best_ckpt["epoch"],
            best_ckpt["val_metrics"]["macro_f1"],
        )
        model.load_state_dict(best_ckpt["model_state_dict"])

    if test_loader is not None:
        best_threshold = float(best_metrics.get("decision_threshold", 0.5))
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            use_amp=use_amp,
            rationale_loss_weight=rationale_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            structure_loss_weight=structure_loss_weight,
            use_causal_debias=use_causal_debias,
            causal_kl_weight=causal_kl_weight,
            causal_prior=causal_prior,
            causal_conf_weight=causal_conf_weight,
            causal_struct_weight=causal_struct_weight,
            causal_min_weight=causal_min_weight,
            causal_max_weight=causal_max_weight,
            decision_threshold=best_threshold,
        )
        logger.info("=" * 60)
        logger.info("测试集最终结果:")
        logger.info("  Decision Threshold: %.3f", best_threshold)
        logger.info("  Accuracy:  %.4f", test_metrics["accuracy"])
        logger.info("  Macro F1:  %.4f", test_metrics["macro_f1"])
        logger.info("  F1 (Real): %.4f", test_metrics["f1_real"])
        logger.info("  F1 (Fake): %.4f", test_metrics["f1_fake"])
        logger.info("=" * 60)
        best_metrics["test"] = test_metrics

    return best_metrics
