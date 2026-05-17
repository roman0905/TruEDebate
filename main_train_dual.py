"""
路 C：双流融合模型训练入口。

复用现有的 PAMD 与 TED 已生成数据：
  output/en_train      / output/en_train_pamd
  output/en_val        / output/en_val_pamd
  output/en_test       / output/en_test_pamd

每个样本会同时取 TED 二元辩论图 + PAMD 多视角辩论图，送入 DualStreamClassifier。
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

import config
from insight_flow.dataset import PairedDebateDataset, paired_collate
from insight_flow.dual_networks import DualStreamClassifier
from insight_flow.dual_train import train_dual

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.ROOT_DIR / "train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="TruEDebate 路 C — 双流融合训练")
    parser.add_argument("--dataset", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--grad_accum", type=int, default=config.GRAD_ACCUM_STEPS)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--freeze_layers", type=int, default=config.BERT_FREEZE_LAYERS)
    parser.add_argument("--no_typed_edges", action="store_true")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--checkpoint_dir", type=str, default=str(config.CHECKPOINT_DIR / "dual"))
    # 后缀（默认 TED 数据不带后缀、PAMD 数据带 _pamd）
    parser.add_argument("--ted_suffix", type=str, default="")
    parser.add_argument("--pamd_suffix", type=str, default="_pamd")

    # V8 模块开关（与 main_train.py 一致）
    parser.add_argument("--no_xpcr", action="store_true")
    parser.add_argument("--xpcr_weight", type=float, default=None)
    parser.add_argument("--no_datr", action="store_true")
    parser.add_argument("--datr_alpha", type=float, default=None)
    parser.add_argument("--use_kdpe", action="store_true")
    parser.add_argument("--kdpe_weight", type=float, default=None)
    parser.add_argument("--kdpe_teacher", type=str, default=None)
    parser.add_argument("--no_focal", action="store_true")
    parser.add_argument("--no_swa", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--patience", type=int, default=None)

    args = parser.parse_args()

    # 应用 config 覆盖
    if args.no_focal:
        config.USE_FOCAL_LOSS = False
    if args.no_swa:
        config.USE_SWA = False
    if args.no_ema:
        config.USE_EMA = False
    if args.no_xpcr:
        config.USE_XPCR = False
    if args.xpcr_weight is not None:
        config.XPCR_WEIGHT = args.xpcr_weight
    if args.no_datr:
        config.USE_DATR = False
    if args.datr_alpha is not None:
        config.DATR_ALPHA = args.datr_alpha
    if args.use_kdpe:
        config.USE_KDPE = True
    if args.kdpe_weight is not None:
        config.KDPE_WEIGHT = args.kdpe_weight
    if args.kdpe_teacher is not None:
        config.KDPE_TEACHER_PROBS_PATH = args.kdpe_teacher
    if args.patience is not None:
        config.EARLY_STOPPING_PATIENCE = args.patience

    set_seed(args.seed)

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"使用设备: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载配对数据
    logger.info("加载配对数据集 (TED 二元辩论 + PAMD 多视角辩论)...")
    ted_train = config.OUTPUT_DIR / f"{args.dataset}_train{args.ted_suffix}"
    pamd_train = config.OUTPUT_DIR / f"{args.dataset}_train{args.pamd_suffix}"
    ted_val = config.OUTPUT_DIR / f"{args.dataset}_val{args.ted_suffix}"
    pamd_val = config.OUTPUT_DIR / f"{args.dataset}_val{args.pamd_suffix}"
    ted_test = config.OUTPUT_DIR / f"{args.dataset}_test{args.ted_suffix}"
    pamd_test = config.OUTPUT_DIR / f"{args.dataset}_test{args.pamd_suffix}"

    train_ds = PairedDebateDataset(ted_train, pamd_train, lang=args.dataset)
    val_ds = PairedDebateDataset(ted_val, pamd_val, lang=args.dataset)
    test_ds = None
    if ted_test.exists() and pamd_test.exists():
        test_ds = PairedDebateDataset(ted_test, pamd_test, lang=args.dataset)

    logger.info(f"训练集: {len(train_ds)} 配对")
    logger.info(f"验证集: {len(val_ds)} 配对")
    if test_ds:
        logger.info(f"测试集: {len(test_ds)} 配对")

    train_loader = TorchDataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=paired_collate, drop_last=False,
    )
    val_loader = TorchDataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=paired_collate,
    )
    test_loader = None
    if test_ds:
        test_loader = TorchDataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=paired_collate,
        )

    # 模型
    logger.info("创建 DualStreamClassifier（共享 BERT + 双 GAT + 融合头）...")
    model = DualStreamClassifier(
        lang=args.dataset,
        freeze_layers=args.freeze_layers,
        use_typed_edges=not args.no_typed_edges,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总量: {total_params:,}")
    logger.info(f"可训练参数:   {trainable_params:,}")
    logger.info(f"冻结参数:     {total_params - trainable_params:,}")

    config.GRAD_ACCUM_STEPS = args.grad_accum

    logger.info("=" * 60)
    logger.info("开始双流训练")
    logger.info(f"  Epochs:      {args.epochs}")
    logger.info(f"  Batch Size:  {args.batch_size}")
    logger.info(f"  Grad Accum:  {args.grad_accum}")
    logger.info(f"  LR:          {args.lr}")
    logger.info(f"  WD:          bert={args.weight_decay}, other={config.WEIGHT_DECAY_OTHER}")
    logger.info(f"  AMP:         {not args.no_amp}")
    logger.info(f"  BERT Freeze: {args.freeze_layers} layers")
    logger.info(f"  TED Suffix:  '{args.ted_suffix}'  PAMD Suffix: '{args.pamd_suffix}'")
    logger.info(f"  Checkpoint:  {args.checkpoint_dir}")
    logger.info(f"  XPCR:        {config.USE_XPCR} (w={config.XPCR_WEIGHT})")
    logger.info(f"  DATR:        {config.USE_DATR} (a={config.DATR_ALPHA})")
    logger.info(f"  KDPE:        {config.USE_KDPE} (w={config.KDPE_WEIGHT})")
    logger.info("=" * 60)

    best_metrics = train_dual(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir,
        grad_accum_steps=args.grad_accum,
    )

    logger.info("=" * 60)
    logger.info("[Dual] 训练完成！最佳验证结果:")
    logger.info(f"  Best Epoch / Winner: {best_metrics.get('epoch', '?')} / {best_metrics.get('winner', '?')}")
    logger.info(f"  Threshold:   {best_metrics.get('threshold', 0.5):.3f}")
    logger.info(f"  Accuracy:    {best_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Macro F1:    {best_metrics.get('macro_f1', 0):.4f}")
    logger.info(f"  F1 (Real):   {best_metrics.get('f1_real', 0):.4f}")
    logger.info(f"  F1 (Fake):   {best_metrics.get('f1_fake', 0):.4f}")

    if "test" in best_metrics:
        test = best_metrics["test"]
        logger.info("-" * 40)
        logger.info("测试集最终结果:")
        logger.info(f"  Accuracy:    {test['accuracy']:.4f}")
        logger.info(f"  Macro F1:    {test['macro_f1']:.4f}")
        logger.info(f"  F1 (Real):   {test['f1_real']:.4f}")
        logger.info(f"  F1 (Fake):   {test['f1_fake']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
