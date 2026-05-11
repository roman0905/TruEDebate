"""
TruEDebate (TED) — 阶段 2: 本地模型训练与评估
加载已生成的辩论 JSON 文件，使用 BERT + GAT + MHA 图神经网络进行假新闻分类训练。

用法:
    python main_train.py --dataset en --epochs 20 --batch_size 4
    python main_train.py --dataset zh --epochs 10 --batch_size 2 --device cpu
"""

import argparse
import logging
import random
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import config
from insight_flow.dataset import DebateGraphDataset
from insight_flow.networks import TEDClassifier
from insight_flow.train import train

# ──────────────────────────────── 日志配置 ────────────────────────────────

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
    """固定随机种子，提升实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="TruEDebate — 阶段 2: 图神经网络训练与评估"
    )
    parser.add_argument(
        "--dataset", type=str, default="en", choices=["en", "zh"],
        help="数据集语言 (en=ARG-EN, zh=ARG-CN)"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE,
        help="批量大小"
    )
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE,
        help="学习率"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=config.WEIGHT_DECAY,
        help="权重衰减"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="训练设备 (cuda / cpu，默认自动检测)"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=config.GRAD_ACCUM_STEPS,
        help="梯度累积步数"
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="禁用混合精度训练"
    )
    parser.add_argument(
        "--freeze_layers", type=int, default=config.BERT_FREEZE_LAYERS,
        help="冻结 BERT 前 N 层"
    )
    parser.add_argument(
        "--data_suffix", type=str, default="",
        help="辩论输出目录后缀，例如 _pamd 表示读取 output/en_train_pamd"
    )
    parser.add_argument(
        "--no_typed_edges", action="store_true",
        help="禁用 edge_type 边类型嵌入，用于消融实验"
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED,
        help="随机种子"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=str(config.CHECKPOINT_DIR),
        help="模型与指标保存目录"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # ── 设备选择 ──
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # ── 加载数据集 ──
    logger.info("加载数据集...")

    train_dir = config.OUTPUT_DIR / f"{args.dataset}_train{args.data_suffix}"
    val_dir = config.OUTPUT_DIR / f"{args.dataset}_val{args.data_suffix}"
    test_dir = config.OUTPUT_DIR / f"{args.dataset}_test{args.data_suffix}"

    train_dataset = DebateGraphDataset(train_dir, lang=args.dataset)
    val_dataset = DebateGraphDataset(val_dir, lang=args.dataset)

    test_dataset = None
    if test_dir.exists() and any(test_dir.glob("*.json")):
        test_dataset = DebateGraphDataset(test_dir, lang=args.dataset)

    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    if test_dataset:
        logger.info(f"测试集: {len(test_dataset)} 样本")
    else:
        logger.info("测试集: 未找到或为空")

    # ── 创建 DataLoader ──
    # PyG DataLoader 自动处理图的 batching (合并 edge_index, 生成 batch vector)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows 下避免多进程问题
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

    # ── 创建模型 ──
    logger.info("创建 TEDClassifier 模型...")
    model = TEDClassifier(
        lang=args.dataset,
        freeze_layers=args.freeze_layers,
        use_typed_edges=not args.no_typed_edges,
    )

    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总量: {total_params:,}")
    logger.info(f"可训练参数:   {trainable_params:,}")
    logger.info(f"冻结参数:     {total_params - trainable_params:,}")

    # ── 训练 ──
    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info(f"  Epochs:      {args.epochs}")
    logger.info(f"  Batch Size:  {args.batch_size}")
    logger.info(f"  Grad Accum:  {args.grad_accum}")
    logger.info(f"  Effective BS: {args.batch_size * args.grad_accum}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Weight Decay: {args.weight_decay}")
    logger.info(f"  AMP:         {not args.no_amp}")
    logger.info(f"  BERT Freeze: {args.freeze_layers} layers")
    logger.info(f"  Data Suffix: {args.data_suffix or '(none)'}")
    logger.info(f"  Edge Type:   {not args.no_typed_edges}")
    logger.info(f"  Seed:        {args.seed}")
    logger.info(f"  Checkpoint:  {args.checkpoint_dir}")
    logger.info("=" * 60)

    # 更新全局梯度累积配置
    config.GRAD_ACCUM_STEPS = args.grad_accum

    best_metrics = train(
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

    # ── 输出最终结果 ──
    logger.info("=" * 60)
    logger.info("训练完成！最佳验证结果:")
    logger.info(f"  Best Epoch:  {best_metrics.get('epoch', 'N/A')}")
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
