"""
TruEDebate (TED) — 阶段 2: 本地模型训练与评估
加载已生成的辩论 JSON 文件，使用 BERT + GAT + MHA 图神经网络进行假新闻分类训练。

用法:
    python main_train.py --dataset en --epochs 20 --batch_size 4
    python main_train.py --dataset zh --epochs 10 --batch_size 2 --device cpu
"""

import argparse
import logging
import os
import sys

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


def main():
    default_workers = 0 if os.name == "nt" else min(config.MAX_WORKERS, os.cpu_count() or 1)

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
        "--label_smoothing", type=float, default=config.LABEL_SMOOTHING,
        help="交叉熵标签平滑系数"
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=config.EARLY_STOPPING_PATIENCE,
        help="早停耐心轮数"
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
        "--lambda_rel", type=float, default=config.RTED_RELIABILITY_LOSS_WEIGHT,
        help="R-TED rationale reliability 辅助损失权重"
    )
    parser.add_argument(
        "--lambda_cons", type=float, default=config.RTED_CONSISTENCY_LOSS_WEIGHT,
        help="R-TED debate-rationale consistency 辅助损失权重"
    )
    parser.add_argument(
        "--lambda_struct", type=float, default=config.EVITED_STRUCTURE_LOSS_WEIGHT,
        help="EviTED relation-aware graph structure 正则权重"
    )
    parser.add_argument(
        "--disable_causal_debias", action="store_true",
        help="关闭 4.7 causal debias 样本级加权"
    )
    parser.add_argument(
        "--lambda_kl", type=float, default=config.EVITED_CAUSAL_KL_WEIGHT,
        help="Causal posterior KL 正则权重"
    )
    parser.add_argument(
        "--causal_prior", type=float, default=config.EVITED_CAUSAL_PRIOR,
        help="样本可靠性隐变量的 Bernoulli prior"
    )
    parser.add_argument(
        "--causal_conf_weight", type=float, default=config.EVITED_CAUSAL_CONF_WEIGHT,
        help="分类置信度在 causal posterior 中的权重"
    )
    parser.add_argument(
        "--causal_struct_weight", type=float, default=config.EVITED_CAUSAL_STRUCT_WEIGHT,
        help="结构似然在 causal posterior 中的权重"
    )
    parser.add_argument(
        "--causal_min_weight", type=float, default=config.EVITED_CAUSAL_MIN_WEIGHT,
        help="样本级 causal weight 下界"
    )
    parser.add_argument(
        "--causal_max_weight", type=float, default=config.EVITED_CAUSAL_MAX_WEIGHT,
        help="样本级 causal weight 上界"
    )
    parser.add_argument(
        "--num_workers", type=int, default=default_workers,
        help="DataLoader 并行 worker 数"
    )
    args = parser.parse_args()

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

    train_dir = config.OUTPUT_DIR / f"{args.dataset}_train"
    val_dir = config.OUTPUT_DIR / f"{args.dataset}_val"
    test_dir = config.OUTPUT_DIR / f"{args.dataset}_test"

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
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_kwargs,
        )

    # ── 创建模型 ──
    logger.info("创建 TEDClassifier 模型...")
    model = TEDClassifier(
        lang=args.dataset,
        freeze_layers=args.freeze_layers,
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
    logger.info(f"  Label Smoothing: {args.label_smoothing}")
    logger.info(f"  Early Stop Patience: {args.early_stopping_patience}")
    logger.info(f"  AMP:         {not args.no_amp}")
    logger.info(f"  BERT Freeze: {args.freeze_layers} layers")
    logger.info(f"  Lambda Rel:  {args.lambda_rel}")
    logger.info(f"  Lambda Cons: {args.lambda_cons}")
    logger.info(f"  Lambda Struct: {args.lambda_struct}")
    logger.info(f"  Causal Debias: {not args.disable_causal_debias}")
    logger.info(f"  Lambda KL:   {args.lambda_kl}")
    logger.info(f"  Causal Prior: {args.causal_prior}")
    logger.info(f"  Causal Fusion Weights: conf={args.causal_conf_weight}, struct={args.causal_struct_weight}")
    logger.info(f"  Causal Weight Clip: [{args.causal_min_weight}, {args.causal_max_weight}]")
    logger.info(f"  Num Workers: {args.num_workers}")
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
        early_stopping_patience=args.early_stopping_patience,
        label_smoothing=args.label_smoothing,
        rationale_loss_weight=args.lambda_rel,
        consistency_loss_weight=args.lambda_cons,
        structure_loss_weight=args.lambda_struct,
        use_causal_debias=not args.disable_causal_debias,
        causal_kl_weight=args.lambda_kl,
        causal_prior=args.causal_prior,
        causal_conf_weight=args.causal_conf_weight,
        causal_struct_weight=args.causal_struct_weight,
        causal_min_weight=args.causal_min_weight,
        causal_max_weight=args.causal_max_weight,
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
