"""
TruEDebate (TED/PAMD) — V6 多种子 Ensemble 脚本

执行流程：
  1. 用不同 seed 串行训练 N 个模型，每个模型把 val/test probs 落盘。
  2. 训练全部完成后，加载 N 份 probs 做平均（softmax 概率平均）。
  3. 在平均后的 val probs 上搜阈值，应用到 test。
  4. 打印 ensemble 指标，并把结果写入 `<base_dir>/ensemble_metrics.json`。

用法示例：
  python main_ensemble.py --dataset en --data_suffix _pamd \
      --seeds 42,123,456,789,2024 --base_dir checkpoints/ensemble_v6

参数与 main_train.py 一致；--seeds 用逗号分隔。
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import config
from insight_flow.dataset import DebateGraphDataset
from insight_flow.networks import TEDClassifier
from insight_flow.train import (
    metrics_from_probs,
    search_best_threshold,
    train,
)

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
    parser = argparse.ArgumentParser(description="TruEDebate — V6 多种子 Ensemble")
    parser.add_argument("--dataset", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--data_suffix", type=str, default="_pamd")
    parser.add_argument(
        "--seeds", type=str, default="42,123,456,789,2024",
        help="逗号分隔的随机种子列表",
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--grad_accum", type=int, default=config.GRAD_ACCUM_STEPS)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--freeze_layers", type=int, default=config.BERT_FREEZE_LAYERS)
    parser.add_argument("--no_typed_edges", action="store_true")
    parser.add_argument("--base_dir", type=str, default=str(config.CHECKPOINT_DIR / "ensemble_v6"))
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="若 seed 目录下已存在 test_probs.npy 则跳过该 seed 的训练（断点续训）",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Ensemble device: {device}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Base dir: {base_dir}")

    # ── 加载数据（只载一次） ──
    train_dir = config.OUTPUT_DIR / f"{args.dataset}_train{args.data_suffix}"
    val_dir = config.OUTPUT_DIR / f"{args.dataset}_val{args.data_suffix}"
    test_dir = config.OUTPUT_DIR / f"{args.dataset}_test{args.data_suffix}"

    train_dataset = DebateGraphDataset(train_dir, lang=args.dataset)
    val_dataset = DebateGraphDataset(val_dir, lang=args.dataset)
    test_dataset = None
    if test_dir.exists() and any(test_dir.glob("*.json")):
        test_dataset = DebateGraphDataset(test_dir, lang=args.dataset)

    logger.info(
        f"训练集: {len(train_dataset)}，验证集: {len(val_dataset)}，"
        f"测试集: {len(test_dataset) if test_dataset else 0}"
    )

    # ── 训练每个 seed ──
    config.GRAD_ACCUM_STEPS = args.grad_accum

    for seed in seeds:
        seed_dir = base_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and (seed_dir / "test_probs.npy").exists():
            logger.info(f"[seed {seed}] 已存在 test_probs.npy，跳过训练。")
            continue

        logger.info("=" * 60)
        logger.info(f"开始训练 seed={seed}")
        logger.info("=" * 60)
        set_seed(seed)

        # DataLoader 在每个 seed 下重新创建（shuffle 顺序不同）
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
            )

        model = TEDClassifier(
            lang=args.dataset,
            freeze_layers=args.freeze_layers,
            use_typed_edges=not args.no_typed_edges,
        )
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_amp=not args.no_amp,
            checkpoint_dir=seed_dir,
            grad_accum_steps=args.grad_accum,
        )

    # ── 聚合 probs ──
    val_probs_list = []
    test_probs_list = []
    val_labels = None
    test_labels = None
    used_seeds = []
    for seed in seeds:
        seed_dir = base_dir / f"seed_{seed}"
        vp = seed_dir / "val_probs.npy"
        tp = seed_dir / "test_probs.npy"
        if not (vp.exists() and tp.exists()):
            logger.warning(f"[seed {seed}] 缺少概率文件，跳过。")
            continue
        val_probs_list.append(np.load(vp))
        test_probs_list.append(np.load(tp))
        if val_labels is None:
            val_labels = np.load(seed_dir / "val_labels.npy")
            test_labels = np.load(seed_dir / "test_labels.npy")
        used_seeds.append(seed)

    if not val_probs_list:
        logger.error("没有可用的 seed 概率，无法做 ensemble。")
        return

    val_avg = np.mean(val_probs_list, axis=0)
    test_avg = np.mean(test_probs_list, axis=0)

    # 单模型 baseline 对照（仅打印 first seed）
    if len(val_probs_list) > 0:
        single_val = metrics_from_probs(val_probs_list[0], val_labels, 0.5)
        single_test = metrics_from_probs(test_probs_list[0], test_labels, 0.5)
        logger.info(
            f"[单模型 seed {used_seeds[0]} @thr=0.5] "
            f"Val macF1={single_val['macro_f1']:.4f} | Test macF1={single_test['macro_f1']:.4f}"
        )

    # Ensemble 阈值搜索
    best_thr, _ = search_best_threshold(val_avg, val_labels)
    val_metrics = metrics_from_probs(val_avg, val_labels, threshold=best_thr)
    test_metrics = metrics_from_probs(test_avg, test_labels, threshold=best_thr)
    test_at_05 = metrics_from_probs(test_avg, test_labels, threshold=0.5)
    val_at_05 = metrics_from_probs(val_avg, val_labels, threshold=0.5)

    logger.info("=" * 60)
    logger.info(f"Ensemble 结果（{len(used_seeds)} seeds: {used_seeds}，阈值={best_thr:.3f}）")
    logger.info("─" * 60)
    logger.info("Val:")
    logger.info(f"  macF1@best_thr: {val_metrics['macro_f1']:.4f}")
    logger.info(f"  macF1@0.5:      {val_at_05['macro_f1']:.4f}")
    logger.info(f"  F1_fake:        {val_metrics['f1_fake']:.4f}")
    logger.info(
        f"  Fake P/R:       {val_metrics['fake_precision']:.4f}/"
        f"{val_metrics['fake_recall']:.4f}"
    )
    logger.info("Test:")
    logger.info(f"  Accuracy:       {test_metrics['accuracy']:.4f}")
    logger.info(f"  macF1@best_thr: {test_metrics['macro_f1']:.4f}")
    logger.info(f"  macF1@0.5:      {test_at_05['macro_f1']:.4f}")
    logger.info(f"  F1_real:        {test_metrics['f1_real']:.4f}")
    logger.info(f"  F1_fake:        {test_metrics['f1_fake']:.4f}")
    logger.info(
        f"  Fake P/R:       {test_metrics['fake_precision']:.4f}/"
        f"{test_metrics['fake_recall']:.4f}"
    )
    logger.info(
        f"  Confusion:      TN={test_metrics['tn']} FP={test_metrics['fp']} "
        f"FN={test_metrics['fn']} TP={test_metrics['tp']}"
    )
    logger.info("=" * 60)

    out = {
        "seeds": used_seeds,
        "threshold": best_thr,
        "val": val_metrics,
        "val_at_0_5": val_at_05,
        "test": test_metrics,
        "test_at_0_5": test_at_05,
    }
    out_path = base_dir / "ensemble_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"Ensemble 指标已保存: {out_path}")

    # 也把 ensemble 概率落盘，便于后续做 stacking/post-hoc 分析
    np.save(base_dir / "ensemble_val_probs.npy", val_avg)
    np.save(base_dir / "ensemble_test_probs.npy", test_avg)


if __name__ == "__main__":
    main()
