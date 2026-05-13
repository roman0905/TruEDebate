"""
V8 KDPE 准备脚本：用已有 V6 ensemble 的 N 个 seed checkpoints，对训练集做推理并保存
softmax 平均概率，作为 V8 训练的 teacher 软标签。

用法：
  python main_kd_prepare.py \
      --ensemble_dir checkpoints/ensemble_v6 \
      --dataset en --data_suffix _pamd \
      --output checkpoints/ensemble_v6/train_probs_aligned.npy

注意：
- 输出概率 shape = [num_train_samples, 2]，顺序与 DebateGraphDataset.file_paths 一致。
- 训练集不能 shuffle，确保对齐。
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import config
from insight_flow.dataset import DebateGraphDataset
from insight_flow.networks import TEDClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.ROOT_DIR / "train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def infer_probs(model, loader, device):
    model.eval()
    probs = []
    for batch_data in loader:
        batch_data = batch_data.to(device)
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
        labels = batch_data.y

        if news_input_ids.dim() == 2 and news_input_ids.shape[0] != labels.shape[0]:
            bs = labels.shape[0]
            seq_len = news_input_ids.shape[-1]
            news_input_ids = news_input_ids.view(bs, seq_len)
            news_attention_mask = news_attention_mask.view(bs, seq_len)

        logits = model(
            node_input_ids=node_input_ids,
            node_attention_mask=node_attention_mask,
            role_ids=role_ids,
            node_group_ids=node_group_ids,
            numeric_features=numeric_features,
            edge_index=edge_index,
            edge_type=edge_type,
            batch=batch_vec,
            news_input_ids=news_input_ids,
            news_attention_mask=news_attention_mask,
        )
        probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--data_suffix", type=str, default="_pamd")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出文件路径；不填则写到 <ensemble_dir>/train_probs_aligned.npy",
    )
    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    seed_dirs = sorted(
        [d for d in ensemble_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
    )
    if not seed_dirs:
        logger.error(f"在 {ensemble_dir} 找不到 seed_* 子目录")
        sys.exit(1)
    logger.info(f"将聚合 {len(seed_dirs)} 个 seed checkpoints: {[d.name for d in seed_dirs]}")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"使用设备: {device}")

    train_dir = config.OUTPUT_DIR / f"{args.dataset}_train{args.data_suffix}"
    train_dataset = DebateGraphDataset(train_dir, lang=args.dataset)
    logger.info(f"训练集样本数: {len(train_dataset)}")

    # 重要：不 shuffle，保证概率顺序与 file_paths 一致。
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    accumulated = None
    for seed_dir in seed_dirs:
        ckpt_path = seed_dir / "best_model.pt"
        if not ckpt_path.exists():
            logger.warning(f"跳过 {seed_dir.name}：缺少 best_model.pt")
            continue
        logger.info(f"[{seed_dir.name}] 加载 {ckpt_path}")
        model = TEDClassifier(lang=args.dataset).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

        probs = infer_probs(model, train_loader, device)  # [N, 2]
        logger.info(
            f"[{seed_dir.name}] 推理完成 shape={probs.shape}, "
            f"fake_prob 均值={probs[:, 1].mean():.4f}"
        )
        accumulated = probs if accumulated is None else accumulated + probs

        del model
        torch.cuda.empty_cache()

    if accumulated is None:
        logger.error("没有任何可用 seed checkpoint，退出。")
        sys.exit(1)

    ensemble_probs = accumulated / len(seed_dirs)
    out_path = Path(args.output) if args.output else (ensemble_dir / "train_probs_aligned.npy")
    np.save(out_path, ensemble_probs)
    logger.info(
        f"Teacher train probs 已保存: {out_path} "
        f"shape={ensemble_probs.shape}, fake_prob 均值={ensemble_probs[:, 1].mean():.4f}"
    )


if __name__ == "__main__":
    main()
