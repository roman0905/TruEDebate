"""
TruEDebate (TED) — 阶段 1: 离线辩论生成
使用多线程并发调用 OpenAI API，为每条新闻生成完整的辩论记录并保存为 JSON。
支持断点续跑（已存在的 JSON 文件会被跳过）。

用法:
    python main_generate.py --dataset en --split train --max_workers 4
    python main_generate.py --dataset zh --split val --max_workers 2 --max_samples 10
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import config
from debate_flow.model import DebateModel

# ──────────────────────────────── 日志配置 ────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.ROOT_DIR / "generate.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────── 数据加载 ────────────────────────────────

def load_dataset(dataset: str, split: str) -> list[dict]:
    """
    加载原始新闻数据集。

    实际数据格式:
    - 路径: data/{en,zh}/{train,val,test}.json
    - 字段: {"content": "...", "label": 0/1 或 "real"/"fake", ...}

    Args:
        dataset: "en" 或 "zh"
        split: "train", "val", 或 "test"

    Returns:
        数据列表 [{"text": ..., "label": ..., "id": ...}, ...]
    """
    data_dir = config.DATA_DIR

    # 数据集实际存储在子目录中: data/en/train.json, data/zh/val.json 等
    possible_paths = [
        data_dir / dataset / f"{split}.json",
        data_dir / dataset / f"{split}.jsonl",
        data_dir / f"{dataset}_{split}.json",
        data_dir / f"{dataset}_{split}.jsonl",
    ]

    file_path = None
    for candidate in possible_paths:
        if candidate.exists():
            file_path = candidate
            break

    if file_path is None:
        raise FileNotFoundError(
            f"在 {data_dir} 中找不到数据集文件。\n"
            f"尝试过的路径: {[str(p) for p in possible_paths]}\n"
            f"请确保数据文件存在于 data/{dataset}/ 目录下。"
        )

    logger.info(f"加载数据集: {file_path}")

    records = []
    if file_path.suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item.setdefault("id", i)
                records.append(item)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data):
                item.setdefault("id", i)
                records.append(item)
        else:
            raise ValueError(f"JSON 文件格式异常: 期望 list，得到 {type(data)}")

    # 标准化字段: 将 "content" 映射为 "text"，标签映射为整数
    for item in records:
        # 文本字段: 数据集使用 "content"，统一为 "text"
        if "text" not in item and "content" in item:
            item["text"] = item["content"]
        # 标签字段: 将字符串标签映射为整数
        if "label" in item:
            item["label"] = config.LABEL_MAP.get(item["label"], item["label"])

    logger.info(f"加载完成: {len(records)} 条记录")
    return records


# ──────────────────────────────── 单条处理 ────────────────────────────────

def process_single_news(
    item: dict,
    output_dir: Path,
    dataset: str,
    split: str,
) -> str:
    """
    为一条新闻生成完整的辩论记录。

    Args:
        item: {"text": ..., "label": ..., "id": ...}
        output_dir: 输出目录
        dataset: 数据集标识
        split: 数据集划分

    Returns:
        输出文件路径
    """
    item_id = item["id"]
    news_text = item["text"]
    label = item["label"]

    # 输出文件路径
    output_file = output_dir / f"{dataset}_{split}_{item_id:06d}.json"

    # 断点续跑：如果文件已存在，跳过
    if output_file.exists():
        return f"SKIP: {output_file.name}"

    try:
        # 运行三阶段辩论 + Synthesis
        debate_model = DebateModel(news_text)
        debate_model.step()

        # 导出辩论记录
        debate_record = debate_model.get_debate_record()

        # 组装完整输出
        output = {
            "id": item_id,
            "news_text": news_text,
            "label": label,
            "dataset": dataset,
            "split": split,
            "nodes": debate_record["nodes"],
            "edge_index": debate_record["edge_index"],
            "synthesis": debate_record["synthesis"],
        }

        # 保存为 JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return f"OK: {output_file.name}"

    except Exception as e:
        logger.error(f"处理失败 (id={item_id}): {e}")
        return f"ERROR: {item_id} - {str(e)}"


# ──────────────────────────────── 主函数 ────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TruEDebate — 阶段 1: 多线程辩论生成"
    )
    parser.add_argument(
        "--dataset", type=str, default="en", choices=["en", "zh"],
        help="数据集语言 (en=ARG-EN, zh=ARG-CN)"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val", "test"],
        help="数据集划分"
    )
    parser.add_argument(
        "--max_workers", type=int, default=config.MAX_WORKERS,
        help="最大并发线程数"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="最多处理的样本数 (用于调试，默认全部处理)"
    )
    args = parser.parse_args()

    # 创建输出目录
    output_dir = config.OUTPUT_DIR / f"{args.dataset}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    records = load_dataset(args.dataset, args.split)

    # 截取样本数
    if args.max_samples is not None:
        records = records[: args.max_samples]
        logger.info(f"截取前 {args.max_samples} 条样本")

    logger.info(
        f"开始生成辩论记录: dataset={args.dataset}, split={args.split}, "
        f"samples={len(records)}, workers={args.max_workers}"
    )

    # 多线程并发处理
    results = {"OK": 0, "SKIP": 0, "ERROR": 0}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_single_news, item, output_dir, args.dataset, args.split
            ): item["id"]
            for item in records
        }

        with tqdm(total=len(futures), desc="Generating debates") as pbar:
            for future in as_completed(futures):
                result = future.result()
                status = result.split(":")[0]
                results[status] = results.get(status, 0) + 1
                pbar.set_postfix(
                    ok=results["OK"],
                    skip=results["SKIP"],
                    err=results["ERROR"],
                )
                pbar.update(1)

    # 统计结果
    logger.info("=" * 60)
    logger.info("生成完成统计:")
    logger.info(f"  成功: {results['OK']}")
    logger.info(f"  跳过: {results['SKIP']}")
    logger.info(f"  失败: {results['ERROR']}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
