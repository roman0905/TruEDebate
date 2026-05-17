"""
TruEDebate (TED) — PyG Dataset 定义
加载生成的辩论 JSON 文件，构建 PyG Data 对象（节点特征、边索引、标签）。
在 __getitem__ 中动态计算 Token ID 以节省内存。
"""

import json
import logging
import math
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from transformers import AutoTokenizer

import config

logger = logging.getLogger(__name__)


# 数值特征维度（必须与 config.NUMERIC_FEATURE_DIM 一致）。
# 0  planner_conf            该样本所有 perspective 的平均规划置信度
# 1  priority_norm           该 perspective 的优先级归一化分数（1 - (priority-1)/(k-1)）
# 2  report_conf             节点文本中解析出的 confidence 值
# 3  self_reflective_conf    self-reflective judge 的置信度
# 4  consistency             role-reversal judge 的一致性分数
# 5  final_conf              final judge 置信度
# 6  uncertainty_signal      文本是否包含 "uncertain"
# 7  is_perspective_node     是否 perspective agent 节点
# ── 扩充字段 ──
# 8  is_planner_node
# 9  is_coordinator_node
# 10 is_judge_node
# 11 node_text_log_length    log(len(text)+1)/10
# 12 has_numbers_or_dates    节点文本是否含数字/日期
# 13 role_reversal_flip      role-reversal 是否触发标签翻转信号
# 14 sample_perspectives_real_ratio  全样本视角投真比例（共享于所有节点）
# 15 sample_perspectives_fake_ratio  全样本视角投假比例（共享于所有节点）


class DebateGraphDataset(TorchDataset):
    """
    辩论图数据集。

    每条样本对应一篇新闻的完整辩论记录（PAMD 模式 9 节点 / 原 TED 7 节点）。
    """

    _NUMBER_RE = re.compile(r"\d")
    _DATE_RE = re.compile(
        r"\b(?:\d{4}|\d{1,2}/\d{1,2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
        re.IGNORECASE,
    )

    def __init__(self, data_dir: str | Path, lang: str = "en"):
        self.data_dir = Path(data_dir)
        self.lang = lang

        self.file_paths = sorted(self.data_dir.glob("*.json"))
        if len(self.file_paths) == 0:
            logger.warning(f"数据目录 {data_dir} 中没有找到 JSON 文件！")

        logger.info(f"加载数据集: {len(self.file_paths)} 个样本 (lang={lang})")

        bert_name = config.BERT_MODELS.get(lang, config.BERT_MODELS["en"])
        tokenizer_path = self._resolve_tokenizer_path(bert_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    @staticmethod
    def _resolve_tokenizer_path(bert_name: str) -> str:
        dir_name = bert_name.split("/")[-1]
        local_path = config.BERT_LOCAL_DIR / dir_name
        if local_path.exists() and (local_path / "tokenizer_config.json").exists():
            return str(local_path)
        full_local = config.BERT_LOCAL_DIR / bert_name
        if full_local.exists() and (full_local / "tokenizer_config.json").exists():
            return str(full_local)
        return bert_name

    def __len__(self) -> int:
        return len(self.file_paths)

    @staticmethod
    def _score_from_text(text: str, keys: tuple[str, ...]) -> float:
        lowered = text.lower()
        for key in keys:
            pattern = rf"{re.escape(key.lower())}\s*(?:\*\*)?\s*[:：]?\s*(?:\*\*)?\s*[-–]?\s*([01](?:\.\d+)?)"
            match = re.search(pattern, lowered)
            if match:
                try:
                    return max(0.0, min(1.0, float(match.group(1))))
                except ValueError:
                    return 0.0
        return 0.0

    @staticmethod
    def _label_from_text(text: str, keys: tuple[str, ...]) -> int | None:
        """从结构化文本中解析 fake/real/true/false 标签。"""
        lowered = text.lower()
        for key in keys:
            pattern = rf"{re.escape(key.lower())}\s*[:：]?\s*\"?(real|fake|true|false)\"?"
            match = re.search(pattern, lowered)
            if match:
                token = match.group(1)
                return 1 if token in ("fake", "false") else 0
        return None

    @staticmethod
    def _sanitize_node_text(text: str) -> str:
        """删除 LLM 伪标签字段，避免直接学到 final_label。"""
        if not config.SANITIZE_FINAL_LABEL_TEXT:
            return text
        sanitized_lines = []
        label_patterns = (
            "final_label",
            "reflective_label",
            "provisional label",
            "verdict_hint",
        )
        for line in text.splitlines():
            lowered = line.lower()
            if any(pattern in lowered for pattern in label_patterns):
                continue
            sanitized_lines.append(line)
        sanitized = "\n".join(sanitized_lines).strip()
        return sanitized or text

    @classmethod
    def _has_numbers_or_dates(cls, text: str) -> float:
        if cls._DATE_RE.search(text):
            return 1.0
        if cls._NUMBER_RE.search(text):
            return 1.0
        return 0.0

    @classmethod
    def _perspective_vote_summary(
        cls, record: dict, nodes: list[dict]
    ) -> tuple[float, float, float]:
        """从 perspective 节点文本里粗略统计 fake/real 倾向，并探测 role-reversal flip。"""
        real_votes = 0
        fake_votes = 0
        total = 0
        for node in nodes:
            if not node.get("perspective_key"):
                continue
            total += 1
            label = cls._label_from_text(
                node.get("text", ""),
                ("verdict", "perspective_verdict", "label", "stance"),
            )
            if label == 0:
                real_votes += 1
            elif label == 1:
                fake_votes += 1
        denom = max(total, 1)
        real_ratio = real_votes / denom
        fake_ratio = fake_votes / denom

        role_reversal = record.get("role_reversal") or {}
        flip_signal = 0.0
        if isinstance(role_reversal, dict):
            consistency = role_reversal.get("consistency_score") or role_reversal.get(
                "consistency"
            )
            try:
                if consistency is not None and float(consistency) < 0.5:
                    flip_signal = 1.0
            except (TypeError, ValueError):
                flip_signal = 0.0
            flip_text = " ".join(
                str(v) for v in role_reversal.values() if isinstance(v, str)
            ).lower()
            if "flip" in flip_text or "inconsistent" in flip_text:
                flip_signal = 1.0
        return real_ratio, fake_ratio, flip_signal

    @classmethod
    def _build_numeric_features(cls, record: dict, nodes: list[dict]) -> torch.Tensor:
        """
        构造 16 维节点数值特征。每行代表一个节点。
        共享样本级特征（投票比例、role-reversal flip）会复制到所有节点。
        """
        selected = record.get("selected_perspectives", [])
        selected_by_key = {
            item.get("key"): item
            for item in selected
            if isinstance(item, dict) and item.get("key")
        }
        selected_conf = [
            float(item.get("confidence", 0.0))
            for item in selected
            if isinstance(item, dict)
        ]
        avg_planner_conf = (
            sum(selected_conf) / len(selected_conf) if selected_conf else 0.0
        )
        top_k = max(len(selected), 1)

        real_ratio, fake_ratio, flip_signal = cls._perspective_vote_summary(record, nodes)

        features = []
        for node in nodes:
            role_name = str(node.get("role_name", ""))
            text = str(node.get("text", ""))
            perspective_key = node.get("perspective_key")
            selected_item = selected_by_key.get(perspective_key, {})

            planner_conf = 0.0
            priority_norm = 0.0
            report_conf = 0.0
            self_conf = 0.0
            consistency = 0.0
            final_conf = 0.0

            is_planner = 1.0 if role_name == "perspective_planner" else 0.0
            is_perspective = 1.0 if perspective_key else 0.0
            is_coordinator = 1.0 if role_name == "perspective_coordinator" else 0.0
            is_judge = (
                1.0
                if role_name in ("self_reflective_judge", "role_reversal_judge", "final_judge")
                else 0.0
            )

            if is_planner > 0.5:
                planner_conf = avg_planner_conf
            elif is_perspective > 0.5:
                planner_conf = float(selected_item.get("confidence", 0.0) or 0.0)
                priority = float(selected_item.get("priority", top_k) or top_k)
                priority_norm = 1.0 - ((priority - 1.0) / max(top_k - 1, 1))
                report_conf = cls._score_from_text(text, ("confidence",))
            elif is_coordinator > 0.5:
                planner_conf = avg_planner_conf
            elif role_name == "self_reflective_judge":
                self_conf = cls._score_from_text(text, ("confidence_score",))
            elif role_name == "role_reversal_judge":
                consistency = cls._score_from_text(text, ("consistency_score",))
            elif role_name == "final_judge":
                final_conf = cls._score_from_text(text, ("confidence_score",))

            uncertainty_signal = 1.0 if "uncertain" in text.lower() else 0.0
            text_log_length = min(math.log(max(len(text), 1) + 1) / 10.0, 1.0)
            has_numdate = cls._has_numbers_or_dates(text)

            features.append(
                [
                    planner_conf,
                    priority_norm,
                    report_conf,
                    self_conf,
                    consistency,
                    final_conf,
                    uncertainty_signal,
                    is_perspective,
                    is_planner,
                    is_coordinator,
                    is_judge,
                    text_log_length,
                    has_numdate,
                    flip_signal,
                    real_ratio,
                    fake_ratio,
                ]
            )

        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def _build_node_group_ids(nodes: list[dict]) -> torch.Tensor:
        """根据 role_id 映射到 4 个语义组。"""
        groups = []
        for node in nodes:
            role_id = int(node.get("role_id", -1))
            group = config.ROLE_TO_GROUP.get(role_id, config.NODE_GROUP_PERSPECTIVE)
            groups.append(group)
        return torch.tensor(groups, dtype=torch.long)

    def __getitem__(self, idx: int) -> Data:
        with open(self.file_paths[idx], "r", encoding="utf-8") as f:
            record = json.load(f)

        nodes = record["nodes"]
        edge_index = record["edge_index"]
        edge_type = record.get("edge_type")
        label = record["label"]
        news_text = record["news_text"]

        if isinstance(label, str):
            label = config.LABEL_MAP.get(label, label)
        label = int(label)

        # Tokenize 每个节点文本
        node_texts = [self._sanitize_node_text(n["text"]) for n in nodes]
        node_encodings = self.tokenizer(
            node_texts,
            padding="max_length",
            truncation=True,
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt",
        )

        role_ids = torch.tensor(
            [n["role_id"] for n in nodes], dtype=torch.long
        )
        node_group_ids = self._build_node_group_ids(nodes)
        numeric_features = self._build_numeric_features(record, nodes)

        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        if not (edge_index_tensor.dim() == 2 and edge_index_tensor.shape[0] == 2):
            raise ValueError(
                f"edge_index 形状异常: {edge_index_tensor.shape}, 期望 [2, num_edges]"
            )

        num_edges = edge_index_tensor.shape[1]
        if edge_type is None:
            edge_type_tensor = torch.zeros(num_edges, dtype=torch.long)
        else:
            if record.get("debate_mode") == "perspective":
                edge_type = self._remap_perspective_edge_types(
                    edge_type, edge_index, nodes
                )
            edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
            if edge_type_tensor.numel() != num_edges:
                raise ValueError(
                    f"edge_type 长度异常: {edge_type_tensor.numel()}, 期望 {num_edges}"
                )

        news_encoding = self.tokenizer(
            news_text,
            padding="max_length",
            truncation=True,
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt",
        )

        data = Data(
            node_input_ids=node_encodings["input_ids"],
            node_attention_mask=node_encodings["attention_mask"],
            role_ids=role_ids,
            node_group_ids=node_group_ids,
            numeric_features=numeric_features,
            edge_index=edge_index_tensor,
            edge_type=edge_type_tensor,
            news_input_ids=news_encoding["input_ids"],
            news_attention_mask=news_encoding["attention_mask"],
            y=torch.tensor(label, dtype=torch.long),
            num_nodes=len(nodes),
            # V8 KDPE：把样本在 file_paths 里的索引带上，便于 train 时取 teacher probs。
            sample_idx=torch.tensor([idx], dtype=torch.long),
        )
        return data

    @staticmethod
    def _remap_perspective_edge_types(
        edge_type: list[int],
        edge_index: list[list[int]],
        nodes: list[dict],
    ) -> list[int]:
        attack_id = config.EDGE_TYPE_IDS["attack"]
        cross_id = config.EDGE_TYPE_IDS["cross_perspective"]
        remapped = list(edge_type)
        src_list, dst_list = edge_index
        for idx, (src, dst) in enumerate(zip(src_list, dst_list)):
            if remapped[idx] != attack_id:
                continue
            src_node = nodes[src]
            dst_node = nodes[dst]
            if src_node.get("perspective_key") and dst_node.get("perspective_key"):
                remapped[idx] = cross_id
        return remapped


# ──────────────────────────────── 路 C：双流配对数据 ────────────────────────────────


class PairedDebateDataset(TorchDataset):
    """同一条新闻的 TED 二元辩论数据 + PAMD 多视角辩论数据配对。

    两套数据在 output/ 目录下按 split 一一对应（en_train ↔ en_train_pamd 等），
    文件名按 id 排序后顺序一致，因此用同一个 idx 即可对齐。
    """

    def __init__(self, ted_dir: str | Path, pamd_dir: str | Path, lang: str = "en"):
        self.ted_dataset = DebateGraphDataset(ted_dir, lang=lang)
        self.pamd_dataset = DebateGraphDataset(pamd_dir, lang=lang)
        if len(self.ted_dataset) != len(self.pamd_dataset):
            raise ValueError(
                f"TED 与 PAMD 数据集长度不一致: "
                f"ted={len(self.ted_dataset)}, pamd={len(self.pamd_dataset)}"
            )
        # 共享底层 tokenizer，二者从同一 BERT_MODELS[lang] 加载。
        self.lang = lang

    def __len__(self) -> int:
        return len(self.ted_dataset)

    def __getitem__(self, idx: int) -> dict:
        return {
            "ted": self.ted_dataset[idx],
            "pamd": self.pamd_dataset[idx],
        }


def paired_collate(items: list[dict]) -> dict:
    """与 PyG DataLoader.collate_fn 兼容的配对 batching。"""
    from torch_geometric.data import Batch
    return {
        "ted": Batch.from_data_list([item["ted"] for item in items]),
        "pamd": Batch.from_data_list([item["pamd"] for item in items]),
    }
