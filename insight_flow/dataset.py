"""
TruEDebate (TED) — PyG Dataset 定义
加载生成的辩论 JSON 文件，构建 PyG Data 对象（节点特征、边索引、标签）。
在 __getitem__ 中动态计算 Token ID 以节省内存。
"""

import json
import logging
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from transformers import AutoTokenizer

import config

logger = logging.getLogger(__name__)


class DebateGraphDataset(TorchDataset):
    """
    辩论图数据集。

    每条样本对应一篇新闻的完整辩论记录（8 个节点 + 边索引 + 标签）。
    节点文本在 __getitem__ 中动态 tokenize，避免预先计算占用过多内存。
    """

    def __init__(self, data_dir: str | Path, lang: str = "en"):
        """
        Args:
            data_dir: 存储辩论 JSON 文件的目录路径
            lang: 语言选择 ("en" 或 "zh")，决定 BERT Tokenizer
        """
        self.data_dir = Path(data_dir)
        self.lang = lang

        # 加载所有 JSON 文件路径
        self.file_paths = sorted(self.data_dir.glob("*.json"))
        if len(self.file_paths) == 0:
            logger.warning(f"数据目录 {data_dir} 中没有找到 JSON 文件！")

        logger.info(f"加载数据集: {len(self.file_paths)} 个样本 (lang={lang})")

        # 初始化 BERT Tokenizer (优先使用本地模型目录)
        bert_name = config.BERT_MODELS.get(lang, config.BERT_MODELS["en"])
        tokenizer_path = self._resolve_tokenizer_path(bert_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    @staticmethod
    def _resolve_tokenizer_path(bert_name: str) -> str:
        """解析 Tokenizer 路径: 优先使用本地 models/ 目录。"""
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
        """从 LLM 结构化文本中解析 0-1 分数。解析失败返回 0。"""
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
    def _sanitize_node_text(text: str) -> str:
        """
        删除 LLM 伪标签字段，避免模型直接学习低精度 final_label /
        provisional_label / verdict_hint 文本。
        """
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
    def _build_numeric_features(cls, record: dict, nodes: list[dict]) -> torch.Tensor:
        """
        将 PAMD 的 planner confidence、priority、agent confidence、
        self-reflection confidence、role-reversal consistency、final confidence
        显式转为节点数值特征。
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

            if role_name == "perspective_planner":
                planner_conf = avg_planner_conf
            elif perspective_key:
                planner_conf = float(selected_item.get("confidence", 0.0) or 0.0)
                priority = float(selected_item.get("priority", top_k) or top_k)
                priority_norm = 1.0 - ((priority - 1.0) / max(top_k - 1, 1))
                report_conf = cls._score_from_text(text, ("confidence",))
            elif role_name == "perspective_coordinator":
                planner_conf = avg_planner_conf
            elif role_name == "self_reflective_judge":
                self_conf = cls._score_from_text(text, ("confidence_score",))
            elif role_name == "role_reversal_judge":
                consistency = cls._score_from_text(text, ("consistency_score",))
            elif role_name == "final_judge":
                final_conf = cls._score_from_text(text, ("confidence_score",))

            uncertainty_signal = 1.0 if "uncertain" in text.lower() else 0.0
            features.append(
                [
                    planner_conf,
                    priority_norm,
                    report_conf,
                    self_conf,
                    consistency,
                    final_conf,
                    uncertainty_signal,
                    1.0 if perspective_key else 0.0,
                ]
            )

        return torch.tensor(features, dtype=torch.float)

    def __getitem__(self, idx: int) -> Data:
        """
        加载第 idx 条辩论记录并构建 PyG Data 对象。

        Returns:
            Data 对象包含:
                - node_input_ids:      [num_nodes, max_length] Token IDs
                - node_attention_mask: [num_nodes, max_length] Attention masks
                - role_ids:            [num_nodes] 角色 ID
                - edge_index:          [2, num_edges] 有向边索引
                - edge_type:           [num_edges] 边类型 ID，旧数据缺失时补 0
                - news_input_ids:      [1, max_length] 新闻 Token IDs
                - news_attention_mask: [1, max_length] 新闻 Attention masks
                - y:                   标量标签 (0=real, 1=fake)
        """
        # 1. 读取 JSON 文件
        with open(self.file_paths[idx], "r", encoding="utf-8") as f:
            record = json.load(f)

        nodes = record["nodes"]
        edge_index = record["edge_index"]
        edge_type = record.get("edge_type")
        label = record["label"]
        news_text = record["news_text"]

        # 标签标准化: 将字符串标签映射为整数
        if isinstance(label, str):
            label = config.LABEL_MAP.get(label, label)
        label = int(label)

        # 2. Tokenize 每个节点文本
        node_texts = [self._sanitize_node_text(n["text"]) for n in nodes]
        node_encodings = self.tokenizer(
            node_texts,
            padding="max_length",
            truncation=True,
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt",
        )
        # node_input_ids: [num_nodes, max_length]
        # node_attention_mask: [num_nodes, max_length]

        # 3. 提取角色 ID
        role_ids = torch.tensor(
            [n["role_id"] for n in nodes], dtype=torch.long
        )
        numeric_features = self._build_numeric_features(record, nodes)

        # 4. 构建 edge_index Tensor
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        # 确保形状为 [2, num_edges]
        if edge_index_tensor.dim() == 2 and edge_index_tensor.shape[0] == 2:
            pass  # 已经是正确形状
        else:
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
                    f"edge_type 长度异常: {edge_type_tensor.numel()}, "
                    f"期望 {num_edges}"
                )

        # 5. Tokenize 新闻文本
        news_encoding = self.tokenizer(
            news_text,
            padding="max_length",
            truncation=True,
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt",
        )
        # news_input_ids: [1, max_length]
        # news_attention_mask: [1, max_length]

        # 6. 构建 PyG Data 对象
        data = Data(
            # 节点文本 Token 信息
            node_input_ids=node_encodings["input_ids"],
            node_attention_mask=node_encodings["attention_mask"],
            # 角色 ID
            role_ids=role_ids,
            numeric_features=numeric_features,
            # 图结构
            edge_index=edge_index_tensor,
            edge_type=edge_type_tensor,
            # 新闻文本 Token 信息
            news_input_ids=news_encoding["input_ids"],
            news_attention_mask=news_encoding["attention_mask"],
            # 标签
            y=torch.tensor(label, dtype=torch.long),
            # 节点数 (供 PyG batching 使用)
            num_nodes=len(nodes),
        )

        return data

    @staticmethod
    def _remap_perspective_edge_types(
        edge_type: list[int],
        edge_index: list[list[int]],
        nodes: list[dict],
    ) -> list[int]:
        """把中立 perspective agent 之间的 attack 边改成 cross_perspective。"""
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
