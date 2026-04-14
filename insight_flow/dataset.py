"""
TruEDebate (TED) — PyG Dataset 定义
加载生成的辩论 JSON 文件，构建 PyG Data 对象（节点特征、边索引、标签）。
在 __getitem__ 中动态计算 Token ID 以节省内存。
"""

import json
import logging
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

    def __getitem__(self, idx: int) -> Data:
        """
        加载第 idx 条辩论记录并构建 PyG Data 对象。

        Returns:
            Data 对象包含:
                - node_input_ids:      [num_nodes, max_length] Token IDs
                - node_attention_mask: [num_nodes, max_length] Attention masks
                - role_ids:            [num_nodes] 角色 ID
                - edge_index:          [2, num_edges] 有向边索引
                - news_input_ids:      [1, max_length] 新闻 Token IDs
                - news_attention_mask: [1, max_length] 新闻 Attention masks
                - y:                   标量标签 (0=real, 1=fake)
        """
        # 1. 读取 JSON 文件
        with open(self.file_paths[idx], "r", encoding="utf-8") as f:
            record = json.load(f)

        nodes = record["nodes"]
        edge_index = record["edge_index"]
        label = record["label"]
        news_text = record["news_text"]

        # 标签标准化: 将字符串标签映射为整数
        if isinstance(label, str):
            label = config.LABEL_MAP.get(label, label)
        label = int(label)

        # 2. Tokenize 每个节点文本
        node_texts = [n["text"] for n in nodes]
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

        # 4. 构建 edge_index Tensor
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        # 确保形状为 [2, num_edges]
        if edge_index_tensor.dim() == 2 and edge_index_tensor.shape[0] == 2:
            pass  # 已经是正确形状
        else:
            raise ValueError(
                f"edge_index 形状异常: {edge_index_tensor.shape}, 期望 [2, num_edges]"
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
            # 图结构
            edge_index=edge_index_tensor,
            # 新闻文本 Token 信息
            news_input_ids=news_encoding["input_ids"],
            news_attention_mask=news_encoding["attention_mask"],
            # 标签
            y=torch.tensor(label, dtype=torch.long),
            # 节点数 (供 PyG batching 使用)
            num_nodes=len(nodes),
        )

        return data
