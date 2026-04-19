"""
TruEDebate (TED) — PyG Dataset 定义 (V4)

【V4 重要改进】根据论文 Algorithm 1 修正图结构：
- 论文中 V = {h_i}，节点来自辩论交互 n_i ∈ D，仅 6 个辩论节点
- Synthesis (S) 仅用于可解释性 R = {S, D}，不是图节点
- 因此 Dataset 自动从 7 节点 JSON 中分离：
    - 前 6 节点 (role_id 0-5) → 辩论图
    - 第 7 节点 (role_id 6)   → 独立的 Synthesis 辅助信息
- 使用 config.EDGE_LIST 作为辩论图边 (不使用 JSON 里的 edge_index)
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
    辩论图数据集 (V4)。

    每条样本对应一篇新闻，包含：
        - 6 个辩论节点 (Pro/Opp × 开篇/质询/结案)
        - 1 个 Synthesis 辅助文本
        - 1 个原始新闻文本
        - 标签 (real=0, fake=1)

    图结构: 仅 6 个辩论节点，边由 config.EDGE_LIST 定义。
    Synthesis 作为独立辅助信息，在模型中与新闻一起参与 Cross-Attention。
    """

    SYNTHESIS_ROLE_ID = 6

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

        # 预计算辩论图的 edge_index (所有样本共享相同的图结构)
        src_list = [e[0] for e in config.EDGE_LIST]
        dst_list = [e[1] for e in config.EDGE_LIST]
        self.debate_edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )  # [2, num_edges]

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

    def __getitem__(self, idx: int) -> Data:
        """
        构造单条样本。

        Returns PyG Data 对象，包含：
            - node_input_ids:      [6, max_length] 辩论节点的 Token IDs
            - node_attention_mask: [6, max_length]
            - role_ids:            [6] 辩论节点角色 ID (0-5)
            - edge_index:          [2, num_edges] 辩论图边 (来自 config.EDGE_LIST)
            - news_input_ids:      [1, max_length] 新闻 Token
            - news_attention_mask: [1, max_length]
            - synth_input_ids:     [1, max_length] Synthesis 辅助文本 Token
            - synth_attention_mask:[1, max_length]
            - y:                   标量标签
        """
        # 1. 读取 JSON
        with open(self.file_paths[idx], "r", encoding="utf-8") as f:
            record = json.load(f)

        nodes = record["nodes"]
        label = record["label"]
        news_text = record["news_text"]

        # 2. 分离辩论节点 (role_id 0-5) 与 Synthesis 节点 (role_id 6)
        debate_nodes = []
        synth_text = ""
        for n in nodes:
            if n["role_id"] == self.SYNTHESIS_ROLE_ID:
                synth_text = n["text"]
            else:
                debate_nodes.append(n)

        # 按 role_id 排序，确保顺序一致 (0, 1, 2, 3, 4, 5)
        debate_nodes.sort(key=lambda n: n["role_id"])

        # 如果没有 Synthesis 文本 (极少数数据异常)，使用空字符串 + 警告
        if not synth_text:
            synth_text = record.get("synthesis", "")

        # 如果辩论节点不足 6 个，报错 (数据异常)
        if len(debate_nodes) != 6:
            raise ValueError(
                f"文件 {self.file_paths[idx]} 含有 {len(debate_nodes)} 个辩论节点，"
                f"期望 6 个。请检查数据生成。"
            )

        # 3. 标签标准化
        if isinstance(label, str):
            label = config.LABEL_MAP.get(label, label)
        label = int(label)

        # 4. Tokenize 辩论节点文本 [6, max_length]
        node_texts = [n["text"] for n in debate_nodes]
        node_encodings = self.tokenizer(
            node_texts,
            padding="max_length",
            truncation=True,
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt",
        )

        # 5. 角色 ID [6]
        role_ids = torch.tensor(
            [n["role_id"] for n in debate_nodes], dtype=torch.long
        )

        # 6. Tokenize 新闻文本
        news_encoding = self.tokenizer(
            news_text,
            padding="max_length",
            truncation=True,
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt",
        )

        # 7. Tokenize Synthesis 文本
        synth_encoding = self.tokenizer(
            synth_text,
            padding="max_length",
            truncation=True,
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt",
        )

        # 8. 构建 PyG Data 对象
        data = Data(
            node_input_ids=node_encodings["input_ids"],
            node_attention_mask=node_encodings["attention_mask"],
            role_ids=role_ids,
            edge_index=self.debate_edge_index.clone(),  # 辩论图边 (6 节点)
            news_input_ids=news_encoding["input_ids"],
            news_attention_mask=news_encoding["attention_mask"],
            synth_input_ids=synth_encoding["input_ids"],
            synth_attention_mask=synth_encoding["attention_mask"],
            y=torch.tensor(label, dtype=torch.long),
            num_nodes=len(debate_nodes),  # 6
        )

        return data
