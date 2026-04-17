"""
TruEDebate (TED) — Analysis Agent 网络架构
包含: Role-aware Encoder (BERT+Role Embedding), GAT, Debate-News MHA, Classifier
严格按照论文模块 C 的网络结构实现。
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import AutoModel

import config


class TEDClassifier(nn.Module):
    """
    TruEDebate Analysis Agent 完整网络。

    架构:
        1. Role-aware Encoder: BERT(text) || Linear(RoleEmbed) → node features
        2. GAT: 多层 GATConv + global_mean_pool → graph-level debate repr
        3. Debate-News MHA: Query(news), Key/Value(debate graph repr)
        4. Classifier: Linear([debate_proj; mha_out]) → 2-class logits
    """

    def __init__(
        self,
        lang: str = "en",
        num_roles: int = config.NUM_ROLES,
        role_embed_dim: int = config.ROLE_EMBED_DIM,
        role_proj_dim: int = config.ROLE_PROJ_DIM,
        gat_hidden_dim: int = config.GAT_HIDDEN_DIM,
        gat_heads: int = config.GAT_HEADS,
        gat_layers: int = config.GAT_LAYERS,
        gat_dropout: float = config.GAT_DROPOUT,
        proj_dim: int = config.PROJ_DIM,
        mha_heads: int = config.MHA_HEADS,
        classifier_dropout: float = config.CLASSIFIER_DROPOUT,
        freeze_layers: int = config.BERT_FREEZE_LAYERS,
    ):
        super().__init__()

        self.lang = lang
        bert_name = config.BERT_MODELS.get(lang, config.BERT_MODELS["en"])
        bert_hidden = config.BERT_HIDDEN_DIM

        # 解析 BERT 模型路径: 优先使用本地目录，否则使用 HuggingFace 名称
        bert_path = self._resolve_bert_path(bert_name)

        # ═══════ Sub-module 1: Role-aware Encoder ═══════

        # Text Encoder: BERT
        self.bert = AutoModel.from_pretrained(bert_path)
        self._freeze_bert_layers(freeze_layers)

        # Role Embedding + Projection
        self.role_embedding = nn.Embedding(num_roles, role_embed_dim)
        self.role_proj = nn.Linear(role_embed_dim, role_proj_dim, bias=False)

        # Node feature dimension after concatenation
        node_dim = bert_hidden + role_proj_dim

        # ═══════ Sub-module 2: GAT ═══════

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        # 第一层 GAT: node_dim → gat_hidden_dim (multi-head, concat)
        self.gat_layers.append(
            GATConv(
                in_channels=node_dim,
                out_channels=gat_hidden_dim,
                heads=gat_heads,
                concat=True,
                dropout=gat_dropout,
            )
        )
        self.gat_norms.append(nn.LayerNorm(gat_hidden_dim * gat_heads))

        # 中间层 (如果 gat_layers > 2)
        for _ in range(gat_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=gat_hidden_dim * gat_heads,
                    out_channels=gat_hidden_dim,
                    heads=gat_heads,
                    concat=True,
                    dropout=gat_dropout,
                )
            )
            self.gat_norms.append(nn.LayerNorm(gat_hidden_dim * gat_heads))

        # 最后一层 GAT: → gat_hidden_dim (single-head, no concat)
        self.gat_layers.append(
            GATConv(
                in_channels=gat_hidden_dim * gat_heads,
                out_channels=gat_hidden_dim,
                heads=1,
                concat=False,
                dropout=gat_dropout,
            )
        )
        self.gat_norms.append(nn.LayerNorm(gat_hidden_dim))

        self.gat_activation = nn.ELU()
        self.gat_dropout = nn.Dropout(gat_dropout)

        # ═══════ Sub-module 3: Debate-News Interactive Attention ═══════

        # 投影层: 将 debate 图表征和 news 表征映射到相同维度
        self.debate_proj = nn.Linear(gat_hidden_dim, proj_dim)
        self.news_proj = nn.Linear(bert_hidden, proj_dim)

        # Multi-Head Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=mha_heads,
            batch_first=True,
        )
        self.mha_norm = nn.LayerNorm(proj_dim)

        # ═══════ Sub-module 4: Classifier (Eq.12) ═══════
        # 论文: y_hat = softmax(W_fc * h + b_fc)
        # CrossEntropyLoss 内置 softmax，所以输出 logits 即可
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(proj_dim * 2, 2)

    @staticmethod
    def _resolve_bert_path(bert_name: str) -> str:
        """
        解析 BERT 模型加载路径。
        优先检查本地 models/ 目录，若存在则使用本地路径，否则使用 HuggingFace 名称。

        本地目录结构示例:
            models/bert-base-uncased/config.json, model.safetensors, ...
            models/chinese-bert-wwm-ext/config.json, model.safetensors, ...
        """
        from pathlib import Path

        # 从 HuggingFace 名称提取目录名 (e.g. "hfl/chinese-bert-wwm-ext" → "chinese-bert-wwm-ext")
        dir_name = bert_name.split("/")[-1]
        local_path = config.BERT_LOCAL_DIR / dir_name

        if local_path.exists() and (local_path / "config.json").exists():
            return str(local_path)

        # 也尝试完整路径 (e.g. "models/hfl/chinese-bert-wwm-ext")
        full_local = config.BERT_LOCAL_DIR / bert_name
        if full_local.exists() and (full_local / "config.json").exists():
            return str(full_local)

        # 本地未找到，使用 HuggingFace 名称 (会自动下载)
        return bert_name

    def _freeze_bert_layers(self, freeze_layers: int) -> None:
        """冻结 BERT 的 embedding 层和前 N 个 encoder 层。"""
        # 冻结 embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # 冻结前 freeze_layers 个 encoder 层
        for layer in self.bert.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def _encode_texts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用 BERT 编码文本，返回 [CLS] 向量。

        Args:
            input_ids: [num_texts, seq_len]
            attention_mask: [num_texts, seq_len]

        Returns:
            cls_features: [num_texts, bert_hidden_dim]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # 取 [CLS] token 的输出 (第 0 个位置)
        cls_features = outputs.last_hidden_state[:, 0, :]  # [num_texts, 768]
        return cls_features

    def forward(
        self,
        node_input_ids: torch.Tensor,
        node_attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        news_input_ids: torch.Tensor,
        news_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        完整前向传播。

        Args:
            node_input_ids:      [total_nodes, seq_len]  所有图中节点的 token IDs
            node_attention_mask: [total_nodes, seq_len]  对应的 attention mask
            role_ids:            [total_nodes]           节点角色 ID
            edge_index:          [2, total_edges]        所有图中的边索引 (已由 PyG 合并)
            batch:               [total_nodes]           节点到图的映射 (PyG batch vector)
            news_input_ids:      [batch_size, seq_len]   每条新闻的 token IDs
            news_attention_mask: [batch_size, seq_len]   对应的 attention mask

        Returns:
            logits: [batch_size, 2]  二分类 logits
        """
        # ── 1. Role-aware Encoder ──

        # 1a. BERT 编码所有节点文本 → [total_nodes, 768]
        node_text_features = self._encode_texts(node_input_ids, node_attention_mask)

        # 1b. Role embedding + projection → [total_nodes, role_proj_dim]
        role_emb = self.role_embedding(role_ids)        # [total_nodes, role_embed_dim]
        role_proj = self.role_proj(role_emb)             # [total_nodes, role_proj_dim]

        # 1c. 拼接文本特征与角色特征 → [total_nodes, node_dim]
        node_features = torch.cat([node_text_features, role_proj], dim=-1)

        # ── 2. GAT 消息传播 ──

        x = node_features
        for i, (gat_layer, norm) in enumerate(
            zip(self.gat_layers, self.gat_norms)
        ):
            x = gat_layer(x, edge_index)       # GATConv
            x = norm(x)                         # LayerNorm
            if i < len(self.gat_layers) - 1:    # 非最后一层加激活和 dropout
                x = self.gat_activation(x)
                x = self.gat_dropout(x)

        # Global mean pooling → [batch_size, gat_hidden_dim]
        graph_repr = global_mean_pool(x, batch)

        # ── 3. Debate-News Interactive Attention ──

        # 3a. BERT 编码新闻文本 → [batch_size, 768]
        news_features = self._encode_texts(news_input_ids, news_attention_mask)

        # 3b. 线性投影到相同维度
        g_proj = self.debate_proj(graph_repr)   # [batch_size, proj_dim]
        e_proj = self.news_proj(news_features)  # [batch_size, proj_dim]

        # 3c. MHA: Query=news, Key/Value=图级 debate 表示
        q = e_proj.unsqueeze(1)     # [batch_size, 1, proj_dim]
        kv = g_proj.unsqueeze(1)    # [batch_size, 1, proj_dim]

        attn_output, _ = self.mha(q, kv, kv)   # [batch_size, 1, proj_dim]
        attn_output = attn_output.squeeze(1)     # [batch_size, proj_dim]
        attn_output = self.mha_norm(attn_output)

        # ── 4. 分类器 ──

        # 拼接 debate projection 和 MHA 交互表征
        combined = torch.cat([g_proj, attn_output], dim=-1)  # [batch_size, proj_dim*2]
        combined = self.classifier_dropout(combined)
        logits = self.classifier(combined)                    # [batch_size, 2]

        return logits
