"""
TruEDebate (TED/PAMD) — Analysis Agent 网络架构（创新版本）

相对原始实现的核心改进：

1. 分层节点感知（Hierarchical Node-Group Pooling）
   - 不再用 global_mean_pool 把 9 个角色相异的节点平均成一个向量；
   - 改为按 planner / perspective / coordinator / judge 四个语义组分别池化。

2. 新闻条件注意力池化（News-Conditional Attention Pooling）
   - 每个组的池化权重由"新闻 [CLS] 表征"决定，避免无关节点稀释关键信号。

3. 新闻 ↔ Perspective Cross-Attention
   - 让新闻原文显式对每个 perspective 节点做交互，捕获细粒度对齐。

4. Perspective 辅助分类头（Multi-task Aux Loss）
   - 每个 perspective 节点都有自己的二分类头，强制视角层面学到判别信号。

5. 节点 / 边 Dropout（GraphDropout 正则化）
   - 训练时随机 mask perspective 节点特征和图边，提升泛化。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel

import config


# ──────────────────────────────── 工具模块 ────────────────────────────────


class NewsConditionalAttentionPool(nn.Module):
    """以 news 表征为 query，对一组节点做加权池化。

    Args:
        nodes: [B, N, D]
        news:  [B, D]
        mask:  [B, N]  True 表示该位置有效

    Returns:
        pooled: [B, D]，对于全部节点都被 mask 掉的样本输出零向量。
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim 必须能整除 num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        nodes: torch.Tensor,
        news: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = nodes.shape
        h = self.num_heads
        hd = self.head_dim

        q = self.q_proj(news).view(b, h, 1, hd)             # [B, H, 1, hd]
        k = self.k_proj(nodes).view(b, n, h, hd).transpose(1, 2)  # [B, H, N, hd]
        v = self.v_proj(nodes).view(b, n, h, hd).transpose(1, 2)  # [B, H, N, hd]

        scores = (q @ k.transpose(-1, -2)) * self.scale     # [B, H, 1, N]

        # 对全部无效的样本，给出全零输出，避免 softmax 对 -inf 行产生 NaN。
        all_invalid = ~mask.any(dim=-1)                     # [B]
        # 临时把全无效样本的第 0 个位置改为有效，softmax 后再 zero 掉。
        adjusted_mask = mask.clone()
        if all_invalid.any():
            adjusted_mask[all_invalid, 0] = True
        scores = scores.masked_fill(
            ~adjusted_mask.unsqueeze(1).unsqueeze(1), float("-inf")
        )
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v                                       # [B, H, 1, hd]
        out = out.transpose(1, 2).contiguous().view(b, 1, d)
        out = self.out_proj(out).squeeze(1)                  # [B, D]
        if all_invalid.any():
            out = out * (~all_invalid).float().unsqueeze(-1)
        return out


def masked_mean(
    nodes: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """[B, N, D] + [B, N] → [B, D]"""
    mask_f = mask.float().unsqueeze(-1)
    summed = (nodes * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(eps)
    return summed / denom


def masked_max(nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B, N, D] + [B, N] → [B, D]"""
    neg_inf = torch.finfo(nodes.dtype).min
    filled = nodes.masked_fill(~mask.unsqueeze(-1), neg_inf)
    out = filled.max(dim=1).values
    # 对全部 False 的样本，把 -inf 替换为 0
    invalid = ~mask.any(dim=-1, keepdim=True)
    out = torch.where(invalid, torch.zeros_like(out), out)
    return out


# ──────────────────────────────── 主模型 ────────────────────────────────


class TEDClassifier(nn.Module):
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
        freeze_layers: int = config.BERT_FREEZE_LAYERS,
        use_typed_edges: bool = True,
        numeric_feature_dim: int = config.NUMERIC_FEATURE_DIM,
        numeric_feature_proj_dim: int = config.NUMERIC_FEATURE_PROJ_DIM,
        node_dropout_p: float = config.NODE_DROPOUT_P,
        edge_dropout_p: float = config.EDGE_DROPOUT_P,
        use_aux_loss: bool = config.USE_AUX_LOSS,
    ):
        super().__init__()

        self.lang = lang
        self.use_typed_edges = use_typed_edges
        self.node_dropout_p = node_dropout_p
        self.edge_dropout_p = edge_dropout_p
        self.use_aux_loss = use_aux_loss
        self.proj_dim = proj_dim

        bert_name = config.BERT_MODELS.get(lang, config.BERT_MODELS["en"])
        bert_hidden = config.BERT_HIDDEN_DIM

        bert_path = self._resolve_bert_path(bert_name)

        # ═══════ Sub-module 1: Role-aware Encoder ═══════
        self.bert = AutoModel.from_pretrained(bert_path)
        self._freeze_bert_layers(freeze_layers)

        self.role_embedding = nn.Embedding(num_roles, role_embed_dim)
        self.role_proj = nn.Linear(role_embed_dim, role_proj_dim)

        self.numeric_feature_dim = numeric_feature_dim
        self.numeric_feature_proj = nn.Sequential(
            nn.Linear(numeric_feature_dim, numeric_feature_proj_dim),
            nn.LayerNorm(numeric_feature_proj_dim),
            nn.ReLU(),
            nn.Dropout(gat_dropout),
        )

        node_dim = bert_hidden + role_proj_dim + numeric_feature_proj_dim

        # ═══════ Sub-module 2: GAT ═══════
        if self.use_typed_edges:
            self.edge_type_embedding = nn.Embedding(
                config.NUM_EDGE_TYPES,
                config.EDGE_TYPE_EMBED_DIM,
            )
            edge_dim = config.EDGE_TYPE_EMBED_DIM
        else:
            self.edge_type_embedding = None
            edge_dim = None

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        self.gat_layers.append(
            GATConv(
                in_channels=node_dim,
                out_channels=gat_hidden_dim,
                heads=gat_heads,
                concat=True,
                dropout=gat_dropout,
                edge_dim=edge_dim,
            )
        )
        self.gat_norms.append(nn.LayerNorm(gat_hidden_dim * gat_heads))

        for _ in range(gat_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=gat_hidden_dim * gat_heads,
                    out_channels=gat_hidden_dim,
                    heads=gat_heads,
                    concat=True,
                    dropout=gat_dropout,
                    edge_dim=edge_dim,
                )
            )
            self.gat_norms.append(nn.LayerNorm(gat_hidden_dim * gat_heads))

        self.gat_layers.append(
            GATConv(
                in_channels=gat_hidden_dim * gat_heads,
                out_channels=gat_hidden_dim,
                heads=1,
                concat=False,
                dropout=gat_dropout,
                edge_dim=edge_dim,
            )
        )
        self.gat_norms.append(nn.LayerNorm(gat_hidden_dim))

        self.gat_activation = nn.ELU()
        self.gat_dropout = nn.Dropout(gat_dropout)

        # ═══════ Sub-module 3: 分层池化与交互 ═══════
        self.node_proj = nn.Linear(gat_hidden_dim, proj_dim)
        self.news_proj = nn.Linear(bert_hidden, proj_dim)

        # 4 个分组各自一个 news-conditional attention pool。
        self.plan_pool = NewsConditionalAttentionPool(proj_dim, mha_heads, gat_dropout)
        self.persp_pool = NewsConditionalAttentionPool(proj_dim, mha_heads, gat_dropout)
        self.coord_pool = NewsConditionalAttentionPool(proj_dim, mha_heads, gat_dropout)
        self.judge_pool = NewsConditionalAttentionPool(proj_dim, mha_heads, gat_dropout)

        # 新闻 ↔ Perspective Cross-Attention（双向：让新闻去 query 各 perspective）。
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=mha_heads,
            dropout=gat_dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(proj_dim)

        # ═══════ Sub-module 4: 分类头 ═══════
        # 拼接特征：news / plan_pool / persp_pool / coord_pool / judge_pool
        #         / global_mean / persp_max / cross_attn / |news-persp| / news*persp
        # 共 10 * proj_dim 维。
        combined_dim = proj_dim * 10
        self.combined_dim = combined_dim
        # V5：减一层 MLP，降低头部容量（V4 五层 → 三层），缓解过拟合。
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(proj_dim, 2),
        )

        # Perspective 辅助分类头（共享权重）：[B*N_persp, proj_dim] → [B*N_persp, 2]
        self.aux_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(proj_dim, 2),
        )

    # ──────────────────────────────── 辅助函数 ────────────────────────────────

    @staticmethod
    def _resolve_bert_path(bert_name: str) -> str:
        from pathlib import Path

        dir_name = bert_name.split("/")[-1]
        local_path = config.BERT_LOCAL_DIR / dir_name
        if local_path.exists() and (local_path / "config.json").exists():
            return str(local_path)
        full_local = config.BERT_LOCAL_DIR / bert_name
        if full_local.exists() and (full_local / "config.json").exists():
            return str(full_local)
        return bert_name

    def _freeze_bert_layers(self, freeze_layers: int) -> None:
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def _encode_texts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def _drop_perspective_nodes(
        self,
        node_features: torch.Tensor,
        node_group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """训练时随机 mask perspective 节点的特征。"""
        if not self.training or self.node_dropout_p <= 0:
            return node_features
        is_perspective = node_group_ids == config.NODE_GROUP_PERSPECTIVE
        if not is_perspective.any():
            return node_features
        keep_prob = 1.0 - self.node_dropout_p
        rand_mask = torch.rand(
            node_features.shape[0], 1,
            device=node_features.device,
            dtype=node_features.dtype,
        )
        drop = (rand_mask >= keep_prob) & is_perspective.unsqueeze(-1)
        return node_features.masked_fill(drop, 0.0)

    def _drop_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.training or self.edge_dropout_p <= 0:
            return edge_index, edge_type
        if edge_index.shape[1] == 0:
            return edge_index, edge_type
        keep = torch.rand(edge_index.shape[1], device=edge_index.device) > self.edge_dropout_p
        if not keep.any():
            return edge_index, edge_type
        new_edge_index = edge_index[:, keep]
        new_edge_type = edge_type[keep] if edge_type is not None else None
        return new_edge_index, new_edge_type

    # ──────────────────────────────── Forward ────────────────────────────────

    def forward(
        self,
        node_input_ids: torch.Tensor,
        node_attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        node_group_ids: torch.Tensor,
        numeric_features: torch.Tensor | None,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None,
        batch: torch.Tensor,
        news_input_ids: torch.Tensor,
        news_attention_mask: torch.Tensor,
        return_aux: bool = False,
        return_features: bool = False,
        mixup_features: torch.Tensor | None = None,
    ):
        # ── 1. Role-aware Encoder ──
        node_text_features = self._encode_texts(node_input_ids, node_attention_mask)
        role_emb = self.role_embedding(role_ids)
        role_proj = self.role_proj(role_emb)

        if numeric_features is None:
            numeric_features = torch.zeros(
                node_text_features.shape[0],
                self.numeric_feature_dim,
                dtype=node_text_features.dtype,
                device=node_text_features.device,
            )
        numeric_features = numeric_features.to(
            device=node_text_features.device,
            dtype=node_text_features.dtype,
        )
        numeric_proj = self.numeric_feature_proj(numeric_features)

        node_features = torch.cat(
            [node_text_features, role_proj, numeric_proj], dim=-1
        )

        # 节点 dropout：训练时随机 mask perspective 节点。
        node_features = self._drop_perspective_nodes(node_features, node_group_ids)

        # ── 2. GAT 消息传播（含边 dropout）──
        edge_index_used, edge_type_used = self._drop_edges(edge_index, edge_type)

        edge_attr = None
        if self.use_typed_edges:
            if edge_type_used is None:
                edge_type_used = torch.zeros(
                    edge_index_used.shape[1],
                    dtype=torch.long,
                    device=edge_index_used.device,
                )
            edge_attr = self.edge_type_embedding(edge_type_used)

        x = node_features
        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            x = gat_layer(x, edge_index_used, edge_attr=edge_attr)
            x = norm(x)
            if i < len(self.gat_layers) - 1:
                x = self.gat_activation(x)
                x = self.gat_dropout(x)

        # ── 3. 节点投影到统一维度 ──
        node_proj_feat = self.node_proj(x)                          # [N, D]

        dense_nodes, node_mask = to_dense_batch(node_proj_feat, batch)
        # node_mask: [B, max_N], True 表示该位置是真实节点。

        # 把 node_group_ids 也做 dense_batch。to_dense_batch 需要浮点输入。
        group_dense, _ = to_dense_batch(
            node_group_ids.float().unsqueeze(-1), batch, fill_value=-1.0
        )
        group_dense = group_dense.squeeze(-1).long()                # [B, max_N]

        plan_mask = node_mask & (group_dense == config.NODE_GROUP_PLANNER)
        persp_mask = node_mask & (group_dense == config.NODE_GROUP_PERSPECTIVE)
        coord_mask = node_mask & (group_dense == config.NODE_GROUP_COORDINATOR)
        judge_mask = node_mask & (group_dense == config.NODE_GROUP_JUDGE)

        # ── 4. 新闻编码 ──
        news_features = self._encode_texts(news_input_ids, news_attention_mask)
        news_proj_feat = self.news_proj(news_features)              # [B, D]

        # ── 5. 分组 news-conditional pool ──
        plan_pool = self.plan_pool(dense_nodes, news_proj_feat, plan_mask)
        persp_pool = self.persp_pool(dense_nodes, news_proj_feat, persp_mask)
        coord_pool = self.coord_pool(dense_nodes, news_proj_feat, coord_mask)
        judge_pool = self.judge_pool(dense_nodes, news_proj_feat, judge_mask)

        # ── 6. 全局均值与 perspective 最大值 ──
        global_mean = masked_mean(dense_nodes, node_mask)
        persp_max = masked_max(dense_nodes, persp_mask)

        # ── 7. 新闻 → Perspective Cross-Attention ──
        # 用 perspective 节点做 K/V，让新闻 query 出对其最相关的 perspective 表征。
        q_news = news_proj_feat.unsqueeze(1)                        # [B, 1, D]
        # 若某样本无 perspective（理论不存在），退化为对所有节点做 attn。
        kv_mask = persp_mask if persp_mask.any(dim=-1).all() else node_mask
        cross_out, _ = self.cross_attn(
            q_news,
            dense_nodes,
            dense_nodes,
            key_padding_mask=~kv_mask,
        )
        cross_out = self.cross_norm(cross_out.squeeze(1))           # [B, D]

        # ── 8. 拼接送入分类头 ──
        combined = torch.cat(
            [
                news_proj_feat,
                plan_pool,
                persp_pool,
                coord_pool,
                judge_pool,
                global_mean,
                persp_max,
                cross_out,
                torch.abs(news_proj_feat - persp_pool),
                news_proj_feat * persp_pool,
            ],
            dim=-1,
        )

        # V5：Mixup 直接替换 combined（外层负责混合 features 和 labels）。
        if mixup_features is not None:
            combined = mixup_features

        logits = self.classifier(combined)                          # [B, 2]

        if return_features and not return_aux:
            return logits, combined

        if return_aux and self.use_aux_loss:
            persp_indices = persp_mask.nonzero(as_tuple=False)
            if persp_indices.numel() == 0:
                aux_logits = logits.new_zeros((0, 2))
                aux_batch_ids = torch.empty(0, dtype=torch.long, device=logits.device)
            else:
                b_idx = persp_indices[:, 0]
                n_idx = persp_indices[:, 1]
                persp_feats = dense_nodes[b_idx, n_idx]
                aux_logits = self.aux_head(persp_feats)
                aux_batch_ids = b_idx
            if return_features:
                return logits, aux_logits, aux_batch_ids, combined
            return logits, aux_logits, aux_batch_ids

        return logits
