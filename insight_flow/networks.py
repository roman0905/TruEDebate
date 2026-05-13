"""
TruEDebate (TED/PAMD) — Analysis Agent 网络架构（V7 创新版本）

相对原始 TED 的核心模型创新：

1. 分层节点感知（Hierarchical Node-Group Pooling） — V4
2. 新闻条件注意力池化（News-Conditional Attention Pooling） — V4
3. 节点 / 边 Dropout 正则化 — V4
4. 辅助 Perspective 分类损失（Multi-task Aux Loss） — V4
5. BERT4 trick: 最后 N 层 CLS 平均 — V6

V7 新增三大方法学贡献（model-level，对应 TED 三个具体漏洞）：

6. **FNACA**（Fine-Grained News-Argument Co-Attention）
   修补 TED 漏洞 W2：原 TED 把 graph 池化为单向量再做 MHA，注意力名存实亡。
   FNACA 让每个 debate turn 的 CLS 在 GAT 之前对 news token 序列做
   多头 cross-attention，捕获 fine-grained 新闻-论据对齐。

7. **SCCG**（Self-Consistency Credibility Gating）
   修补 TED 漏洞 W3：原 TED 等权聚合所有 turn，hallucination 一并进入 verdict。
   SCCG 从三信号（c_internal / c_grounded / c_cross）算出每个 turn 的可信度
   c_i，门控 GAT 输入特征与图池化阶段。

8. **SCRA**（Stance-Contrastive Representation Auxiliary）
   修补 TED 漏洞 W5：原 TED BERT 编码与立场无关。
   SCRA 在 graph-level 表示上加 SupCon 对比损失，强迫表示空间按真假分簇。
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


# ──────────────────────────────── V7: FNACA ────────────────────────────────


class FNACA(nn.Module):
    """Fine-Grained News-Argument Co-Attention.

    每个 turn 的 CLS 表示对 news 的 token 序列做多头 cross-attention，
    获得 news-anchored turn 表示。修补 TED 的 W2：原 TED 的 MHA 把
    graph 池化为单向量再做 attention，无法做 token-level 对齐。

    Forward:
        turn_cls    : [N, D]      所有 turn 的 CLS 表示（已 batch 拼接）。
        news_seq    : [B, L, D]   每个样本新闻的 token 表示序列。
        news_mask   : [B, L]      news 的 attention_mask（True = 有效 token）。
        batch_vec   : [N]         每个 turn 属于哪个 sample。

    Returns:
        enriched    : [N, D]      news-enriched turn 表示（residual）。
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        turn_cls: torch.Tensor,
        news_seq: torch.Tensor,
        news_mask: torch.Tensor,
        batch_vec: torch.Tensor,
    ) -> torch.Tensor:
        # 把 news 序列按 batch_vec 广播到每个 turn。
        news_per_turn = news_seq[batch_vec]                   # [N, L, D]
        mask_per_turn = news_mask[batch_vec]                  # [N, L]

        q = turn_cls.unsqueeze(1)                             # [N, 1, D]
        attn_out, _ = self.cross_attn(
            q, news_per_turn, news_per_turn,
            key_padding_mask=~mask_per_turn.bool(),
        )
        attn_out = attn_out.squeeze(1)                        # [N, D]
        # Residual + Norm
        x = self.norm(turn_cls + attn_out)
        x = self.ffn_norm(x + self.ffn(x))
        return x


# ──────────────────────────────── V7: SCCG ────────────────────────────────


class CredibilityGate(nn.Module):
    """Self-Consistency Credibility Gating.

    用三个信号衡量每个 turn 的可信度：
      c_internal : 该 turn 内部前半段 vs 后半段的语义自洽度（cosine sim）。
      c_grounded : 该 turn 与所属样本新闻 CLS 的相似度（事实接地）。
      c_cross    : 该 turn 与同 sample 其他 turn 平均表示的相似度（跨论据共识）。

    将这三个信号通过小 MLP 融合，sigmoid 输出 c_i ∈ (0, 1)。
    初始化偏置使训练初期 c_i ≈ 1（不要立即砍掉信号）。

    Returns:
        c_i        : [N, 1]   可信度门控值。
        c_features : [N, 3]   三个原始信号，便于解释和监控。
    """

    def __init__(self, init_bias: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        # 让最后一层 bias 偏正，初始 c_i 接近 1（sigmoid(2) ≈ 0.88）。
        with torch.no_grad():
            self.mlp[-1].bias.fill_(init_bias)

    def forward(
        self,
        turn_cls: torch.Tensor,
        first_pool: torch.Tensor,
        second_pool: torch.Tensor,
        news_cls: torch.Tensor,
        batch_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # c_internal: cos(first_half, second_half)
        c_internal = F.cosine_similarity(first_pool, second_pool, dim=-1)

        # c_grounded: cos(turn_cls, news_cls_of_this_sample)
        news_per_turn = news_cls[batch_vec]                   # [N, D]
        c_grounded = F.cosine_similarity(turn_cls, news_per_turn, dim=-1)

        # c_cross: cos(turn_cls, mean(other turns in same sample))
        # 用 index_add 计算每个 sample 的 turn 平均向量。
        batch_size = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1
        d = turn_cls.shape[-1]
        sums = torch.zeros(batch_size, d, device=turn_cls.device, dtype=turn_cls.dtype)
        counts = torch.zeros(batch_size, device=turn_cls.device, dtype=turn_cls.dtype)
        sums.index_add_(0, batch_vec, turn_cls)
        counts.index_add_(
            0, batch_vec,
            torch.ones_like(batch_vec, dtype=turn_cls.dtype),
        )
        sample_means = sums / counts.unsqueeze(-1).clamp_min(1.0)
        means_per_turn = sample_means[batch_vec]              # [N, D]
        c_cross = F.cosine_similarity(turn_cls, means_per_turn, dim=-1)

        features = torch.stack([c_internal, c_grounded, c_cross], dim=-1)  # [N, 3]
        c = torch.sigmoid(self.mlp(features))                 # [N, 1]
        return c, features


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
        # V6：BERT4 trick — 是否对最后 N 层 CLS 取平均。
        self.use_bert4 = getattr(config, "USE_BERT4", False)
        self.bert4_layers = getattr(config, "BERT4_LAYERS", 4)
        # V7：三个创新模块的启用开关。
        self.use_fnaca = getattr(config, "USE_FNACA", False)
        self.use_sccg = getattr(config, "USE_SCCG", False)
        self.sccg_gate_gat_input = getattr(config, "SCCG_GATE_GAT_INPUT", True)
        self.sccg_gate_gat_output = getattr(config, "SCCG_GATE_GAT_OUTPUT", True)

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

        # ═══════ V7 Sub-module 1.5: FNACA + SCCG ═══════
        if self.use_fnaca:
            self.fnaca = FNACA(
                dim=bert_hidden,
                num_heads=getattr(config, "FNACA_HEADS", 4),
                dropout=getattr(config, "FNACA_DROPOUT", 0.1),
            )
        else:
            self.fnaca = None

        if self.use_sccg:
            self.credibility_gate = CredibilityGate(
                init_bias=getattr(config, "SCCG_INIT_BIAS", 2.0),
                dropout=gat_dropout,
            )
        else:
            self.credibility_gate = None

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
        # V6：恢复 V4 的 5 层分类 MLP（V5 减到 3 层后头部表征不足）。
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(proj_dim * 2, proj_dim),
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
        return_rich: bool = False,
    ):
        """编码文本。

        - return_rich=False（默认，向后兼容）：仅返回 [N, D] 的 CLS 表征。
        - return_rich=True：返回 dict，包含
            cls          : [N, D]      CLS 表征（BERT4 trick 已生效）。
            first_pool   : [N, D]      tokens 前半段（mask-aware）mean-pool。
            second_pool  : [N, D]      tokens 后半段 mean-pool。
            seq          : [N, L, D]   最后层（或 BERT4 平均）的 token 序列。
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.use_bert4,
        )
        last_hidden = outputs.last_hidden_state
        if self.use_bert4:
            last_n = outputs.hidden_states[-self.bert4_layers:]
            stacked = torch.stack(last_n, dim=0)
            last_hidden = stacked.mean(dim=0)

        cls = last_hidden[:, 0, :]
        if not return_rich:
            return cls

        seq_len = last_hidden.shape[1]
        mid = seq_len // 2
        mask_f = attention_mask.float().unsqueeze(-1)        # [N, L, 1]
        masked = last_hidden * mask_f
        first_pool = (
            masked[:, :mid].sum(dim=1)
            / mask_f[:, :mid].sum(dim=1).clamp_min(1e-6)
        )
        second_pool = (
            masked[:, mid:].sum(dim=1)
            / mask_f[:, mid:].sum(dim=1).clamp_min(1e-6)
        )
        return {
            "cls": cls,
            "first_pool": first_pool,
            "second_pool": second_pool,
            "seq": last_hidden,
        }

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
        # ── 1. Role-aware Encoder（V7：富信息编码以支持 FNACA / SCCG）──
        need_rich = self.use_fnaca or self.use_sccg
        if need_rich:
            node_enc = self._encode_texts(
                node_input_ids, node_attention_mask, return_rich=True
            )
            node_text_features = node_enc["cls"]
            node_first_pool = node_enc["first_pool"]
            node_second_pool = node_enc["second_pool"]
        else:
            node_text_features = self._encode_texts(node_input_ids, node_attention_mask)
            node_first_pool = node_second_pool = None

        # ── 1b. 新闻编码（V7：提前到此处，供 FNACA / SCCG 使用）──
        if self.use_fnaca or self.use_sccg:
            news_enc = self._encode_texts(
                news_input_ids, news_attention_mask, return_rich=True
            )
            news_features = news_enc["cls"]
            news_seq = news_enc["seq"]
        else:
            news_features = self._encode_texts(news_input_ids, news_attention_mask)
            news_seq = None

        # ── 1c. V7 FNACA：每个 turn 的 CLS 对 news token 序列做 cross-attention ──
        if self.use_fnaca and self.fnaca is not None and news_seq is not None:
            node_text_features = self.fnaca(
                turn_cls=node_text_features,
                news_seq=news_seq,
                news_mask=news_attention_mask.bool(),
                batch_vec=batch,
            )

        # ── 1d. V7 SCCG：算每个 turn 的可信度 c_i ──
        credibility = None
        if self.use_sccg and self.credibility_gate is not None:
            credibility, _c_feats = self.credibility_gate(
                turn_cls=node_text_features,
                first_pool=node_first_pool,
                second_pool=node_second_pool,
                news_cls=news_features,
                batch_vec=batch,
            )

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

        # V7 SCCG：用 c_i 门控 GAT 输入特征（低可信度的 turn 信号被衰减）。
        if (
            self.use_sccg
            and self.sccg_gate_gat_input
            and credibility is not None
        ):
            # credibility: [N, 1]，element-wise 缩放
            node_features = node_features * credibility

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

        # V7 SCCG：再次用 c_i 门控 GAT 输出，确保池化阶段也尊重可信度。
        if (
            self.use_sccg
            and self.sccg_gate_gat_output
            and credibility is not None
        ):
            x = x * credibility

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

        # ── 4. 新闻投影（V7：news_features 已在 step 1b 取到）──
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

        if return_aux:
            # 即使 aux loss 未启用，也按调用约定返回 4-tuple，方便 train 端解包。
            if self.use_aux_loss:
                persp_indices = persp_mask.nonzero(as_tuple=False)
                if persp_indices.numel() == 0:
                    aux_logits = logits.new_zeros((0, 2))
                    aux_batch_ids = torch.empty(
                        0, dtype=torch.long, device=logits.device
                    )
                else:
                    b_idx = persp_indices[:, 0]
                    n_idx = persp_indices[:, 1]
                    persp_feats = dense_nodes[b_idx, n_idx]
                    aux_logits = self.aux_head(persp_feats)
                    aux_batch_ids = b_idx
            else:
                aux_logits = logits.new_zeros((0, 2))
                aux_batch_ids = torch.empty(
                    0, dtype=torch.long, device=logits.device
                )
            if return_features:
                return logits, aux_logits, aux_batch_ids, combined
            return logits, aux_logits, aux_batch_ids

        if return_features:
            return logits, combined

        return logits
