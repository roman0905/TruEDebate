"""
TruEDebate (TED) — Analysis Agent 网络架构 V4 (Synthesis-as-Auxiliary-Query)

【V4 核心修正】根据论文 Algorithm 1 重新设计图结构与 Synthesis 的使用方式：

1. 【图结构修正】图 V 只含 6 个辩论节点（论文原意），不包含 Synthesis
2. 【Synthesis 作为辅助 Query】不作为图节点，而是：
   - 独立 BERT 编码得到 e_synth
   - 作为第二个 Query 对辩论图做 Cross-Attention: c_synth = MHA(e_synth, node_dense, node_dense)
   - 与 News Query 的交互表示 c_news 并列融合
   - 体现"Synthesis bridges the debate discourse and analytical processes" (论文原话)

3. 【保留双分支立场池化】显式建模 Pro vs Opp 的对抗
4. 【修复 MHA 退化】News 和 Synthesis 都对节点级表征做真正的交互注意力

最终分类器输入：[g_global; c_news; c_synth; divergence]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel

import config


class TEDClassifier(nn.Module):
    """
    TruEDebate Analysis Agent V4。

    输入:
        - node_input_ids: [6 辩论节点]  (不含 Synthesis)
        - news_input_ids: [新闻]
        - synth_input_ids: [Synthesis 辅助文本]

    输出: [batch_size, 2] 二分类 logits
    """

    PRO_ROLE_IDS = [0, 2, 4]
    OPP_ROLE_IDS = [1, 3, 5]

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

        bert_path = self._resolve_bert_path(bert_name)

        # ═══════ Sub-module 1: Role-aware Encoder (BERT + Role Embedding) ═══════

        # BERT 在辩论节点、新闻、Synthesis 间共享 (参数效率 + 语义一致性)
        self.bert = AutoModel.from_pretrained(bert_path)
        self._freeze_bert_layers(freeze_layers)

        self.role_embedding = nn.Embedding(num_roles, role_embed_dim)
        self.role_proj = nn.Linear(role_embed_dim, role_proj_dim, bias=False)

        node_dim = bert_hidden + role_proj_dim

        # ═══════ Sub-module 2: Debate Graph GAT (仅 6 辩论节点) ═══════

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        # 第一层: node_dim → gat_hidden_dim (multi-head concat)
        self.gat_layers.append(
            GATv2Conv(
                in_channels=node_dim,
                out_channels=gat_hidden_dim,
                heads=gat_heads,
                concat=True,
                dropout=gat_dropout,
                add_self_loops=True,
            )
        )
        self.gat_norms.append(nn.LayerNorm(gat_hidden_dim * gat_heads))

        # 第二层: gat_hidden_dim*heads → gat_hidden_dim (multi-head average)
        self.gat_layers.append(
            GATv2Conv(
                in_channels=gat_hidden_dim * gat_heads,
                out_channels=gat_hidden_dim,
                heads=gat_heads,
                concat=False,
                dropout=gat_dropout,
                add_self_loops=True,
            )
        )
        self.gat_norms.append(nn.LayerNorm(gat_hidden_dim))

        self.gat_activation = nn.ELU()
        self.gat_dropout = nn.Dropout(gat_dropout)

        # GAT 第一层残差连接 (维度对齐)
        self.gat_input_proj = nn.Linear(
            node_dim, gat_hidden_dim * gat_heads, bias=False
        )

        # ═══════ Sub-module 3: 投影层 (节点/新闻/Synthesis → proj_dim) ═══════

        self.node_proj = nn.Linear(gat_hidden_dim, proj_dim)
        self.news_proj = nn.Linear(bert_hidden, proj_dim)
        self.synth_proj = nn.Linear(bert_hidden, proj_dim)

        # ═══════ Sub-module 4: 双路 Cross-Attention ═══════

        # 【关键】News Cross-Attention: News 作为 Query，辩论节点作为 K/V
        self.news_cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=mha_heads,
            dropout=gat_dropout,
            batch_first=True,
        )
        self.news_attn_norm = nn.LayerNorm(proj_dim)

        # 【关键】Synthesis Cross-Attention: Synthesis 作为 Query，辩论节点作为 K/V
        # 这体现了论文 "Synthesis bridges the debate and analytical processes"
        self.synth_cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=mha_heads,
            dropout=gat_dropout,
            batch_first=True,
        )
        self.synth_attn_norm = nn.LayerNorm(proj_dim)

        # ═══════ Sub-module 5: 多视图融合分类器 ═══════

        # 输入: [g_global; c_news; c_synth; divergence] = 4 * proj_dim
        fusion_input_dim = proj_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(proj_dim, 2),
        )

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
        """BERT 编码，返回 [CLS] token 向量。"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state[:, 0, :]

    def _masked_mean_pool(
        self,
        node_dense: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """基于 mask 的安全均值池化。"""
        mask_f = mask.unsqueeze(-1).float()
        weighted = node_dense * mask_f
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return weighted.sum(dim=1) / denom

    def forward(
        self,
        node_input_ids: torch.Tensor,
        node_attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        news_input_ids: torch.Tensor,
        news_attention_mask: torch.Tensor,
        synth_input_ids: torch.Tensor,
        synth_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # ── 1. Role-aware Encoder (6 辩论节点) ──
        node_text_features = self._encode_texts(
            node_input_ids, node_attention_mask
        )  # [total_nodes=B*6, 768]
        role_emb = self.role_embedding(role_ids)
        role_proj = self.role_proj(role_emb)
        node_features = torch.cat(
            [node_text_features, role_proj], dim=-1
        )  # [total_nodes, node_dim]

        # ── 2. GAT 消息传播 (带残差) ──
        x = node_features
        x_residual = self.gat_input_proj(node_features)

        for i, (gat_layer, norm) in enumerate(
            zip(self.gat_layers, self.gat_norms)
        ):
            x_in = x
            x = gat_layer(x, edge_index)
            x = norm(x)

            if i == 0:
                x = x + x_residual
            elif x_in.shape == x.shape:
                x = x + x_in

            if i < len(self.gat_layers) - 1:
                x = self.gat_activation(x)
                x = self.gat_dropout(x)

        # x: [total_nodes, gat_hidden_dim]

        # ── 3. 节点级投影 + 稠密批 ──
        node_proj = self.node_proj(x)  # [total_nodes, P]
        node_dense, node_mask = to_dense_batch(node_proj, batch)
        # node_dense: [B, 6, P]; node_mask: [B, 6] (全为 True，因为每图 6 节点)
        role_dense, _ = to_dense_batch(role_ids, batch, fill_value=-1)

        # ── 4. 新闻编码 + 投影 ──
        news_features = self._encode_texts(news_input_ids, news_attention_mask)
        e_news = self.news_proj(news_features)  # [B, P]

        # ── 5. Synthesis 编码 + 投影 ──
        synth_features = self._encode_texts(synth_input_ids, synth_attention_mask)
        e_synth = self.synth_proj(synth_features)  # [B, P]

        # ── 6. 【核心1】News Cross-Attention ──
        q_news = e_news.unsqueeze(1)  # [B, 1, P]
        c_news, _ = self.news_cross_attn(
            q_news,
            node_dense,
            node_dense,
            key_padding_mask=~node_mask,
        )
        c_news = c_news.squeeze(1)  # [B, P]
        c_news = self.news_attn_norm(c_news + e_news)  # 残差

        # ── 7. 【核心2】Synthesis Cross-Attention ──
        q_synth = e_synth.unsqueeze(1)  # [B, 1, P]
        c_synth, _ = self.synth_cross_attn(
            q_synth,
            node_dense,
            node_dense,
            key_padding_mask=~node_mask,
        )
        c_synth = c_synth.squeeze(1)  # [B, P]
        c_synth = self.synth_attn_norm(c_synth + e_synth)  # 残差

        # ── 8. 全图均值池化 ──
        g_global = self._masked_mean_pool(node_dense, node_mask)  # [B, P]

        # ── 9. 【核心3】立场感知双分支聚合 ──
        pro_mask = torch.zeros_like(node_mask)
        for r in self.PRO_ROLE_IDS:
            pro_mask = pro_mask | (role_dense == r)
        pro_mask = pro_mask & node_mask

        opp_mask = torch.zeros_like(node_mask)
        for r in self.OPP_ROLE_IDS:
            opp_mask = opp_mask | (role_dense == r)
        opp_mask = opp_mask & node_mask

        g_pro = self._masked_mean_pool(node_dense, pro_mask)
        g_opp = self._masked_mean_pool(node_dense, opp_mask)

        # 立场分歧信号
        divergence = torch.abs(g_pro - g_opp)  # [B, P]

        # ── 10. 多视图融合 + 分类 ──
        combined = torch.cat(
            [g_global, c_news, c_synth, divergence],
            dim=-1,
        )  # [B, 4P]

        logits = self.classifier(combined)  # [B, 2]

        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    注意：V4 默认使用 CrossEntropy + class_weight (config.USE_FOCAL_LOSS=False)
    因为 V3 实验显示 Focal Loss 反而过拟合、val acc 下降。
    保留此类以便消融实验。
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                soft_targets = torch.full_like(
                    log_probs,
                    self.label_smoothing / (num_classes - 1),
                )
                soft_targets.scatter_(
                    1,
                    targets.unsqueeze(1),
                    1.0 - self.label_smoothing,
                )
            ce = -(soft_targets * log_probs).sum(dim=-1)
            with torch.no_grad():
                p_t = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            ce = F.nll_loss(log_probs, targets, reduction="none")
            p_t = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - p_t).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * focal_weight * ce
        else:
            loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
