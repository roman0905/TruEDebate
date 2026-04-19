"""
TruEDebate (TED) — Analysis Agent 网络架构 V3 (Innovation Edition)

针对原论文实现的关键问题，进行了三大创新改进：

【创新点 1】Node-level Cross-Attention (修复论文公式退化)
原论文 Eq.10: c = MHA(e_F^proj, g^proj, g^proj) 中 g^proj 是单一向量，
导致 MHA 退化为简单线性变换。本实现将 K/V 改为 GAT 输出的所有节点表征，
让 News 真正"注意"到不同的辩论发言。

【创新点 2】Stance-Aware Dual-Branch Pooling (利用辩论的二分图结构)
辩论天然是 Proponent vs Opponent 的二分结构。本实现：
  - 分别聚合正方节点 (roles 0,2,4) 得到 g_pro
  - 分别聚合反方节点 (roles 1,3,5) 得到 g_opp
  - 计算立场分歧 divergence = |g_pro - g_opp| 作为显式信号

【创新点 3】Multi-View Feature Fusion (多视图特征融合)
最终分类器输入：[g_global; c_attention; divergence]
  - g_global: 全图聚合表示 (论文原始)
  - c_attention: 节点级交互注意力 (创新点1)
  - divergence: 立场分歧信号 (创新点2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel

import config


class TEDClassifier(nn.Module):
    """
    TruEDebate Analysis Agent 增强版本 (V3)。

    核心创新：
    1. 修复 MHA 退化：使用节点级 K/V 进行真正的交互注意力
    2. 立场感知双分支：分别聚合 Pro/Opp 阵营，计算分歧信号
    3. 多视图融合：全图 + 注意力 + 立场分歧 三路特征
    4. GATv2Conv：比 GATConv 更具表达力的注意力机制
    """

    # Pro/Opp 角色 ID 定义 (与 config.ROLE_IDS 对应)
    PRO_ROLE_IDS = [0, 2, 4]  # proponent_opening, proponent_questioner, proponent_closing
    OPP_ROLE_IDS = [1, 3, 5]  # opponent_opening, opponent_questioner, opponent_closing
    SYNTHESIS_ROLE_ID = 6

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

        # ═══════ Sub-module 1: Role-aware Encoder ═══════

        self.bert = AutoModel.from_pretrained(bert_path)
        self._freeze_bert_layers(freeze_layers)

        self.role_embedding = nn.Embedding(num_roles, role_embed_dim)
        self.role_proj = nn.Linear(role_embed_dim, role_proj_dim, bias=False)

        node_dim = bert_hidden + role_proj_dim

        # ═══════ Sub-module 2: GAT (使用 GATv2 提升表达力) ═══════

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        # 第一层：node_dim → gat_hidden_dim (multi-head concat)
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

        # 第二层：gat_hidden_dim*heads → gat_hidden_dim (multi-head average)
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

        # GAT 输入维度对齐（用于残差连接）
        self.gat_input_proj = nn.Linear(node_dim, gat_hidden_dim * gat_heads, bias=False)

        # ═══════ Sub-module 3: 节点级投影 + 多路注意力 ═══════

        # 节点投影：将 GAT 输出映射到 proj_dim
        self.node_proj = nn.Linear(gat_hidden_dim, proj_dim)
        self.news_proj = nn.Linear(bert_hidden, proj_dim)

        # 【创新1】节点级交互注意力（修复 MHA 退化）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=mha_heads,
            dropout=gat_dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(proj_dim)

        # ═══════ Sub-module 4: 多视图分类器 ═══════

        # 输入特征维度: [g_global(P) + c_attn(P) + divergence(P)] = 3P
        # 其中 P = proj_dim
        fusion_input_dim = proj_dim * 3

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
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_features = outputs.last_hidden_state[:, 0, :]
        return cls_features

    def _masked_mean_pool(
        self,
        node_dense: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于 mask 的安全均值池化。

        Args:
            node_dense: [B, N, D] 节点稠密表征
            mask: [B, N] 布尔 mask (True 表示有效节点)

        Returns:
            pooled: [B, D] 池化后的图级表征
        """
        mask_f = mask.unsqueeze(-1).float()  # [B, N, 1]
        weighted = node_dense * mask_f
        denom = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
        pooled = weighted.sum(dim=1) / denom
        return pooled

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
        # ── 1. Role-aware Encoder ──
        node_text_features = self._encode_texts(node_input_ids, node_attention_mask)
        role_emb = self.role_embedding(role_ids)
        role_proj = self.role_proj(role_emb)
        node_features = torch.cat([node_text_features, role_proj], dim=-1)

        # ── 2. GAT 消息传播 (带残差连接) ──
        x = node_features
        x_residual = self.gat_input_proj(node_features)  # 用于第一层残差

        for i, (gat_layer, norm) in enumerate(
            zip(self.gat_layers, self.gat_norms)
        ):
            x_in = x
            x = gat_layer(x, edge_index)
            x = norm(x)

            # 残差连接：第一层用 gat_input_proj，第二层用前层输出
            if i == 0:
                x = x + x_residual
            elif x_in.shape == x.shape:
                x = x + x_in

            if i < len(self.gat_layers) - 1:
                x = self.gat_activation(x)
                x = self.gat_dropout(x)

        # x: [total_nodes, gat_hidden_dim]

        # ── 3. 节点级投影 + 转为稠密批 ──
        node_proj = self.node_proj(x)  # [total_nodes, proj_dim]
        # 转换为 dense 形式: [B, max_nodes, proj_dim] + mask
        node_dense, node_mask = to_dense_batch(node_proj, batch)
        # role_ids 也转为 dense 形式以便构造 Pro/Opp mask
        role_dense, _ = to_dense_batch(role_ids, batch, fill_value=-1)

        # ── 4. 新闻编码 + 投影 ──
        news_features = self._encode_texts(news_input_ids, news_attention_mask)
        e_proj = self.news_proj(news_features)  # [B, proj_dim]

        # ── 5. 【创新1】节点级 Cross-Attention ──
        # News 作为 Query，所有辩论节点作为 K/V
        # KV 是 [B, max_nodes, P]，真正实现多对一的交互注意力
        q = e_proj.unsqueeze(1)  # [B, 1, P]
        # key_padding_mask: True 表示需要被忽略
        attn_output, _ = self.cross_attn(
            q,
            node_dense,
            node_dense,
            key_padding_mask=~node_mask,
        )
        c_attn = attn_output.squeeze(1)  # [B, P]
        # 残差 + LayerNorm，稳定训练
        c_attn = self.cross_attn_norm(c_attn + e_proj)

        # ── 6. 全图均值池化（论文原始路径）──
        g_global = self._masked_mean_pool(node_dense, node_mask)  # [B, P]

        # ── 7. 【创新2】立场感知双分支聚合 ──
        # 构造 Pro/Opp 掩码（基于 role_id）
        pro_mask = torch.zeros_like(node_mask)
        for r in self.PRO_ROLE_IDS:
            pro_mask = pro_mask | (role_dense == r)
        pro_mask = pro_mask & node_mask

        opp_mask = torch.zeros_like(node_mask)
        for r in self.OPP_ROLE_IDS:
            opp_mask = opp_mask | (role_dense == r)
        opp_mask = opp_mask & node_mask

        g_pro = self._masked_mean_pool(node_dense, pro_mask)  # [B, P]
        g_opp = self._masked_mean_pool(node_dense, opp_mask)  # [B, P]

        # 立场分歧信号（绝对差异）
        divergence = torch.abs(g_pro - g_opp)  # [B, P]

        # ── 8. 多视图特征融合 + 分类 ──
        combined = torch.cat([g_global, c_attn, divergence], dim=-1)  # [B, 3P]
        logits = self.classifier(combined)  # [B, 2]

        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    针对 ARG-EN 的严重类别不平衡 (Real:Fake ≈ 3:1 训练集，4:1 验证/测试集)
    Focal Loss 对 hard examples (即模型预测不准的样本) 给予更高权重，
    比单纯的 class weight 更适合处理动态难度。

    支持 label smoothing。
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
        """
        Args:
            logits: [B, num_classes]
            targets: [B] 整数类标签
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]

        # Label smoothing 转为软标签
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
            ce = -(soft_targets * log_probs).sum(dim=-1)  # [B]
            # 计算 p_t (真实类的预测概率)
            with torch.no_grad():
                p_t = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            ce = F.nll_loss(log_probs, targets, reduction="none")  # [B]
            p_t = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t).pow(self.gamma)

        # Class weight (alpha)
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
