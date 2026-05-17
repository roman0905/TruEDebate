"""
路 C：TED + PAMD 双流融合分类器（Dual-Stream Classifier）

设计动机
--------
- V2 baseline 显示：用同一个 Analysis Agent 跑 TED 二元辩论数据时 Test macF1 ≈ 0.803，
  但跑 PAMD 多视角数据时只有 0.7671（-3.6 点）；两种数据在分类信号上互补但都有缺陷。
- TED 二元辩论：pro/con 立场对比强，BERT 容易学到 stance 判别；但视角单一。
- PAMD 多视角：能识别细粒度操纵类型（事实/因果/时序/情绪/意图/语言风格），
  但所有 perspective 同源于一个 LLM，judge 节点存在信号稀释。
- 双流融合：让两套互补的辩论信号在 graph 表征层联合训练，论文同时保住两个创新点
  并显式建模"互补性"。

架构
----
- 共享 BERT 编码器（避免 220M 参数翻倍，3884 样本承受不起）。
- 共享 role_embedding / role_proj / numeric_proj / news_proj（同维度同语义空间）。
- 两路独立 GAT + 分层池化（边结构、节点数完全不同）。
- 两路各自得到 [B, 10·D] 拼接特征，再拼接成 [B, 20·D] 送入 fusion 分类器。
- aux loss / XPCR / DATR 信号都来自 PAMD 分支（TED 没有 perspective 节点）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config
from insight_flow.networks import TEDClassifier


class DualStreamClassifier(nn.Module):
    """TED + PAMD 双流融合分类器。"""

    def __init__(
        self,
        lang: str = "en",
        freeze_layers: int = config.BERT_FREEZE_LAYERS,
        use_typed_edges: bool = True,
    ):
        super().__init__()
        self.lang = lang

        # ── 两路 TEDClassifier 分支 ──
        # TED 分支：没有 perspective 节点，关闭 aux loss
        self.ted_branch = TEDClassifier(
            lang=lang,
            freeze_layers=freeze_layers,
            use_typed_edges=use_typed_edges,
            use_aux_loss=False,
        )
        # PAMD 分支：保留 aux loss，配合 V8 的 XPCR / DATR
        self.pamd_branch = TEDClassifier(
            lang=lang,
            freeze_layers=freeze_layers,
            use_typed_edges=use_typed_edges,
            use_aux_loss=True,
        )

        # ── 共享参数：BERT + role + numeric + news 投影 ──
        # 直接覆盖 pamd_branch 的模块为 ted_branch 的引用，相当于共享权重。
        self.pamd_branch.bert = self.ted_branch.bert
        self.pamd_branch.role_embedding = self.ted_branch.role_embedding
        self.pamd_branch.role_proj = self.ted_branch.role_proj
        self.pamd_branch.numeric_feature_proj = self.ted_branch.numeric_feature_proj
        self.pamd_branch.news_proj = self.ted_branch.news_proj
        # BERT 内部冻结状态已设置；共享后保持一致。

        # ── Fusion 头 ──
        ted_dim = self.ted_branch.combined_dim
        pamd_dim = self.pamd_branch.combined_dim
        combined_dim = ted_dim + pamd_dim
        proj_dim = config.PROJ_DIM

        self.fusion = nn.Sequential(
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

        # 对外接口属性
        self.combined_dim = combined_dim
        self.classifier = self.fusion          # 兼容 mixup
        self.use_aux_loss = True               # aux 信号经 PAMD 分支提供

    # ──────────────────────────────────────────────────────

    @staticmethod
    def _call_branch(branch: TEDClassifier, fields: dict, **kwargs):
        """统一调用 TEDClassifier.forward。"""
        return branch(
            node_input_ids=fields["node_input_ids"],
            node_attention_mask=fields["node_attention_mask"],
            role_ids=fields["role_ids"],
            node_group_ids=fields["node_group_ids"],
            numeric_features=fields["numeric_features"],
            edge_index=fields["edge_index"],
            edge_type=fields["edge_type"],
            batch=fields["batch"],
            news_input_ids=fields["news_input_ids"],
            news_attention_mask=fields["news_attention_mask"],
            **kwargs,
        )

    def forward(
        self,
        ted_fields: dict,
        pamd_fields: dict,
        return_aux: bool = False,
        return_features: bool = False,
        mixup_features: torch.Tensor | None = None,
    ):
        # ── TED 分支：取 combined（无 aux）──
        ted_out = self._call_branch(self.ted_branch, ted_fields, return_features=True)
        # ted_out 是 (logits, combined)
        _, ted_combined = ted_out

        # ── PAMD 分支：取 combined + 可选 aux ──
        if return_aux:
            pamd_out = self._call_branch(
                self.pamd_branch, pamd_fields,
                return_aux=True, return_features=True,
            )
            # (logits, aux_logits, aux_batch_ids, combined)
            _, aux_logits, aux_batch_ids, pamd_combined = pamd_out
        else:
            pamd_out = self._call_branch(
                self.pamd_branch, pamd_fields, return_features=True
            )
            _, pamd_combined = pamd_out
            aux_logits = pamd_combined.new_zeros((0, 2))
            aux_batch_ids = torch.empty(
                0, dtype=torch.long, device=pamd_combined.device
            )

        # ── 拼接两路特征 ──
        combined = torch.cat([ted_combined, pamd_combined], dim=-1)

        # Mixup（manifold mixup）支持
        if mixup_features is not None:
            combined = mixup_features

        logits = self.fusion(combined)

        if return_aux:
            if return_features:
                return logits, aux_logits, aux_batch_ids, combined
            return logits, aux_logits, aux_batch_ids
        if return_features:
            return logits, combined
        return logits
