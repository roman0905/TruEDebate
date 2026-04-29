"""
R-TED — Analysis Agent 网络

使用关系感知图卷积近似异构论证图推理，并加入独立 news/meta 分支。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel

import config


class TEDClassifier(nn.Module):
    """R-TED classifier."""

    NODE_TYPE_IDS = {
        "claim": 0,
        "td_rationale": 1,
        "cs_rationale": 2,
        "argument": 3,
        "synthesis": 4,
        "evidence": 5,
    }
    PRO_SPEAKER_ROLE_IDS = {1, 3, 5}
    OPP_SPEAKER_ROLE_IDS = {2, 4, 6}

    def __init__(
        self,
        lang: str = "en",
        node_type_embed_dim: int = config.ROLE_EMBED_DIM,
        speaker_role_embed_dim: int = config.ROLE_EMBED_DIM,
        rgcn_hidden_dim: int = config.GAT_HIDDEN_DIM,
        rgcn_layers: int = config.GAT_LAYERS,
        proj_dim: int = config.PROJ_DIM,
        classifier_dropout: float = config.CLASSIFIER_DROPOUT,
        freeze_layers: int = config.BERT_FREEZE_LAYERS,
    ):
        super().__init__()

        self.lang = lang
        bert_name = config.BERT_MODELS.get(lang, config.BERT_MODELS["en"])
        bert_hidden = config.BERT_HIDDEN_DIM
        bert_path = self._resolve_bert_path(bert_name)

        self.bert = AutoModel.from_pretrained(bert_path)
        self._freeze_bert_layers(freeze_layers)

        self.node_type_embedding = nn.Embedding(len(self.NODE_TYPE_IDS), node_type_embed_dim)
        self.speaker_role_embedding = nn.Embedding(len(config.ROLE_IDS) + 1, speaker_role_embed_dim)
        node_input_dim = bert_hidden + node_type_embed_dim + speaker_role_embed_dim
        self.input_proj = nn.Linear(node_input_dim, rgcn_hidden_dim)
        self.input_norm = nn.LayerNorm(rgcn_hidden_dim)

        self.rgcn_layers = nn.ModuleList()
        self.rgcn_norms = nn.ModuleList()
        for _ in range(max(rgcn_layers, 2)):
            self.rgcn_layers.append(
                RGCNConv(
                    rgcn_hidden_dim,
                    rgcn_hidden_dim,
                    num_relations=config.EVITED_NUM_EDGE_TYPES,
                )
            )
            self.rgcn_norms.append(nn.LayerNorm(rgcn_hidden_dim))

        self.dropout = nn.Dropout(config.GAT_DROPOUT)
        self.activation = nn.GELU()
        self.node_proj = nn.Linear(rgcn_hidden_dim, proj_dim)
        self.news_proj = nn.Sequential(
            nn.Linear(bert_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.source_embedding = nn.Embedding(config.SOURCE_EMBED_BUCKETS, proj_dim // 2)
        self.source_encoder = nn.Sequential(
            nn.Linear(proj_dim // 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(config.TIME_FEATURE_DIM, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.meta_fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.teacher_encoder = nn.Sequential(
            nn.Linear(config.TEACHER_FEATURE_DIM, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.evidence_interactive_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=config.MHA_HEADS,
            dropout=classifier_dropout,
            batch_first=True,
        )
        self.evidence_attn_norm = nn.LayerNorm(proj_dim)
        self.node_type_attention_bias = nn.Embedding(len(self.NODE_TYPE_IDS), 1)
        nn.init.zeros_(self.node_type_attention_bias.weight)
        self.relation_embedding = nn.Embedding(config.EVITED_NUM_EDGE_TYPES, proj_dim)
        self.edge_pair_encoder = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
        )
        self.edge_scorer = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(proj_dim, 1),
        )

        gate_input_dim = proj_dim * 10
        self.alpha_news = self._build_gate(gate_input_dim, proj_dim, classifier_dropout)
        self.alpha_debate = self._build_gate(gate_input_dim, proj_dim, classifier_dropout)
        self.alpha_td = self._build_gate(gate_input_dim, proj_dim, classifier_dropout)
        self.alpha_cs = self._build_gate(gate_input_dim, proj_dim, classifier_dropout)
        self.alpha_evidence = self._build_gate(gate_input_dim, proj_dim, classifier_dropout)
        self.alpha_graph = self._build_gate(gate_input_dim, proj_dim, classifier_dropout)

        self.debate_fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.conflict_fusion = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.final_proj = nn.Sequential(
            nn.Linear(proj_dim * 6, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(proj_dim, 2),
        )

    @staticmethod
    def _build_gate(input_dim: int, proj_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _resolve_bert_path(bert_name: str) -> str:
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

    def _encode_texts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    @staticmethod
    def _masked_mean_pool(node_dense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        weighted = node_dense * mask_f
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return weighted.sum(dim=1) / denom

    def _pool_by_node_type(
        self,
        node_dense: torch.Tensor,
        node_mask: torch.Tensor,
        node_type_dense: torch.Tensor,
        node_type: str,
    ) -> torch.Tensor:
        mask = (node_type_dense == self.NODE_TYPE_IDS[node_type]) & node_mask
        return self._masked_mean_pool(node_dense, mask)

    def _pool_argument_side(
        self,
        node_dense: torch.Tensor,
        node_mask: torch.Tensor,
        node_type_dense: torch.Tensor,
        speaker_role_dense: torch.Tensor,
        target_roles: set[int],
    ) -> torch.Tensor:
        arg_mask = (node_type_dense == self.NODE_TYPE_IDS["argument"]) & node_mask
        role_mask = torch.zeros_like(node_mask)
        for role_id in target_roles:
            role_mask = role_mask | (speaker_role_dense == role_id)
        return self._masked_mean_pool(node_dense, arg_mask & role_mask)

    def _edge_reconstruction_stats(
        self,
        node_repr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        if edge_index.numel() == 0:
            graph_nll = node_repr.new_zeros(num_graphs)
            return {
                "structure_loss": node_repr.new_zeros(()),
                "structure_nll": graph_nll,
                "structure_likelihood": torch.ones_like(graph_nll),
            }

        pair_to_relations: dict[tuple[int, int], set[int]] = {}
        src_list = edge_index[0].detach().cpu().tolist()
        dst_list = edge_index[1].detach().cpu().tolist()
        type_list = edge_type.clamp(min=0, max=config.EVITED_NUM_EDGE_TYPES - 1).detach().cpu().tolist()
        for src, dst, rel_type in zip(src_list, dst_list, type_list):
            pair_to_relations.setdefault((int(src), int(dst)), set()).add(int(rel_type))

        negative_pairs = self._sample_negative_pairs(
            positive_pairs=set(pair_to_relations),
            batch=batch,
            max_pairs=len(pair_to_relations),
        )
        all_pairs = list(pair_to_relations) + negative_pairs
        if not all_pairs:
            graph_nll = node_repr.new_zeros(num_graphs)
            return {
                "structure_loss": node_repr.new_zeros(()),
                "structure_nll": graph_nll,
                "structure_likelihood": torch.ones_like(graph_nll),
            }

        pair_tensor = torch.tensor(all_pairs, dtype=torch.long, device=node_repr.device)
        features = torch.cat([node_repr[pair_tensor[:, 0]], node_repr[pair_tensor[:, 1]]], dim=-1)
        logits = self._predict_edge_relations(features)
        targets = logits.new_zeros((len(all_pairs), config.EVITED_NUM_EDGE_TYPES))
        for row, pair in enumerate(pair_to_relations):
            rel_ids = list(pair_to_relations[pair])
            targets[row, rel_ids] = 1.0

        pair_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        ).mean(dim=-1)
        pair_graph = batch[pair_tensor[:, 0]]
        loss_sum = pair_loss.new_zeros(num_graphs)
        counts = pair_loss.new_zeros(num_graphs)
        loss_sum.index_add_(0, pair_graph, pair_loss)
        counts.index_add_(0, pair_graph, torch.ones_like(pair_loss))
        graph_nll = loss_sum / counts.clamp(min=1.0)
        return {
            "structure_loss": pair_loss.mean(),
            "structure_nll": graph_nll,
            "structure_likelihood": torch.exp(-graph_nll).clamp(min=1e-4, max=1.0),
        }

    def _predict_edge_relations(self, pair_features: torch.Tensor) -> torch.Tensor:
        pair_repr = self.edge_pair_encoder(pair_features)
        relation_repr = self.relation_embedding.weight.unsqueeze(0).expand(
            pair_repr.size(0),
            -1,
            -1,
        )
        pair_repr = pair_repr.unsqueeze(1).expand(-1, config.EVITED_NUM_EDGE_TYPES, -1)
        relation_features = torch.cat(
            [pair_repr, relation_repr, pair_repr * relation_repr],
            dim=-1,
        )
        return self.edge_scorer(relation_features).squeeze(-1)

    @staticmethod
    def _sample_negative_pairs(
        positive_pairs: set[tuple[int, int]],
        batch: torch.Tensor,
        max_pairs: int,
    ) -> list[tuple[int, int]]:
        if max_pairs <= 0:
            return []

        batch_list = batch.detach().cpu().tolist()
        nodes_by_graph: dict[int, list[int]] = {}
        for node_idx, graph_id in enumerate(batch_list):
            nodes_by_graph.setdefault(int(graph_id), []).append(node_idx)

        negative_pairs: list[tuple[int, int]] = []
        for src, dst in positive_pairs:
            graph_nodes = nodes_by_graph.get(batch_list[src], [])
            if len(graph_nodes) < 2:
                continue
            try:
                start = (graph_nodes.index(dst) + 1) % len(graph_nodes)
            except ValueError:
                start = 0
            for offset in range(len(graph_nodes)):
                cand = graph_nodes[(start + offset) % len(graph_nodes)]
                pair = (src, cand)
                if cand != src and pair not in positive_pairs:
                    negative_pairs.append(pair)
                    break
            if len(negative_pairs) >= max_pairs:
                break
        return negative_pairs

    def _evidence_aware_attention(
        self,
        h_news: torch.Tensor,
        node_dense: torch.Tensor,
        node_mask: torch.Tensor,
        node_type_dense: torch.Tensor,
        node_reliability_dense: torch.Tensor,
    ) -> torch.Tensor:
        type_ids = node_type_dense.clamp(min=0).long()
        type_bias = self.node_type_attention_bias(type_ids).squeeze(-1)
        reliability_bias = node_reliability_dense.clamp(min=1e-4, max=1.0).log()
        attn_bias = (type_bias + reliability_bias).masked_fill(~node_mask, -1e4)
        attn_mask = attn_bias[:, None, :].repeat_interleave(
            self.evidence_interactive_attn.num_heads,
            dim=0,
        )
        attn_mask = attn_mask.to(dtype=h_news.dtype, device=h_news.device)
        h_attn, _ = self.evidence_interactive_attn(
            h_news.unsqueeze(1),
            node_dense,
            node_dense,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return self.evidence_attn_norm(h_attn.squeeze(1) + h_news)

    def forward(
        self,
        node_input_ids: torch.Tensor,
        node_attention_mask: torch.Tensor,
        news_input_ids: torch.Tensor,
        news_attention_mask: torch.Tensor,
        node_type_ids: torch.Tensor,
        speaker_role_ids: torch.Tensor,
        node_reliability: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        source_bucket: torch.Tensor,
        time_features: torch.Tensor,
        teacher_features: torch.Tensor,
        batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_features = self._encode_texts(node_input_ids, node_attention_mask)
        type_features = self.node_type_embedding(node_type_ids)
        role_features = self.speaker_role_embedding(speaker_role_ids)
        x = torch.cat([text_features, type_features, role_features], dim=-1)
        x = self.input_norm(self.input_proj(x))

        for layer, norm in zip(self.rgcn_layers, self.rgcn_norms):
            x_in = x
            x = layer(x, edge_index, edge_type)
            x = norm(x + x_in)
            x = self.activation(x)
            x = self.dropout(x)

        node_proj = self.node_proj(x)
        node_dense, node_mask = to_dense_batch(node_proj, batch)
        node_type_dense, _ = to_dense_batch(node_type_ids, batch, fill_value=-1)
        speaker_role_dense, _ = to_dense_batch(speaker_role_ids, batch, fill_value=0)
        node_reliability_dense, _ = to_dense_batch(node_reliability, batch, fill_value=0.0)

        news_features = self._encode_texts(news_input_ids, news_attention_mask)
        h_news = self.news_proj(news_features)
        h_claim = self._pool_by_node_type(node_dense, node_mask, node_type_dense, "claim")
        h_td = self._pool_by_node_type(node_dense, node_mask, node_type_dense, "td_rationale")
        h_cs = self._pool_by_node_type(node_dense, node_mask, node_type_dense, "cs_rationale")
        h_evidence = self._pool_by_node_type(node_dense, node_mask, node_type_dense, "evidence")
        h_synth = self._pool_by_node_type(node_dense, node_mask, node_type_dense, "synthesis")
        h_graph = self._masked_mean_pool(node_dense, node_mask)

        h_arg = self._pool_by_node_type(node_dense, node_mask, node_type_dense, "argument")
        h_debate = self.debate_fusion(torch.cat([h_arg, h_synth], dim=-1))
        h_source = self.source_encoder(self.source_embedding(source_bucket))
        h_time = self.time_encoder(time_features.float())
        h_meta = self.meta_fusion(torch.cat([h_source, h_time], dim=-1))
        h_teacher = self.teacher_encoder(teacher_features.float())
        h_attn = self._evidence_aware_attention(
            h_news,
            node_dense,
            node_mask,
            node_type_dense,
            node_reliability_dense,
        )

        g_pro = self._pool_argument_side(
            node_dense,
            node_mask,
            node_type_dense,
            speaker_role_dense,
            self.PRO_SPEAKER_ROLE_IDS,
        )
        g_opp = self._pool_argument_side(
            node_dense,
            node_mask,
            node_type_dense,
            speaker_role_dense,
            self.OPP_SPEAKER_ROLE_IDS,
        )
        divergence = torch.abs(g_pro - g_opp)
        conflict_repr = self.conflict_fusion(
            torch.cat([g_pro, g_opp, divergence, g_pro * g_opp], dim=-1)
        )

        gate_context = torch.cat(
            [h_news, h_attn, h_debate, h_td, h_cs, h_evidence, h_claim, h_meta, h_teacher, h_graph],
            dim=-1,
        )
        alpha_news = self.alpha_news(gate_context)
        alpha_debate = self.alpha_debate(gate_context)
        alpha_td = self.alpha_td(gate_context)
        alpha_cs = self.alpha_cs(gate_context)
        alpha_evidence = self.alpha_evidence(gate_context)
        alpha_graph = self.alpha_graph(gate_context)

        weighted_sum = (
            alpha_news * h_news
            + alpha_debate * h_debate
            + alpha_td * h_td
            + alpha_cs * h_cs
            + alpha_evidence * h_evidence
            + alpha_graph * h_graph
        )

        final_input = torch.cat([weighted_sum, h_attn, h_claim, h_meta, h_teacher, conflict_repr], dim=-1)
        logits = self.final_proj(final_input)
        structure_stats = self._edge_reconstruction_stats(node_proj, edge_index, edge_type, batch)
        return {
            "logits": logits,
            **structure_stats,
            "alpha_news": alpha_news.squeeze(-1),
            "alpha_debate": alpha_debate.squeeze(-1),
            "alpha_td": alpha_td.squeeze(-1),
            "alpha_cs": alpha_cs.squeeze(-1),
            "alpha_evidence": alpha_evidence.squeeze(-1),
            "alpha_graph": alpha_graph.squeeze(-1),
        }


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

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

    def per_sample_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                soft_targets = torch.full_like(
                    log_probs,
                    self.label_smoothing / (num_classes - 1),
                )
                soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(soft_targets * log_probs).sum(dim=-1)
            p_t = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            ce = F.nll_loss(log_probs, targets, reduction="none")
            p_t = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = (1.0 - p_t).pow(self.gamma) * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss
        return loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.per_sample_loss(logits, targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
