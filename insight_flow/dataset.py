"""
R-TED — PyG Dataset 定义

图中仅保留高信息密度文本节点：
- claim
- td/cs rationale
- retrieval evidence
- debate arguments
- synthesis

news 通过独立编码器处理，source/time 通过元信息分支处理。
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from transformers import AutoTokenizer

import config
from debate_flow.prompts import (
    build_claim_records,
    build_internal_evidence_cards,
    build_rationale_cards,
    normalize_evidence_cards,
    render_synthesis_text,
)

logger = logging.getLogger(__name__)


class DebateGraphDataset(TorchDataset):
    """R-TED 图数据集。"""

    NODE_TYPE_IDS = {
        "claim": 0,
        "td_rationale": 1,
        "cs_rationale": 2,
        "argument": 3,
        "synthesis": 4,
        "evidence": 5,
    }
    EDGE_TYPE_IDS = {
        "claim_supported_by_td": 0,
        "td_supports_claim": 1,
        "claim_supported_by_cs": 2,
        "cs_supports_claim": 3,
        "argument_cites_claim": 4,
        "claim_cited_by_argument": 5,
        "argument_cites_td": 6,
        "td_cited_by_argument": 7,
        "argument_cites_cs": 8,
        "cs_cited_by_argument": 9,
        "argument_supports_claim": 10,
        "claim_supported_by_argument": 11,
        "argument_refutes_claim": 12,
        "claim_refuted_by_argument": 13,
        "argument_attacks_argument": 14,
        "synthesis_mentions_claim": 15,
        "claim_mentioned_by_synthesis": 16,
        "synthesis_uses_td": 17,
        "td_used_by_synthesis": 18,
        "synthesis_uses_cs": 19,
        "cs_used_by_synthesis": 20,
        "synthesis_reviews_argument": 21,
        "argument_reviewed_by_synthesis": 22,
        "claim_supported_by_evidence": 23,
        "evidence_supports_claim": 24,
        "claim_refuted_by_evidence": 25,
        "evidence_refutes_claim": 26,
        "claim_related_to_evidence": 27,
        "evidence_related_to_claim": 28,
        "argument_cites_evidence": 29,
        "evidence_cited_by_argument": 30,
    }

    ARGUMENT_ROLE_NAMES = [
        "proponent_opening",
        "opponent_opening",
        "proponent_questioner",
        "opponent_questioner",
        "proponent_closing",
        "opponent_closing",
    ]

    PRO_ROLE_IDS = {
        config.ROLE_IDS["proponent_opening"],
        config.ROLE_IDS["proponent_questioner"],
        config.ROLE_IDS["proponent_closing"],
    }
    OPP_ROLE_IDS = {
        config.ROLE_IDS["opponent_opening"],
        config.ROLE_IDS["opponent_questioner"],
        config.ROLE_IDS["opponent_closing"],
    }

    def __init__(self, data_dir: str | Path, lang: str = "en"):
        self.data_dir = Path(data_dir)
        self.lang = lang
        self.split = self._infer_split_from_dir()

        self.file_paths = sorted(self.data_dir.glob("*.json"))
        if len(self.file_paths) == 0:
            logger.warning("数据目录 %s 中没有找到 JSON 文件！", data_dir)

        logger.info("加载数据集: %s 个样本 (lang=%s)", len(self.file_paths), lang)

        bert_name = config.BERT_MODELS.get(lang, config.BERT_MODELS["en"])
        tokenizer_path = self._resolve_tokenizer_path(bert_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.source_record_map = self._load_source_record_map()

    def _infer_split_from_dir(self) -> str | None:
        dir_name = self.data_dir.name
        for split in ("train", "val", "test"):
            if dir_name.endswith(f"_{split}") or dir_name == split:
                return split
        return None

    def _load_source_record_map(self) -> dict[int, dict[str, Any]]:
        if self.split is None:
            return {}

        possible_paths = [
            config.DATA_DIR / self.lang / f"{self.split}.json",
            config.DATA_DIR / self.lang / f"{self.split}.jsonl",
            config.DATA_DIR / f"{self.lang}_{self.split}.json",
            config.DATA_DIR / f"{self.lang}_{self.split}.jsonl",
        ]
        file_path = next((p for p in possible_paths if p.exists()), None)
        if file_path is None:
            return {}

        records: list[dict[str, Any]] = []
        if file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    item.setdefault("id", i)
                    records.append(item)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    item.setdefault("id", i)
                    records.append(item)

        for item in records:
            if "text" not in item and "content" in item:
                item["text"] = item["content"]
            if "label" in item:
                item["label"] = config.LABEL_MAP.get(item["label"], item["label"])

        return {int(item["id"]): item for item in records}

    def _merge_source_record(self, record: dict[str, Any]) -> dict[str, Any]:
        source = self.source_record_map.get(int(record.get("id", -1)))
        if not source:
            return record

        merged = dict(record)
        fallback_keys = (
            "td_rationale",
            "cs_rationale",
            "td_pred",
            "cs_pred",
            "td_acc",
            "cs_acc",
            "time",
            "source_id",
        )
        for key in fallback_keys:
            value = merged.get(key)
            if value in (None, "", -1):
                if key in source:
                    merged[key] = source[key]

        if not merged.get("news_text") and source.get("text"):
            merged["news_text"] = source["text"]
        return merged

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

    @staticmethod
    def _normalize_label(label: Any) -> int:
        if isinstance(label, str):
            label = config.LABEL_MAP.get(label, label)
        return int(label)

    @staticmethod
    def _safe_float(value: Any, default: float = -1.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_time_features(time_str: Any) -> list[float]:
        from datetime import datetime

        time_text = str(time_str or "").strip()
        if not time_text:
            return [0.0] * config.TIME_FEATURE_DIM

        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
            try:
                parsed = datetime.strptime(time_text, fmt)
                break
            except ValueError:
                continue

        if parsed is None:
            return [0.0] * config.TIME_FEATURE_DIM

        year_norm = max(0.0, min((parsed.year - 2000) / 30.0, 1.0))
        month_norm = (parsed.month - 1) / 11.0
        day_norm = (parsed.day - 1) / 30.0
        hour_norm = parsed.hour / 23.0 if hasattr(parsed, "hour") else 0.0
        return [year_norm, month_norm, day_norm, hour_norm, 1.0]

    @staticmethod
    def _hash_source_bucket(source_id: Any) -> int:
        try:
            source_value = int(source_id)
        except (TypeError, ValueError):
            return 0
        if source_value < 0:
            return 0
        return (source_value % (config.SOURCE_EMBED_BUCKETS - 1)) + 1

    @staticmethod
    def _teacher_features(td_pred: float, cs_pred: float) -> list[float]:
        td_valid = 1.0 if td_pred in (0.0, 1.0) else 0.0
        cs_valid = 1.0 if cs_pred in (0.0, 1.0) else 0.0
        td_fake = 1.0 if td_pred == 1.0 else 0.0
        cs_fake = 1.0 if cs_pred == 1.0 else 0.0
        both_valid = td_valid * cs_valid
        agreement = 1.0 if both_valid and td_pred == cs_pred else 0.0
        disagreement = 1.0 if both_valid and td_pred != cs_pred else 0.0
        fake_count = td_fake + cs_fake
        fake_ratio = fake_count / max(td_valid + cs_valid, 1.0)
        return [td_valid, cs_valid, agreement, disagreement, fake_count / 2.0, fake_ratio]

    @staticmethod
    def _clamp_reliability(value: Any, default: float = 0.5) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = default
        return max(0.05, min(score, 1.0))

    def _evidence_reliability(self, card: dict[str, Any]) -> float:
        credibility = self._clamp_reliability(card.get("credibility_score", 0.5))
        retrieval = self._clamp_reliability(card.get("retrieval_score", 0.5))
        redundancy = self._clamp_reliability(card.get("redundancy_score", 0.0), default=0.0)
        score = 0.45 * credibility + 0.45 * retrieval + 0.10 * (1.0 - redundancy)
        return self._clamp_reliability(score)

    @staticmethod
    def _synthesis_conflict_text(synthesis_structured: dict[str, Any], fallback: str) -> str:
        pieces = [
            str(synthesis_structured.get("conflict_points", "")).strip(),
            str(synthesis_structured.get("explanation", "")).strip(),
        ]
        text = "\n".join(piece for piece in pieces if piece).strip()
        return text or fallback

    def __len__(self) -> int:
        return len(self.file_paths)

    def _ensure_claims(self, record: dict[str, Any], news_text: str) -> list[dict[str, str]]:
        claims = record.get("claims")
        if isinstance(claims, list) and claims:
            normalized = []
            for idx, claim in enumerate(claims, start=1):
                if isinstance(claim, dict):
                    claim_id = str(claim.get("id", f"c{idx}")).strip() or f"c{idx}"
                    content = str(claim.get("content", "")).strip()
                else:
                    claim_id = f"c{idx}"
                    content = str(claim).strip()
                if content:
                    normalized.append({"id": claim_id, "content": content})
            if normalized:
                return normalized

        claim_texts = record.get("claim_texts", [])
        if isinstance(claim_texts, str):
            claim_texts = [claim_texts]
        if not claim_texts:
            claim_texts = [" ".join(news_text.split()[:48]).strip()]
        return build_claim_records([str(x).strip() for x in claim_texts if str(x).strip()])

    def _ensure_rationale_cards(
        self,
        record: dict[str, Any],
        claims: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        cards = record.get("rationale_cards")
        if isinstance(cards, list) and cards:
            normalized = []
            for card in cards:
                if not isinstance(card, dict):
                    continue
                card_id = str(card.get("id", "")).strip()
                content = str(card.get("content", "")).strip()
                if not card_id or not content:
                    continue
                related_claims = card.get("related_claims", [])
                if not isinstance(related_claims, list):
                    related_claims = []
                normalized.append(
                    {
                        "id": card_id,
                        "type": str(card.get("type", "rationale")).strip(),
                        "content": content,
                        "related_claims": [str(x).strip() for x in related_claims if str(x).strip()],
                        "stance_pred": card.get("stance_pred", -1),
                    }
                )
            if normalized:
                return normalized

        return build_rationale_cards(
            claims,
            str(record.get("td_rationale", "") or ""),
            str(record.get("cs_rationale", "") or ""),
            td_pred=int(record.get("td_pred", -1)),
            cs_pred=int(record.get("cs_pred", -1)),
        )

    def _ensure_synthesis_structured(self, record: dict[str, Any]) -> dict[str, Any]:
        synthesis_structured = record.get("synthesis_structured")
        if isinstance(synthesis_structured, dict):
            return synthesis_structured
        return {
            "proponent_summary": "",
            "opponent_summary": "",
            "supported_claims": [],
            "questionable_claims": [],
            "used_td_rationales": [],
            "used_cs_rationales": [],
            "used_evidence": [],
            "conflict_points": "",
            "final_debate_tendency": "uncertain",
            "explanation": str(record.get("synthesis", "")).strip(),
        }

    def _ensure_evidence_cards(
        self,
        record: dict[str, Any],
        claims: list[dict[str, str]],
        rationale_cards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        cards = record.get("evidence_cards", [])
        internal_cards = build_internal_evidence_cards(
            rationale_cards,
            publish_time=str(record.get("time", "")),
        )
        if not isinstance(cards, list):
            cards = []
        return normalize_evidence_cards(cards + internal_cards, claims)

    def _build_graph(
        self,
        record: dict[str, Any],
        claims: list[dict[str, str]],
        rationale_cards: list[dict[str, Any]],
        evidence_cards: list[dict[str, Any]],
        synthesis_structured: dict[str, Any],
    ) -> tuple[list[str], list[int], list[int], list[float], list[int], list[int], list[int]]:
        node_texts: list[str] = []
        node_type_ids: list[int] = []
        speaker_role_ids: list[int] = []
        node_reliability: list[float] = []

        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_types: list[int] = []

        def add_node(text: str, node_type: str, reliability: float = 0.5) -> int:
            node_texts.append(text)
            node_type_ids.append(self.NODE_TYPE_IDS[node_type])
            speaker_role_ids.append(0)
            node_reliability.append(self._clamp_reliability(reliability))
            return len(node_texts) - 1

        def add_edge(src: int, dst: int, edge_type: str) -> None:
            edge_src.append(src)
            edge_dst.append(dst)
            edge_types.append(self.EDGE_TYPE_IDS[edge_type])

        claim_idx_map: dict[str, int] = {}
        for claim in claims:
            claim_idx_map[claim["id"]] = add_node(claim["content"], "claim", reliability=0.75)

        card_idx_map: dict[str, int] = {}
        for card in rationale_cards:
            node_type = "td_rationale" if card["id"].startswith("td_") else "cs_rationale"
            card_idx = add_node(card["content"], node_type, reliability=0.65)
            card_idx_map[card["id"]] = card_idx
            for claim_id in card.get("related_claims", []):
                if claim_id not in claim_idx_map:
                    continue
                if node_type == "td_rationale":
                    add_edge(claim_idx_map[claim_id], card_idx, "claim_supported_by_td")
                    add_edge(card_idx, claim_idx_map[claim_id], "td_supports_claim")
                else:
                    add_edge(claim_idx_map[claim_id], card_idx, "claim_supported_by_cs")
                    add_edge(card_idx, claim_idx_map[claim_id], "cs_supports_claim")

        evidence_idx_map: dict[str, int] = {}
        evidence_idx_by_rationale: dict[str, int] = {}
        for card in evidence_cards:
            text = str(card.get("evidence_text", "")).strip()
            if not text:
                continue
            stance = str(card.get("stance", "neutral")).strip().lower()
            source = str(card.get("source", "")).strip()
            evidence_text = f"Stance: {stance}\nSource: {source}\nEvidence: {text}".strip()
            evidence_idx = add_node(evidence_text, "evidence", reliability=self._evidence_reliability(card))
            evidence_idx_map[card["id"]] = evidence_idx
            rationale_id = str(card.get("rationale_id", "")).strip()
            if rationale_id:
                evidence_idx_by_rationale[rationale_id] = evidence_idx
            for claim_id in card.get("related_claims", []):
                if claim_id not in claim_idx_map:
                    continue
                if stance == "support":
                    add_edge(claim_idx_map[claim_id], evidence_idx, "claim_supported_by_evidence")
                    add_edge(evidence_idx, claim_idx_map[claim_id], "evidence_supports_claim")
                elif stance == "refute":
                    add_edge(claim_idx_map[claim_id], evidence_idx, "claim_refuted_by_evidence")
                    add_edge(evidence_idx, claim_idx_map[claim_id], "evidence_refutes_claim")
                else:
                    add_edge(claim_idx_map[claim_id], evidence_idx, "claim_related_to_evidence")
                    add_edge(evidence_idx, claim_idx_map[claim_id], "evidence_related_to_claim")

        nodes = record["nodes"]
        synthesis_text = str(record.get("synthesis", "") or "").strip()
        debate_nodes = []
        for node in nodes:
            if node.get("role_id") == config.ROLE_IDS["synthesis"]:
                if not synthesis_text:
                    synthesis_text = str(node.get("text", "")).strip()
                continue
            debate_nodes.append(node)
        debate_nodes.sort(key=lambda n: n["role_id"])

        argument_idx_by_role_id: dict[int, int] = {}
        for node in debate_nodes:
            role_id = int(node["role_id"])
            arg_idx = add_node(
                str(node.get("text", "")).strip(),
                "argument",
                reliability=self._clamp_reliability(node.get("confidence", 0.5), default=0.5),
            )
            argument_idx_by_role_id[role_id] = arg_idx
            speaker_role_ids[arg_idx] = role_id + 1

            for claim_id in node.get("referenced_claims", []) or []:
                if claim_id not in claim_idx_map:
                    continue
                add_edge(arg_idx, claim_idx_map[claim_id], "argument_cites_claim")
                add_edge(claim_idx_map[claim_id], arg_idx, "claim_cited_by_argument")
                if role_id in self.PRO_ROLE_IDS:
                    add_edge(arg_idx, claim_idx_map[claim_id], "argument_supports_claim")
                    add_edge(claim_idx_map[claim_id], arg_idx, "claim_supported_by_argument")
                else:
                    add_edge(arg_idx, claim_idx_map[claim_id], "argument_refutes_claim")
                    add_edge(claim_idx_map[claim_id], arg_idx, "claim_refuted_by_argument")

            for card_id in node.get("referenced_rationales", []) or []:
                if card_id not in card_idx_map:
                    continue
                if card_id.startswith("td_"):
                    add_edge(arg_idx, card_idx_map[card_id], "argument_cites_td")
                    add_edge(card_idx_map[card_id], arg_idx, "td_cited_by_argument")
                elif card_id.startswith("cs_"):
                    add_edge(arg_idx, card_idx_map[card_id], "argument_cites_cs")
                    add_edge(card_idx_map[card_id], arg_idx, "cs_cited_by_argument")
                if card_id in evidence_idx_by_rationale:
                    add_edge(arg_idx, evidence_idx_by_rationale[card_id], "argument_cites_evidence")
                    add_edge(evidence_idx_by_rationale[card_id], arg_idx, "evidence_cited_by_argument")

            for evidence_id in node.get("evidence_ids", []) or []:
                if evidence_id in evidence_idx_map:
                    add_edge(arg_idx, evidence_idx_map[evidence_id], "argument_cites_evidence")
                    add_edge(evidence_idx_map[evidence_id], arg_idx, "evidence_cited_by_argument")

        for src_role_id, dst_role_id in config.EDGE_LIST:
            src_idx = argument_idx_by_role_id.get(src_role_id)
            dst_idx = argument_idx_by_role_id.get(dst_role_id)
            if src_idx is not None and dst_idx is not None:
                add_edge(src_idx, dst_idx, "argument_attacks_argument")

        if not synthesis_text:
            synthesis_text = render_synthesis_text(synthesis_structured, lang=self.lang)
        synthesis_text = self._synthesis_conflict_text(synthesis_structured, synthesis_text)
        synthesis_idx = add_node(synthesis_text, "synthesis", reliability=0.7)
        for claim_id in synthesis_structured.get("supported_claims", []) + synthesis_structured.get("questionable_claims", []):
            if claim_id in claim_idx_map:
                add_edge(synthesis_idx, claim_idx_map[claim_id], "synthesis_mentions_claim")
                add_edge(claim_idx_map[claim_id], synthesis_idx, "claim_mentioned_by_synthesis")

        for card_id in synthesis_structured.get("used_td_rationales", []):
            if card_id in card_idx_map:
                add_edge(synthesis_idx, card_idx_map[card_id], "synthesis_uses_td")
                add_edge(card_idx_map[card_id], synthesis_idx, "td_used_by_synthesis")
                if card_id in evidence_idx_by_rationale:
                    add_edge(synthesis_idx, evidence_idx_by_rationale[card_id], "argument_cites_evidence")
                    add_edge(evidence_idx_by_rationale[card_id], synthesis_idx, "evidence_cited_by_argument")
        for card_id in synthesis_structured.get("used_cs_rationales", []):
            if card_id in card_idx_map:
                add_edge(synthesis_idx, card_idx_map[card_id], "synthesis_uses_cs")
                add_edge(card_idx_map[card_id], synthesis_idx, "cs_used_by_synthesis")
                if card_id in evidence_idx_by_rationale:
                    add_edge(synthesis_idx, evidence_idx_by_rationale[card_id], "argument_cites_evidence")
                    add_edge(evidence_idx_by_rationale[card_id], synthesis_idx, "evidence_cited_by_argument")
        for evidence_id in synthesis_structured.get("used_evidence", []):
            if evidence_id in evidence_idx_map:
                add_edge(synthesis_idx, evidence_idx_map[evidence_id], "argument_cites_evidence")
                add_edge(evidence_idx_map[evidence_id], synthesis_idx, "evidence_cited_by_argument")

        for arg_idx in argument_idx_by_role_id.values():
            add_edge(synthesis_idx, arg_idx, "synthesis_reviews_argument")
            add_edge(arg_idx, synthesis_idx, "argument_reviewed_by_synthesis")

        return node_texts, node_type_ids, speaker_role_ids, node_reliability, edge_src, edge_dst, edge_types

    def __getitem__(self, idx: int) -> Data:
        with open(self.file_paths[idx], "r", encoding="utf-8") as f:
            record = json.load(f)
        record = self._merge_source_record(record)

        news_text = str(record["news_text"])
        label = self._normalize_label(record["label"])
        claims = self._ensure_claims(record, news_text)
        rationale_cards = self._ensure_rationale_cards(record, claims)
        evidence_cards = self._ensure_evidence_cards(record, claims, rationale_cards)
        synthesis_structured = self._ensure_synthesis_structured(record)

        (
            node_texts,
            node_type_ids,
            speaker_role_ids,
            node_reliability,
            edge_src,
            edge_dst,
            edge_types,
        ) = self._build_graph(record, claims, rationale_cards, evidence_cards, synthesis_structured)

        node_encodings = self.tokenizer(
            node_texts,
            padding="max_length",
            truncation=True,
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt",
        )
        news_encoding = self.tokenizer(
            news_text,
            padding="max_length",
            truncation=True,
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt",
        )

        td_pred = self._safe_float(record.get("td_pred", -1))
        cs_pred = self._safe_float(record.get("cs_pred", -1))
        td_acc = self._safe_float(record.get("td_acc", -1))
        cs_acc = self._safe_float(record.get("cs_acc", -1))
        source_bucket = self._hash_source_bucket(record.get("source_id", -1))
        time_features = self._parse_time_features(record.get("time", ""))
        teacher_features = self._teacher_features(td_pred, cs_pred)

        data = Data(
            node_input_ids=node_encodings["input_ids"],
            node_attention_mask=node_encodings["attention_mask"],
            news_input_ids=news_encoding["input_ids"],
            news_attention_mask=news_encoding["attention_mask"],
            node_type_ids=torch.tensor(node_type_ids, dtype=torch.long),
            speaker_role_ids=torch.tensor(speaker_role_ids, dtype=torch.long),
            node_reliability=torch.tensor(node_reliability, dtype=torch.float32),
            edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
            edge_type=torch.tensor(edge_types, dtype=torch.long),
            source_bucket=torch.tensor(source_bucket, dtype=torch.long),
            time_features=torch.tensor(time_features, dtype=torch.float32),
            teacher_features=torch.tensor(teacher_features, dtype=torch.float32),
            td_pred=torch.tensor(td_pred, dtype=torch.float32),
            cs_pred=torch.tensor(cs_pred, dtype=torch.float32),
            td_acc=torch.tensor(td_acc, dtype=torch.float32),
            cs_acc=torch.tensor(cs_acc, dtype=torch.float32),
            y=torch.tensor(label, dtype=torch.long),
            num_nodes=len(node_type_ids),
        )
        return data
