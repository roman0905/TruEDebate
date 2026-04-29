"""
TruEDebate (TED) — Mesa Agent 定义
每个 DebateAgent 代表辩论中的一个角色（开篇/质询/结案），
由 DebateModel 调度并在 step() 中调用 LLM 生成发言。
"""

import logging
from mesa import Agent

from debate_flow.prompts import (
    call_llm,
    format_opening_prompt,
    format_cross_exam_prompt,
    format_closing_prompt,
    parse_structured_argument,
    render_argument_text,
)
import config

logger = logging.getLogger(__name__)


class DebateAgent(Agent):
    """
    辩论智能体。

    Attributes:
        side: "Proponent" 或 "Opponent"
        role: "opening" / "questioner" / "closing"
        role_id: 角色数字编码 (0-5)
        speech: 生成的发言文本
    """

    def __init__(self, model, side: str, role: str, role_id: int):
        """
        Args:
            model: Mesa Model 实例
            side: "Proponent" 或 "Opponent"
            role: "opening" / "questioner" / "closing"
            role_id: 配置中的角色 ID (0-5)
        """
        super().__init__(model)
        self.side = side
        self.role = role
        self.role_id = role_id
        self.speech: str = ""
        self.structured_speech: dict = {}

    def step(self) -> None:
        """
        根据当前角色和阶段，调用 LLM 生成发言。
        所需的上下文信息（前序发言）从 model 中获取。
        """
        news_text = self.model.news_text

        if self.role == "opening":
            self._do_opening(news_text)
        elif self.role == "questioner":
            self._do_cross_exam(news_text)
        elif self.role == "closing":
            self._do_closing(news_text)
        else:
            raise ValueError(f"Unknown role: {self.role}")

    def _do_opening(self, news_text: str) -> None:
        """Stage 1: 开篇立论"""
        system_msg, prompt = format_opening_prompt(
            news_text,
            self.side,
            claims=getattr(self.model, "claims", []),
            rationale_cards=getattr(self.model, "rationale_cards", []),
            evidence_cards=getattr(self.model, "evidence_cards", []),
            lang=getattr(self.model, "lang", "en"),
        )
        logger.info(f"[{self.side} Opening] 生成发言中...")
        raw = call_llm(prompt, system_msg, generation_key="opening")
        self.structured_speech = parse_structured_argument(
            raw,
            claim_ids=[claim["id"] for claim in getattr(self.model, "claims", [])],
            rationale_ids=[card["id"] for card in getattr(self.model, "rationale_cards", [])],
            evidence_ids=[card["id"] for card in getattr(self.model, "evidence_cards", [])],
            fallback_text=raw,
        )
        self.speech = render_argument_text(self.structured_speech, lang=getattr(self.model, "lang", "en"))
        logger.info(f"[{self.side} Opening] 发言完成 ({len(self.speech)} chars)")

    def _do_cross_exam(self, news_text: str) -> None:
        """Stage 2: 质询反驳"""
        # 需要 Stage 1 的发言作为上下文
        pro_opening = self.model.get_speech("proponent_opening")
        opp_opening = self.model.get_speech("opponent_opening")

        system_msg, prompt = format_cross_exam_prompt(
            news_text,
            self.side,
            pro_opening,
            opp_opening,
            claims=getattr(self.model, "claims", []),
            rationale_cards=getattr(self.model, "rationale_cards", []),
            evidence_cards=getattr(self.model, "evidence_cards", []),
            lang=getattr(self.model, "lang", "en"),
        )
        logger.info(f"[{self.side} Questioner] 生成发言中...")
        raw = call_llm(prompt, system_msg, generation_key="questioner")
        self.structured_speech = parse_structured_argument(
            raw,
            claim_ids=[claim["id"] for claim in getattr(self.model, "claims", [])],
            rationale_ids=[card["id"] for card in getattr(self.model, "rationale_cards", [])],
            evidence_ids=[card["id"] for card in getattr(self.model, "evidence_cards", [])],
            fallback_text=raw,
        )
        self.speech = render_argument_text(self.structured_speech, lang=getattr(self.model, "lang", "en"))
        logger.info(
            f"[{self.side} Questioner] 发言完成 "
            f"(structured={len(self.speech)} chars, raw={len(raw)} chars)"
        )

    def _do_closing(self, news_text: str) -> None:
        """Stage 3: 结案陈词"""
        pro_opening = self.model.get_speech("proponent_opening")
        opp_opening = self.model.get_speech("opponent_opening")
        pro_cross = self.model.get_speech("proponent_questioner")
        opp_cross = self.model.get_speech("opponent_questioner")

        system_msg, prompt = format_closing_prompt(
            news_text,
            self.side,
            pro_opening,
            opp_opening,
            pro_cross,
            opp_cross,
            claims=getattr(self.model, "claims", []),
            rationale_cards=getattr(self.model, "rationale_cards", []),
            evidence_cards=getattr(self.model, "evidence_cards", []),
            lang=getattr(self.model, "lang", "en"),
        )
        logger.info(f"[{self.side} Closing] 生成发言中...")
        raw = call_llm(prompt, system_msg, generation_key="closing")
        self.structured_speech = parse_structured_argument(
            raw,
            claim_ids=[claim["id"] for claim in getattr(self.model, "claims", [])],
            rationale_ids=[card["id"] for card in getattr(self.model, "rationale_cards", [])],
            evidence_ids=[card["id"] for card in getattr(self.model, "evidence_cards", [])],
            fallback_text=raw,
        )
        self.speech = render_argument_text(self.structured_speech, lang=getattr(self.model, "lang", "en"))
        logger.info(f"[{self.side} Closing] 发言完成 ({len(self.speech)} chars)")

    def to_node(self) -> dict:
        return {
            "text": self.speech,
            "role_id": self.role_id,
            "role_name": f"{self.side.lower()}_{self.role}",
            "side": self.side,
            "stage": self.role,
            "argument": self.structured_speech.get("argument", ""),
            "referenced_claims": self.structured_speech.get("referenced_claims", []),
            "referenced_rationales": self.structured_speech.get("referenced_rationales", []),
            "evidence_ids": self.structured_speech.get("evidence_ids", []),
            "reasoning": self.structured_speech.get("reasoning", ""),
            "weakness": self.structured_speech.get("weakness", ""),
            "confidence": self.structured_speech.get("confidence", 0.5),
            "attack_points": self.structured_speech.get("attack_points", []),
            "counter_evidence": self.structured_speech.get("counter_evidence", []),
        }

    def __repr__(self) -> str:
        return f"DebateAgent(side={self.side}, role={self.role}, id={self.role_id})"
