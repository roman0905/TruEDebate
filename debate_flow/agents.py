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
        system_msg, prompt = format_opening_prompt(news_text, self.side)
        logger.info(f"[{self.side} Opening] 生成发言中...")
        self.speech = call_llm(prompt, system_msg)
        logger.info(f"[{self.side} Opening] 发言完成 ({len(self.speech)} chars)")

    def _do_cross_exam(self, news_text: str) -> None:
        """Stage 2: 质询反驳"""
        # 需要 Stage 1 的发言作为上下文
        pro_opening = self.model.get_speech("proponent_opening")
        opp_opening = self.model.get_speech("opponent_opening")

        system_msg, prompt = format_cross_exam_prompt(
            news_text, self.side, pro_opening, opp_opening
        )
        logger.info(f"[{self.side} Questioner] 生成发言中...")
        self.speech = call_llm(prompt, system_msg)
        logger.info(f"[{self.side} Questioner] 发言完成 ({len(self.speech)} chars)")

    def _do_closing(self, news_text: str) -> None:
        """Stage 3: 结案陈词"""
        pro_opening = self.model.get_speech("proponent_opening")
        opp_opening = self.model.get_speech("opponent_opening")
        pro_cross = self.model.get_speech("proponent_questioner")
        opp_cross = self.model.get_speech("opponent_questioner")

        system_msg, prompt = format_closing_prompt(
            news_text, self.side, pro_opening, opp_opening, pro_cross, opp_cross
        )
        logger.info(f"[{self.side} Closing] 生成发言中...")
        self.speech = call_llm(prompt, system_msg)
        logger.info(f"[{self.side} Closing] 发言完成 ({len(self.speech)} chars)")

    def __repr__(self) -> str:
        return f"DebateAgent(side={self.side}, role={self.role}, id={self.role_id})"
