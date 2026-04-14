"""
TruEDebate (TED) — Mesa Model 定义
DebateModel 控制三阶段辩论流程，按 Algorithm 1 顺序调度智能体，
并在最后调用 Synthesis Agent 生成总结报告。
"""

import logging
from mesa import Model

from debate_flow.agents import DebateAgent
from debate_flow.prompts import call_llm, format_synthesis_prompt
import config

logger = logging.getLogger(__name__)


class DebateModel(Model):
    """
    辩论模拟模型。

    按照论文 Algorithm 1 的逻辑，依次执行三个辩论阶段和一个综合总结阶段。
    输入一篇新闻文本，输出完整的辩论记录（含图结构信息）。
    """

    def __init__(self, news_text: str):
        """
        Args:
            news_text: 待辩论的原始新闻文本
        """
        super().__init__()
        self.news_text = news_text
        self.synthesis_text: str = ""

        # 存储各角色到 Agent 的映射，便于按角色名检索发言
        self._agent_map: dict[str, DebateAgent] = {}

        # 创建 6 个辩论智能体
        agent_configs = [
            ("Proponent", "opening",    config.ROLE_IDS["proponent_opening"]),
            ("Opponent",  "opening",    config.ROLE_IDS["opponent_opening"]),
            ("Proponent", "questioner", config.ROLE_IDS["proponent_questioner"]),
            ("Opponent",  "questioner", config.ROLE_IDS["opponent_questioner"]),
            ("Proponent", "closing",    config.ROLE_IDS["proponent_closing"]),
            ("Opponent",  "closing",    config.ROLE_IDS["opponent_closing"]),
        ]

        for side, role, role_id in agent_configs:
            agent = DebateAgent(self, side=side, role=role, role_id=role_id)
            key = f"{side.lower()}_{role}"
            self._agent_map[key] = agent

    def get_speech(self, role_key: str) -> str:
        """
        获取指定角色的发言文本。

        Args:
            role_key: 角色键，如 "proponent_opening", "opponent_questioner"

        Returns:
            该角色的发言文本
        """
        agent = self._agent_map.get(role_key)
        if agent is None:
            raise KeyError(f"Unknown role_key: {role_key}")
        return agent.speech

    def step(self) -> None:
        """
        执行完整的三阶段辩论流程 + Synthesis（Algorithm 1）。

        Stage 1: Opening Statements (正反方开篇立论)
        Stage 2: Cross-examination (正反方质询反驳)
        Stage 3: Closing Statements (正反方结案陈词)
        Synthesis: 综合总结报告
        """
        # ── Stage 1: Opening Statement ──
        logger.info("=" * 60)
        logger.info("Stage 1: Opening Statements")
        logger.info("=" * 60)
        self._agent_map["proponent_opening"].step()
        self._agent_map["opponent_opening"].step()

        # ── Stage 2: Cross-examination ──
        logger.info("=" * 60)
        logger.info("Stage 2: Cross-Examination and Rebuttal")
        logger.info("=" * 60)
        self._agent_map["proponent_questioner"].step()
        self._agent_map["opponent_questioner"].step()

        # ── Stage 3: Closing Statement ──
        logger.info("=" * 60)
        logger.info("Stage 3: Closing Statements")
        logger.info("=" * 60)
        self._agent_map["proponent_closing"].step()
        self._agent_map["opponent_closing"].step()

        # ── Synthesis ──
        logger.info("=" * 60)
        logger.info("Synthesis Agent: Generating Summary")
        logger.info("=" * 60)
        self._generate_synthesis()

    def _generate_synthesis(self) -> None:
        """调用 Synthesis Agent 生成综合总结。"""
        system_msg, prompt = format_synthesis_prompt(
            news_text=self.news_text,
            pro_opening=self.get_speech("proponent_opening"),
            opp_opening=self.get_speech("opponent_opening"),
            pro_cross=self.get_speech("proponent_questioner"),
            opp_cross=self.get_speech("opponent_questioner"),
            pro_closing=self.get_speech("proponent_closing"),
            opp_closing=self.get_speech("opponent_closing"),
        )
        self.synthesis_text = call_llm(prompt, system_msg)
        logger.info(f"[Synthesis] 总结完成 ({len(self.synthesis_text)} chars)")

    def get_debate_record(self) -> dict:
        """
        导出完整的辩论记录，包含图结构信息。

        Returns:
            dict 包含：
                - nodes: [{text, role_id, role_name}] (8 个节点)
                - edge_index: [[src...], [dst...]]
                - synthesis: 综合总结文本
        """
        # 按 role_id 顺序组装节点 (0-7)
        nodes = []

        # 角色节点 (0-5)
        role_order = [
            "proponent_opening",
            "opponent_opening",
            "proponent_questioner",
            "opponent_questioner",
            "proponent_closing",
            "opponent_closing",
        ]
        for role_key in role_order:
            agent = self._agent_map[role_key]
            nodes.append({
                "text": agent.speech,
                "role_id": agent.role_id,
                "role_name": role_key,
            })

        # Synthesis 节点 (6)
        nodes.append({
            "text": self.synthesis_text,
            "role_id": config.ROLE_IDS["synthesis"],
            "role_name": "synthesis",
        })

        # Original News 节点 (7)
        nodes.append({
            "text": self.news_text,
            "role_id": config.ROLE_IDS["original_news"],
            "role_name": "original_news",
        })

        # 构建 edge_index: [2, num_edges]
        src_list = [e[0] for e in config.EDGE_LIST]
        dst_list = [e[1] for e in config.EDGE_LIST]

        return {
            "nodes": nodes,
            "edge_index": [src_list, dst_list],
            "synthesis": self.synthesis_text,
        }
