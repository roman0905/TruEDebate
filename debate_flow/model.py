"""
TruEDebate (TED) — Mesa Model 定义
DebateModel 控制三阶段辩论流程，按 Algorithm 1 顺序调度智能体，
并在最后调用 Synthesis Agent 生成总结报告。
"""

import json
import logging
from mesa import Model

from debate_flow.agents import DebateAgent, PerspectiveAgent
from debate_flow.prompts import (
    call_llm,
    format_final_judge_prompt,
    format_perspective_coordinator_prompt,
    format_perspective_planner_prompt,
    format_role_reversal_prompt,
    format_self_reflective_judge_prompt,
    format_synthesis_prompt,
)
import config

logger = logging.getLogger(__name__)


class DebateModel(Model):
    """
    辩论模拟模型。

    按照论文 Algorithm 1 的逻辑，依次执行三个辩论阶段和一个综合总结阶段。
    输入一篇新闻文本，输出完整的辩论记录（含图结构信息）。
    """

    def __init__(
        self,
        news_text: str,
        debate_mode: str = config.DEFAULT_DEBATE_MODE,
        top_k_perspectives: int = config.DEFAULT_PERSPECTIVE_TOP_K,
        fixed_perspectives: bool = False,
        enable_role_reversal: bool = True,
    ):
        """
        Args:
            news_text: 待辩论的原始新闻文本
            debate_mode: "ted" 或 "perspective"
            top_k_perspectives: 自适应选择视角数
            fixed_perspectives: True 时使用固定前 K 个视角，便于消融实验
            enable_role_reversal: 是否启用角色反转一致性检查
        """
        super().__init__()
        self.news_text = news_text
        self.debate_mode = debate_mode
        self.top_k_perspectives = top_k_perspectives
        self.fixed_perspectives = fixed_perspectives
        self.enable_role_reversal = enable_role_reversal
        self.synthesis_text: str = ""

        # 存储各角色到 Agent 的映射，便于按角色名检索发言
        self._agent_map: dict[str, DebateAgent] = {}
        self._perspective_agents: list[PerspectiveAgent] = []

        self.planner_text: str = ""
        self.planner_summary: str = ""
        self.selected_perspectives: list[dict] = []
        self.coordinator_synthesis: str = ""
        self.self_reflection_text: str = ""
        self.role_reversal_text: str = ""
        self.final_judgment_text: str = ""

        if debate_mode not in {"ted", "perspective"}:
            raise ValueError(f"Unknown debate_mode: {debate_mode}")

        if debate_mode == "ted":
            self._init_ted_agents()

    def _init_ted_agents(self) -> None:
        """创建原始 TED 的 6 个正反方辩论智能体。"""
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
        if self.debate_mode == "perspective":
            self._run_perspective_debate()
            return

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

    def _run_perspective_debate(self) -> None:
        """
        执行 Perspective-Adaptive Multi-Agent Debate。

        流程:
        1. Perspective Planner 自适应选择风险视角。
        2. 多个 Perspective Agent 并行式生成中立分析报告。
        3. Coordinator 做视角置信度加权聚合。
        4. Self-Reflective Judge 审计论点可靠性。
        5. Role-Reversal Judge 检查角色反转一致性。
        6. Final Judge 输出最终融合判断。
        """
        logger.info("=" * 60)
        logger.info("Perspective Planner: Selecting Risk Perspectives")
        logger.info("=" * 60)
        self._generate_perspective_plan()

        logger.info("=" * 60)
        logger.info("Perspective Agents: Generating Reports")
        logger.info("=" * 60)
        self._create_perspective_agents()
        for agent in self._perspective_agents:
            agent.step()

        logger.info("=" * 60)
        logger.info("Coordinator: Aggregating Perspective Reports")
        logger.info("=" * 60)
        self._generate_perspective_coordinator()

        logger.info("=" * 60)
        logger.info("Self-Reflective Judge: Auditing Reliability")
        logger.info("=" * 60)
        self._generate_self_reflection()

        logger.info("=" * 60)
        logger.info("Role-Reversal Judge: Checking Consistency")
        logger.info("=" * 60)
        self._generate_role_reversal()

        logger.info("=" * 60)
        logger.info("Final Judge: Producing Fused Judgment")
        logger.info("=" * 60)
        self._generate_final_judgment()
        self.synthesis_text = self.final_judgment_text

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

    def _generate_perspective_plan(self) -> None:
        """调用 Planner 选择当前样本最相关的风险视角。"""
        if self.fixed_perspectives:
            selected = list(config.PERSPECTIVE_AGENT_DEFINITIONS.keys())[
                : self.top_k_perspectives
            ]
            self.selected_perspectives = [
                {
                    "key": key,
                    "reason": "固定多视角消融设置",
                    "priority": idx + 1,
                    "confidence": 1.0,
                }
                for idx, key in enumerate(selected)
            ]
            self.planner_summary = "固定多视角设置，未进行自适应选择。"
            self.planner_text = json.dumps(
                {
                    "selected_perspectives": self.selected_perspectives,
                    "planner_summary": self.planner_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            return

        catalog = self._format_perspective_catalog()
        system_msg, prompt = format_perspective_planner_prompt(
            self.news_text, catalog, self.top_k_perspectives
        )
        self.planner_text = call_llm(prompt, system_msg)
        parsed = self._parse_json_object(self.planner_text)
        self.selected_perspectives = self._sanitize_selected_perspectives(
            parsed.get("selected_perspectives", [])
        )
        self.planner_summary = str(parsed.get("planner_summary", "")).strip()

        if not self.selected_perspectives:
            fallback_keys = list(config.PERSPECTIVE_AGENT_DEFINITIONS.keys())[
                : self.top_k_perspectives
            ]
            self.selected_perspectives = [
                {
                    "key": key,
                    "reason": "Planner 输出无法解析，使用默认高覆盖视角。",
                    "priority": idx + 1,
                    "confidence": 0.5,
                }
                for idx, key in enumerate(fallback_keys)
            ]
            self.planner_summary = "Planner 输出解析失败，回退到默认视角集合。"

        logger.info(
            "Planner 选择视角: %s",
            ", ".join(item["key"] for item in self.selected_perspectives),
        )

    def _create_perspective_agents(self) -> None:
        """根据 Planner 输出创建多视角专家。"""
        self._perspective_agents = []
        for item in self.selected_perspectives:
            perspective_key = item["key"]
            definition = config.PERSPECTIVE_AGENT_DEFINITIONS[perspective_key]
            agent = PerspectiveAgent(
                self,
                perspective_key=perspective_key,
                role_name=definition["role_name"],
                focus=definition["focus"],
                role_id=definition["role_id"],
            )
            self._perspective_agents.append(agent)

    def _generate_perspective_coordinator(self) -> None:
        """调用 Coordinator 聚合各视角报告。"""
        reports = self._format_perspective_reports()
        system_msg, prompt = format_perspective_coordinator_prompt(
            self.news_text, reports
        )
        self.coordinator_synthesis = call_llm(prompt, system_msg)
        logger.info(
            f"[Coordinator] 聚合完成 ({len(self.coordinator_synthesis)} chars)"
        )

    def _generate_self_reflection(self) -> None:
        """调用自反思裁判审计论证可靠性。"""
        reports = self._format_perspective_reports()
        system_msg, prompt = format_self_reflective_judge_prompt(
            self.news_text, self.coordinator_synthesis, reports
        )
        self.self_reflection_text = call_llm(prompt, system_msg)
        logger.info(
            f"[Self-Reflective Judge] 审计完成 ({len(self.self_reflection_text)} chars)"
        )

    def _generate_role_reversal(self) -> None:
        """调用角色反转一致性裁判。"""
        if not self.enable_role_reversal:
            self.role_reversal_text = (
                "Role-reversal consistency check disabled for ablation."
            )
            return
        system_msg, prompt = format_role_reversal_prompt(
            self.news_text, self.coordinator_synthesis, self.self_reflection_text
        )
        self.role_reversal_text = call_llm(prompt, system_msg)
        logger.info(
            f"[Role-Reversal Judge] 一致性检查完成 ({len(self.role_reversal_text)} chars)"
        )

    def _generate_final_judgment(self) -> None:
        """调用最终裁判融合多视角、自反思和一致性结果。"""
        system_msg, prompt = format_final_judge_prompt(
            self.news_text,
            self.coordinator_synthesis,
            self.self_reflection_text,
            self.role_reversal_text,
        )
        self.final_judgment_text = call_llm(prompt, system_msg)
        logger.info(f"[Final Judge] 最终判断完成 ({len(self.final_judgment_text)} chars)")

    @staticmethod
    def _format_perspective_catalog() -> str:
        """将可选视角格式化给 Planner。"""
        lines = []
        for key, definition in config.PERSPECTIVE_AGENT_DEFINITIONS.items():
            lines.append(
                f"- {key}: {definition['role_name']} | {definition['focus']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        """从 LLM 输出中尽量解析 JSON 对象。"""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    return {}
        return {}

    def _sanitize_selected_perspectives(self, raw_items: object) -> list[dict]:
        """校验 Planner 选择，过滤未知视角并截断到 top-k。"""
        if not isinstance(raw_items, list):
            return []

        valid_keys = set(config.PERSPECTIVE_AGENT_DEFINITIONS)
        selected = []
        seen = set()
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if key not in valid_keys or key in seen:
                continue
            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            selected.append(
                {
                    "key": key,
                    "reason": str(item.get("reason", "")).strip(),
                    "priority": int(item.get("priority", idx + 1)),
                    "confidence": confidence,
                }
            )
            seen.add(key)
            if len(selected) >= self.top_k_perspectives:
                break
        return selected

    def _format_perspective_reports(self) -> str:
        """将各视角报告拼接为裁判上下文。"""
        blocks = []
        reason_by_key = {
            item["key"]: item.get("reason", "") for item in self.selected_perspectives
        }
        confidence_by_key = {
            item["key"]: item.get("confidence", 0.0)
            for item in self.selected_perspectives
        }
        for agent in self._perspective_agents:
            blocks.append(
                "\n".join(
                    [
                        f"### {agent.role_name} ({agent.perspective_key})",
                        f"Planner reason: {reason_by_key.get(agent.perspective_key, '')}",
                        f"Planner confidence: {confidence_by_key.get(agent.perspective_key, 0.0)}",
                        agent.report,
                    ]
                )
            )
        return "\n\n".join(blocks)

    def get_debate_record(self) -> dict:
        """
        导出完整的辩论记录，包含图结构信息。

        论文中原始新闻不作为图节点，而是通过 Encoder(F) 编码后
        与图表示做 Interactive Attention (Eq.10)。

        Returns:
            dict 包含：
                - nodes: [{text, role_id, role_name}] (7 个节点: 6 辩论 + 1 综合)
                - edge_index: [[src...], [dst...]]
                - synthesis: 综合总结文本
        """
        if self.debate_mode == "perspective":
            return self._get_perspective_record()

        # 按 role_id 顺序组装节点 (0-6)
        nodes = []

        # 辩论角色节点 (0-5)
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

        # 构建 edge_index: [2, num_edges]
        src_list = [e[0] for e in config.EDGE_LIST]
        dst_list = [e[1] for e in config.EDGE_LIST]

        return {
            "nodes": nodes,
            "edge_index": [src_list, dst_list],
            "synthesis": self.synthesis_text,
            "debate_mode": "ted",
            "graph_schema_version": 1,
        }

    def _get_perspective_record(self) -> dict:
        """导出多视角辩论记录，使用动态节点图。"""
        nodes = [
            {
                "text": self.planner_text,
                "role_id": config.ROLE_IDS["perspective_planner"],
                "role_name": "perspective_planner",
                "metadata": {
                    "selected_perspectives": self.selected_perspectives,
                    "planner_summary": self.planner_summary,
                },
            }
        ]

        for agent in self._perspective_agents:
            nodes.append(
                {
                    "text": agent.report,
                    "role_id": agent.role_id,
                    "role_name": f"perspective_{agent.perspective_key}",
                    "perspective_key": agent.perspective_key,
                }
            )

        coordinator_idx = len(nodes)
        nodes.append(
            {
                "text": self.coordinator_synthesis,
                "role_id": config.ROLE_IDS["perspective_coordinator"],
                "role_name": "perspective_coordinator",
            }
        )

        self_reflection_idx = len(nodes)
        nodes.append(
            {
                "text": self.self_reflection_text,
                "role_id": config.ROLE_IDS["self_reflective_judge"],
                "role_name": "self_reflective_judge",
            }
        )

        role_reversal_idx = len(nodes)
        nodes.append(
            {
                "text": self.role_reversal_text,
                "role_id": config.ROLE_IDS["role_reversal_judge"],
                "role_name": "role_reversal_judge",
            }
        )

        final_judge_idx = len(nodes)
        nodes.append(
            {
                "text": self.final_judgment_text,
                "role_id": config.ROLE_IDS["final_judge"],
                "role_name": "final_judge",
            }
        )

        edges: list[tuple[int, int]] = []
        edge_types: list[int] = []

        def add_edge(src: int, dst: int, edge_type: str) -> None:
            edges.append((src, dst))
            edge_types.append(config.EDGE_TYPE_IDS[edge_type])

        perspective_indices = list(range(1, coordinator_idx))
        for idx in perspective_indices:
            add_edge(0, idx, "plan")
            add_edge(idx, coordinator_idx, "synthesis")

        for src in perspective_indices:
            for dst in perspective_indices:
                if src != dst:
                    add_edge(src, dst, "attack")

        add_edge(coordinator_idx, self_reflection_idx, "judge")
        add_edge(coordinator_idx, role_reversal_idx, "consistency")
        add_edge(self_reflection_idx, role_reversal_idx, "consistency")
        add_edge(coordinator_idx, final_judge_idx, "judge")
        add_edge(self_reflection_idx, final_judge_idx, "judge")
        add_edge(role_reversal_idx, final_judge_idx, "consistency")

        src_list = [e[0] for e in edges]
        dst_list = [e[1] for e in edges]

        return {
            "nodes": nodes,
            "edge_index": [src_list, dst_list],
            "edge_type": edge_types,
            "synthesis": self.final_judgment_text,
            "debate_mode": "perspective",
            "graph_schema_version": config.GRAPH_SCHEMA_VERSION,
            "selected_perspectives": self.selected_perspectives,
            "planner": self.planner_text,
            "coordinator_synthesis": self.coordinator_synthesis,
            "self_reflection": self.self_reflection_text,
            "role_reversal": self.role_reversal_text,
            "final_judgment": self.final_judgment_text,
        }
