"""
TruEDebate (TED) — Prompt 模板与 LLM 调用工具
包含辩论三阶段的 Prompt 模板和 Synthesis Agent 的总结模板。
"""

import logging
import os
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

# ──────────────────────────────── OpenAI Client ────────────────────────────────

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """获取或创建 OpenAI 客户端单例。"""
    global _client
    if _client is None:
        api_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        base_url = config.OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 未设置。请通过环境变量或 config.py 配置。"
            )
        # 构建客户端参数
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        _client = OpenAI(**client_kwargs)
    return _client


def call_llm(prompt: str, system_msg: str = "You are a helpful assistant.") -> str:
    """
    调用 OpenAI GPT-4o-mini 生成回复。

    Args:
        prompt: 用户提示词
        system_msg: 系统消息

    Returns:
        LLM 生成的文本回复
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.OPENAI_MAX_TOKENS,
            temperature=config.OPENAI_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# 辩论阶段 Prompt 模板
# ═══════════════════════════════════════════════════════════════════════════════

# ──────────────── Stage 1: Opening Statement (开篇立论) ────────────────

OPENING_SYSTEM = (
    "You are a debate participant. You must argue {stance} the authenticity "
    "of the given news article. You are the Opening Speaker for the "
    "{side} side. Present a clear, structured opening statement."
)

OPENING_PROMPT = """
## Debate Topic
Evaluate the authenticity of the following news article.

## News Article
{news_text}

## Your Role
You are the **Opening Speaker** for the **{side}** side.
Your stance: The news is **{stance_label}**.

## Instructions
1. Present your opening argument in 3-5 concise paragraphs.
2. Support your position with logical reasoning based on the news content.
3. Analyze the characteristics of the news (sources, tone, details, verifiability).
4. Be persuasive but maintain a professional debate tone.

## Your Opening Statement:
"""

# ──────────────── Stage 2: Cross-examination (质询反驳) ────────────────

CROSS_EXAM_SYSTEM = (
    "You are a debate participant. You must argue {stance} the authenticity "
    "of the given news article. You are the Questioner for the "
    "{side} side. Challenge the opposing side's arguments."
)

CROSS_EXAM_PROMPT = """
## Debate Topic
Evaluate the authenticity of the following news article.

## News Article
{news_text}

## Previous Opening Statements
### Proponent (argues news is REAL):
{pro_opening}

### Opponent (argues news is FAKE):
{opp_opening}

## Your Role
You are the **Questioner** for the **{side}** side.
Your stance: The news is **{stance_label}**.

## Instructions
1. Directly challenge the key arguments from the opposing side's opening statement.
2. Point out logical flaws, unsupported claims, or missing evidence in their argument.
3. Reinforce your own side's position with counter-arguments.
4. Ask 1-2 pointed rhetorical questions that highlight weaknesses in the other side's case.

## Your Cross-Examination:
"""

# ──────────────── Stage 3: Closing Statement (结案陈词) ────────────────

CLOSING_SYSTEM = (
    "You are a debate participant. You must argue {stance} the authenticity "
    "of the given news article. You are the Closing Speaker for the "
    "{side} side. Deliver a compelling closing argument."
)

CLOSING_PROMPT = """
## Debate Topic
Evaluate the authenticity of the following news article.

## News Article
{news_text}

## Full Debate Record
### Stage 1 — Opening Statements
**Proponent Opening:** {pro_opening}
**Opponent Opening:** {opp_opening}

### Stage 2 — Cross-Examination
**Proponent Questioner:** {pro_cross}
**Opponent Questioner:** {opp_cross}

## Your Role
You are the **Closing Speaker** for the **{side}** side.
Your stance: The news is **{stance_label}**.

## Instructions
1. Summarize the most compelling arguments from your side throughout the debate.
2. Address the strongest points raised by the opposing side and explain why they are insufficient.
3. Deliver a clear, persuasive conclusion about the news article's authenticity.
4. End with a strong statement reinforcing your position.

## Your Closing Statement:
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Synthesis Agent Prompt（严格按照论文 3.4.1 节）
# ═══════════════════════════════════════════════════════════════════════════════

SYNTHESIS_SYSTEM = (
    "You are the Synthesis Agent. Your job is to objectively analyze a debate "
    "about the authenticity of a news article and produce a structured summary."
)

SYNTHESIS_PROMPT = """
## News Article
{news_text}

## Full Debate Record
### Stage 1 — Opening Statements
**Proponent (REAL):** {pro_opening}
**Opponent (FAKE):** {opp_opening}

### Stage 2 — Cross-Examination
**Proponent Questioner:** {pro_cross}
**Opponent Questioner:** {opp_cross}

### Stage 3 — Closing Statements
**Proponent Closing:** {pro_closing}
**Opponent Closing:** {opp_closing}

## Your Task
Produce a detailed assessment of this debate. It should contain a detailed \
explanation of your assessment of the debate. Focus on evaluating the \
authenticity of the news involved in the topic by checking the following:
1. Whether the news contains specific details and verifiable information.
2. Whether the news cites reliable sources or news organizations.
3. The tone and style of the news, with real news generally being more \
objective and neutral.
4. Any use of emotional language, which might be a characteristic of fake news.
5. Whether the information in the news can be confirmed through other \
reliable channels.

## Your Synthesis Report:
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Perspective-Adaptive Multi-Agent Debate Prompt
# ═══════════════════════════════════════════════════════════════════════════════

PERSPECTIVE_PLANNER_SYSTEM = (
    "You are the Perspective Planner for fake-news detection. Select the most "
    "useful analytical perspectives for the current news sample. Do not verify "
    "with external retrieval; reason only from the text."
)

PERSPECTIVE_PLANNER_PROMPT = """
## News Article
{news_text}

## Available Perspectives
{perspective_catalog}

## Task
Select the top {top_k} perspectives that are most relevant for judging this article.
Return strict JSON only, using this schema:
{{
  "selected_perspectives": [
    {{"key": "perspective_key", "reason": "why this perspective is relevant", "priority": 1, "confidence": 0.0}}
  ],
  "planner_summary": "short risk summary"
}}
"""

PERSPECTIVE_AGENT_SYSTEM = (
    "You are {role_name}. Analyze fake-news risk from your own perspective. "
    "You are not assigned a fixed REAL/FAKE side; you must report evidence for "
    "both authenticity and falsity."
)

PERSPECTIVE_AGENT_PROMPT = """
## News Article
{news_text}

## Planner Context
{planner_summary}

## Your Perspective
Key: {perspective_key}
Name: {role_name}
Focus: {focus}

## Task
Produce a concise structured report with these fields:
1. suspicious_points: concrete risk points from your perspective.
2. arguments_for_real: reasons that support authenticity.
3. arguments_for_fake: reasons that support falsity or manipulation.
4. confidence: score from 0.0 to 1.0 for how diagnostic this perspective is.
5. verdict_hint: one of real/fake/uncertain.

Keep the analysis grounded in the article text and explicitly mark speculation.
"""

PERSPECTIVE_COORDINATOR_SYSTEM = (
    "You are the Coordinator for a multi-perspective fake-news debate. Aggregate "
    "specialist reports without letting fluency or role pressure dominate."
)

PERSPECTIVE_COORDINATOR_PROMPT = """
## News Article
{news_text}

## Perspective Reports
{perspective_reports}

## Task
Aggregate the reports into a balanced synthesis:
1. Key accepted risk signals.
2. Key authenticity-supporting signals.
3. Cross-perspective conflicts or gaps.
4. Perspective confidence weighting.
5. Provisional label: real/fake/uncertain.
6. Short rationale for the provisional label.
"""

SELF_REFLECTION_SYSTEM = (
    "You are a self-reflective judge. Before producing a final decision, audit "
    "the debate reliability, unsupported claims, contradictions, and uncertainty."
)

SELF_REFLECTION_PROMPT = """
## News Article
{news_text}

## Coordinator Synthesis
{coordinator_synthesis}

## Perspective Reports
{perspective_reports}

## Task
Perform a reliability audit before classification. Cover:
1. factual_arguments: claims that are directly checkable from the text.
2. subjective_or_speculative_arguments: claims that go beyond the text.
3. contradictions: conflicts between reports or within the article.
4. unanswered_points: important issues not resolved by the debate.
5. hallucination_risk: arguments that may be invented or overconfident.
6. uncertainty_sources: why the final conclusion may be uncertain.

Then output:
- reflective_label: real/fake/uncertain
- confidence_score: 0.0 to 1.0
- key_accepted_arguments
- rejected_arguments
"""

ROLE_REVERSAL_SYSTEM = (
    "You are a role-reversal consistency judge. Test whether the conclusion is "
    "stable when the strongest REAL-supporting and FAKE-supporting arguments are "
    "forced to argue the opposite side."
)

ROLE_REVERSAL_PROMPT = """
## News Article
{news_text}

## Coordinator Synthesis
{coordinator_synthesis}

## Self-Reflective Audit
{self_reflection}

## Task
Create a lightweight role-reversal consistency check:
1. Reverse the strongest authenticity-supporting arguments into fake-supporting challenges.
2. Reverse the strongest fake-supporting arguments into authenticity-supporting challenges.
3. Judge whether the original conclusion remains stable.
4. Assign consistency_score from 0.0 to 1.0.
5. Explain whether confidence should be reduced and why.
"""

FINAL_JUDGE_SYSTEM = (
    "You are the final judge for fake-news detection. Fuse the coordinator "
    "synthesis, self-reflection audit, and role-reversal consistency check."
)

FINAL_JUDGE_PROMPT = """
## News Article
{news_text}

## Coordinator Synthesis
{coordinator_synthesis}

## Self-Reflective Audit
{self_reflection}

## Role-Reversal Consistency
{role_reversal}

## Task
Return the final judgment in a compact structured form.
Important: this is a binary fake-news detection task, so final_label must be
exactly one of real or fake. Do not output uncertain as final_label. If evidence
is weak or mixed, choose the more likely label and reflect uncertainty through
confidence_score and uncertainty_source.

- final_label: real/fake
- confidence_score: 0.0 to 1.0
- consistency_adjustment: how role reversal changed confidence
- key_accepted_arguments
- rejected_arguments
- uncertainty_source
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt 格式化工具
# ═══════════════════════════════════════════════════════════════════════════════

def format_opening_prompt(news_text: str, side: str) -> tuple[str, str]:
    """格式化开篇立论 Prompt。

    Args:
        news_text: 原始新闻文本
        side: "Proponent" 或 "Opponent"

    Returns:
        (system_msg, user_prompt)
    """
    stance = "for" if side == "Proponent" else "against"
    stance_label = "REAL (True)" if side == "Proponent" else "FAKE (False)"
    system_msg = OPENING_SYSTEM.format(stance=stance, side=side)
    prompt = OPENING_PROMPT.format(
        news_text=news_text, side=side, stance_label=stance_label
    )
    return system_msg, prompt


def format_cross_exam_prompt(
    news_text: str, side: str, pro_opening: str, opp_opening: str
) -> tuple[str, str]:
    """格式化质询反驳 Prompt。"""
    stance = "for" if side == "Proponent" else "against"
    stance_label = "REAL (True)" if side == "Proponent" else "FAKE (False)"
    system_msg = CROSS_EXAM_SYSTEM.format(stance=stance, side=side)
    prompt = CROSS_EXAM_PROMPT.format(
        news_text=news_text,
        side=side,
        stance_label=stance_label,
        pro_opening=pro_opening,
        opp_opening=opp_opening,
    )
    return system_msg, prompt


def format_closing_prompt(
    news_text: str,
    side: str,
    pro_opening: str,
    opp_opening: str,
    pro_cross: str,
    opp_cross: str,
) -> tuple[str, str]:
    """格式化结案陈词 Prompt。"""
    stance = "for" if side == "Proponent" else "against"
    stance_label = "REAL (True)" if side == "Proponent" else "FAKE (False)"
    system_msg = CLOSING_SYSTEM.format(stance=stance, side=side)
    prompt = CLOSING_PROMPT.format(
        news_text=news_text,
        side=side,
        stance_label=stance_label,
        pro_opening=pro_opening,
        opp_opening=opp_opening,
        pro_cross=pro_cross,
        opp_cross=opp_cross,
    )
    return system_msg, prompt


def format_synthesis_prompt(
    news_text: str,
    pro_opening: str,
    opp_opening: str,
    pro_cross: str,
    opp_cross: str,
    pro_closing: str,
    opp_closing: str,
) -> tuple[str, str]:
    """格式化 Synthesis Agent 总结 Prompt。"""
    prompt = SYNTHESIS_PROMPT.format(
        news_text=news_text,
        pro_opening=pro_opening,
        opp_opening=opp_opening,
        pro_cross=pro_cross,
        opp_cross=opp_cross,
        pro_closing=pro_closing,
        opp_closing=opp_closing,
    )
    return SYNTHESIS_SYSTEM, prompt


def format_perspective_planner_prompt(
    news_text: str,
    perspective_catalog: str,
    top_k: int,
) -> tuple[str, str]:
    """格式化 Perspective Planner Prompt。"""
    prompt = PERSPECTIVE_PLANNER_PROMPT.format(
        news_text=news_text,
        perspective_catalog=perspective_catalog,
        top_k=top_k,
    )
    return PERSPECTIVE_PLANNER_SYSTEM, prompt


def format_perspective_agent_prompt(
    news_text: str,
    perspective_key: str,
    role_name: str,
    focus: str,
    planner_summary: str,
) -> tuple[str, str]:
    """格式化多视角专家 Prompt。"""
    system_msg = PERSPECTIVE_AGENT_SYSTEM.format(role_name=role_name)
    prompt = PERSPECTIVE_AGENT_PROMPT.format(
        news_text=news_text,
        perspective_key=perspective_key,
        role_name=role_name,
        focus=focus,
        planner_summary=planner_summary,
    )
    return system_msg, prompt


def format_perspective_coordinator_prompt(
    news_text: str,
    perspective_reports: str,
) -> tuple[str, str]:
    """格式化多视角聚合 Prompt。"""
    prompt = PERSPECTIVE_COORDINATOR_PROMPT.format(
        news_text=news_text,
        perspective_reports=perspective_reports,
    )
    return PERSPECTIVE_COORDINATOR_SYSTEM, prompt


def format_self_reflective_judge_prompt(
    news_text: str,
    coordinator_synthesis: str,
    perspective_reports: str,
) -> tuple[str, str]:
    """格式化自反思裁判 Prompt。"""
    prompt = SELF_REFLECTION_PROMPT.format(
        news_text=news_text,
        coordinator_synthesis=coordinator_synthesis,
        perspective_reports=perspective_reports,
    )
    return SELF_REFLECTION_SYSTEM, prompt


def format_role_reversal_prompt(
    news_text: str,
    coordinator_synthesis: str,
    self_reflection: str,
) -> tuple[str, str]:
    """格式化角色反转一致性 Prompt。"""
    prompt = ROLE_REVERSAL_PROMPT.format(
        news_text=news_text,
        coordinator_synthesis=coordinator_synthesis,
        self_reflection=self_reflection,
    )
    return ROLE_REVERSAL_SYSTEM, prompt


def format_final_judge_prompt(
    news_text: str,
    coordinator_synthesis: str,
    self_reflection: str,
    role_reversal: str,
) -> tuple[str, str]:
    """格式化最终裁判 Prompt。"""
    prompt = FINAL_JUDGE_PROMPT.format(
        news_text=news_text,
        coordinator_synthesis=coordinator_synthesis,
        self_reflection=self_reflection,
        role_reversal=role_reversal,
    )
    return FINAL_JUDGE_SYSTEM, prompt
