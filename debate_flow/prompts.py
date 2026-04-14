"""
TruEDebate (TED) — Prompt 模板与 LLM 调用工具
包含辩论三阶段的 Prompt 模板和 Synthesis Agent 的总结模板。
"""

import os
import logging
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
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 未设置。请通过环境变量或 config.py 配置。"
            )
        _client = OpenAI(api_key=api_key)
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
