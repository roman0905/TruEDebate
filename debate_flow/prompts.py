"""
TruEDebate (TED) — Prompt 模板与 LLM 调用工具
包含辩论三阶段的 Prompt 模板和 Synthesis Agent 的总结模板。
"""

import os
import re
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

## Chain-of-Thought Output Format (MANDATORY)
Think step-by-step and structure your response in EXACTLY three sections. Each
section MUST begin with the exact marker on its own line (including the square
brackets and Chinese characters). Do not add any prose before the first marker.

[逻辑漏洞定位]
Identify 2-3 specific logical flaws, unsupported assumptions, or internal
contradictions in the opposing side's opening statement. For each flaw, briefly
explain *why* it undermines their position.

[事实反证]
Provide 2-3 concrete counter-evidence points grounded in the news article
itself — cite sources, tone, verifiable details, citation patterns, or
emotional-language cues. Tie each item back to a weakness from the previous
section.

[反驳发言]
Write a polished, persuasive 3-5 paragraph cross-examination speech directed
at the opposing side. Leverage the analysis above, but write naturally — do
NOT mention the earlier markers, brackets, or meta-analysis. End with 1-2
pointed rhetorical questions that expose the opposing side's weakest point.

## Output:
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


_REBUTTAL_MARKER_RE = re.compile(r"\[\s*反驳发言\s*\]\s*(.*)", re.DOTALL)
_NEXT_MARKER_RE = re.compile(r"\n\s*\[[^\]]{1,30}\]\s*\n")


def extract_rebuttal(cot_output: str) -> str:
    """从 CoT 输出中提取 [反驳发言] 段落（只将此部分喂给 BERT）。

    期望的 LLM 输出结构:
        [逻辑漏洞定位] ...
        [事实反证] ...
        [反驳发言] ...

    - 定位最后一个 [反驳发言] 标记之后的文本。
    - 若该文本后还出现其他方括号标记（LLM 偶尔会追加脚注），截断到下一个标记前。
    - 标记缺失时回退到原始文本，避免数据生成失败。
    """
    match = _REBUTTAL_MARKER_RE.search(cot_output)
    if match:
        rebuttal = match.group(1)
        tail = _NEXT_MARKER_RE.search(rebuttal)
        if tail:
            rebuttal = rebuttal[: tail.start()]
        rebuttal = rebuttal.strip()
        if rebuttal:
            return rebuttal

    logger.warning("CoT 解析未能定位 [反驳发言]，回退到原始输出。")
    return cot_output.strip()


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
