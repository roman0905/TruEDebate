"""Prompt templates and OpenAI helpers for debate generation."""

import logging
import os
import re

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

LOGIC_MARKER = "[LOGIC_FLAWS]"
EVIDENCE_MARKER = "[COUNTER_EVIDENCE]"
REBUTTAL_MARKER = "[REBUTTAL_SPEECH]"
COMPLETE_END_CHARS = '.!?"\')]}。！？；：”’）】》'

_INCOMPLETE_END_RE = re.compile(
    r"\b(the|and|or|to|of|in|for|with|that|which|a|an|is|are|was|were|be|by|from|as|on|at|this|these|those)\s*$",
    re.IGNORECASE,
)
_REBUTTAL_MARKER_RE = re.compile(
    rf"{re.escape(REBUTTAL_MARKER)}\s*(.*)",
    re.DOTALL,
)
_NEXT_MARKER_RE = re.compile(
    r"\n\s*\[(?:LOGIC_FLAWS|COUNTER_EVIDENCE|REBUTTAL_SPEECH)\]\s*\n"
)


def _get_client() -> OpenAI:
    """Return a shared OpenAI client."""
    global _client
    if _client is None:
        api_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        base_url = config.OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. Set it in the environment or config.py."
            )
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        _client = OpenAI(**client_kwargs)
    return _client


def _looks_incomplete(text: str) -> bool:
    """Detect obvious truncation before persisting text."""
    stripped = text.rstrip()
    if not stripped:
        return True
    if stripped[-1] not in COMPLETE_END_CHARS:
        return True
    if _INCOMPLETE_END_RE.search(stripped):
        return True
    return False


def _build_retry_prompt(prompt: str) -> str:
    """Ask the model to regenerate a compact but complete answer."""
    if re.search(r"[\u4e00-\u9fff]", prompt):
        return (
            f"{prompt}\n\n"
            "## 重试约束\n"
            "你上一次回答被截断或不完整。请从头重新生成完整答案，"
            "保留关键证据和推理，但删除空话，避免寒暄，避免重复，"
            "并以一句完整的话结束。必须严格遵守上面的输出结构。"
        )
    return (
        f"{prompt}\n\n"
        "## Retry Constraint\n"
        "Your previous answer was cut off or incomplete. Regenerate the full answer "
        "from scratch. Keep all key evidence and reasoning, but remove filler, avoid "
        "ceremonial openings, avoid repetition, and finish with a complete final sentence. "
        "Follow the required structure exactly."
    )


def call_llm(
    prompt: str,
    system_msg: str = "You are a helpful assistant.",
    generation_key: str | None = None,
) -> str:
    """Call the chat completion API with stage-aware token budgets and retries."""
    client = _get_client()
    stage_key = generation_key or "default"
    initial_max_tokens = config.OPENAI_STAGE_MAX_TOKENS.get(
        stage_key, config.OPENAI_MAX_TOKENS
    )
    retry_max_tokens = config.OPENAI_STAGE_RETRY_MAX_TOKENS.get(
        stage_key, initial_max_tokens
    )
    current_prompt = prompt
    current_max_tokens = initial_max_tokens

    for attempt in range(1, config.OPENAI_MAX_RETRIES_ON_LENGTH + 2):
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": current_prompt},
                ],
                max_tokens=current_max_tokens,
                temperature=config.OPENAI_TEMPERATURE,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        finish_reason = choice.finish_reason or "unknown"
        incomplete = finish_reason == "length" or _looks_incomplete(content)

        logger.info(
            "[LLM/%s] attempt=%s finish_reason=%s chars=%s max_tokens=%s",
            stage_key,
            attempt,
            finish_reason,
            len(content),
            current_max_tokens,
        )
        if not incomplete:
            return content

        if attempt > config.OPENAI_MAX_RETRIES_ON_LENGTH:
            logger.warning(
                "[LLM/%s] output still looks incomplete after retries "
                "(finish_reason=%s, chars=%s)",
                stage_key,
                finish_reason,
                len(content),
            )
            return content

        logger.warning(
            "[LLM/%s] detected truncated or incomplete output; retrying "
            "(attempt=%s, finish_reason=%s)",
            stage_key,
            attempt,
            finish_reason,
        )
        current_prompt = _build_retry_prompt(prompt)
        current_max_tokens = min(
            retry_max_tokens,
            max(current_max_tokens + 128, int(current_max_tokens * 1.35)),
        )


OPENING_SYSTEM = (
    "You are a debate participant. You must argue {stance} the authenticity "
    "of the given news article. You are the Opening Speaker for the {side} side. "
    "Present a compact, evidence-first opening statement."
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
1. Write 2-3 short paragraphs, 160-220 words total.
2. Use only the 2-3 strongest arguments grounded in the news text.
3. Prioritize evidence about sources, tone, specific details, and verifiability.
4. Do not use ceremonial salutations, stage directions, or generic debate filler.
5. End with one direct sentence stating your conclusion.

## Your Opening Statement:
"""

CROSS_EXAM_SYSTEM = (
    "You are a debate participant. You must argue {stance} the authenticity "
    "of the given news article. You are the Questioner for the {side} side. "
    "Challenge the opposing side's arguments with concise analysis."
)

CROSS_EXAM_PROMPT = f"""
## Debate Topic
Evaluate the authenticity of the following news article.

## News Article
{{news_text}}

## Previous Opening Statements
### Proponent (argues news is REAL):
{{pro_opening}}

### Opponent (argues news is FAKE):
{{opp_opening}}

## Your Role
You are the **Questioner** for the **{{side}}** side.
Your stance: The news is **{{stance_label}}**.

## Chain-of-Thought Output Format (MANDATORY)
Think step by step and structure your response in EXACTLY three sections. Each
section MUST begin with the exact marker on its own line. Do not add any prose
before the first marker.

{LOGIC_MARKER}
Identify exactly 2 logical flaws, unsupported assumptions, or internal
contradictions in the opposing opening statement. Use 2 bullet points. Keep
each bullet to at most 35 words and explain why it weakens the argument.

{EVIDENCE_MARKER}
Provide exactly 2 counter-evidence points grounded in the news article itself.
Use 2 bullet points. Keep each bullet to at most 40 words. Reference sources,
tone, verifiable details, citation patterns, or emotional-language cues.

{REBUTTAL_MARKER}
Write a polished cross-examination speech in 2 short paragraphs, 140-190 words
total. Use the strongest points above. Do not mention the markers or
meta-analysis. Do not use ceremonial salutations or filler. End with exactly 1
pointed rhetorical question.

## Output:
"""

CLOSING_SYSTEM = (
    "You are a debate participant. You must argue {stance} the authenticity "
    "of the given news article. You are the Closing Speaker for the {side} side. "
    "Deliver a compact closing argument."
)

CLOSING_PROMPT = """
## Debate Topic
Evaluate the authenticity of the following news article.

## News Article
{news_text}

## Full Debate Record
### Stage 1 - Opening Statements
**Proponent Opening:** {pro_opening}
**Opponent Opening:** {opp_opening}

### Stage 2 - Cross-Examination
**Proponent Questioner:** {pro_cross}
**Opponent Questioner:** {opp_cross}

## Your Role
You are the **Closing Speaker** for the **{side}** side.
Your stance: The news is **{stance_label}**.

## Instructions
1. Write 2-3 short paragraphs, 180-240 words total.
2. Keep only the 2 strongest arguments from your side.
3. Address the strongest opposing point once and explain why it fails.
4. Do not restate the entire debate or repeat earlier phrasing.
5. Do not use ceremonial salutations or filler.
6. End with one clear concluding sentence.

## Your Closing Statement:
"""

SYNTHESIS_SYSTEM = (
    "You are the Synthesis Agent. Your job is to objectively analyze a debate "
    "about the authenticity of a news article and produce a compact structured summary."
)

SYNTHESIS_PROMPT = """
## News Article
{news_text}

## Full Debate Record
### Stage 1 - Opening Statements
**Proponent (REAL):** {pro_opening}
**Opponent (FAKE):** {opp_opening}

### Stage 2 - Cross-Examination
**Proponent Questioner:** {pro_cross}
**Opponent Questioner:** {opp_cross}

### Stage 3 - Closing Statements
**Proponent Closing:** {pro_closing}
**Opponent Closing:** {opp_closing}

## Your Task
Produce a compact assessment of the debate that preserves the key semantic
content without filler.

## Output Format (MANDATORY)
Write exactly 6 labeled sections in this order:
1. Verifiable Details:
2. Source Reliability:
3. Tone and Style:
4. Emotional Language:
5. Cross-Verification:
Conclusion:

## Constraints
1. Total length: 220-320 words.
2. Each numbered section: 1-2 sentences only.
3. Conclusion: 1-2 sentences only.
4. No introduction, no recap of the debate history, no bullet lists, no filler.
5. Focus on the strongest evidence and disagreements only.

## Your Synthesis Report:
"""

OPENING_SYSTEM_ZH = (
    "你是一名辩论参与者，必须围绕给定新闻的真实性进行{stance}立场论证。"
    "你是{side}的一辩开篇陈词者。请给出紧凑、证据优先的开篇陈词。"
)

OPENING_PROMPT_ZH = """
## 辩题
评估以下新闻文章的真实性。

## 新闻文章
{news_text}

## 你的角色
你是**{side}**的**开篇陈词者**。
你的立场：该新闻是**{stance_label}**。

## 写作要求
1. 写 2-3 个短段落，总长度约 260-380 个中文字符。
2. 只使用新闻文本中最强的 2-3 个论点。
3. 优先分析来源、语气、具体细节和可核验性。
4. 不要使用寒暄、舞台说明或泛泛的辩论套话。
5. 最后用一句直接的话明确你的结论。

## 你的开篇陈词：
"""

CROSS_EXAM_SYSTEM_ZH = (
    "你是一名辩论参与者，必须围绕给定新闻的真实性进行{stance}立场论证。"
    "你是{side}的质询者。请用简洁分析挑战对方论点。"
)

CROSS_EXAM_PROMPT_ZH = f"""
## 辩题
评估以下新闻文章的真实性。

## 新闻文章
{{news_text}}

## 前序开篇陈词
### 正方（主张新闻为真）：
{{pro_opening}}

### 反方（主张新闻为假）：
{{opp_opening}}

## 你的角色
你是**{{side}}**的**质询者**。
你的立场：该新闻是**{{stance_label}}**。

## 思维链输出格式（必须遵守）
请分步骤分析，并严格输出下面三个部分。每个部分必须以独占一行的指定标记开始。
第一个标记前不要添加任何文字。

{LOGIC_MARKER}
指出对方开篇陈词中恰好 2 个逻辑漏洞、无依据假设或内部矛盾。
使用 2 个项目符号。每条不超过 55 个中文字符，并说明其为何削弱论证。

{EVIDENCE_MARKER}
给出恰好 2 个基于新闻文本本身的反证点。
使用 2 个项目符号。每条不超过 65 个中文字符。可引用来源、语气、可核验细节、引用模式或情绪化语言线索。

{REBUTTAL_MARKER}
写一段成熟的质询反驳发言，分 2 个短段落，总长度约 220-340 个中文字符。
使用上面最强的要点。不要提及这些标记或元分析。不要寒暄或堆砌套话。
最后必须以 1 个尖锐的反问句结尾。

## 输出：
"""

CLOSING_SYSTEM_ZH = (
    "你是一名辩论参与者，必须围绕给定新闻的真实性进行{stance}立场论证。"
    "你是{side}的总结陈词者。请给出紧凑的结案陈词。"
)

CLOSING_PROMPT_ZH = """
## 辩题
评估以下新闻文章的真实性。

## 新闻文章
{news_text}

## 完整辩论记录
### 第一阶段 - 开篇陈词
**正方开篇：** {pro_opening}
**反方开篇：** {opp_opening}

### 第二阶段 - 质询反驳
**正方质询：** {pro_cross}
**反方质询：** {opp_cross}

## 你的角色
你是**{side}**的**总结陈词者**。
你的立场：该新闻是**{stance_label}**。

## 写作要求
1. 写 2-3 个短段落，总长度约 300-430 个中文字符。
2. 只保留己方最强的 2 个论点。
3. 回应对方最强观点一次，并说明其为何不成立。
4. 不要完整复述辩论过程，也不要重复早先措辞。
5. 不要使用寒暄或套话。
6. 最后用一句清晰的话总结结论。

## 你的总结陈词：
"""

SYNTHESIS_SYSTEM_ZH = (
    "你是综合分析智能体。你的任务是客观分析一场关于新闻真实性的辩论，"
    "并输出紧凑的结构化总结。"
)

SYNTHESIS_PROMPT_ZH = """
## 新闻文章
{news_text}

## 完整辩论记录
### 第一阶段 - 开篇陈词
**正方（真实）：** {pro_opening}
**反方（虚假）：** {opp_opening}

### 第二阶段 - 质询反驳
**正方质询：** {pro_cross}
**反方质询：** {opp_cross}

### 第三阶段 - 总结陈词
**正方总结：** {pro_closing}
**反方总结：** {opp_closing}

## 任务
生成一份紧凑评估，保留关键语义内容，不添加空泛套话。

## 输出格式（必须遵守）
严格按以下顺序写 6 个带标签部分：
1. 可核验细节：
2. 来源可靠性：
3. 语气和风格：
4. 情绪化语言：
5. 交叉验证：
结论：

## 约束
1. 总长度约 360-520 个中文字符。
2. 每个编号部分只写 1-2 句。
3. 结论只写 1-2 句。
4. 不要引言，不要复述辩论历史，不要使用项目符号，不要套话。
5. 只关注最强证据和关键分歧。

## 你的综合报告：
"""


def _is_zh(lang: str) -> bool:
    return lang.lower() in {"zh", "cn", "chinese"}


def _side_label(side: str, lang: str) -> str:
    if not _is_zh(lang):
        return side
    return "正方" if side == "Proponent" else "反方"


def format_opening_prompt(
    news_text: str, side: str, lang: str = "en"
) -> tuple[str, str]:
    """Format the opening statement prompt."""
    if _is_zh(lang):
        stance = "支持" if side == "Proponent" else "反对"
        stance_label = "真实" if side == "Proponent" else "虚假"
        template = OPENING_PROMPT_ZH
        system_template = OPENING_SYSTEM_ZH
    else:
        stance = "for" if side == "Proponent" else "against"
        stance_label = "REAL (True)" if side == "Proponent" else "FAKE (False)"
        template = OPENING_PROMPT
        system_template = OPENING_SYSTEM

    side_label = _side_label(side, lang)
    system_msg = system_template.format(stance=stance, side=side_label)
    prompt = template.format(
        news_text=news_text,
        side=side_label,
        stance_label=stance_label,
    )
    return system_msg, prompt


def format_cross_exam_prompt(
    news_text: str,
    side: str,
    pro_opening: str,
    opp_opening: str,
    lang: str = "en",
) -> tuple[str, str]:
    """Format the cross-examination prompt."""
    if _is_zh(lang):
        stance = "支持" if side == "Proponent" else "反对"
        stance_label = "真实" if side == "Proponent" else "虚假"
        template = CROSS_EXAM_PROMPT_ZH
        system_template = CROSS_EXAM_SYSTEM_ZH
    else:
        stance = "for" if side == "Proponent" else "against"
        stance_label = "REAL (True)" if side == "Proponent" else "FAKE (False)"
        template = CROSS_EXAM_PROMPT
        system_template = CROSS_EXAM_SYSTEM

    side_label = _side_label(side, lang)
    system_msg = system_template.format(stance=stance, side=side_label)
    prompt = template.format(
        news_text=news_text,
        side=side_label,
        stance_label=stance_label,
        pro_opening=pro_opening,
        opp_opening=opp_opening,
    )
    return system_msg, prompt


def extract_rebuttal(cot_output: str) -> str:
    """Extract only the final rebuttal speech from the structured CoT output."""
    match = _REBUTTAL_MARKER_RE.search(cot_output)
    if match:
        rebuttal = match.group(1)
        tail = _NEXT_MARKER_RE.search(rebuttal)
        if tail:
            rebuttal = rebuttal[: tail.start()]
        rebuttal = rebuttal.strip()
        if rebuttal:
            return rebuttal

    logger.warning(
        "Failed to locate the rebuttal marker in CoT output; falling back to raw output."
    )
    return cot_output.strip()


def format_closing_prompt(
    news_text: str,
    side: str,
    pro_opening: str,
    opp_opening: str,
    pro_cross: str,
    opp_cross: str,
    lang: str = "en",
) -> tuple[str, str]:
    """Format the closing statement prompt."""
    if _is_zh(lang):
        stance = "支持" if side == "Proponent" else "反对"
        stance_label = "真实" if side == "Proponent" else "虚假"
        template = CLOSING_PROMPT_ZH
        system_template = CLOSING_SYSTEM_ZH
    else:
        stance = "for" if side == "Proponent" else "against"
        stance_label = "REAL (True)" if side == "Proponent" else "FAKE (False)"
        template = CLOSING_PROMPT
        system_template = CLOSING_SYSTEM

    side_label = _side_label(side, lang)
    system_msg = system_template.format(stance=stance, side=side_label)
    prompt = template.format(
        news_text=news_text,
        side=side_label,
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
    lang: str = "en",
) -> tuple[str, str]:
    """Format the synthesis prompt."""
    system_msg = SYNTHESIS_SYSTEM_ZH if _is_zh(lang) else SYNTHESIS_SYSTEM
    template = SYNTHESIS_PROMPT_ZH if _is_zh(lang) else SYNTHESIS_PROMPT
    prompt = template.format(
        news_text=news_text,
        pro_opening=pro_opening,
        opp_opening=opp_opening,
        pro_cross=pro_cross,
        opp_cross=opp_cross,
        pro_closing=pro_closing,
        opp_closing=opp_closing,
    )
    return system_msg, prompt
