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
    if stripped[-1] not in '.!?"\')]}':
        return True
    if _INCOMPLETE_END_RE.search(stripped):
        return True
    return False


def _build_retry_prompt(prompt: str) -> str:
    """Ask the model to regenerate a compact but complete answer."""
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


def format_opening_prompt(news_text: str, side: str) -> tuple[str, str]:
    """Format the opening statement prompt."""
    stance = "for" if side == "Proponent" else "against"
    stance_label = "REAL (True)" if side == "Proponent" else "FAKE (False)"
    system_msg = OPENING_SYSTEM.format(stance=stance, side=side)
    prompt = OPENING_PROMPT.format(
        news_text=news_text,
        side=side,
        stance_label=stance_label,
    )
    return system_msg, prompt


def format_cross_exam_prompt(
    news_text: str, side: str, pro_opening: str, opp_opening: str
) -> tuple[str, str]:
    """Format the cross-examination prompt."""
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
) -> tuple[str, str]:
    """Format the closing statement prompt."""
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
    """Format the synthesis prompt."""
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
