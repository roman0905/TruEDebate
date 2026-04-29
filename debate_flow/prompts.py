"""Prompt templates and OpenAI helpers for debate generation."""

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

LOGIC_MARKER = "[LOGIC_FLAWS]"
EVIDENCE_MARKER = "[RATIONALE_EVIDENCE]"
REBUTTAL_MARKER = "[REBUTTAL_SPEECH]"
COMPLETE_END_CHARS = '.!?"\')]}。！？；：”’）】》0123456789'

_INCOMPLETE_END_RE = re.compile(
    r"\b(the|and|or|to|of|in|for|with|that|which|a|an|is|are|was|were|be|by|from|as|on|at|this|these|those)\s*$",
    re.IGNORECASE,
)
_REBUTTAL_MARKER_RE = re.compile(
    rf"{re.escape(REBUTTAL_MARKER)}\s*(.*)",
    re.DOTALL,
)
_NEXT_MARKER_RE = re.compile(
    r"\n\s*\[(?:LOGIC_FLAWS|RATIONALE_EVIDENCE|REBUTTAL_SPEECH)\]\s*\n"
)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


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
    stripped = text.rstrip()
    if not stripped:
        return True
    if stripped[-1] not in COMPLETE_END_CHARS:
        return True
    if _INCOMPLETE_END_RE.search(stripped):
        return True
    return False


def _build_retry_prompt(prompt: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", prompt):
        return (
            f"{prompt}\n\n"
            "## 重试约束\n"
            "你上一次回答被截断或不完整。请从头重新生成完整答案，"
            "保留关键推理和证据，不要空话，不要重复，并以一句完整的话结束。"
            "必须严格遵守上面的输出结构。"
        )
    return (
        f"{prompt}\n\n"
        "## Retry Constraint\n"
        "Your previous answer was cut off or incomplete. Regenerate the full answer "
        "from scratch. Keep the key reasoning and evidence, remove filler, avoid "
        "repetition, and finish with a complete final sentence. Follow the required "
        "structure exactly."
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
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": current_prompt},
            ],
            max_tokens=current_max_tokens,
            temperature=config.OPENAI_TEMPERATURE,
        )

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

        current_prompt = _build_retry_prompt(prompt)
        current_max_tokens = min(
            retry_max_tokens,
            max(current_max_tokens + 128, int(current_max_tokens * 1.35)),
        )


CLAIM_SYSTEM = (
    "You are a claim decomposition agent. Extract only concise, verifiable factual "
    "claims from the news article."
)

CLAIM_PROMPT = """
## News Article
{news_text}

## Task
Extract 3-5 concise factual claims that are central to verifying this article.

## Constraints
1. Each claim must be one line only.
2. Use a numbered list: `1.`, `2.`, `3.` ...
3. Keep each claim within 8-22 words.
4. Prefer claims that mention people, events, dates, locations, institutions, or explicit actions.
5. Do not add commentary, probability language, or explanations.

## Output:
"""

CLAIM_SYSTEM_ZH = "你是声明拆分智能体。请从新闻中抽取简洁、可核验的事实性子声明。"

CLAIM_PROMPT_ZH = """
## 新闻文章
{news_text}

## 任务
抽取 3-5 条对核验该新闻最关键的事实性子声明。

## 约束
1. 每条子声明单独一行。
2. 使用编号列表：`1.`、`2.`、`3.` ...
3. 每条尽量控制在 12-35 个中文字符。
4. 优先保留人物、事件、时间、地点、机构、行为等可核验信息。
5. 不要加解释、评论或概率表述。

## 输出：
"""

OPENING_SYSTEM = (
    "You are a debate participant. You must argue {stance} the authenticity "
    "of the given news article. You are the Opening Speaker for the {side} side. "
    "Present a compact rationale-grounded opening statement."
)

OPENING_PROMPT = """
## Debate Topic
Evaluate the authenticity of the following news article.

## News Article
{news_text}
{claim_block}
{rationale_block}
{evidence_block}

## Your Role
You are the **Opening Speaker** for the **{side}** side.
Your stance: The news is **{stance_label}**.

## Output Format (MANDATORY)
Use the exact field labels below. Keep all labels in English. The content can be
written in English.

Argument:
One compact argument in 2-3 short paragraphs, 150-220 words total.

Referenced Claims:
Comma-separated claim IDs such as `c1, c3`.

Referenced Rationales:
Comma-separated rationale IDs such as `td_1, cs_1`.

Evidence IDs:
Comma-separated evidence IDs such as `e1, e3`. Use `none` only if no evidence cards are provided.

Reasoning:
2-3 concise sentences explaining why the referenced claims and rationales support your stance.

Weakness:
1-2 concise sentences naming the main uncertainty or vulnerability in your own position.

Confidence:
A single number between 0.00 and 1.00.

## Constraints
1. Cite 2-3 strongest claims only.
2. Cite at least 1 rationale card.
3. If evidence cards are provided, cite 1-3 evidence IDs and ground your argument in them.
4. Focus on specificity, verifiability, source cues, and commonsense plausibility.
5. Do not add any fields beyond the required ones.
"""

CROSS_EXAM_SYSTEM = (
    "You are a debate participant. You must argue {stance} the authenticity "
    "of the given news article. You are the Questioner for the {side} side. "
    "Challenge the opposing side's arguments with concise rationale-grounded analysis."
)

CROSS_EXAM_PROMPT = f"""
## Debate Topic
Evaluate the authenticity of the following news article.

## News Article
{{news_text}}
{{claim_block}}
{{rationale_block}}
{{evidence_block}}

## Previous Opening Statements
### Proponent (argues news is REAL):
{{pro_opening}}

### Opponent (argues news is FAKE):
{{opp_opening}}

## Your Role
You are the **Questioner** for the **{{side}}** side.
Your stance: The news is **{{stance_label}}**.

## Output Format (MANDATORY)
Use the exact field labels below. Keep all labels in English. The content can be
written in English.

Attack Points:
Exactly 2 bullet points identifying weaknesses, unsupported assumptions, or contradictions in the opponent's argument.

Counter Evidence:
Exactly 2 bullet points grounded in the claim list and rationale cards.

Argument:
One compact rebuttal in 2 short paragraphs, 140-200 words total.

Referenced Claims:
Comma-separated claim IDs such as `c1, c4`.

Referenced Rationales:
Comma-separated rationale IDs such as `td_1, cs_1`.

Evidence IDs:
Comma-separated evidence IDs such as `e1, e3`. Use `none` only if no evidence cards are provided.

Reasoning:
2-3 concise sentences explaining why the cited claims and rationale cards weaken the opposing stance.

Weakness:
1 concise sentence naming the main limitation in your own rebuttal.

Confidence:
A single number between 0.00 and 1.00.

## Constraints
1. Cite at least 2 claims in total.
2. Cite at least 1 rationale card.
3. If evidence cards are provided, cite 1-3 evidence IDs.
4. Do not output any prose outside these fields.
"""

CLOSING_SYSTEM = (
    "You are a debate participant. You must argue {stance} the authenticity "
    "of the given news article. You are the Closing Speaker for the {side} side. "
    "Deliver a compact rationale-grounded closing argument."
)

CLOSING_PROMPT = """
## Debate Topic
Evaluate the authenticity of the following news article.

## News Article
{news_text}
{claim_block}
{rationale_block}
{evidence_block}

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

## Output Format (MANDATORY)
Use the exact field labels below. Keep all labels in English. The content can be
written in English.

Argument:
One compact closing argument in 2-3 short paragraphs, 170-240 words total.

Referenced Claims:
Comma-separated claim IDs such as `c1, c2`.

Referenced Rationales:
Comma-separated rationale IDs such as `td_1, cs_1`.

Evidence IDs:
Comma-separated evidence IDs such as `e1, e3`. Use `none` only if no evidence cards are provided.

Reasoning:
2-3 concise sentences that crystallize your strongest closing logic.

Weakness:
1 concise sentence acknowledging the strongest remaining uncertainty.

Confidence:
A single number between 0.00 and 1.00.

## Constraints
1. Keep only your side's 2 strongest points.
2. Address the strongest opposing point once.
3. Cite at least 1 rationale card.
4. If evidence cards are provided, cite 1-3 evidence IDs.
5. Do not output any prose outside these fields.
"""

SYNTHESIS_SYSTEM = (
    "You are the Synthesis Agent. Your job is to objectively analyze a debate "
    "about the authenticity of a news article and produce a compact structured summary."
)

SYNTHESIS_PROMPT = """
## News Article
{news_text}
{claim_block}
{rationale_block}
{evidence_block}

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
Produce a compact assessment that preserves the key semantic content, the main
claims under dispute, and the strongest rationale signals.

## Output Format (MANDATORY)
Return exactly one JSON object and nothing else:
{{
  "proponent_summary": "...",
  "opponent_summary": "...",
  "supported_claims": ["c1", "c2"],
  "questionable_claims": ["c3"],
  "used_td_rationales": ["td_1"],
  "used_cs_rationales": ["cs_1"],
  "used_evidence": ["e1", "e3"],
  "conflict_points": "...",
  "final_debate_tendency": "real/fake/uncertain",
  "explanation": "..."
}}

## Constraints
1. Keep summaries concise and evidence-focused.
2. Claim and rationale IDs must come from the provided lists only.
3. Evidence IDs must come from the provided evidence cards only.
4. `final_debate_tendency` must be one of `real`, `fake`, or `uncertain`.
5. `explanation` should be 2-4 sentences.
"""

OPENING_SYSTEM_ZH = (
    "你是一名辩论参与者，必须围绕给定新闻的真实性进行{stance}立场论证。"
    "你是{side}的一辩开篇陈词者。请给出紧凑、受理由约束的开篇陈词。"
)

OPENING_PROMPT_ZH = """
## 辩题
评估以下新闻文章的真实性。

## 新闻文章
{news_text}
{claim_block}
{rationale_block}

## 你的角色
你是**{side}**的**开篇陈词者**。
你的立场：该新闻是**{stance_label}**。

## 输出格式（必须遵守）
字段标签必须保持为英文，字段内容可以写中文。

Argument:
2-3 个短段落，总长度约 260-380 个中文字符。

Referenced Claims:
用逗号分隔的 claim ID，例如 `c1, c3`。

Referenced Rationales:
用逗号分隔的 rationale ID，例如 `td_1, cs_1`。

Evidence IDs:
用逗号分隔的 evidence ID，例如 `e1, e3`。如果没有证据卡片，写 `none`。

Reasoning:
2-3 句，解释为什么这些 claim 和理由卡支持你的立场。

Weakness:
1-2 句，指出你方论证里最大的薄弱点。

Confidence:
一个 0.00 到 1.00 之间的小数。

## 约束
1. 只引用最强的 2-3 个 claim。
2. 至少引用 1 个 rationale card。
3. 如果提供了 evidence cards，必须引用 1-3 个 evidence ID。
4. 不要输出多余字段。
"""

CROSS_EXAM_SYSTEM_ZH = (
    "你是一名辩论参与者，必须围绕给定新闻的真实性进行{stance}立场论证。"
    "你是{side}的质询者。请用简洁、受理由约束的分析挑战对方论点。"
)

CROSS_EXAM_PROMPT_ZH = f"""
## 辩题
评估以下新闻文章的真实性。

## 新闻文章
{{news_text}}
{{claim_block}}
{{rationale_block}}
{{evidence_block}}

## 前序开篇陈词
### 正方（主张新闻为真）：
{{pro_opening}}

### 反方（主张新闻为假）：
{{opp_opening}}

## 你的角色
你是**{{side}}**的**质询者**。
你的立场：该新闻是**{{stance_label}}**。

## 输出格式（必须遵守）
字段标签必须保持为英文，字段内容可以写中文。

Attack Points:
恰好 2 个项目符号，指出对方论证中的漏洞、跳步或矛盾。

Counter Evidence:
恰好 2 个项目符号，必须基于子声明和理由卡片。

Argument:
一段成熟的质询反驳发言，分 2 个短段落，总长度约 220-340 个中文字符。

Referenced Claims:
用逗号分隔的 claim ID。

Referenced Rationales:
用逗号分隔的 rationale ID。

Evidence IDs:
用逗号分隔的 evidence ID。如果没有证据卡片，写 `none`。

Reasoning:
2-3 句，解释为什么引用的 claim 和理由削弱对方立场。

Weakness:
1 句，指出你方反驳仍存在的限制。

Confidence:
一个 0.00 到 1.00 之间的小数。

## 约束
1. 至少引用 2 个 claim。
2. 至少引用 1 个 rationale card。
3. 如果提供了 evidence cards，必须引用 1-3 个 evidence ID。
4. 不要输出多余字段。
"""

CLOSING_SYSTEM_ZH = (
    "你是一名辩论参与者，必须围绕给定新闻的真实性进行{stance}立场论证。"
    "你是{side}的总结陈词者。请给出紧凑、受理由约束的结案陈词。"
)

CLOSING_PROMPT_ZH = """
## 辩题
评估以下新闻文章的真实性。

## 新闻文章
{news_text}
{claim_block}
{rationale_block}
{evidence_block}

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

## 输出格式（必须遵守）
字段标签必须保持为英文，字段内容可以写中文。

Argument:
2-3 个短段落，总长度约 300-430 个中文字符。

Referenced Claims:
用逗号分隔的 claim ID。

Referenced Rationales:
用逗号分隔的 rationale ID。

Evidence IDs:
用逗号分隔的 evidence ID。如果没有证据卡片，写 `none`。

Reasoning:
2-3 句，概括你方最强的结案逻辑。

Weakness:
1 句，指出剩余的不确定性。

Confidence:
一个 0.00 到 1.00 之间的小数。

## 约束
1. 只保留己方最强的 2 个论点。
2. 回应对方最强观点一次。
3. 至少引用 1 个 rationale card。
4. 如果提供了 evidence cards，必须引用 1-3 个 evidence ID。
5. 不要输出多余字段。
"""

SYNTHESIS_SYSTEM_ZH = (
    "你是综合分析智能体。你的任务是客观分析一场关于新闻真实性的辩论，"
    "并输出紧凑的结构化总结。"
)

SYNTHESIS_PROMPT_ZH = """
## 新闻文章
{news_text}
{claim_block}
{rationale_block}
{evidence_block}

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
生成一份紧凑评估，保留关键语义内容、核心子声明和最强理由信号。

## 输出格式（必须遵守）
只返回一个 JSON 对象，不要输出任何额外文字：
{{
  "proponent_summary": "...",
  "opponent_summary": "...",
  "supported_claims": ["c1", "c2"],
  "questionable_claims": ["c3"],
  "used_td_rationales": ["td_1"],
  "used_cs_rationales": ["cs_1"],
  "used_evidence": ["e1", "e3"],
  "conflict_points": "...",
  "final_debate_tendency": "real/fake/uncertain",
  "explanation": "..."
}}

## 约束
1. 摘要要紧凑，只保留关键证据和关键分歧。
2. claim 和 rationale ID 只能使用题面里出现过的。
3. evidence ID 只能使用题面里出现过的。
4. `final_debate_tendency` 只能是 `real`、`fake`、`uncertain` 之一。
5. `explanation` 用 2-4 句话说明最终判断依据。
"""


def _is_zh(lang: str) -> bool:
    return lang.lower() in {"zh", "cn", "chinese"}


def _side_label(side: str, lang: str) -> str:
    if not _is_zh(lang):
        return side
    return "正方" if side == "Proponent" else "反方"


def _normalize_id_list(value: Any, valid_ids: list[str]) -> list[str]:
    valid_set = set(valid_ids)
    if isinstance(value, list):
        candidates = value
    else:
        text = str(value or "").strip()
        if not text:
            return []
        candidates = re.split(r"[,;/，、\s]+", text)

    cleaned = []
    for item in candidates:
        item_text = str(item).strip().strip(".")
        if item_text in valid_set and item_text not in cleaned:
            cleaned.append(item_text)
    return cleaned


def _extract_overlap_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text or "")
        if token.strip()
    }


def build_claim_records(claim_texts: list[str]) -> list[dict[str, str]]:
    records = []
    for idx, claim in enumerate(claim_texts[: config.RTED_MAX_CLAIMS], start=1):
        claim_text = str(claim).strip()
        if not claim_text:
            continue
        records.append({"id": f"c{idx}", "content": claim_text})
    return records


def _infer_related_claim_ids(
    rationale_text: str,
    claims: list[dict[str, str]],
) -> list[str]:
    if not rationale_text.strip() or not claims:
        return []

    rationale_tokens = _extract_overlap_tokens(rationale_text)
    scored: list[tuple[int, str]] = []
    for claim in claims:
        claim_tokens = _extract_overlap_tokens(claim["content"])
        overlap = len(rationale_tokens & claim_tokens)
        scored.append((overlap, claim["id"]))

    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = [claim_id for score, claim_id in scored if score > 0][:3]
    if chosen:
        return chosen
    return [claim["id"] for claim in claims[: min(2, len(claims))]]


def build_rationale_cards(
    claims: list[dict[str, str]],
    td_rationale: str,
    cs_rationale: str,
    td_pred: int = -1,
    cs_pred: int = -1,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if td_rationale.strip():
        cards.append(
            {
                "id": "td_1",
                "type": "textual_detail",
                "content": td_rationale.strip(),
                "related_claims": _infer_related_claim_ids(td_rationale, claims),
                "stance_pred": td_pred,
            }
        )
    if cs_rationale.strip():
        cards.append(
            {
                "id": "cs_1",
                "type": "commonsense_verifiability",
                "content": cs_rationale.strip(),
                "related_claims": _infer_related_claim_ids(cs_rationale, claims),
                "stance_pred": cs_pred,
            }
        )
    return cards


def build_internal_evidence_cards(
    rationale_cards: list[dict[str, Any]],
    publish_time: str = "",
) -> list[dict[str, Any]]:
    """Convert local TD/CS rationale cards into no-API internal evidence cards."""
    cards: list[dict[str, Any]] = []
    for idx, card in enumerate(rationale_cards, start=1):
        content = str(card.get("content", "")).strip()
        if not content:
            continue

        rationale_id = str(card.get("id", "")).strip()
        rationale_type = str(card.get("type", "rationale")).strip() or "rationale"
        if rationale_id.startswith("td_"):
            evidence_id = f"e_internal_td_{idx}"
        elif rationale_id.startswith("cs_"):
            evidence_id = f"e_internal_cs_{idx}"
        else:
            evidence_id = f"e_internal_{idx}"

        cards.append(
            {
                "id": evidence_id,
                "source": rationale_type,
                "url": "",
                "publish_time": publish_time,
                "credibility_score": 0.6,
                "stance": "neutral",
                "query_intent": "internal_rationale",
                "evidence_type": "internal_rationale",
                "related_claims": card.get("related_claims", []),
                "evidence_text": content,
                "retrieval_score": 1.0,
                "redundancy_score": 0.0,
                "rationale_id": rationale_id,
                "teacher_pred": card.get("stance_pred", -1),
            }
        )
    return cards


def _format_claim_block(claims: list[dict[str, Any]] | list[str], lang: str) -> str:
    if not claims:
        return ""
    label = "## Key Claims" if not _is_zh(lang) else "## 核心子声明"
    lines = [label]
    if claims and isinstance(claims[0], dict):
        for claim in claims:
            lines.append(f"- {claim['id']}: {claim['content']}")
    else:
        for idx, claim in enumerate(claims, start=1):
            lines.append(f"- c{idx}: {claim}")
    return "\n" + "\n".join(lines)


def _format_rationale_block(
    rationale_cards: list[dict[str, Any]],
    lang: str,
) -> str:
    if not rationale_cards:
        return ""

    if _is_zh(lang):
        header = "## 理由卡片"
    else:
        header = "## Rationale Cards"

    parts = ["", header]
    for card in rationale_cards:
        card_title = f"### {card['id']} ({card['type']})"
        related_claims = ", ".join(card.get("related_claims", [])) or "none"
        parts.extend(
            [
                card_title,
                f"Related Claims: {related_claims}",
                f"Stance Prediction: {card.get('stance_pred', -1)}",
                f"Content: {card.get('content', '').strip()}",
            ]
        )
    return "\n".join(parts)


def _infer_evidence_related_claims(
    evidence_text: str,
    claims: list[dict[str, str]],
    raw_related: Any = None,
) -> list[str]:
    claim_ids = [claim["id"] for claim in claims]
    normalized = _normalize_id_list(raw_related, claim_ids)
    if normalized:
        return normalized
    return _infer_related_claim_ids(evidence_text, claims)


def normalize_evidence_cards(
    raw_cards: list[dict[str, Any]] | None,
    claims: list[dict[str, str]],
    max_cards: int = config.EVITED_MAX_EVIDENCE_PER_SAMPLE,
) -> list[dict[str, Any]]:
    """Normalize external or internal evidence cards into EviTED evidence records."""
    if not raw_cards:
        return []

    cards: list[dict[str, Any]] = []
    for idx, card in enumerate(raw_cards, start=1):
        if not isinstance(card, dict):
            continue

        evidence_text = str(
            card.get("evidence_text")
            or card.get("snippet")
            or card.get("content")
            or card.get("text")
            or ""
        ).strip()
        if not evidence_text:
            continue

        evidence_id = str(card.get("id", f"e{idx}")).strip() or f"e{idx}"
        if not evidence_id.startswith("e"):
            evidence_id = f"e{idx}"

        stance = str(card.get("stance", "neutral")).strip().lower()
        if stance not in {"support", "refute", "neutral"}:
            stance = "neutral"

        def as_float(key: str, default: float) -> float:
            try:
                return float(card.get(key, default))
            except (TypeError, ValueError):
                return default

        related_claims = _infer_evidence_related_claims(
            evidence_text,
            claims,
            card.get("related_claims", card.get("related_claim")),
        )
        cards.append(
            {
                "id": evidence_id,
                "source": str(card.get("source", card.get("url", ""))).strip(),
                "url": str(card.get("url", "")).strip(),
                "publish_time": str(card.get("publish_time", card.get("published_date", ""))).strip(),
                "credibility_score": max(0.0, min(as_float("credibility_score", 0.5), 1.0)),
                "stance": stance,
                "query_intent": str(card.get("query_intent", "")).strip(),
                "evidence_type": str(card.get("evidence_type", "")).strip(),
                "rationale_id": str(card.get("rationale_id", "")).strip(),
                "teacher_pred": card.get("teacher_pred", card.get("stance_pred", -1)),
                "related_claims": related_claims,
                "evidence_text": evidence_text,
                "retrieval_score": max(0.0, min(as_float("retrieval_score", card.get("score", 0.5)), 1.0)),
                "redundancy_score": max(0.0, min(as_float("redundancy_score", 0.0), 1.0)),
            }
        )
        if len(cards) >= max_cards:
            break
    return cards


def _format_evidence_block(
    evidence_cards: list[dict[str, Any]],
    lang: str,
) -> str:
    if not evidence_cards:
        return ""

    header = "## Evidence Cards" if not _is_zh(lang) else "## 证据卡片"
    parts = ["", header]
    for card in evidence_cards:
        related_claims = ", ".join(card.get("related_claims", [])) or "none"
        parts.extend(
            [
                f"### {card['id']} ({card.get('stance', 'neutral')})",
                f"Related Claims: {related_claims}",
                f"Source: {card.get('source', '')}",
                f"Credibility Score: {card.get('credibility_score', 0.5)}",
                f"Retrieval Score: {card.get('retrieval_score', 0.5)}",
                f"Evidence: {card.get('evidence_text', '').strip()}",
            ]
        )
    return "\n".join(parts)


def _parse_labeled_sections(text: str, labels: list[str]) -> dict[str, str]:
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(rf"^(?P<label>{label_pattern})\s*:\s*(?P<inline>.*)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group("label")
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        inline = match.group("inline").strip()
        value = "\n".join(part for part in [inline, body] if part).strip()
        sections[label] = value
    return sections


def _parse_list_block(value: str) -> list[str]:
    if not value:
        return []
    items = []
    for line in value.splitlines():
        line = re.sub(r"^\s*[-*]\s*", "", line.strip())
        if line:
            items.append(line)
    if items:
        return items
    return [part.strip() for part in re.split(r"[;,；]+", value) if part.strip()]


def _parse_confidence(value: str) -> float:
    match = re.search(r"([01](?:\.\d+)?)", value or "")
    if not match:
        return 0.5
    conf = float(match.group(1))
    return max(0.0, min(conf, 1.0))


def _fallback_reasoning(argument: str) -> str:
    text = re.sub(r"\s+", " ", str(argument or "")).strip()
    if not text:
        return ""

    sentence_like = [
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?。！？])\s+", text)
        if chunk.strip()
    ]
    if len(sentence_like) >= 2:
        return " ".join(sentence_like[-2:])
    if sentence_like:
        return sentence_like[-1]

    words = text.split()
    if len(words) <= 24:
        return text
    return " ".join(words[-24:])


def render_argument_text(structured: dict[str, Any], lang: str = "en") -> str:
    sections = [
        ("Argument", structured.get("argument", "")),
        ("Referenced Claims", ", ".join(structured.get("referenced_claims", []))),
        ("Referenced Rationales", ", ".join(structured.get("referenced_rationales", []))),
        ("Evidence IDs", ", ".join(structured.get("evidence_ids", []))),
        ("Reasoning", structured.get("reasoning", "")),
        ("Weakness", structured.get("weakness", "")),
        ("Confidence", f"{structured.get('confidence', 0.5):.2f}"),
    ]
    if structured.get("attack_points"):
        sections.insert(0, ("Attack Points", "\n".join(f"- {x}" for x in structured["attack_points"])))
    if structured.get("counter_evidence"):
        insert_at = 1 if structured.get("attack_points") else 0
        sections.insert(
            insert_at + 1,
            ("Counter Evidence", "\n".join(f"- {x}" for x in structured["counter_evidence"])),
        )
    return "\n".join(f"{label}:\n{value}".strip() for label, value in sections if value)


def parse_structured_argument(
    raw_output: str,
    claim_ids: list[str],
    rationale_ids: list[str],
    evidence_ids: list[str] | None = None,
    fallback_text: str = "",
) -> dict[str, Any]:
    labels = [
        "Attack Points",
        "Counter Evidence",
        "Argument",
        "Referenced Claims",
        "Referenced Rationales",
        "Evidence IDs",
        "Reasoning",
        "Weakness",
        "Confidence",
    ]
    sections = _parse_labeled_sections(raw_output, labels)

    argument = sections.get("Argument", "").strip()
    if not argument:
        argument = fallback_text.strip() or raw_output.strip()

    referenced_claims = _normalize_id_list(sections.get("Referenced Claims", ""), claim_ids)
    if not referenced_claims and claim_ids:
        referenced_claims = claim_ids[: min(2, len(claim_ids))]

    referenced_rationales = _normalize_id_list(
        sections.get("Referenced Rationales", ""),
        rationale_ids,
    )
    if not referenced_rationales and rationale_ids:
        referenced_rationales = rationale_ids[:1]

    referenced_evidence = _normalize_id_list(
        sections.get("Evidence IDs", ""),
        evidence_ids or [],
    )

    reasoning = sections.get("Reasoning", "").strip()
    if not reasoning:
        reasoning = _fallback_reasoning(argument)

    return {
        "attack_points": _parse_list_block(sections.get("Attack Points", "")),
        "counter_evidence": _parse_list_block(sections.get("Counter Evidence", "")),
        "argument": argument,
        "referenced_claims": referenced_claims,
        "referenced_rationales": referenced_rationales,
        "evidence_ids": referenced_evidence,
        "reasoning": reasoning,
        "weakness": sections.get("Weakness", "").strip(),
        "confidence": _parse_confidence(sections.get("Confidence", "")),
    }


def _extract_json_text(raw_output: str) -> str:
    fenced = _JSON_BLOCK_RE.search(raw_output)
    if fenced:
        return fenced.group(1).strip()
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_output[start : end + 1]
    return raw_output.strip()


def render_synthesis_text(synthesis: dict[str, Any], lang: str = "en") -> str:
    tendency = synthesis.get("final_debate_tendency", "uncertain")
    supported = ", ".join(synthesis.get("supported_claims", []))
    questionable = ", ".join(synthesis.get("questionable_claims", []))
    td_used = ", ".join(synthesis.get("used_td_rationales", []))
    cs_used = ", ".join(synthesis.get("used_cs_rationales", []))
    evidence_used = ", ".join(synthesis.get("used_evidence", []))
    return (
        f"Proponent Summary: {synthesis.get('proponent_summary', '').strip()}\n"
        f"Opponent Summary: {synthesis.get('opponent_summary', '').strip()}\n"
        f"Supported Claims: {supported}\n"
        f"Questionable Claims: {questionable}\n"
        f"Used TD Rationales: {td_used}\n"
        f"Used CS Rationales: {cs_used}\n"
        f"Used Evidence: {evidence_used}\n"
        f"Conflict Points: {synthesis.get('conflict_points', '').strip()}\n"
        f"Final Debate Tendency: {tendency}\n"
        f"Explanation: {synthesis.get('explanation', '').strip()}"
    ).strip()


def parse_synthesis_output(
    raw_output: str,
    claim_ids: list[str],
    rationale_ids: list[str],
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    json_text = _extract_json_text(raw_output)
    try:
        payload = json.loads(json_text)
    except Exception:
        payload = {}

    td_ids = [rid for rid in rationale_ids if rid.startswith("td_")]
    cs_ids = [rid for rid in rationale_ids if rid.startswith("cs_")]
    tendency = str(payload.get("final_debate_tendency", "")).strip().lower()
    if tendency not in {"real", "fake", "uncertain"}:
        tendency = "uncertain"

    explanation = str(payload.get("explanation", "")).strip()
    if not explanation:
        explanation = raw_output.strip()

    return {
        "proponent_summary": str(payload.get("proponent_summary", "")).strip(),
        "opponent_summary": str(payload.get("opponent_summary", "")).strip(),
        "supported_claims": _normalize_id_list(payload.get("supported_claims", []), claim_ids),
        "questionable_claims": _normalize_id_list(payload.get("questionable_claims", []), claim_ids),
        "used_td_rationales": _normalize_id_list(payload.get("used_td_rationales", []), td_ids),
        "used_cs_rationales": _normalize_id_list(payload.get("used_cs_rationales", []), cs_ids),
        "used_evidence": _normalize_id_list(payload.get("used_evidence", []), evidence_ids or []),
        "conflict_points": str(payload.get("conflict_points", "")).strip(),
        "final_debate_tendency": tendency,
        "explanation": explanation,
    }


def format_claim_prompt(news_text: str, lang: str = "en") -> tuple[str, str]:
    if _is_zh(lang):
        return CLAIM_SYSTEM_ZH, CLAIM_PROMPT_ZH.format(news_text=news_text)
    return CLAIM_SYSTEM, CLAIM_PROMPT.format(news_text=news_text)


def parse_claims(raw_output: str, max_claims: int = 5) -> list[str]:
    claims = []
    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        if line:
            claims.append(line)
    if not claims:
        claims = [chunk.strip() for chunk in re.split(r"[。！？.!?]\s*", raw_output) if chunk.strip()]
    cleaned = []
    for claim in claims:
        claim = re.sub(r"\s+", " ", claim).strip()
        if claim and claim not in cleaned:
            cleaned.append(claim)
    return cleaned[:max_claims]


def format_opening_prompt(
    news_text: str,
    side: str,
    claims: list[dict[str, Any]] | list[str] | None = None,
    rationale_cards: list[dict[str, Any]] | None = None,
    evidence_cards: list[dict[str, Any]] | None = None,
    lang: str = "en",
) -> tuple[str, str]:
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
        claim_block=_format_claim_block(claims or [], lang),
        rationale_block=_format_rationale_block(rationale_cards or [], lang),
        evidence_block=_format_evidence_block(evidence_cards or [], lang),
        side=side_label,
        stance_label=stance_label,
    )
    return system_msg, prompt


def format_cross_exam_prompt(
    news_text: str,
    side: str,
    pro_opening: str,
    opp_opening: str,
    claims: list[dict[str, Any]] | list[str] | None = None,
    rationale_cards: list[dict[str, Any]] | None = None,
    evidence_cards: list[dict[str, Any]] | None = None,
    lang: str = "en",
) -> tuple[str, str]:
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
        claim_block=_format_claim_block(claims or [], lang),
        rationale_block=_format_rationale_block(rationale_cards or [], lang),
        evidence_block=_format_evidence_block(evidence_cards or [], lang),
        side=side_label,
        stance_label=stance_label,
        pro_opening=pro_opening,
        opp_opening=opp_opening,
    )
    return system_msg, prompt


def extract_rebuttal(cot_output: str) -> str:
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
    claims: list[dict[str, Any]] | list[str] | None = None,
    rationale_cards: list[dict[str, Any]] | None = None,
    evidence_cards: list[dict[str, Any]] | None = None,
    lang: str = "en",
) -> tuple[str, str]:
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
        claim_block=_format_claim_block(claims or [], lang),
        rationale_block=_format_rationale_block(rationale_cards or [], lang),
        evidence_block=_format_evidence_block(evidence_cards or [], lang),
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
    claims: list[dict[str, Any]] | list[str] | None = None,
    rationale_cards: list[dict[str, Any]] | None = None,
    evidence_cards: list[dict[str, Any]] | None = None,
    lang: str = "en",
) -> tuple[str, str]:
    system_msg = SYNTHESIS_SYSTEM_ZH if _is_zh(lang) else SYNTHESIS_SYSTEM
    template = SYNTHESIS_PROMPT_ZH if _is_zh(lang) else SYNTHESIS_PROMPT
    prompt = template.format(
        news_text=news_text,
        claim_block=_format_claim_block(claims or [], lang),
        rationale_block=_format_rationale_block(rationale_cards or [], lang),
        evidence_block=_format_evidence_block(evidence_cards or [], lang),
        pro_opening=pro_opening,
        opp_opening=opp_opening,
        pro_cross=pro_cross,
        opp_cross=opp_cross,
        pro_closing=pro_closing,
        opp_closing=opp_closing,
    )
    return system_msg, prompt
