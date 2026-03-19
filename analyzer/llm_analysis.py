"""
LLM analysis: uses Claude API for theme extraction, risk flagging,
Q&A summarisation, and sentiment interpretation.
"""

import json
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS, FIXED_CATEGORIES


client = Anthropic(api_key=ANTHROPIC_API_KEY)


def _call_claude(system_prompt: str, user_prompt: str) -> str:
    """Make a single Claude API call and return the text response."""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def extract_themes(transcript_text: str) -> list[dict]:
    """
    Extract the top 5-7 key themes from the earnings call.

    Returns: [{"theme": str, "description": str, "relevance": str}, ...]
    """
    system = (
        "You are a senior equity research analyst. Extract key themes from earnings "
        "call transcripts. Be specific and concrete — avoid generic themes like "
        "'strong performance'. Focus on what would change an analyst's view. "
        "Return valid JSON only, no markdown."
    )
    prompt = f"""Analyze this earnings call transcript and extract the 5-7 most important themes.

For each theme, provide:
- "theme": a concise label (3-6 words)
- "description": one sentence explaining what was said
- "relevance": one sentence on why this matters for investors

Return a JSON array of objects. No other text.

TRANSCRIPT:
{transcript_text[:12000]}
"""
    raw = _call_claude(system, prompt)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return [{"theme": "Parse error", "description": raw[:500], "relevance": "N/A"}]


def extract_topic_texts(transcript_text: str) -> dict[str, str]:
    """
    Extract text passages relevant to each fixed framework category + any emergent topics.
    Used as input for FinBERT sentiment scoring.

    Returns: {"topic_name": "relevant text excerpt", ...}
    """
    categories_str = ", ".join(FIXED_CATEGORIES)
    system = (
        "You are a financial transcript analyst. Extract relevant text passages "
        "for each topic category. Return valid JSON only, no markdown."
    )
    prompt = f"""From this earnings call transcript, extract the most relevant passages
for each of the following categories. Also identify up to 3 additional emergent topics
that don't fit these categories but were prominently discussed.

Fixed categories: {categories_str}

For each topic (fixed + emergent), return the key 2-4 sentences from the transcript
that best represent management's position on that topic.

Return a JSON object where keys are topic names and values are the extracted text passages.
No other text.

TRANSCRIPT:
{transcript_text[:12000]}
"""
    raw = _call_claude(system, prompt)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return {"Parse Error": raw[:500]}


def flag_risks(transcript_text: str) -> list[dict]:
    """
    Identify risk flags and hedging language.

    Returns: [{"category": str, "flag": str, "quote": str, "severity": str}, ...]
    """
    system = (
        "You are a skeptical credit analyst reviewing an earnings call transcript. "
        "Your job is to flag language that should concern investors. "
        "Be specific — cite exact quotes. Don't flag things that are normal. "
        "Only flag genuine red flags or hedging. Return valid JSON only, no markdown."
    )
    prompt = f"""Review this earnings call transcript and flag any risk indicators.

Flag categories:
1. EVASIVE LANGUAGE: Non-answers to direct questions, shifting from concrete metrics to qualitative language, "we're monitoring/exploring" without specifics
2. GUIDANCE SIGNALS: Widened guidance ranges, quietly dropped metrics, redefined non-GAAP measures
3. RED FLAG PHRASES: Recurring "one-time" charges, blaming macro for underperformance, "investing for the long term" when margins compress
4. STRUCTURAL TELLS: CEO answering CFO questions (or vice versa), unusually short Q&A, analyst follow-ups being cut off

For each flag, provide:
- "category": one of the four categories above
- "flag": concise description of the issue (one sentence)
- "quote": the exact words from the transcript (keep under 30 words)
- "severity": "high", "medium", or "low"

Return a JSON array. If nothing is flagged, return an empty array. No other text.

TRANSCRIPT:
{transcript_text[:12000]}
"""
    raw = _call_claude(system, prompt)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return []


def summarize_qa(qa_exchanges: list[dict]) -> list[dict]:
    """
    Summarize Q&A exchanges with quality tags.

    Returns: [{"analyst": str, "topic": str, "summary": str, "quality": str}, ...]
    """
    if not qa_exchanges:
        return []

    exchanges_text = ""
    for i, ex in enumerate(qa_exchanges[:15], 1):  # Cap at 15 exchanges
        exchanges_text += (
            f"\n--- Exchange {i} ---\n"
            f"Analyst: {ex.get('analyst', 'Unknown')}\n"
            f"Question: {ex.get('question', '')[:500]}\n"
            f"Response by: {ex.get('response_by', 'Unknown')}\n"
            f"Response: {ex.get('response', '')[:500]}\n"
        )

    system = (
        "You are a senior analyst summarizing the Q&A section of an earnings call. "
        "Be brutally honest about whether management actually answered the question. "
        "Return valid JSON only, no markdown."
    )
    prompt = f"""Summarize each Q&A exchange below. For each, provide:
- "analyst": analyst name and firm
- "topic": what the question was about (3-5 words)
- "summary": one sentence combining the question and response
- "quality": one of "DIRECT ANSWER", "PARTIAL ANSWER", or "DEFLECTION"

A DEFLECTION is when management redirects, gives a non-answer, or answers a different question.
A PARTIAL ANSWER addresses the question but omits key specifics.
A DIRECT ANSWER fully addresses what was asked.

Return a JSON array. No other text.

Q&A EXCHANGES:
{exchanges_text}
"""
    raw = _call_claude(system, prompt)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return []


def generate_questions(
    themes: list[dict],
    risk_flags: list[dict],
    qa_summary: list[dict],
) -> list[str]:
    """
    Generate follow-up questions a sharp analyst would ask.
    """
    context = f"""
THEMES IDENTIFIED:
{json.dumps(themes, indent=2)[:3000]}

RISK FLAGS:
{json.dumps(risk_flags, indent=2)[:3000]}

Q&A QUALITY SUMMARY:
{json.dumps(qa_summary, indent=2)[:3000]}
"""
    system = (
        "You are a veteran buy-side analyst. Based on the analysis of an earnings call, "
        "generate the questions you would ask if you had 5 minutes with the CEO. "
        "Focus on gaps, deflections, and concerning signals. Be specific and pointed — "
        "no softball questions. Return valid JSON only, no markdown."
    )
    prompt = f"""Based on this earnings call analysis, generate 5-7 pointed follow-up questions
that a sharp analyst would ask. Each question should target a specific gap, deflection,
or risk signal identified in the analysis.

Return a JSON array of strings (the questions). No other text.

{context}
"""
    raw = _call_claude(system, prompt)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return [raw[:500]]


def interpret_sentiment_shifts(
    current_scores: dict[str, dict],
    prior_scores: dict[str, dict],
    current_quarter: str,
    prior_quarter: str,
) -> list[dict]:
    """
    Use Claude to interpret the FinBERT sentiment score shifts between quarters.

    Returns: [{"topic": str, "shift": str, "interpretation": str}, ...]
    """
    shifts_data = []
    all_topics = set(list(current_scores.keys()) + list(prior_scores.keys()))

    for topic in all_topics:
        curr = current_scores.get(topic, {}).get("score", 0)
        prev = prior_scores.get(topic, {}).get("score", 0)
        delta = curr - prev
        shifts_data.append({
            "topic": topic,
            "prior_score": prev,
            "current_score": curr,
            "delta": round(delta, 3),
        })

    # Sort by absolute delta — biggest shifts first
    shifts_data.sort(key=lambda x: abs(x["delta"]), reverse=True)

    system = (
        "You are a financial analyst interpreting sentiment changes between quarterly "
        "earnings calls. Provide concise, specific interpretations. "
        "Return valid JSON only, no markdown."
    )
    prompt = f"""These are FinBERT sentiment scores by topic, comparing {prior_quarter} to {current_quarter}.
Scores range from -1 (very negative) to +1 (very positive).

{json.dumps(shifts_data, indent=2)}

For each topic with a meaningful shift (absolute delta > 0.05), provide:
- "topic": the topic name
- "prior_score": the score from {prior_quarter}
- "current_score": the score from {current_quarter}
- "delta": the change
- "shift": "improved", "declined", "stable", or "new_topic" / "dropped_topic"
- "interpretation": one sentence explaining what likely drove the change

Return a JSON array. No other text.
"""
    raw = _call_claude(system, prompt)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return shifts_data


def _clean_json(text: str) -> str:
    """Strip markdown code fences and whitespace from LLM JSON output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
