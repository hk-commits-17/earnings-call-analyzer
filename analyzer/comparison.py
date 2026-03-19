"""
Comparison engine: runs the full analysis pipeline on two transcripts
and produces the quarter-over-quarter comparison.
"""

from analyzer.parser import parse_transcript
from analyzer.sentiment import score_sections_by_topic
from analyzer.llm_analysis import (
    extract_themes,
    extract_topic_texts,
    flag_risks,
    summarize_qa,
    generate_questions,
    interpret_sentiment_shifts,
)


def analyze_single_transcript(transcript_text: str, quarter_label: str) -> dict:
    """
    Run the full analysis pipeline on a single transcript.

    Returns a dict with all analysis outputs for one quarter.
    """
    # Parse the transcript into sections
    parsed = parse_transcript(transcript_text)

    # Extract topic texts for sentiment scoring
    topic_texts = extract_topic_texts(transcript_text)

    # Score sentiment per topic using FinBERT
    sentiment_scores = score_sections_by_topic(topic_texts)

    # Extract key themes
    themes = extract_themes(transcript_text)

    # Flag risks
    risks = flag_risks(transcript_text)

    # Summarize Q&A
    qa_summary = summarize_qa(parsed["qa_exchanges"])

    return {
        "quarter": quarter_label,
        "parsed": parsed,
        "topic_texts": topic_texts,
        "sentiment_scores": sentiment_scores,
        "themes": themes,
        "risks": risks,
        "qa_summary": qa_summary,
    }


def run_comparison(
    current_text: str,
    prior_text: str,
    current_quarter: str,
    prior_quarter: str,
) -> dict:
    """
    Run the full comparison pipeline across two quarters.

    Returns the complete analysis report data.
    """
    # Analyze each transcript independently
    current = analyze_single_transcript(current_text, current_quarter)
    prior = analyze_single_transcript(prior_text, prior_quarter)

    # Interpret sentiment shifts using Claude
    sentiment_shifts = interpret_sentiment_shifts(
        current["sentiment_scores"],
        prior["sentiment_scores"],
        current_quarter,
        prior_quarter,
    )

    # Generate follow-up questions based on the current quarter analysis
    questions = generate_questions(
        current["themes"],
        current["risks"],
        current["qa_summary"],
    )

    # Identify new and dropped topics
    current_topics = set(current["sentiment_scores"].keys())
    prior_topics = set(prior["sentiment_scores"].keys())
    new_topics = current_topics - prior_topics
    dropped_topics = prior_topics - current_topics

    return {
        "current": current,
        "prior": prior,
        "current_quarter": current_quarter,
        "prior_quarter": prior_quarter,
        "sentiment_shifts": sentiment_shifts,
        "new_topics": list(new_topics),
        "dropped_topics": list(dropped_topics),
        "questions": questions,
    }
