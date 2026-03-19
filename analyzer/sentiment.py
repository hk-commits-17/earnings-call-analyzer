"""
Sentiment analysis using FinBERT for quantitative scoring.
Produces numerical sentiment scores per text segment.

Uses Streamlit's cache_resource to keep the model loaded across reruns.
Falls back to a keyword-based scorer if FinBERT can't load (e.g. memory limits).
"""

import os
import numpy as np

# Set HuggingFace cache to a writable location for Streamlit Cloud
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

_finbert_available = None


def _load_model():
    """
    Load FinBERT model and tokenizer with Streamlit caching.
    Returns (model, tokenizer) or (None, None) if loading fails.
    """
    global _finbert_available

    if _finbert_available is False:
        return None, None

    try:
        import streamlit as st

        @st.cache_resource(show_spinner="Loading FinBERT sentiment model...")
        def _cached_load():
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            return model, tokenizer

        result = _cached_load()
        _finbert_available = True
        return result

    except Exception:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            _finbert_available = True
            return model, tokenizer
        except Exception as e:
            print(f"FinBERT failed to load: {e}. Using keyword-based fallback.")
            _finbert_available = False
            return None, None


def score_sentiment(text: str) -> dict:
    """
    Score a text segment using FinBERT (or fallback).

    Returns:
        {
            "score": float (-1 to +1, negative to positive),
            "positive": float (probability),
            "negative": float (probability),
            "neutral": float (probability),
            "label": str ("positive", "negative", or "neutral"),
            "method": str ("finbert" or "keyword")
        }
    """
    model, tokenizer = _load_model()

    if model is not None and tokenizer is not None:
        return _score_with_finbert(text, model, tokenizer)
    else:
        return _score_with_keywords(text)


def _score_with_finbert(text: str, model, tokenizer) -> dict:
    """Score using FinBERT model."""
    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    probs = probabilities[0].numpy()

    positive_prob = float(probs[0])
    negative_prob = float(probs[1])
    neutral_prob = float(probs[2])

    score = positive_prob - negative_prob

    labels = ["positive", "negative", "neutral"]
    label = labels[int(np.argmax(probs))]

    return {
        "score": round(score, 3),
        "positive": round(positive_prob, 3),
        "negative": round(negative_prob, 3),
        "neutral": round(neutral_prob, 3),
        "label": label,
        "method": "finbert",
    }


def _score_with_keywords(text: str) -> dict:
    """
    Simple keyword-based sentiment fallback.
    Not as accurate as FinBERT but works without ML dependencies.
    """
    text_lower = text.lower()
    words = text_lower.split()
    total = max(len(words), 1)

    positive_words = {
        "strong", "growth", "exceeded", "outperformed", "record", "momentum",
        "accelerat", "robust", "confident", "optimistic", "beat", "upside",
        "expand", "increased", "improved", "positive", "encouraging", "solid",
        "resilient", "favorable", "surpassed", "raised", "upgrade",
    }
    negative_words = {
        "decline", "challenging", "headwind", "pressure", "weakness", "miss",
        "below", "deteriorat", "uncertain", "risk", "concern", "slowdown",
        "restructur", "impair", "loss", "difficult", "volatile", "contraction",
        "disappoint", "lowered", "downgrade", "cautious", "soft", "reduced",
    }

    pos_count = sum(1 for w in words if any(p in w for p in positive_words))
    neg_count = sum(1 for w in words if any(n in w for n in negative_words))

    pos_ratio = pos_count / total
    neg_ratio = neg_count / total

    score = round(pos_ratio - neg_ratio, 3)
    score = max(-1.0, min(1.0, score * 10))

    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {
        "score": round(score, 3),
        "positive": round(max(0, score), 3),
        "negative": round(max(0, -score), 3),
        "neutral": round(1 - abs(score), 3),
        "label": label,
        "method": "keyword",
    }


def score_sections_by_topic(topic_texts: dict[str, str]) -> dict[str, dict]:
    """
    Score sentiment for multiple topics.

    Args:
        topic_texts: {"topic_name": "relevant text excerpt", ...}

    Returns:
        {"topic_name": {"score": float, "label": str, ...}, ...}
    """
    results = {}
    for topic, text in topic_texts.items():
        if text and len(text.strip()) > 20:
            results[topic] = score_sentiment(text)
        else:
            results[topic] = {
                "score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "label": "neutral",
                "method": "none",
            }
    return results


def score_full_transcript(text: str, chunk_size: int = 3000) -> dict:
    """
    Score the overall sentiment of a full transcript by averaging chunk scores.
    """
    from analyzer.parser import get_text_chunks

    chunks = get_text_chunks(text, max_chars=chunk_size)
    if not chunks:
        return {"score": 0.0, "label": "neutral"}

    scores = []
    for chunk in chunks:
        result = score_sentiment(chunk)
        scores.append(result["score"])

    avg_score = float(np.mean(scores))
    label = "positive" if avg_score > 0.1 else ("negative" if avg_score < -0.1 else "neutral")

    return {
        "score": round(avg_score, 3),
        "label": label,
        "chunk_scores": [round(s, 3) for s in scores],
        "num_chunks": len(chunks),
    }
