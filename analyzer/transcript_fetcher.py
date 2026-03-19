"""
Transcript fetcher: pulls earnings call transcripts via API or parses uploads.
"""

import requests
import fitz  # PyMuPDF
from config import FMP_API_KEY


def fetch_transcripts_by_ticker(ticker: str) -> list[dict]:
    """
    Fetch the two most recent quarterly transcripts for a US-listed ticker
    using the Financial Modeling Prep API.

    Returns a list of up to 2 dicts, each with:
        - quarter: str (e.g. "Q4 2025")
        - date: str
        - content: str (full transcript text)
        - speakers: list of sections with speaker info
    """
    if not FMP_API_KEY:
        raise ValueError("FMP_API_KEY not set. Add it to your .env file.")

    # Step 1: Get available transcript dates for this ticker
    dates_url = (
        f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}"
        f"?apikey={FMP_API_KEY}"
    )
    resp = requests.get(dates_url, timeout=15)
    resp.raise_for_status()
    available = resp.json()

    if not available or not isinstance(available, list):
        return []

    # Take the two most recent
    recent = available[:2]
    transcripts = []

    for item in recent:
        quarter = item.get("quarter", "")
        year = item.get("year", "")
        date = item.get("date", "")
        content = item.get("content", "")

        if content:
            transcripts.append({
                "quarter": f"Q{quarter} {year}",
                "date": date,
                "content": content,
                "raw": item,
            })

    return transcripts


def parse_uploaded_file(uploaded_file) -> str:
    """
    Parse an uploaded file (PDF or text) and return the transcript text.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return _parse_pdf(uploaded_file)
    elif filename.endswith(".txt") or filename.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="replace")
    else:
        # Try reading as text
        return uploaded_file.read().decode("utf-8", errors="replace")


def _parse_pdf(uploaded_file) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)
