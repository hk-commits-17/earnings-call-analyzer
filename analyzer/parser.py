"""
Transcript parser: splits raw transcript text into structured sections.
Identifies prepared remarks vs Q&A, and segments by speaker.
"""

import re


def parse_transcript(raw_text: str) -> dict:
    """
    Parse a raw transcript into structured sections.

    Returns:
        {
            "full_text": str,
            "prepared_remarks": str,
            "qa_section": str,
            "speakers": [{"name": str, "role": str, "text": str}, ...],
            "qa_exchanges": [{"analyst": str, "question": str, "response_by": str, "response": str}, ...]
        }
    """
    result = {
        "full_text": raw_text,
        "prepared_remarks": "",
        "qa_section": "",
        "speakers": [],
        "qa_exchanges": [],
    }

    # Try to split into prepared remarks and Q&A
    qa_markers = [
        r"question[s]?\s*(?:and|&)\s*answer",
        r"q\s*(?:and|&)\s*a\s*session",
        r"q&a",
        r"we (?:will|would) now (?:like to |)open.*(?:line|floor|call).*(?:for |to )question",
        r"(?:let's|let us) open.*(?:for |to )question",
        r"operator.*first question",
    ]

    qa_split_pos = None
    for marker in qa_markers:
        match = re.search(marker, raw_text, re.IGNORECASE)
        if match:
            qa_split_pos = match.start()
            break

    if qa_split_pos:
        result["prepared_remarks"] = raw_text[:qa_split_pos].strip()
        result["qa_section"] = raw_text[qa_split_pos:].strip()
    else:
        # If we can't find the split, treat everything as prepared remarks
        result["prepared_remarks"] = raw_text
        result["qa_section"] = ""

    # Extract speaker segments
    result["speakers"] = _extract_speakers(raw_text)

    # Extract Q&A exchanges if Q&A section exists
    if result["qa_section"]:
        result["qa_exchanges"] = _extract_qa_exchanges(result["qa_section"])

    return result


def _extract_speakers(text: str) -> list[dict]:
    """
    Attempt to identify speaker turns in the transcript.
    Common patterns:
        - "John Smith -- CEO" or "John Smith - CEO"
        - "John Smith, CEO:"
        - Speaker names in ALL CAPS followed by content
    """
    # Pattern: Name followed by title/role delimiter then content
    speaker_pattern = re.compile(
        r"(?:^|\n\n?)([A-Z][a-zA-Z\s\.\-']+?)\s*(?:--|—|-|,)\s*"
        r"((?:CEO|CFO|COO|CTO|President|Chairman|Vice President|VP|"
        r"Chief\s+\w+\s*Officer|Director|Analyst|Managing Director|"
        r"Head of|Senior|Executive|Operator|Moderator)[^\n]*?)"
        r"\s*\n(.*?)(?=\n[A-Z][a-zA-Z\s\.\-']+?\s*(?:--|—|-|,)\s*(?:CEO|CFO|COO|CTO|President|Chairman|"
        r"Vice President|VP|Chief|Director|Analyst|Managing|Head|Senior|Executive|Operator|Moderator)|\Z)",
        re.DOTALL | re.MULTILINE,
    )

    speakers = []
    for match in speaker_pattern.finditer(text):
        name = match.group(1).strip()
        role = match.group(2).strip()
        spoken_text = match.group(3).strip()
        if len(spoken_text) > 20:  # Filter out noise
            speakers.append({
                "name": name,
                "role": role,
                "text": spoken_text,
            })

    return speakers


def _extract_qa_exchanges(qa_text: str) -> list[dict]:
    """
    Extract analyst question / management response pairs from Q&A section.
    This is a best-effort extraction — transcript formats vary widely.
    """
    exchanges = []

    # Split by speaker turns and try to pair analyst questions with management responses
    # This is simplified — real transcripts have inconsistent formatting
    lines = qa_text.split("\n")
    current_speaker = ""
    current_role = ""
    current_text = []
    segments = []

    for line in lines:
        # Check if this line starts a new speaker
        speaker_match = re.match(
            r"^([A-Z][a-zA-Z\s\.\-']+?)\s*(?:--|—|-|,)\s*(.+?)$", line.strip()
        )
        if speaker_match and len(line.strip()) < 150:
            if current_speaker and current_text:
                segments.append({
                    "name": current_speaker,
                    "role": current_role,
                    "text": " ".join(current_text).strip(),
                })
            current_speaker = speaker_match.group(1).strip()
            current_role = speaker_match.group(2).strip()
            current_text = []
        else:
            current_text.append(line.strip())

    # Don't forget the last segment
    if current_speaker and current_text:
        segments.append({
            "name": current_speaker,
            "role": current_role,
            "text": " ".join(current_text).strip(),
        })

    # Pair analyst questions with management responses
    i = 0
    while i < len(segments):
        seg = segments[i]
        is_analyst = any(
            kw in seg["role"].lower()
            for kw in ["analyst", "research", "capital", "securities", "bank", "partners"]
        )
        if is_analyst and i + 1 < len(segments):
            exchanges.append({
                "analyst": f"{seg['name']} ({seg['role']})",
                "question": seg["text"],
                "response_by": f"{segments[i + 1]['name']} ({segments[i + 1]['role']})",
                "response": segments[i + 1]["text"],
            })
            i += 2
        else:
            i += 1

    return exchanges


def get_text_chunks(text: str, max_chars: int = 4000) -> list[str]:
    """Split text into chunks for FinBERT processing (max ~512 tokens)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sentence)
        current_len += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
