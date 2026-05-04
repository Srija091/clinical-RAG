"""
ingestion/chunker.py
====================
Sentence-aware text chunking for clinical notes.
Splits on sentences but keeps chunks at a good size for embedding.
"""

import re
from typing import List


def chunk_text(
    text: str,
    doc_id: str = "doc",
    chunk_size: int = 300,
    overlap: int = 50
) -> List[dict]:
    """
    Split text into overlapping chunks, sentence-aware.

    Args:
        text       : anonymized clinical note text
        doc_id     : document identifier
        chunk_size : target words per chunk
        overlap    : words of overlap between chunks

    Returns:
        List of { text, doc_id, chunk_id, word_count }
    """
    if not text or not text.strip():
        return []

    # Split into sentences (simple but effective for clinical notes)
    sentences = _split_sentences(text)

    chunks = []
    current_words = []
    current_sentences = []
    chunk_id = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        current_words.extend(words)
        current_sentences.append(sentence)

        if len(current_words) >= chunk_size:
            chunk_text_str = " ".join(current_sentences).strip()
            if chunk_text_str:
                chunks.append({
                    "text": chunk_text_str,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "word_count": len(current_words),
                    "id": f"{doc_id}_chunk_{chunk_id}"
                })
                chunk_id += 1

            # Overlap: keep last N words worth of sentences
            overlap_words = 0
            overlap_sentences = []
            for s in reversed(current_sentences):
                s_words = s.split()
                if overlap_words + len(s_words) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_words += len(s_words)
                else:
                    break

            current_sentences = overlap_sentences
            current_words = " ".join(current_sentences).split()

    # Last chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences).strip()
        if chunk_text_str and len(chunk_text_str.split()) > 10:
            chunks.append({
                "text": chunk_text_str,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "word_count": len(current_words),
                "id": f"{doc_id}_chunk_{chunk_id}"
            })

    return chunks


def _split_sentences(text: str) -> List[str]:
    """
    Split clinical note text into sentences.
    Handles common clinical abbreviations (e.g. Dr., mg., vs.)
    """
    # Protect common abbreviations from being split
    abbreviations = [
        "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.",
        "mg.", "mcg.", "mL.", "mmHg.", "vs.",
        "approx.", "est.", "ref.", "temp.",
        "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."
    ]
    protected = text
    placeholders = {}
    for i, abbr in enumerate(abbreviations):
        ph = f"__ABBR{i}__"
        placeholders[ph] = abbr
        protected = protected.replace(abbr, ph)

    # Split on sentence-ending punctuation followed by whitespace + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', protected)

    # Restore abbreviations
    restored = []
    for s in sentences:
        for ph, original in placeholders.items():
            s = s.replace(ph, original)
        s = s.strip()
        if s:
            restored.append(s)

    # Also split on newlines with section headers
    final = []
    for s in restored:
        # Split on lines that look like clinical section headers
        sub = re.split(r'\n(?=[A-Z][a-zA-Z\s]+:)', s)
        final.extend([x.strip() for x in sub if x.strip()])

    return final
