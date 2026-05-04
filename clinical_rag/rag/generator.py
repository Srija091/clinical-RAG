"""
rag/generator.py
================
Generates answers from retrieved clinical note chunks using Groq (free API).
Includes source citation, confidence scoring, and refusal for out-of-context questions.

Sign up free at https://groq.com (no credit card needed).
"""

import os
from typing import List
from groq import Groq


SYSTEM_PROMPT = """You are a clinical AI assistant that answers physician questions
based strictly on the clinical notes provided as context.

RULES:
1. Answer ONLY using information found in the provided context.
2. If the answer is not in the context, say "This information is not available in the provided notes."
3. Never invent, assume, or hallucinate clinical information — patient safety depends on accuracy.
4. Be concise and clinical in tone.
5. When referencing specific values (vitals, labs, medications), quote them exactly as they appear.
6. Note any PII placeholders (e.g. [PATIENT_NAME]) as anonymized for privacy.
7. At the end of your answer, rate your confidence: HIGH, MEDIUM, or LOW based on how directly the context answers the question."""


def generate_answer(
    question: str,
    chunks: List[dict],
    model: str = "llama-3.3-70b-versatile"
) -> dict:
    """
    Generate a clinical answer from retrieved chunks using Groq.

    Args:
        question : physician's question
        chunks   : retrieved context chunks from ChromaDB
        model    : Groq model name (all free on Groq)

    Returns:
        { answer, confidence, sources_used, error }
    """
    result = {
        "answer": "",
        "confidence": "medium",
        "sources_used": [],
        "error": None
    }

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        result["error"] = "GROQ_API_KEY not set. Get a free key at groq.com"
        result["answer"] = "Error: No API key configured."
        return result

    if not chunks:
        result["answer"] = "No relevant context found in indexed notes."
        result["confidence"] = "low"
        return result

    # Build context block with source labels
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1} | Doc: {chunk['doc_id']} | "
            f"Relevance: {chunk['score']:.0%}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""Clinical Notes Context:
{context}

---

Physician Question: {question}

Please answer based strictly on the context above. End with your confidence level (HIGH/MEDIUM/LOW)."""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.1  # Low temperature for clinical accuracy
        )

        raw_answer = response.choices[0].message.content.strip()

        # Extract confidence from end of answer
        confidence = _extract_confidence(raw_answer)
        clean_answer = _clean_answer(raw_answer)

        result["answer"] = clean_answer
        result["confidence"] = confidence
        result["sources_used"] = [c["doc_id"] for c in chunks]

    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "auth" in err.lower():
            result["error"] = "Invalid Groq API key. Check your key at groq.com"
        elif "model" in err.lower():
            result["error"] = f"Model '{model}' not available on Groq free tier."
        elif "rate" in err.lower():
            result["error"] = "Rate limit hit. Wait a moment and try again."
        else:
            result["error"] = f"Groq API error: {err}"
        result["answer"] = f"Generation failed: {result['error']}"

    return result


def _extract_confidence(text: str) -> str:
    """Extract confidence level from end of LLM response."""
    text_upper = text.upper()
    if "CONFIDENCE: HIGH" in text_upper or "CONFIDENCE IS HIGH" in text_upper:
        return "high"
    elif "CONFIDENCE: LOW" in text_upper or "CONFIDENCE IS LOW" in text_upper:
        return "low"
    elif "CONFIDENCE: MEDIUM" in text_upper or "CONFIDENCE IS MEDIUM" in text_upper:
        return "medium"
    # Heuristic: if answer is detailed and specific, assume medium
    return "medium"


def _clean_answer(text: str) -> str:
    """Remove the confidence line from the end of the answer for cleaner display."""
    lines = text.strip().split("\n")
    # Remove trailing confidence line
    cleaned = []
    for line in lines:
        if line.strip().upper().startswith("CONFIDENCE:"):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip() or text
