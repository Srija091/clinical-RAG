"""
ingestion/pii_anonymizer.py
===========================
Detects and anonymizes PII in clinical notes using:
  - spaCy NER (PERSON, DATE, GPE, ORG, CARDINAL)
  - Regex for SSN, MRN, phone, email, DOB, zip codes

Runs 100% offline — no API calls.
Install: pip install spacy && python -m spacy download en_core_web_sm
"""

import re
import spacy
from typing import Tuple

# Load spaCy model (small = fast, good enough for NER)
# If you want higher accuracy: python -m spacy download en_core_web_trf
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


# ── Regex patterns for clinical PII ──────────────────────────────────────────

PII_PATTERNS = [
    # SSN: 123-45-6789 or 123456789
    (r'\b\d{3}-\d{2}-\d{4}\b',              "[SSN]"),
    (r'\b\d{9}\b',                           "[SSN]"),

    # MRN: MRN 12345 or MRN: 12345-A
    (r'\bMRN\s*:?\s*[\w\-]+\b',             "[MRN]"),
    (r'\bMedical Record\s*(?:Number|#|No)?\s*:?\s*[\w\-]+\b', "[MRN]"),

    # Phone numbers
    (r'\b(?:\+1[\s\-]?)?\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}\b', "[PHONE]"),

    # Email
    (r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', "[EMAIL]"),

    # Date of birth patterns
    (r'\b(?:DOB|Date of Birth|D\.O\.B\.?)\s*:?\s*[\w/\-,\s]+\d{4}\b', "[DOB]"),

    # Dates: MM/DD/YYYY, MM-DD-YYYY, Month DD YYYY
    (r'\b(?:January|February|March|April|May|June|July|August|September|'
     r'October|November|December)\s+\d{1,2},?\s+\d{4}\b',     "[DATE]"),
    (r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',                  "[DATE]"),

    # ZIP codes (standalone 5-digit or ZIP+4)
    (r'\b\d{5}(?:-\d{4})?\b',               "[ZIP]"),

    # Street addresses: 123 Oak Street, 456 N. Main Ave
    (r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+'
     r'(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|'
     r'Drive|Dr|Lane|Ln|Court|Ct|Place|Pl|Way)\b', "[ADDRESS]"),

    # Age references: "46-year-old" (keep age, anonymize if needed)
    # We keep age as it's clinical — only remove exact DOB

    # NPI / DEA numbers
    (r'\b(?:NPI|DEA)\s*:?\s*[A-Z0-9]{9,10}\b', "[PROVIDER_ID]"),
]


# ── spaCy entity labels to anonymize ─────────────────────────────────────────

SPACY_PII_LABELS = {
    "PERSON":   "[PATIENT_NAME]",
    "GPE":      "[LOCATION]",     # cities, states, countries
    "ORG":      "[ORGANIZATION]", # hospital names etc — keep for clinical context
    "FAC":      "[FACILITY]",
    "LOC":      "[LOCATION]",
}

# Labels we intentionally keep (clinical context)
KEEP_LABELS = {"DATE", "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY",
               "TIME", "NORP", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
               "LANGUAGE"}


def anonymize(text: str) -> Tuple[str, list]:
    """
    Anonymize PII in clinical text.

    Returns:
        anon_text  : anonymized string
        pii_found  : list of (original_text, label) tuples
    """
    pii_found = []
    result = text

    # ── Step 1: spaCy NER ────────────────────────────────────────────────────
    nlp = _get_nlp()
    doc = nlp(text)

    # Process entities in reverse order to preserve character offsets
    replacements = []
    for ent in doc.ents:
        if ent.label_ in SPACY_PII_LABELS:
            tag = SPACY_PII_LABELS[ent.label_]
            replacements.append((ent.start_char, ent.end_char, ent.text, tag))
            pii_found.append((ent.text, ent.label_))

    # Apply replacements in reverse order
    for start, end, original, tag in sorted(replacements, key=lambda x: x[0], reverse=True):
        result = result[:start] + tag + result[end:]

    # ── Step 2: Regex patterns ────────────────────────────────────────────────
    for pattern, replacement in PII_PATTERNS:
        matches = re.findall(pattern, result, flags=re.IGNORECASE)
        for m in matches:
            if isinstance(m, str) and len(m) > 2:
                pii_found.append((m, replacement.strip("[]")))
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result, pii_found


def get_pii_report(text: str) -> list:
    """
    Return list of (pii_text, label) found in original text.
    Used for the UI explainability panel.
    """
    _, pii_found = anonymize(text)
    # Deduplicate
    seen = set()
    unique = []
    for item in pii_found:
        key = (item[0][:30], item[1])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def anonymize_batch(texts: list) -> list:
    """Anonymize a list of texts. Returns list of anonymized strings."""
    return [anonymize(t)[0] for t in texts]
