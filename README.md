# Clinical Note Summarizer with Privacy-Safe RAG

**100% free stack** — spaCy NER + ChromaDB + sentence-transformers + Groq free API

Upload clinical notes → PII automatically stripped → ask physician questions → get cited answers → every query HIPAA audit logged.

---

## Quick Start (5 minutes)

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Get free Groq API key
Sign up at **https://groq.com** — no credit card, takes 60 seconds.

### 3. Run the web app
```bash
streamlit run app.py
```

### 4. Or run the FastAPI backend
```bash
uvicorn api.main:app --reload
# Docs at http://localhost:8000/docs
```

---

## How It Works

```
Clinical Note (with PII)
        ↓
[spaCy NER + Regex] ← strips names, DOB, SSN, MRN, phone, address
        ↓
Anonymized Text
        ↓
[Chunker] ← sentence-aware, overlapping chunks
        ↓
[sentence-transformers] ← local embeddings, no API
        ↓
[ChromaDB] ← persistent vector store, free
        ↓
Physician asks question
        ↓
[Semantic Retrieval] ← top-k cosine similarity
        ↓
[Groq LLM] ← Llama-3.3-70B, free tier
        ↓
Answer + Source Citations
        ↓
[SQLite Audit Log] ← HIPAA compliance, every query logged
```

---

## PII Detected & Removed

| Type | Examples | Replacement |
|---|---|---|
| Patient names | John Smith, Dr. Sarah Johnson | `[PATIENT_NAME]` |
| Dates of birth | March 15, 1978, 01/15/1985 | `[DOB]` / `[DATE]` |
| SSN | 123-45-6789 | `[SSN]` |
| MRN | MRN 78234-A | `[MRN]` |
| Phone | (513) 555-0192 | `[PHONE]` |
| Address | 456 Oak Street, Cincinnati OH | `[ADDRESS]` |
| ZIP codes | 45202 | `[ZIP]` |
| Email | patient@email.com | `[EMAIL]` |

---

## Free Stack

| Component | Tool | Cost |
|---|---|---|
| PII anonymization | spaCy `en_core_web_sm` + regex | Free, offline |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Free, offline |
| Vector store | ChromaDB (persistent) | Free forever |
| LLM generation | Groq free tier (Llama-3.3-70B) | Free |
| Audit log | SQLite | Free |
| UI | Streamlit | Free |
| API | FastAPI + Uvicorn | Free |

---

## Project Structure

```
clinical_rag/
├── app.py                      # Streamlit web UI
├── api/
│   └── main.py                 # FastAPI REST backend
├── ingestion/
│   ├── pii_anonymizer.py       # spaCy NER + regex PII scrubbing
│   └── chunker.py              # Sentence-aware text chunking
├── embeddings/
│   └── chroma_store.py         # ChromaDB + sentence-transformers
├── rag/
│   └── generator.py            # Groq LLM answer generation
├── audit/
│   └── sqlite_logger.py        # HIPAA audit log (SQLite)
├── requirements.txt
└── README.md
```

---

## Resume Bullets

```
• Built a HIPAA-aware RAG pipeline with spaCy NER PII anonymization (names,
  SSN, MRN, DOB, phone, address), ChromaDB vector store, and local
  sentence-transformer embeddings for physician Q&A over clinical notes —
  zero cloud cost.

• Implemented SQLite-based HIPAA audit logging capturing every query,
  response, source passages, model, and confidence level with JSON export
  for compliance reporting.

• Deployed as a dual-interface system (Streamlit UI + FastAPI REST API)
  with source-passage explainability panel showing which note sections
  grounded each answer, reducing hallucination risk in clinical context.
```
