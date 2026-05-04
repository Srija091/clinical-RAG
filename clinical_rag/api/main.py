"""
api/main.py
===========
FastAPI REST API for the Clinical RAG system.
Run: uvicorn api.main:app --reload

Endpoints:
  POST /ingest      — anonymize + index a clinical note
  POST /query       — ask a question, get RAG answer
  GET  /audit       — retrieve audit log
  GET  /stats       — collection + audit stats
  DELETE /clear     — clear all indexed notes
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from ingestion.pii_anonymizer import anonymize, get_pii_report
from ingestion.chunker import chunk_text
from embeddings.chroma_store import index_chunks, retrieve, get_collection_stats, clear_collection
from rag.generator import generate_answer
from audit.sqlite_logger import log_query, get_audit_log, get_audit_stats

app = FastAPI(
    title="Clinical RAG API",
    description="Privacy-safe RAG over clinical notes — HIPAA-aware audit logging",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Request / Response models ───────────────────────────────────────────────────

class IngestRequest(BaseModel):
    doc_id: str
    text: str
    chunk_size: int = 300
    overlap: int = 50


class IngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    pii_removed: int
    message: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4
    model: str = "llama-3.3-70b-versatile"


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    sources: List[dict]
    audit_id: int


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Clinical RAG API",
        "status": "running",
        "docs": "/docs"
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_note(req: IngestRequest):
    """
    Anonymize and index a clinical note.
    PII is stripped before any text enters the vector store.
    """
    if not req.text or len(req.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text too short to index.")

    # Anonymize
    anon_text, pii_found = anonymize(req.text)

    # Chunk
    chunks = chunk_text(anon_text, doc_id=req.doc_id,
                        chunk_size=req.chunk_size, overlap=req.overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to index after chunking.")

    # Index
    count = index_chunks(chunks)

    return IngestResponse(
        doc_id=req.doc_id,
        chunks_indexed=count,
        pii_removed=len(pii_found),
        message=f"Successfully indexed {count} chunks. Removed {len(pii_found)} PII entities."
    )


@app.post("/query", response_model=QueryResponse)
def query_notes(req: QueryRequest, x_groq_api_key: Optional[str] = Header(None)):
    """
    Ask a clinical question. Returns RAG-generated answer with source citations.
    """
    if not req.question or len(req.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question too short.")

    # Allow API key via header or env
    if x_groq_api_key:
        os.environ["GROQ_API_KEY"] = x_groq_api_key

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=401,
            detail="Groq API key required. Set X-Groq-Api-Key header or GROQ_API_KEY env var."
        )

    # Retrieve
    chunks = retrieve(req.question, top_k=req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant notes found. Ingest some notes first.")

    # Generate
    result = generate_answer(
        question=req.question,
        chunks=chunks,
        model=req.model
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    # Audit log
    audit_id = log_query(
        question=req.question,
        answer=result["answer"],
        sources=[f"{c['doc_id']}:chunk{c['chunk_id']}" for c in chunks],
        model=req.model,
        confidence=result["confidence"]
    )

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        confidence=result["confidence"],
        sources=chunks,
        audit_id=audit_id
    )


@app.get("/audit")
def get_audit(limit: int = 50):
    """Retrieve HIPAA audit log entries."""
    return {"entries": get_audit_log(limit=limit)}


@app.get("/stats")
def get_stats():
    """Get collection and audit statistics."""
    return {
        "collection": get_collection_stats(),
        "audit": get_audit_stats()
    }


@app.delete("/clear")
def clear_notes():
    """Clear all indexed notes from ChromaDB."""
    clear_collection()
    return {"message": "All notes cleared from vector store."}
