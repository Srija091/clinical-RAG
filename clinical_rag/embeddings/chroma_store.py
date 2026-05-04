"""
embeddings/chroma_store.py
===========================
ChromaDB vector store + sentence-transformers embeddings.
Both are 100% free and run fully offline — no API keys.

Install:
  pip install chromadb sentence-transformers
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List
import os

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "clinical_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free, 80MB, runs offline

# ── Singletons ─────────────────────────────────────────────────────────────────
_embedder = None
_client = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("Loading sentence-transformer model (first run downloads ~80MB)...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_collection():
    global _client, _collection
    if _collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # cosine similarity
        )
    return _collection


# ── Public API ──────────────────────────────────────────────────────────────────

def index_chunks(chunks: List[dict]) -> int:
    """
    Embed and index a list of chunks into ChromaDB.

    Args:
        chunks: list of { id, text, doc_id, chunk_id, word_count }

    Returns:
        Number of chunks indexed
    """
    if not chunks:
        return 0

    collection = _get_collection()
    embedder = _get_embedder()

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "doc_id": c["doc_id"],
            "chunk_id": str(c["chunk_id"]),
            "word_count": str(c["word_count"])
        }
        for c in chunks
    ]

    # Generate embeddings locally
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    # Upsert — safe to re-index same doc
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def retrieve(query: str, top_k: int = 4) -> List[dict]:
    """
    Semantic search over indexed clinical notes.

    Args:
        query  : clinical question
        top_k  : number of chunks to retrieve

    Returns:
        List of { text, doc_id, chunk_id, score }
    """
    collection = _get_collection()
    embedder = _get_embedder()

    if collection.count() == 0:
        return []

    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]

        # Convert cosine distance to similarity score (0-1)
        score = round(1 - dist, 4)

        chunks.append({
            "text": text,
            "doc_id": meta.get("doc_id", "unknown"),
            "chunk_id": meta.get("chunk_id", "0"),
            "score": score
        })

    # Sort by score descending
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def get_collection_stats() -> dict:
    """Return basic stats about the collection."""
    try:
        collection = _get_collection()
        return {"count": collection.count(), "name": COLLECTION_NAME}
    except Exception:
        return {"count": 0, "name": COLLECTION_NAME}


def clear_collection() -> None:
    """Delete all documents from the collection."""
    global _collection
    try:
        collection = _get_collection()
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        _collection = None
    except Exception as e:
        print(f"Error clearing collection: {e}")
