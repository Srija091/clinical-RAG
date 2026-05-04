"""
audit/sqlite_logger.py
=======================
HIPAA-compliant audit log using SQLite.
Logs every query, response, sources used, model, and timestamp.
Zero cloud cost — everything stored locally.
"""

import sqlite3
import json
import datetime
import os
from typing import List

DB_PATH = "./audit.db"


def _get_conn() -> sqlite3.Connection:
    """Get SQLite connection and ensure table exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            question    TEXT    NOT NULL,
            answer      TEXT    NOT NULL,
            sources     TEXT    NOT NULL,
            model       TEXT    NOT NULL,
            confidence  TEXT    NOT NULL,
            session_id  TEXT
        )
    """)
    conn.commit()
    return conn


def log_query(
    question: str,
    answer: str,
    sources: list,
    model: str,
    confidence: str = "medium",
    session_id: str = None
) -> int:
    """
    Log a query-response pair to the HIPAA audit trail.

    Returns: inserted row id
    """
    conn = _get_conn()
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    cursor = conn.execute(
        """INSERT INTO audit_log
           (timestamp, question, answer, sources, model, confidence, session_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            timestamp,
            question,
            answer,
            json.dumps(sources),
            model,
            confidence,
            session_id or "default"
        )
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_audit_log(limit: int = 100) -> List[dict]:
    """
    Retrieve audit log entries, newest first.
    """
    if not os.path.exists(DB_PATH):
        return []

    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, timestamp, question, answer, sources, model, confidence
           FROM audit_log
           ORDER BY id DESC
           LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()

    entries = []
    for row in rows:
        try:
            sources = json.loads(row[4])
        except Exception:
            sources = [row[4]]

        entries.append({
            "id": row[0],
            "timestamp": row[1],
            "question": row[2],
            "answer": row[3],
            "sources": sources,
            "model": row[5],
            "confidence": row[6]
        })

    return entries


def get_audit_stats() -> dict:
    """Return summary statistics for the audit log."""
    if not os.path.exists(DB_PATH):
        return {"total_queries": 0}

    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    models = conn.execute(
        "SELECT model, COUNT(*) FROM audit_log GROUP BY model"
    ).fetchall()
    conn.close()

    return {
        "total_queries": total,
        "by_model": {m[0]: m[1] for m in models}
    }


def export_audit_log(output_path: str = "audit_export.json") -> str:
    """Export full audit log to JSON file."""
    entries = get_audit_log(limit=10000)
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)
    return output_path
