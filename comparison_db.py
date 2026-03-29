"""A/B comparison logging database.

Stores every retrieval event from all three systems (FTS5, sqlite-vec, contextdb)
with timing, scoring, and token estimates for later analysis.
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH = Path.home() / ".hermes" / "contextdb-ab" / "comparison.db"
_conn: Optional[sqlite3.Connection] = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH))
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
    return _conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS retrievals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    session_id      TEXT,
    query           TEXT    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'session_search',

    -- FTS5 (baseline)
    fts5_result_count   INTEGER DEFAULT 0,
    fts5_latency_ms     REAL    DEFAULT 0,
    fts5_tokens_est     INTEGER DEFAULT 0,

    -- sqlite-vec
    vec_result_count    INTEGER DEFAULT 0,
    vec_latency_ms      REAL    DEFAULT 0,
    vec_tokens_est      INTEGER DEFAULT 0,
    vec_top_similarity  REAL    DEFAULT 0,

    -- contextdb
    cdb_result_count    INTEGER DEFAULT 0,
    cdb_latency_ms      REAL    DEFAULT 0,
    cdb_tokens_est      INTEGER DEFAULT 0,
    cdb_avg_confidence  REAL    DEFAULT 0,
    cdb_avg_score       REAL    DEFAULT 0,
    cdb_top_source      TEXT,
    cdb_top_credibility REAL    DEFAULT 0,

    -- Which system's result was used
    chosen              TEXT    DEFAULT 'contextdb',

    -- LLM summarization savings
    fts5_llm_call       INTEGER DEFAULT 0,
    cdb_llm_call        INTEGER DEFAULT 0,

    -- Raw results (JSON) for later inspection
    fts5_results_json   TEXT,
    vec_results_json    TEXT,
    cdb_results_json    TEXT
);

CREATE TABLE IF NOT EXISTS knowledge_writes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    session_id      TEXT,
    content         TEXT    NOT NULL,
    source_model    TEXT,
    confidence      REAL    DEFAULT 0.5,
    mem_type        TEXT    DEFAULT 'semantic',
    labels          TEXT,
    namespace       TEXT    DEFAULT 'hermes'
);

CREATE TABLE IF NOT EXISTS sessions_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    started_at      REAL    NOT NULL,
    ended_at        REAL,
    model           TEXT,
    platform        TEXT,
    retrieval_count INTEGER DEFAULT 0,
    knowledge_count INTEGER DEFAULT 0
);
"""


@dataclass
class RetrievalLog:
    query: str
    session_id: str = ""
    source: str = "session_search"

    fts5_result_count: int = 0
    fts5_latency_ms: float = 0
    fts5_tokens_est: int = 0

    vec_result_count: int = 0
    vec_latency_ms: float = 0
    vec_tokens_est: int = 0
    vec_top_similarity: float = 0

    cdb_result_count: int = 0
    cdb_latency_ms: float = 0
    cdb_tokens_est: int = 0
    cdb_avg_confidence: float = 0
    cdb_avg_score: float = 0
    cdb_top_source: str = ""
    cdb_top_credibility: float = 0

    chosen: str = "contextdb"
    fts5_llm_call: int = 0
    cdb_llm_call: int = 0

    fts5_results_json: str = "[]"
    vec_results_json: str = "[]"
    cdb_results_json: str = "[]"


def log_retrieval(entry: RetrievalLog):
    """Write a retrieval comparison entry."""
    db = _db()
    db.execute(
        """INSERT INTO retrievals (
            timestamp, session_id, query, source,
            fts5_result_count, fts5_latency_ms, fts5_tokens_est,
            vec_result_count, vec_latency_ms, vec_tokens_est, vec_top_similarity,
            cdb_result_count, cdb_latency_ms, cdb_tokens_est,
            cdb_avg_confidence, cdb_avg_score, cdb_top_source, cdb_top_credibility,
            chosen, fts5_llm_call, cdb_llm_call,
            fts5_results_json, vec_results_json, cdb_results_json
        ) VALUES (
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?
        )""",
        (
            time.time(), entry.session_id, entry.query, entry.source,
            entry.fts5_result_count, entry.fts5_latency_ms, entry.fts5_tokens_est,
            entry.vec_result_count, entry.vec_latency_ms, entry.vec_tokens_est, entry.vec_top_similarity,
            entry.cdb_result_count, entry.cdb_latency_ms, entry.cdb_tokens_est,
            entry.cdb_avg_confidence, entry.cdb_avg_score, entry.cdb_top_source, entry.cdb_top_credibility,
            entry.chosen, entry.fts5_llm_call, entry.cdb_llm_call,
            entry.fts5_results_json, entry.vec_results_json, entry.cdb_results_json,
        ),
    )
    db.commit()


def log_knowledge_write(session_id: str, content: str, source_model: str,
                        confidence: float, mem_type: str, labels: List[str],
                        namespace: str = "hermes"):
    db = _db()
    db.execute(
        """INSERT INTO knowledge_writes
           (timestamp, session_id, content, source_model, confidence, mem_type, labels, namespace)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (time.time(), session_id, content, source_model, confidence, mem_type,
         json.dumps(labels), namespace),
    )
    db.commit()


def log_session(session_id: str, model: str = "", platform: str = ""):
    db = _db()
    db.execute(
        "INSERT INTO sessions_log (session_id, started_at, model, platform) VALUES (?, ?, ?, ?)",
        (session_id, time.time(), model, platform),
    )
    db.commit()


def end_session(session_id: str):
    db = _db()
    db.execute(
        "UPDATE sessions_log SET ended_at=? WHERE session_id=? AND ended_at IS NULL",
        (time.time(), session_id),
    )
    db.commit()


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def get_stats() -> Dict[str, Any]:
    """Get aggregate comparison statistics for the review report."""
    db = _db()

    total = db.execute("SELECT COUNT(*) FROM retrievals").fetchone()[0]
    if total == 0:
        return {"total_retrievals": 0}

    row = db.execute("""
        SELECT
            COUNT(*)                                    AS total,
            AVG(fts5_latency_ms)                        AS avg_fts5_latency,
            AVG(vec_latency_ms)                         AS avg_vec_latency,
            AVG(cdb_latency_ms)                         AS avg_cdb_latency,
            AVG(fts5_result_count)                      AS avg_fts5_results,
            AVG(vec_result_count)                        AS avg_vec_results,
            AVG(cdb_result_count)                        AS avg_cdb_results,
            AVG(fts5_tokens_est)                         AS avg_fts5_tokens,
            AVG(vec_tokens_est)                          AS avg_vec_tokens,
            AVG(cdb_tokens_est)                          AS avg_cdb_tokens,
            SUM(fts5_tokens_est)                         AS total_fts5_tokens,
            SUM(vec_tokens_est)                          AS total_vec_tokens,
            SUM(cdb_tokens_est)                          AS total_cdb_tokens,
            AVG(cdb_avg_confidence)                      AS avg_cdb_confidence,
            AVG(cdb_avg_score)                           AS avg_cdb_score,
            AVG(vec_top_similarity)                      AS avg_vec_similarity,
            SUM(fts5_llm_call)                           AS total_fts5_llm_calls,
            SUM(cdb_llm_call)                            AS total_cdb_llm_calls,
            SUM(CASE WHEN cdb_result_count > 0 AND fts5_result_count = 0 THEN 1 ELSE 0 END) AS cdb_only_hits,
            SUM(CASE WHEN vec_result_count > 0 AND fts5_result_count = 0 THEN 1 ELSE 0 END) AS vec_only_hits,
            SUM(CASE WHEN fts5_result_count > 0 AND cdb_result_count = 0 THEN 1 ELSE 0 END) AS fts5_only_hits
        FROM retrievals
    """).fetchone()

    kw = db.execute("SELECT COUNT(*) FROM knowledge_writes").fetchone()[0]
    sessions = db.execute("SELECT COUNT(*) FROM sessions_log").fetchone()[0]

    return {
        "total_retrievals": row["total"],
        "total_sessions": sessions,
        "total_knowledge_writes": kw,

        "avg_fts5_latency_ms": round(row["avg_fts5_latency"] or 0, 1),
        "avg_vec_latency_ms": round(row["avg_vec_latency"] or 0, 1),
        "avg_cdb_latency_ms": round(row["avg_cdb_latency"] or 0, 1),

        "avg_fts5_results": round(row["avg_fts5_results"] or 0, 1),
        "avg_vec_results": round(row["avg_vec_results"] or 0, 1),
        "avg_cdb_results": round(row["avg_cdb_results"] or 0, 1),

        "avg_fts5_tokens": int(row["avg_fts5_tokens"] or 0),
        "avg_vec_tokens": int(row["avg_vec_tokens"] or 0),
        "avg_cdb_tokens": int(row["avg_cdb_tokens"] or 0),

        "total_fts5_tokens": int(row["total_fts5_tokens"] or 0),
        "total_vec_tokens": int(row["total_vec_tokens"] or 0),
        "total_cdb_tokens": int(row["total_cdb_tokens"] or 0),

        "token_savings_vs_fts5": int((row["total_fts5_tokens"] or 0) - (row["total_cdb_tokens"] or 0)),
        "llm_calls_saved": int((row["total_fts5_llm_calls"] or 0) - (row["total_cdb_llm_calls"] or 0)),

        "avg_cdb_confidence": round(row["avg_cdb_confidence"] or 0, 3),
        "avg_cdb_score": round(row["avg_cdb_score"] or 0, 3),
        "avg_vec_similarity": round(row["avg_vec_similarity"] or 0, 3),

        "semantic_only_hits_vec": int(row["vec_only_hits"] or 0),
        "semantic_only_hits_cdb": int(row["cdb_only_hits"] or 0),
        "fts5_only_hits": int(row["fts5_only_hits"] or 0),

        "cdb_hit_rate": round((row["total"] - (row["fts5_only_hits"] or 0)) / row["total"] * 100, 1)
            if row["total"] > 0 else 0,
        "vec_hit_rate": round((row["total"] - (row["fts5_only_hits"] or 0)) / row["total"] * 100, 1)
            if row["total"] > 0 else 0,
    }
