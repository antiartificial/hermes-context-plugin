"""sqlite-vec store — semantic vector search over hermes session messages.

Uses sqlite-vec extension for ANN vector search alongside the existing
FTS5 full-text search in hermes_state.db.
"""

import json
import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DB_PATH = Path.home() / ".hermes" / "contextdb-ab" / "vectors.db"
_conn: Optional[sqlite3.Connection] = None
_VEC_AVAILABLE = False


def _init_db() -> sqlite3.Connection:
    global _conn, _VEC_AVAILABLE
    if _conn is not None:
        return _conn

    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(str(_DB_PATH))
    _conn.execute("PRAGMA journal_mode=WAL")

    # Try loading sqlite-vec extension
    try:
        import sqlite_vec
        _conn.enable_load_extension(True)
        sqlite_vec.load(_conn)
        _VEC_AVAILABLE = True
        logger.info("sqlite-vec extension loaded")
    except ImportError:
        logger.warning("sqlite-vec not installed — using cosine fallback")
    except Exception as e:
        logger.warning("sqlite-vec failed to load: %s — using cosine fallback", e)

    # Create tables
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS message_vectors (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            message_id  TEXT,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            vector      BLOB   NOT NULL,
            timestamp   REAL   NOT NULL,
            source      TEXT   DEFAULT 'cli'
        )
    """)
    _conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_mv_session ON message_vectors(session_id)
    """)

    if _VEC_AVAILABLE:
        try:
            from .embeddings import EMBED_DIM
            _conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages
                USING vec0(embedding float[{EMBED_DIM}])
            """)
        except Exception as e:
            logger.warning("Failed to create vec0 table: %s", e)
            _VEC_AVAILABLE = False

    _conn.commit()
    return _conn


def _serialize_vec(vec: List[float]) -> bytes:
    """Pack float list into little-endian binary for sqlite-vec."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _deserialize_vec(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def index_message(session_id: str, content: str, role: str,
                  vector: List[float], message_id: str = "",
                  source: str = "cli"):
    """Add a message embedding to the vector store."""
    db = _init_db()
    blob = _serialize_vec(vector)

    cursor = db.execute(
        """INSERT INTO message_vectors
           (session_id, message_id, role, content, vector, timestamp, source)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (session_id, message_id, role, content, blob, time.time(), source),
    )
    row_id = cursor.lastrowid

    if _VEC_AVAILABLE and row_id:
        try:
            db.execute(
                "INSERT INTO vec_messages (rowid, embedding) VALUES (?, ?)",
                (row_id, blob),
            )
        except Exception as e:
            logger.debug("vec_messages insert failed: %s", e)

    db.commit()


def search(query_vec: List[float], limit: int = 10,
           min_similarity: float = 0.3) -> List[Dict[str, Any]]:
    """Search for similar messages by vector.

    Returns list of {content, session_id, role, similarity, timestamp}.
    Uses sqlite-vec ANN if available, falls back to brute-force cosine.
    """
    db = _init_db()
    start = time.time()

    if _VEC_AVAILABLE:
        try:
            results = _search_vec0(db, query_vec, limit, min_similarity)
            latency = (time.time() - start) * 1000
            for r in results:
                r["latency_ms"] = latency
            return results
        except Exception as e:
            logger.debug("vec0 search failed, falling back: %s", e)

    # Brute-force fallback
    results = _search_brute(db, query_vec, limit, min_similarity)
    latency = (time.time() - start) * 1000
    for r in results:
        r["latency_ms"] = latency
    return results


def _search_vec0(db: sqlite3.Connection, query_vec: List[float],
                 limit: int, min_sim: float) -> List[Dict[str, Any]]:
    """ANN search via sqlite-vec."""
    blob = _serialize_vec(query_vec)
    rows = db.execute(
        """SELECT v.rowid, v.distance
           FROM vec_messages v
           WHERE v.embedding MATCH ?
           AND k = ?""",
        (blob, limit * 2),  # over-fetch then filter by similarity
    ).fetchall()

    results = []
    for row in rows:
        sim = 1.0 - row[1]  # vec0 returns distance, convert to similarity
        if sim < min_sim:
            continue
        meta = db.execute(
            "SELECT content, session_id, role, timestamp FROM message_vectors WHERE id=?",
            (row[0],),
        ).fetchone()
        if meta:
            results.append({
                "content": meta[0],
                "session_id": meta[1],
                "role": meta[2],
                "timestamp": meta[3],
                "similarity": round(sim, 4),
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]


def _search_brute(db: sqlite3.Connection, query_vec: List[float],
                  limit: int, min_sim: float) -> List[Dict[str, Any]]:
    """Brute-force cosine search over all stored vectors."""
    rows = db.execute(
        "SELECT id, content, session_id, role, vector, timestamp FROM message_vectors"
    ).fetchall()

    scored = []
    for row in rows:
        stored_vec = _deserialize_vec(row[4])
        sim = _cosine_sim(query_vec, stored_vec)
        if sim >= min_sim:
            scored.append({
                "content": row[1],
                "session_id": row[2],
                "role": row[3],
                "timestamp": row[5],
                "similarity": round(sim, 4),
            })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:limit]


def message_count() -> int:
    """Return total indexed messages."""
    db = _init_db()
    return db.execute("SELECT COUNT(*) FROM message_vectors").fetchone()[0]
