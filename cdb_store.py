"""contextdb knowledge store — backed by the real contextdb server via Python SDK.

Connects to contextdb REST API (default: localhost:7701) backed by Postgres + pgvector.
Supports both sync and async clients for compatibility with Hermes's tool dispatch
(Ollama-backed providers run tool handlers inside async contexts).
Falls back to offline mode (no-op) if the server is unreachable.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CONTEXTDB_URL = os.environ.get("CONTEXTDB_URL", "http://localhost:7701")
_db = None
_async_db = None
_online = False


def _get_db():
    global _db, _online
    if _db is not None:
        return _db

    try:
        from contextdb import ContextDB
        _db = ContextDB(_CONTEXTDB_URL, timeout=5.0)
        _db.ping()
        _online = True
        logger.info("contextdb connected at %s", _CONTEXTDB_URL)
    except Exception as e:
        logger.warning("contextdb unavailable (%s) — running in offline mode", e)
        _db = None
        _online = False
    return _db


def _get_async_db():
    """Get or create an async client for use in async tool dispatch."""
    global _async_db
    if _async_db is not None:
        return _async_db

    try:
        from contextdb import AsyncContextDB
        _async_db = AsyncContextDB(_CONTEXTDB_URL, timeout=5.0)
        return _async_db
    except ImportError:
        logger.debug("AsyncContextDB not available, async calls will use sync fallback")
        return None
    except Exception as e:
        logger.debug("Failed to create async client: %s", e)
        return None


def _ns(namespace: str = "hermes", mode: str = "agent_memory"):
    db = _get_db()
    if db is None:
        return None
    return db.namespace(namespace, mode=mode)


def write_knowledge(content: str, source_id: str, vector: List[float],
                    confidence: float = 0.5, mem_type: str = "semantic",
                    labels: Optional[List[str]] = None,
                    namespace: str = "hermes") -> Optional[str]:
    """Write a knowledge node. Returns the node ID or None if offline."""
    ns = _ns(namespace, mode=_mode_for_type(mem_type))
    if ns is None:
        return None

    try:
        result = ns.write(
            content=content,
            source_id=source_id,
            vector=vector,
            labels=labels or [],
            confidence=confidence,
        )
        return result.node_id if result.admitted else None
    except Exception as e:
        logger.warning("contextdb write failed: %s", e)
        return None


def retrieve(query_vec: List[float], namespace: str = "hermes",
             top_k: int = 10, text: str = "",
             weights: Optional[Dict[str, float]] = None
             ) -> List[Dict[str, Any]]:
    """Retrieve knowledge nodes with multi-dimensional scoring."""
    ns = _ns(namespace)
    if ns is None:
        return []

    try:
        from contextdb.types import ScoreParams
        w = weights or {}
        params = ScoreParams(
            similarity_weight=w.get("similarity", 0.3),
            confidence_weight=w.get("confidence", 0.3),
            recency_weight=w.get("recency", 0.2),
            utility_weight=w.get("credibility", 0.2),
        )

        results = ns.retrieve(
            vector=query_vec if query_vec else None,
            text=text or None,
            top_k=top_k,
            score_params=params,
        )

        return [
            {
                "id": r.id,
                "content": r.properties.get("text", r.properties.get("content", "")),
                "confidence": r.confidence_score,
                "source_id": r.properties.get("source_id", ""),
                "mem_type": _infer_mem_type(r.labels),
                "labels": r.labels,
                "score": r.score,
                "similarity": r.similarity_score,
                "recency": r.recency_score,
                "credibility": r.confidence_score,
            }
            for r in results
        ]
    except Exception as e:
        logger.warning("contextdb retrieve failed: %s", e)
        return []


def label_source(source_id: str, labels: List[str], namespace: str = "hermes"):
    """Set labels on a source for credibility tracking."""
    ns = _ns(namespace)
    if ns is None:
        return
    try:
        ns.label_source(source_id, labels)
    except Exception as e:
        logger.debug("label_source failed: %s", e)


def node_count(namespace: str = "hermes") -> int:
    """Return count of knowledge nodes (approximated from stats)."""
    db = _get_db()
    if db is None:
        return 0
    try:
        stats = db.stats()
        return stats.get("IngestAdmitted", 0)
    except Exception:
        return 0


def source_stats(namespace: str = "hermes") -> List[Dict[str, Any]]:
    """Return source credibility stats. Limited by REST API surface."""
    # The Python SDK doesn't expose source listing yet —
    # return empty until that's added
    return []


def is_online() -> bool:
    """Check if contextdb server is reachable."""
    return _online


# -- Async variants for Ollama/gateway contexts ----------------------------

async def async_write_knowledge(content: str, source_id: str, vector: List[float],
                                confidence: float = 0.5, mem_type: str = "semantic",
                                labels: Optional[List[str]] = None,
                                namespace: str = "hermes") -> Optional[str]:
    """Async version of write_knowledge."""
    adb = _get_async_db()
    if adb is None:
        return write_knowledge(content, source_id, vector, confidence, mem_type, labels, namespace)

    try:
        ns = adb.namespace(namespace, mode=_mode_for_type(mem_type))
        result = await ns.write(
            content=content,
            source_id=source_id,
            vector=vector,
            labels=labels or [],
            confidence=confidence,
        )
        return result.node_id if result.admitted else None
    except Exception as e:
        logger.warning("contextdb async write failed: %s", e)
        return None


async def async_retrieve(query_vec: List[float], namespace: str = "hermes",
                         top_k: int = 10, text: str = "",
                         weights: Optional[Dict[str, float]] = None
                         ) -> List[Dict[str, Any]]:
    """Async version of retrieve."""
    adb = _get_async_db()
    if adb is None:
        return retrieve(query_vec, namespace, top_k, text, weights)

    try:
        from contextdb.types import ScoreParams
        w = weights or {}
        params = ScoreParams(
            similarity_weight=w.get("similarity", 0.3),
            confidence_weight=w.get("confidence", 0.3),
            recency_weight=w.get("recency", 0.2),
            utility_weight=w.get("credibility", 0.2),
        )

        ns = adb.namespace(namespace)
        results = await ns.retrieve(
            vector=query_vec if query_vec else None,
            text=text or None,
            top_k=top_k,
            score_params=params,
        )

        return [
            {
                "id": r.id,
                "content": r.properties.get("text", r.properties.get("content", "")),
                "confidence": r.confidence_score,
                "source_id": r.properties.get("source_id", ""),
                "mem_type": _infer_mem_type(r.labels),
                "labels": r.labels,
                "score": r.score,
                "similarity": r.similarity_score,
                "recency": r.recency_score,
                "credibility": r.confidence_score,
            }
            for r in results
        ]
    except Exception as e:
        logger.warning("contextdb async retrieve failed: %s", e)
        return []


def _infer_mem_type(labels: List[str]) -> str:
    """Infer memory type from labels."""
    for l in labels:
        if l in ("episodic", "semantic", "procedural", "working"):
            return l
    return "semantic"


def _mode_for_type(mem_type: str) -> str:
    """Map memory type to contextdb namespace mode."""
    return {
        "episodic": "agent_memory",
        "working": "agent_memory",
        "procedural": "procedural",
        "semantic": "general",
    }.get(mem_type, "general")
