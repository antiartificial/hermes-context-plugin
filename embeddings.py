"""Lightweight embedding helper — uses OpenAI API (already in hermes venv)."""

import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_CACHE_DB: Optional[sqlite3.Connection] = None
_EMBED_DIM = 256  # text-embedding-3-small supports dimensions param


def _cache_path() -> Path:
    return Path.home() / ".hermes" / "contextdb-ab" / "embed_cache.db"


def _get_cache() -> sqlite3.Connection:
    global _CACHE_DB
    if _CACHE_DB is None:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_DB = sqlite3.connect(str(path))
        _CACHE_DB.execute("PRAGMA journal_mode=WAL")
        _CACHE_DB.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(hash TEXT PRIMARY KEY, vec BLOB)"
        )
        _CACHE_DB.commit()
    return _CACHE_DB


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _cache_get(text: str) -> Optional[List[float]]:
    h = _text_hash(text)
    row = _get_cache().execute("SELECT vec FROM cache WHERE hash=?", (h,)).fetchone()
    if row:
        return json.loads(row[0])
    return None


def _cache_put(text: str, vec: List[float]):
    h = _text_hash(text)
    _get_cache().execute(
        "INSERT OR REPLACE INTO cache (hash, vec) VALUES (?, ?)",
        (h, json.dumps(vec)),
    )
    _get_cache().commit()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts, using cache where possible.

    Uses OpenAI text-embedding-3-small with reduced dimensions for efficiency.
    Falls back to a deterministic hash-based pseudo-embedding if API unavailable.
    """
    results: List[Optional[List[float]]] = [None] * len(texts)
    uncached_indices = []

    for i, text in enumerate(texts):
        cached = _cache_get(text)
        if cached:
            results[i] = cached
        else:
            uncached_indices.append(i)

    if uncached_indices:
        try:
            import openai

            client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
            batch = [texts[i] for i in uncached_indices]
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
                dimensions=_EMBED_DIM,
            )
            for j, idx in enumerate(uncached_indices):
                vec = resp.data[j].embedding
                results[idx] = vec
                _cache_put(texts[idx], vec)
        except Exception as e:
            logger.warning("OpenAI embed failed, using hash fallback: %s", e)
            for idx in uncached_indices:
                vec = _hash_embed(texts[idx])
                results[idx] = vec
                _cache_put(texts[idx], vec)

    return results  # type: ignore


def embed_text(text: str) -> List[float]:
    """Embed a single text."""
    return embed_texts([text])[0]


def _hash_embed(text: str) -> List[float]:
    """Deterministic pseudo-embedding from text hash. Not semantic, but
    consistent and usable for testing when API is unavailable."""
    import struct

    h = hashlib.sha512(text.lower().encode()).digest()
    # Expand to _EMBED_DIM floats via repeated hashing
    raw = b""
    seed = text.encode()
    while len(raw) < _EMBED_DIM * 4:
        seed = hashlib.sha512(seed).digest()
        raw += seed
    floats = list(struct.unpack(f"<{_EMBED_DIM}f", raw[: _EMBED_DIM * 4]))
    # Normalize to unit vector
    norm = sum(x * x for x in floats) ** 0.5
    return [x / norm for x in floats] if norm > 1e-10 else floats


EMBED_DIM = _EMBED_DIM
