"""Tool handlers — the code that runs when the LLM calls each tool."""

import json
import logging
import time

from . import comparison_db as cdb
from . import cdb_store
from . import vec_store
from .embeddings import embed_text

logger = logging.getLogger(__name__)

# Source model → source_id mapping for contextdb credibility tracking
_MODEL_SOURCE_MAP = {
    "opus":   "model:opus",
    "sonnet": "model:sonnet",
    "haiku":  "model:haiku",
}


def knowledge_recall(args: dict, **kwargs) -> str:
    """Search the knowledge base with dual retrieval (vec + contextdb)."""
    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required"})

    namespace = args.get("namespace", "hermes")
    top_k = args.get("top_k", 5)
    session_id = kwargs.get("task_id", "")

    try:
        query_vec = embed_text(query)
    except Exception as e:
        return json.dumps({"error": f"Embedding failed: {e}"})

    # --- Run all three searches ---

    # 1. sqlite-vec semantic search
    t0 = time.time()
    vec_results = vec_store.search(query_vec, limit=top_k)
    vec_ms = (time.time() - t0) * 1000

    # 2. contextdb knowledge retrieval
    t0 = time.time()
    cdb_results = cdb_store.retrieve(query_vec, namespace=namespace, top_k=top_k)
    cdb_ms = (time.time() - t0) * 1000

    # --- Estimate tokens ---
    vec_text = "\n".join(r["content"] for r in vec_results)
    cdb_text = "\n".join(r["content"] for r in cdb_results)
    vec_tokens = cdb.estimate_tokens(vec_text)
    cdb_tokens = cdb.estimate_tokens(cdb_text)

    # --- Log comparison ---
    entry = cdb.RetrievalLog(
        query=query,
        session_id=session_id,
        source="knowledge_recall",

        vec_result_count=len(vec_results),
        vec_latency_ms=round(vec_ms, 1),
        vec_tokens_est=vec_tokens,
        vec_top_similarity=vec_results[0]["similarity"] if vec_results else 0,

        cdb_result_count=len(cdb_results),
        cdb_latency_ms=round(cdb_ms, 1),
        cdb_tokens_est=cdb_tokens,
        cdb_avg_confidence=sum(r["confidence"] for r in cdb_results) / len(cdb_results) if cdb_results else 0,
        cdb_avg_score=sum(r["score"] for r in cdb_results) / len(cdb_results) if cdb_results else 0,
        cdb_top_source=cdb_results[0]["source_id"] if cdb_results else "",
        cdb_top_credibility=cdb_results[0]["credibility"] if cdb_results else 0,

        chosen="contextdb",
        cdb_llm_call=0,  # no LLM summarization needed

        vec_results_json=json.dumps(vec_results[:5]),
        cdb_results_json=json.dumps(cdb_results[:5]),
    )
    cdb.log_retrieval(entry)

    # --- Build response (prefer contextdb, supplement with vec) ---
    response = {
        "knowledge": [
            {
                "content": r["content"],
                "confidence": r["confidence"],
                "source": r["source_id"],
                "credibility": r["credibility"],
                "score": r["score"],
                "mem_type": r["mem_type"],
                "labels": r["labels"],
            }
            for r in cdb_results
        ],
        "supplementary": [
            {
                "content": r["content"],
                "similarity": r["similarity"],
                "source": "session_history",
            }
            for r in vec_results
            # Only include vec results that aren't duplicated in cdb
            if not any(c["content"] == r["content"] for c in cdb_results)
        ][:3],
        "meta": {
            "contextdb_results": len(cdb_results),
            "vec_results": len(vec_results),
            "cdb_latency_ms": round(cdb_ms, 1),
            "vec_latency_ms": round(vec_ms, 1),
        },
    }

    return json.dumps(response)


def knowledge_store(args: dict, **kwargs) -> str:
    """Store a knowledge node in contextdb."""
    content = args.get("content", "").strip()
    if not content:
        return json.dumps({"error": "content is required"})

    confidence = args.get("confidence", 0.7)
    mem_type = args.get("mem_type", "semantic")
    labels = args.get("labels", [])
    source_model = args.get("source_model", "unknown")
    session_id = kwargs.get("task_id", "")

    source_id = _MODEL_SOURCE_MAP.get(source_model, f"model:{source_model}")

    try:
        vector = embed_text(content)
    except Exception as e:
        return json.dumps({"error": f"Embedding failed: {e}"})

    node_id = cdb_store.write_knowledge(
        content=content,
        source_id=source_id,
        vector=vector,
        confidence=confidence,
        mem_type=mem_type,
        labels=labels,
    )

    # Log the write
    cdb.log_knowledge_write(
        session_id=session_id,
        content=content,
        source_model=source_model,
        confidence=confidence,
        mem_type=mem_type,
        labels=labels,
    )

    return json.dumps({
        "stored": True,
        "node_id": node_id,
        "source": source_id,
        "confidence": confidence,
        "mem_type": mem_type,
    })


def ab_report(args: dict, **kwargs) -> str:
    """Generate A/B comparison report."""
    stats = cdb.get_stats()

    if stats["total_retrievals"] == 0:
        return json.dumps({
            "message": "No retrieval data yet. Use the agent for a while to accumulate comparison data.",
            "knowledge_nodes": cdb_store.node_count(),
            "indexed_messages": vec_store.message_count(),
        })

    # Format the report
    report = {
        "summary": {
            "total_retrievals": stats["total_retrievals"],
            "total_sessions": stats["total_sessions"],
            "knowledge_nodes": cdb_store.node_count(),
            "indexed_messages": vec_store.message_count(),
        },
        "latency": {
            "fts5_avg_ms": stats["avg_fts5_latency_ms"],
            "sqlite_vec_avg_ms": stats["avg_vec_latency_ms"],
            "contextdb_avg_ms": stats["avg_cdb_latency_ms"],
        },
        "result_quality": {
            "fts5_avg_results": stats["avg_fts5_results"],
            "sqlite_vec_avg_results": stats["avg_vec_results"],
            "contextdb_avg_results": stats["avg_cdb_results"],
            "contextdb_avg_confidence": stats["avg_cdb_confidence"],
            "contextdb_avg_score": stats["avg_cdb_score"],
            "vec_avg_similarity": stats["avg_vec_similarity"],
        },
        "token_economics": {
            "fts5_total_tokens": stats["total_fts5_tokens"],
            "vec_total_tokens": stats["total_vec_tokens"],
            "contextdb_total_tokens": stats["total_cdb_tokens"],
            "token_savings_vs_fts5": stats["token_savings_vs_fts5"],
            "llm_summarization_calls_saved": stats["llm_calls_saved"],
        },
        "semantic_advantage": {
            "queries_only_vec_found": stats["semantic_only_hits_vec"],
            "queries_only_cdb_found": stats["semantic_only_hits_cdb"],
            "queries_only_fts5_found": stats["fts5_only_hits"],
            "contextdb_hit_rate_pct": stats["cdb_hit_rate"],
        },
        "source_credibility": cdb_store.source_stats(),
    }

    return json.dumps(report, indent=2)
