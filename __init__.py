"""contextdb A/B testing plugin for Hermes.

Runs dual retrieval pipelines (sqlite-vec + contextdb) alongside FTS5,
logs comparison metrics, and prefers contextdb responses.
"""

import json
import logging
import time
from typing import Any

from . import schemas, tools, comparison_db as cdb, vec_store, cdb_store
from .embeddings import embed_text

logger = logging.getLogger(__name__)

_current_session_id = ""
_current_model = ""


def _on_session_start(session_id: str, model: str = "", platform: str = "", **kwargs):
    """Track session for logging."""
    global _current_session_id, _current_model
    _current_session_id = session_id
    _current_model = model
    cdb.log_session(session_id, model, platform)
    logger.info("contextdb-ab: session started (model=%s)", model)


def _on_session_end(session_id: str, **kwargs):
    """Flush session log."""
    cdb.end_session(session_id)
    logger.info("contextdb-ab: session ended")


def _on_post_tool_call(tool_name: str, args: Any, result: str,
                       task_id: str = "", **kwargs):
    """Index session_search results into sqlite-vec for future semantic search.

    Also intercept memory tool writes to capture in contextdb.
    """
    if tool_name == "session_search" and result:
        _index_session_search_result(result, task_id)
    elif tool_name == "memory" and result:
        _capture_memory_write(args, task_id)


def _index_session_search_result(result: str, session_id: str):
    """After a session_search returns, index the content into sqlite-vec
    so future semantic searches can find it."""
    try:
        data = json.loads(result)
        # Session search returns either a list of session summaries or a message
        if isinstance(data, list):
            for item in data:
                content = item.get("summary", item.get("content", ""))
                if content and len(content) > 20:
                    try:
                        vec = embed_text(content[:2000])
                        vec_store.index_message(
                            session_id=session_id,
                            content=content[:2000],
                            role="session_recall",
                            vector=vec,
                        )
                    except Exception as e:
                        logger.debug("Failed to index session result: %s", e)
        elif isinstance(data, dict):
            for key in ("summary", "content", "result"):
                content = data.get(key, "")
                if content and len(content) > 20:
                    try:
                        vec = embed_text(content[:2000])
                        vec_store.index_message(
                            session_id=session_id,
                            content=content[:2000],
                            role="session_recall",
                            vector=vec,
                        )
                    except Exception:
                        pass
    except (json.JSONDecodeError, TypeError):
        pass


def _capture_memory_write(args: Any, session_id: str):
    """When the agent writes to MEMORY.md/USER.md, also store in contextdb."""
    if not isinstance(args, dict):
        return
    action = args.get("action", "")
    content = args.get("content", args.get("text", ""))

    if action == "add" and content and len(content) > 10:
        target = args.get("target", "memory")
        try:
            vec = embed_text(content)
            cdb_store.write_knowledge(
                content=content,
                source_id=f"hermes:{target}",
                vector=vec,
                confidence=0.8,
                mem_type="semantic" if target == "memory" else "episodic",
                labels=[target, "auto-captured"],
            )
            logger.debug("Captured memory write to contextdb: %s", content[:60])
        except Exception as e:
            logger.debug("Failed to capture memory write: %s", e)


def _on_pre_llm_call(session_id: str = "", user_message: str = "",
                     conversation_history: Any = None, is_first_turn: bool = False,
                     **kwargs):
    """Inject relevant contextdb knowledge into the system prompt.

    Returns {"context": "..."} to be appended to the ephemeral system prompt.
    Only injects on first turn (session start) to respect frozen snapshot pattern,
    or when there's a substantive user message to match against.
    """
    if not user_message or len(user_message.strip()) < 10:
        return None

    try:
        query_vec = embed_text(user_message[:500])
        results = cdb_store.retrieve(query_vec, top_k=3)

        if not results:
            return None

        # Only inject if results are meaningfully relevant
        top_score = results[0]["score"]
        if top_score < 0.3:
            return None

        lines = ["[contextdb knowledge recall]"]
        for r in results:
            cred = r["credibility"]
            cred_label = "high" if cred > 0.7 else "medium" if cred > 0.4 else "low"
            lines.append(
                f"- [{r['mem_type']}] (conf={r['confidence']:.0%}, "
                f"src={r['source_id']} cred={cred_label}) {r['content']}"
            )

        context = "\n".join(lines)

        # Log this as a retrieval
        cdb.log_retrieval(cdb.RetrievalLog(
            query=user_message[:200],
            session_id=session_id,
            source="pre_llm_inject",
            cdb_result_count=len(results),
            cdb_latency_ms=0,
            cdb_tokens_est=cdb.estimate_tokens(context),
            cdb_avg_confidence=sum(r["confidence"] for r in results) / len(results),
            cdb_avg_score=top_score,
            cdb_top_source=results[0]["source_id"],
            cdb_top_credibility=results[0]["credibility"],
            chosen="contextdb",
            cdb_results_json=json.dumps([
                {"content": r["content"][:100], "score": r["score"]}
                for r in results
            ]),
        ))

        return {"context": context}

    except Exception as e:
        logger.debug("pre_llm_call knowledge recall failed: %s", e)
        return None


def register(ctx):
    """Wire schemas to handlers and register hooks."""
    # Tools
    ctx.register_tool(
        name="knowledge_recall",
        toolset="contextdb",
        schema=schemas.KNOWLEDGE_RECALL,
        handler=tools.knowledge_recall,
    )
    ctx.register_tool(
        name="knowledge_store",
        toolset="contextdb",
        schema=schemas.KNOWLEDGE_STORE,
        handler=tools.knowledge_store,
    )
    ctx.register_tool(
        name="ab_report",
        toolset="contextdb",
        schema=schemas.AB_REPORT,
        handler=tools.ab_report,
    )

    # Hooks
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)

    logger.info("contextdb-ab plugin registered (3 tools, 4 hooks)")
