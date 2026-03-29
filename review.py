#!/usr/bin/env python3
"""A/B comparison review — run standalone to see how contextdb compares.

Usage:
    python3 -m hermes_context_plugin.review
    # or
    python3 ~/projects/hermes-context-plugin/review.py
"""

import json
import sys
from pathlib import Path

# Allow running standalone — add plugin dir to path and import as package
_plugin_dir = Path(__file__).parent
sys.path.insert(0, str(_plugin_dir.parent))

# Try the hermes_plugins namespace first (when loaded by hermes),
# fall back to direct import for standalone use
try:
    from hermes_plugins.contextdb_ab import comparison_db as cdb, cdb_store, vec_store
except ImportError:
    # Standalone: import from the directory directly
    sys.path.insert(0, str(_plugin_dir))
    import comparison_db as cdb  # type: ignore
    import cdb_store  # type: ignore
    import vec_store  # type: ignore


def _bar(value: float, width: int = 20, fill: str = "#", empty: str = ".") -> str:
    filled = int(value * width)
    return fill * filled + empty * (width - filled)


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def print_report():
    stats = cdb.get_stats()

    print()
    print("=" * 64)
    print("  contextdb A/B Comparison Report")
    print("=" * 64)
    print()

    if stats["total_retrievals"] == 0:
        print("  No retrieval data yet.")
        print(f"  Knowledge nodes:    {cdb_store.node_count()}")
        print(f"  Indexed messages:   {vec_store.message_count()}")
        print()
        print("  Use Hermes for a while to accumulate comparison data.")
        print()
        return

    # --- Summary ---
    print(f"  Retrievals:         {stats['total_retrievals']}")
    print(f"  Sessions:           {stats['total_sessions']}")
    print(f"  Knowledge nodes:    {cdb_store.node_count()}")
    print(f"  Indexed messages:   {vec_store.message_count()}")
    print(f"  Knowledge writes:   {stats['total_knowledge_writes']}")
    print()

    # --- Latency ---
    print("  LATENCY (avg per retrieval)")
    print("  -" * 30)
    max_lat = max(stats["avg_fts5_latency_ms"], stats["avg_vec_latency_ms"],
                  stats["avg_cdb_latency_ms"], 1)
    for label, val in [
        ("FTS5     ", stats["avg_fts5_latency_ms"]),
        ("Vec      ", stats["avg_vec_latency_ms"]),
        ("contextdb", stats["avg_cdb_latency_ms"]),
    ]:
        bar = _bar(val / max_lat)
        print(f"  {label}  {bar}  {val:.1f}ms")
    print()

    # --- Result count ---
    print("  RESULTS (avg per retrieval)")
    print("  -" * 30)
    max_res = max(stats["avg_fts5_results"], stats["avg_vec_results"],
                  stats["avg_cdb_results"], 1)
    for label, val in [
        ("FTS5     ", stats["avg_fts5_results"]),
        ("Vec      ", stats["avg_vec_results"]),
        ("contextdb", stats["avg_cdb_results"]),
    ]:
        bar = _bar(val / max_res)
        print(f"  {label}  {bar}  {val:.1f}")
    print()

    # --- Token economics ---
    print("  TOKEN ECONOMICS (cumulative)")
    print("  -" * 30)
    print(f"  FTS5 total tokens:       {_fmt_tokens(stats['total_fts5_tokens'])}")
    print(f"  sqlite-vec total tokens: {_fmt_tokens(stats['total_vec_tokens'])}")
    print(f"  contextdb total tokens:  {_fmt_tokens(stats['total_cdb_tokens'])}")
    savings = stats["token_savings_vs_fts5"]
    if savings > 0:
        print(f"  Token savings vs FTS5:   {_fmt_tokens(savings)} saved")
    elif savings < 0:
        print(f"  Token delta vs FTS5:     {_fmt_tokens(abs(savings))} more (richer context)")
    print(f"  LLM calls saved:         {stats['llm_calls_saved']}")
    print()

    # --- Quality ---
    print("  QUALITY METRICS")
    print("  -" * 30)
    print(f"  contextdb avg confidence:  {stats['avg_cdb_confidence']:.3f}")
    print(f"  contextdb avg score:       {stats['avg_cdb_score']:.3f}")
    print(f"  sqlite-vec avg similarity: {stats['avg_vec_similarity']:.3f}")
    print()

    # --- Semantic advantage ---
    print("  SEMANTIC ADVANTAGE")
    print("  -" * 30)
    print(f"  Queries only contextdb found:  {stats['semantic_only_hits_cdb']}")
    print(f"  Queries only vec found:        {stats['semantic_only_hits_vec']}")
    print(f"  Queries only FTS5 found:       {stats['fts5_only_hits']}")
    print(f"  contextdb hit rate:            {stats['cdb_hit_rate']:.1f}%")
    print()

    # --- Source credibility ---
    sources = cdb_store.source_stats()
    if sources:
        print("  SOURCE CREDIBILITY")
        print("  -" * 30)
        for s in sources:
            cred = s["credibility"]
            bar = _bar(cred, width=15)
            print(f"  {s['source_id']:<20s}  {bar}  {cred:.1%}  (a={s['alpha']:.0f} b={s['beta']:.0f})")
        print()

    print("=" * 64)
    print()


def print_json():
    stats = cdb.get_stats()
    stats["source_credibility"] = cdb_store.source_stats()
    stats["knowledge_nodes"] = cdb_store.node_count()
    stats["indexed_messages"] = vec_store.message_count()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    if "--json" in sys.argv:
        print_json()
    else:
        print_report()
