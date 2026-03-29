"""Tool schemas — what the LLM sees."""

KNOWLEDGE_RECALL = {
    "name": "knowledge_recall",
    "description": (
        "Search the persistent knowledge base for relevant information from "
        "past research, implementations, and debugging sessions. Returns ranked "
        "results with confidence scores and source credibility. Use this when "
        "you need to recall what was learned in previous sessions about a topic, "
        "codebase, or system."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query describing what you want to recall",
            },
            "namespace": {
                "type": "string",
                "description": "Knowledge namespace to search (default: 'hermes')",
                "default": "hermes",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum results to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

KNOWLEDGE_STORE = {
    "name": "knowledge_store",
    "description": (
        "Store a distilled piece of knowledge for future recall. Use this to "
        "persist important findings, architectural decisions, system behaviors, "
        "or lessons learned. Each entry tracks its source model and confidence."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The knowledge to store (a clear, self-contained statement)",
            },
            "confidence": {
                "type": "number",
                "description": "How confident you are in this knowledge (0.0-1.0)",
                "default": 0.7,
            },
            "mem_type": {
                "type": "string",
                "enum": ["episodic", "semantic", "procedural", "working"],
                "description": "Memory type: episodic (events), semantic (facts), procedural (how-to), working (temporary)",
                "default": "semantic",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorization (e.g., ['auth', 'contextdb', 'deployment'])",
            },
            "source_model": {
                "type": "string",
                "description": "Which model produced this knowledge (e.g., 'opus', 'sonnet', 'haiku')",
                "default": "unknown",
            },
        },
        "required": ["content"],
    },
}

AB_REPORT = {
    "name": "ab_report",
    "description": (
        "Generate a comparison report of retrieval performance between FTS5 "
        "(keyword search), sqlite-vec (semantic search), and contextdb "
        "(epistemic knowledge base). Shows hit rates, latency, token savings, "
        "and quality metrics."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}
