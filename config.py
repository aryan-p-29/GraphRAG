"""
config.py — Centralised settings and language registry.
Loads from environment / .env file at import time.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if it exists next to this file
load_dotenv(Path(__file__).parent / ".env")

# ── Neo4j ─────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag123")

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL        = os.getenv("LLM_MODEL", "models/gemini-3.1-flash-lite")

# ── Language Registry ─────────────────────────────────────────────────────────
# Maps file extensions → parser module + class name (loaded lazily in dispatcher)
LANGUAGE_REGISTRY: dict[str, dict[str, str]] = {
    ".py":  {"module": "parsers.python_parser",     "class": "PythonParser"},
    ".js":  {"module": "parsers.javascript_parser", "class": "JavaScriptParser"},
    ".ts":  {"module": "parsers.javascript_parser", "class": "JavaScriptParser"},
    ".go":  {"module": "parsers.go_parser",         "class": "GoParser"},
    ".cpp": {"module": "parsers.cpp_parser",        "class": "CppParser"},
    ".hpp": {"module": "parsers.cpp_parser",        "class": "CppParser"},
    ".cc":  {"module": "parsers.cpp_parser",        "class": "CppParser"},
    ".h":   {"module": "parsers.cpp_parser",        "class": "CppParser"},
}

# Extensions to discover when walking a repo (subset of registry keys)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(LANGUAGE_REGISTRY.keys())
