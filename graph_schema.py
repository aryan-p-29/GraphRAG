"""
graph_schema.py — Language-agnostic Pydantic models for graph nodes and edges.
All parsers emit lists of NodeData / EdgeData; the ingester writes them to Neo4j.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Allowed node labels (matches plan schema) ─────────────────────────────────
NODE_LABELS = frozenset(
    {
        "File",
        "Class",
        "Function",
        "Method",
        "Struct",
        "Interface",
        "TypeAlias",
        "Module",
    }
)

# ── Allowed relationship types ─────────────────────────────────────────────────
EDGE_TYPES = frozenset(
    {
        "DEFINES",
        "HAS_METHOD",
        "CALLS",
        "IMPORTS",
        "INHERITS",
        "IMPLEMENTS",
        "CONTAINS",
    }
)


class NodeData(BaseModel):
    """Represents a single graph node to be MERGE'd into Neo4j."""

    id: str = Field(
        ...,
        description="Unique identifier: typically '<file_path>::<name>' or just '<name>' for modules.",
    )
    label: str = Field(..., description="One of the allowed node labels.")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key/value properties stored on the node.",
    )
    # Text used to generate the embedding (populated just before ingestion)
    embed_text: Optional[str] = Field(
        default=None,
        description="Text to embed via Gemini. Derived from docstring + signature.",
    )

    def model_post_init(self, __context: Any) -> None:
        if self.label not in NODE_LABELS:
            raise ValueError(f"Unknown node label: {self.label!r}")


class EdgeData(BaseModel):
    """Represents a directed relationship between two nodes."""

    type: str = Field(..., description="One of the allowed relationship types.")
    source_id: str = Field(..., description="ID of the source node.")
    target_id: str = Field(..., description="ID of the target node.")
    properties: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.type not in EDGE_TYPES:
            raise ValueError(f"Unknown edge type: {self.type!r}")
