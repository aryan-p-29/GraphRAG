"""
api.py — FastAPI backend for the GraphRAG Codebase Explorer.

Endpoints
─────────
GET  /api/health           Liveness check
POST /api/ingest           Start ingestion; streams SSE log lines
GET  /api/ingest/status    Current ingestion status
POST /api/query            Run a natural-language query
GET  /api/graph            Pull graph nodes/edges for visualisation
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="GraphRAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ingestion state (simple in-process singleton) ─────────────────────────────
_ingest_lock = threading.Lock()
_ingest_state: dict[str, Any] = {"status": "idle", "log": []}


# ── Request / response models ─────────────────────────────────────────────────
class IngestRequest(BaseModel):
    source: str                          # GitHub URL or local path
    extensions: list[str] = [".py"]


class QueryRequest(BaseModel):
    query: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """Append a log line to the shared state (thread-safe)."""
    logger.info(msg)
    with _ingest_lock:
        _ingest_state["log"].append(msg)


def _run_ingestion(source: str, extensions: list[str]) -> None:
    """Run the full ingestion pipeline in a background thread."""
    with _ingest_lock:
        _ingest_state.update({"status": "running", "log": []})

    try:
        _log(f"▶ Starting ingestion for: {source}")

        from ingest import (
            clone_or_use_local,
            parse_repo,
            attach_embeddings,
            write_to_neo4j,
        )

        repo_path, is_temp = clone_or_use_local(source)
        _log(f"✔ Repo ready at {repo_path}")

        exts = frozenset(extensions)
        nodes, edges = parse_repo(repo_path, exts)
        _log(f"✔ Parsed {len(nodes)} nodes, {len(edges)} edges")

        attach_embeddings(nodes)
        _log("✔ Embeddings generated")

        write_to_neo4j(nodes, edges)
        _log(f"✔ Wrote {len(nodes)} nodes and {len(edges)} edges to Neo4j")

        if is_temp:
            import shutil
            shutil.rmtree(repo_path, ignore_errors=True)

        _log("✅ Ingestion complete")
        with _ingest_lock:
            _ingest_state["status"] = "done"

    except Exception as exc:
        _log(f"❌ Ingestion failed: {exc}")
        with _ingest_lock:
            _ingest_state["status"] = "error"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/ingest/status")
def ingest_status() -> dict:
    with _ingest_lock:
        return {
            "status": _ingest_state["status"],
            "log": list(_ingest_state["log"]),
        }


@app.post("/api/ingest")
async def ingest(req: IngestRequest) -> EventSourceResponse:
    """
    Start ingestion and stream log lines as Server-Sent Events.
    Each SSE event has data = one log line string.
    """
    with _ingest_lock:
        if _ingest_state["status"] == "running":
            raise HTTPException(status_code=409, detail="Ingestion already running")

    # Start the heavy work in a background thread so the async event loop is free
    thread = threading.Thread(
        target=_run_ingestion,
        args=(req.source, req.extensions),
        daemon=True,
    )
    thread.start()

    async def event_stream() -> AsyncGenerator[dict, None]:
        sent = 0
        while True:
            with _ingest_lock:
                log = _ingest_state["log"]
                status = _ingest_state["status"]

            # Emit any new log lines
            while sent < len(log):
                yield {"data": log[sent]}
                sent += 1

            if status in ("done", "error"):
                yield {"data": f"__STATUS__{status}"}
                break

            await asyncio.sleep(0.25)

    return EventSourceResponse(event_stream())


@app.post("/api/query")
async def query(req: QueryRequest) -> dict:
    """Run a query through the hybrid retriever + LLM."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        # Build engine lazily (cached by the module after first call)
        from query_engine import build_query_engine
        engine = build_query_engine()
        response = await asyncio.to_thread(engine.query, req.query)
        
        import re
        answer = str(response)
        # The LLM sometimes hallucinates internal LlamaIndex UUIDs.
        # Strip all UUIDs from the final answer text.
        answer = re.sub(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '', answer)

        # Extract source node IDs from metadata when available
        sources: list[str] = []
        if hasattr(response, "source_nodes"):
            for sn in response.source_nodes:
                nid = (
                    sn.node.metadata.get("id")
                    or sn.node.metadata.get("node_id")
                    or sn.node.id_
                )
                if nid:
                    nid_str = str(nid)
                    # Don't show raw UUIDs as source chips, they are meaningless to the user
                    if not re.match(r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$', nid_str):
                        sources.append(nid_str)
        return {"answer": answer, "sources": sources}
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/graph")
def graph_data() -> dict:
    """
    Return all nodes and edges from Neo4j for the graph visualisation.

    LlamaIndex injects an __Entity__ label on semantic nodes so their label
    list looks like ['Function', '__Entity__']. We skip __Entity__ when
    picking the display label.

    Edges: returned for ALL node pairs in the graph (File nodes included on
    the source side so DEFINES and IMPORTS edges are visible).
    """
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from neo4j import GraphDatabase

    # Labels we consider semantic / display-worthy
    SEMANTIC = {"Class", "Function", "Method", "Module",
                "Struct", "Interface", "TypeAlias", "File"}

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    nodes: list[dict] = []
    edges: list[dict] = []

    try:
        with driver.session() as session:
            # ── All nodes (including File so DEFINES edges have endpoints) ─
            result = session.run(
                """
                MATCH (n)
                RETURN n.id         AS id,
                       n.name       AS name,
                       labels(n)    AS lbls,
                       n.file_path  AS file_path
                LIMIT 500
                """
            )
            for row in result:
                nid = row["id"]
                if not nid:
                    continue
                # Pick first semantic label, skip __Entity__
                lbls = row["lbls"] or []
                label = next((l for l in lbls if l in SEMANTIC), None) \
                        or next((l for l in lbls if l != "__Entity__"), "Node")
                nodes.append({
                    "id":    nid,
                    "name":  row["name"] or nid,
                    "label": label,
                    "file":  row["file_path"] or "",
                })

            # ── All edges between returned nodes ──────────────────────────
            node_ids = [n["id"] for n in nodes]
            result = session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE a.id IN $ids AND b.id IN $ids
                RETURN a.id    AS source,
                       b.id    AS target,
                       type(r) AS type
                LIMIT 2000
                """,
                ids=node_ids,
            )
            seen: set[tuple] = set()
            for row in result:
                key = (row["source"], row["target"], row["type"])
                if key not in seen:
                    seen.add(key)
                    edges.append({
                        "source": row["source"],
                        "target": row["target"],
                        "type":   row["type"],
                    })

    finally:
        driver.close()

    logger.info("Graph payload: %d nodes, %d edges", len(nodes), len(edges))
    return {"nodes": nodes, "edges": edges}

