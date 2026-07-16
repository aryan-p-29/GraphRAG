"""
ingest.py
──────────
Full ingestion pipeline:
  1. Clone / re-use a local copy of a GitHub repo (via GitPython)
  2. Walk the file tree and dispatch each file to the correct language parser
  3. Collect all NodeData + EdgeData objects
  4. Generate embeddings for every node that has embed_text (batched)
  5. Write to Neo4j using MERGE (idempotent) in configurable batch sizes

Usage:
    python ingest.py <github_url_or_local_path> [--extensions .py .js ...]

Key guarantees:
  - MERGE (not CREATE) on every node and relationship → idempotent runs
  - Writes are batched (default WRITE_BATCH_SIZE nodes / rels per tx)
  - Embedding failures fall back to zero-vector (ingestion never aborts)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import git
from neo4j import GraphDatabase, Session

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, SUPPORTED_EXTENSIONS
from graph_schema import NodeData, EdgeData
from parsers import ParserDispatcher
from embeddings import embed_texts

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")

# ── Tunables ──────────────────────────────────────────────────────────────────
WRITE_BATCH_SIZE = 100   # nodes or edges per Neo4j transaction


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Clone / locate repo
# ─────────────────────────────────────────────────────────────────────────────

def clone_or_use_local(source: str) -> tuple[Path, bool]:
    """
    If *source* looks like a URL, shallow-clone it to a temp dir.
    Otherwise treat it as a local path.
    Returns (local_path, is_temp).
    """
    if source.startswith("http://") or source.startswith("https://") or source.startswith("git@"):
        tmp = Path(tempfile.mkdtemp(prefix="graphrag_"))
        logger.info("Cloning %s → %s", source, tmp)
        git.Repo.clone_from(source, tmp, depth=1, no_single_branch=True)
        return tmp, True
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Local path does not exist: {path}")
    logger.info("Using local repo: %s", path)
    return path, False


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Discover files
# ─────────────────────────────────────────────────────────────────────────────

def iter_source_files(
    repo_root: Path,
    extensions: frozenset[str] | None = None,
) -> Iterator[Path]:
    """Yield all source files under repo_root whose extension is supported."""
    exts = extensions or SUPPORTED_EXTENSIONS
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "vendor", "dist", "build"}
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune undesirable directories in-place
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() in exts:
                yield p


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 & 4: Parse + embed
# ─────────────────────────────────────────────────────────────────────────────

def parse_repo(
    repo_root: Path,
    extensions: frozenset[str] | None = None,
) -> tuple[list[NodeData], list[EdgeData]]:
    """Parse all discovered files and return unified node/edge lists."""
    all_nodes: list[NodeData] = []
    all_edges: list[EdgeData] = []
    files = list(iter_source_files(repo_root, extensions))
    logger.info("Discovered %d source file(s)", len(files))

    for fpath in files:
        rel = str(fpath.relative_to(repo_root))
        ext = fpath.suffix.lower()
        try:
            parser = ParserDispatcher.get_parser(ext)
            source = fpath.read_bytes()
            nodes, edges = parser.parse(source, rel)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
        except Exception as exc:
            logger.warning("Skipping %s — %s", rel, exc)

    logger.info("Extracted %d nodes, %d edges", len(all_nodes), len(all_edges))

    logger.info("Running global cross-file method resolution...")
    global_fuzzy_resolve(all_nodes, all_edges)

    return all_nodes, all_edges


def global_fuzzy_resolve(nodes: list[NodeData], edges: list[EdgeData]) -> None:
    """
    Cross-file fuzzy resolution of unmatched CALLS targets.

    The parser now emits full dotted targets such as "crud.create_user" or
    "self._helper". We use the prefix to narrow candidates before applying
    the ambiguity cutoff, preventing both false-positive and false-negative edges.

    Resolution algorithm (per dangling CALLS edge):
    ──────────────────────────────────────────────
    1. Parse the raw callee from the last '::' segment:
         "file.py::crud.create_user" → prefix="crud",  method="create_user"
         "file.py::self._helper"     → prefix="self",  method="_helper"
         "file.py::get_engine"       → prefix=None,    method="get_engine"

    2. Gather all candidates for `method` (Methods first, then Functions).

    3. Apply prefix filter:
         prefix == "self"  → keep only candidates in the SAME source file
                             as the caller.
         prefix == other   → keep candidates whose file_path or parent_name
                             contains the prefix string (case-insensitive).
         prefix == None    → no filter.

    4. If filtered candidates ≤ 3  → reroute.
       If filtered candidates > 3  → still ambiguous; drop.
       If 0 after filter           → fall back to unfiltered list, same rule.
    """
    node_ids: set[str] = {n.id for n in nodes}

    method_map: dict[str, list[str]] = defaultdict(list)
    func_map:   dict[str, list[str]] = defaultdict(list)
    class_map:  dict[str, list[str]] = defaultdict(list)
    init_map:   dict[str, str]       = {}  # class_id -> init_method_id

    # id → properties dict (avoids constructing invalid NodeData sentinels)
    props_by_id: dict[str, dict] = {n.id: n.properties for n in nodes}

    for n in nodes:
        name = n.properties.get("name", "")
        if n.label == "Method":
            method_map[name].append(n.id)
            if name == "__init__":
                # The parent class ID can be derived since method IDs are "class_id::method_name"
                parent_class_id = n.id.rsplit("::", 1)[0]
                init_map[parent_class_id] = n.id
        elif n.label == "Function":
            func_map[name].append(n.id)
        elif n.label == "Class":
            class_map[name].append(n.id)

    new_edges:  list[EdgeData] = []
    remove_idx: set[int]       = set()

    for idx, edge in enumerate(edges):
        if edge.type != "CALLS":
            continue
        if edge.target_id in node_ids:
            continue  # already resolved

        # Parse dotted callee from the last '::' segment
        raw_callee = edge.target_id.split("::")[-1]   # e.g. "crud.create_user"
        parts = raw_callee.split(".", 1)
        if len(parts) == 2:
            prefix, method_name = parts[0], parts[1]
        else:
            prefix, method_name = None, parts[0]

        # Gather all candidates
        all_candidates = method_map.get(method_name) or func_map.get(method_name) or class_map.get(method_name) or []
        if not all_candidates:
            continue

        # Apply prefix-based narrowing
        if prefix and prefix.lower() == "self":
            caller_file = edge.source_id.split("::")[0]
            filtered = [
                c for c in all_candidates
                if props_by_id.get(c, {}).get("file_path", "") == caller_file
            ]
        elif prefix:
            p_lower = prefix.lower()
            filtered = [
                c for c in all_candidates
                if (p_lower in props_by_id.get(c, {}).get("file_path", "").lower()
                    or p_lower in props_by_id.get(c, {}).get("parent_name", "").lower())
            ]
        else:
            filtered = all_candidates

        # Fall back to unfiltered if prefix filter removed everything
        candidates = filtered if filtered else all_candidates

        if len(candidates) > 3:
            continue  # still ambiguous; drop

        remove_idx.add(idx)
        for c_id in candidates:
            # If the candidate is a Class, route to its __init__ method if available
            is_class = any(c_id in class_list for class_list in class_map.values())
            target_id = init_map.get(c_id, c_id) if is_class else c_id

            new_edges.append(EdgeData(
                type="CALLS",
                source_id=edge.source_id,
                target_id=target_id,
                properties={"fuzzy_resolved": True},
            ))

    edges[:] = [e for i, e in enumerate(edges) if i not in remove_idx] + new_edges
    logger.info("Global fuzzy resolution: %d dangling CALLS edge(s) resolved.", len(remove_idx))




def attach_embeddings(nodes: list[NodeData]) -> None:
    """Embed all nodes that have embed_text; mutate nodes in-place."""
    embeddable = [(i, n) for i, n in enumerate(nodes) if n.embed_text]
    if not embeddable:
        return

    texts = [n.embed_text for _, n in embeddable]  # type: ignore[misc]
    logger.info("Generating embeddings for %d node(s)…", len(texts))
    vectors = embed_texts(texts)

    for (i, node), vector in zip(embeddable, vectors):
        node.properties["embedding"] = vector

    logger.info("Embeddings attached ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Write to Neo4j (idempotent, batched)
# ─────────────────────────────────────────────────────────────────────────────

# ── Cypher templates ──────────────────────────────────────────────────────────

_MERGE_NODE_QUERY = """\
UNWIND $batch AS row
MERGE (n {id: row.id})
SET n:{label}
SET n += row.props
"""

# Relationship MERGE uses source/target IDs — no label needed on the rel side
_MERGE_EDGE_QUERY = """\
UNWIND $batch AS row
MATCH (src {id: row.source_id})
MATCH (tgt {id: row.target_id})
MERGE (src)-[r:{rel_type}]->(tgt)
SET r += row.props
"""


def _batches(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i: i + size]


def _write_nodes(session: Session, nodes: list[NodeData], batch_size: int = WRITE_BATCH_SIZE) -> None:
    # Group nodes by label so each batch uses a single Cypher template
    from collections import defaultdict
    by_label: dict[str, list[NodeData]] = defaultdict(list)
    for n in nodes:
        by_label[n.label].append(n)

    for label, label_nodes in by_label.items():
        for batch in _batches(label_nodes, batch_size):
            rows = [
                {
                    "id": n.id,
                    "props": {**n.properties, "id": n.id},
                }
                for n in batch
            ]
            query = f"""
UNWIND $batch AS row
MERGE (n:{label} {{id: row.id}})
SET n += row.props
"""
            session.run(query, batch=rows)
        logger.debug("  MERGE'd %d %s node(s)", len(label_nodes), label)


def _write_edges(session: Session, edges: list[EdgeData], batch_size: int = WRITE_BATCH_SIZE) -> None:
    from collections import defaultdict
    by_type: dict[str, list[EdgeData]] = defaultdict(list)
    for e in edges:
        by_type[e.type].append(e)

    for rel_type, rel_edges in by_type.items():
        for batch in _batches(rel_edges, batch_size):
            rows = [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "props": e.properties,
                }
                for e in batch
            ]
            query = f"""
UNWIND $batch AS row
MATCH (src {{id: row.source_id}})
MATCH (tgt {{id: row.target_id}})
MERGE (src)-[r:{rel_type}]->(tgt)
SET r += row.props
"""
            session.run(query, batch=rows)
        logger.debug("  MERGE'd %d %s edge(s)", len(rel_edges), rel_type)


def _create_indexes(session: Session) -> None:
    """Create per-label uniqueness constraints so MERGE on id is O(1)."""
    labels = ["File", "Class", "Function", "Method", "Struct", "Interface", "TypeAlias", "Module"]
    for label in labels:
        try:
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )
        except Exception as exc:
            logger.debug("Constraint for %s: %s (may already exist)", label, exc)


def _create_vector_index(session: Session) -> None:
    """Create the Neo4j vector index that LlamaIndex's VectorContextRetriever uses."""
    try:
        session.run("""
            CREATE VECTOR INDEX entity IF NOT EXISTS
            FOR (n:__Entity__)
            ON n.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 3072,
                `vector.similarity_function`: "cosine"
            }}
        """)
        logger.info("Vector index 'entity' created (or already exists)")
    except Exception as exc:
        logger.debug("Vector index: %s", exc)


def _label_entity_nodes(session: Session) -> None:
    """
    Add the __Entity__ label to semantically rich nodes (Class, Function, Method,
    Struct, Interface) so LlamaIndex's VectorContextRetriever can find them.
    Deliberately excludes File and Module nodes (poor embed content).
    """
    semantic_labels = ["Class", "Function", "Method", "Struct", "Interface"]
    for label in semantic_labels:
        result = session.run(
            f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
            f"SET n:__Entity__ RETURN count(n) AS cnt"
        )
        cnt = result.single()["cnt"]
        if cnt:
            logger.debug("  Labelled %d %s node(s) as __Entity__", cnt, label)

    total = session.run("MATCH (n:__Entity__) RETURN count(n) AS cnt").single()["cnt"]
    logger.info("Total __Entity__ nodes (for vector search): %d", total)


def write_to_neo4j(nodes: list[NodeData], edges: list[EdgeData]) -> None:
    """Open a Neo4j session and write all nodes then edges idempotently."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            logger.info("Creating indexes…")
            _create_indexes(session)

            logger.info("Writing %d node(s) in batches of %d…", len(nodes), WRITE_BATCH_SIZE)
            _write_nodes(session, nodes)

            # Only write edges where both endpoints actually have nodes
            node_ids = {n.id for n in nodes}
            valid_edges = [
                e for e in edges
                if e.source_id in node_ids and e.target_id in node_ids
            ]
            skipped = len(edges) - len(valid_edges)
            if skipped:
                logger.info("Skipping %d edge(s) with dangling endpoints", skipped)

            logger.info("Writing %d edge(s) in batches of %d…", len(valid_edges), WRITE_BATCH_SIZE)
            _write_edges(session, valid_edges)

            # ── Post-write: vector index + entity labels ──────────────────────
            _create_vector_index(session)
            _label_entity_nodes(session)

    finally:
        driver.close()

    logger.info("Graph write complete ✓")



# ─────────────────────────────────────────────────────────────────────────────
# Verification helper
# ─────────────────────────────────────────────────────────────────────────────

def print_node_counts() -> None:
    """Print a breakdown of node counts per label from Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            print("\n── Neo4j Node Counts ──────────────────────────────")
            result = s.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt "
                "ORDER BY cnt DESC"
            )
            total = 0
            for row in result:
                label = row["label"] or "unlabelled"
                cnt   = row["cnt"]
                total += cnt
                print(f"  {label:<15} {cnt:>6}")
            print(f"  {'TOTAL':<15} {total:>6}")

            print("\n── Neo4j Relationship Counts ──────────────────────")
            result2 = s.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt "
                "ORDER BY cnt DESC"
            )
            rel_total = 0
            for row in result2:
                rt  = row["rel_type"]
                cnt = row["cnt"]
                rel_total += cnt
                print(f"  {rt:<20} {cnt:>6}")
            print(f"  {'TOTAL':<20} {rel_total:>6}")
            print("────────────────────────────────────────────────────\n")
    finally:
        driver.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_ingest(source: str, extensions: frozenset[str] | None = None) -> None:
    repo_path, is_temp = clone_or_use_local(source)
    try:
        nodes, edges = parse_repo(repo_path, extensions)
        attach_embeddings(nodes)
        write_to_neo4j(nodes, edges)
        print_node_counts()
    finally:
        if is_temp:
            shutil.rmtree(repo_path, ignore_errors=True)
            logger.info("Cleaned up temp clone at %s", repo_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG ingestion pipeline")
    parser.add_argument("source", help="GitHub URL or local repo path")
    parser.add_argument(
        "--extensions", nargs="*", default=None,
        help="File extensions to ingest (e.g. --extensions .py .js). Default: all supported.",
    )
    args = parser.parse_args()

    exts = frozenset(args.extensions) if args.extensions else None
    run_ingest(args.source, exts)
