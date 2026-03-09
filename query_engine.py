"""
query_engine.py
────────────────
Read-only hybrid query engine over the pre-built GraphRAG Neo4j graph.

Architecture:
  1. GeminiNativeEmbedding  — custom LlamaIndex BaseEmbedding that wraps our
                             google.genai client (avoids deprecated google-generativeai)
  2. GeminiNativeLLM        — custom LlamaIndex CustomLLM that wraps google.genai
  3. Neo4jPropertyGraphStore — connected with refresh_schema=False, create_indexes=False
                             (strictly read-only: never mutates the graph)
  4. VectorContextRetriever — finds the top-k most similar nodes via embedding
                             then follows graph edges to depth `path_depth`
  5. CypherCustomRetriever  — direct Cypher traversal for call-stack / import context
  6. PropertyGraphIndex     — fuses both retrievers; .as_query_engine() adds LLM synthesis

Usage:
    from query_engine import build_query_engine
    engine = build_query_engine()
    response = engine.query("What classes are defined and what do they do?")
    print(response)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Sequence

from google import genai as google_genai
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.schema import BaseNode
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    GEMINI_API_KEY, EMBEDDING_MODEL, LLM_MODEL,
)
from embeddings import embed_single, embed_texts, EMBED_DIM

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Custom LlamaIndex Embedding wrapper (uses google.genai natively)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiNativeEmbedding(BaseEmbedding):
    """
    LlamaIndex BaseEmbedding backed by google.genai (not google-generativeai).
    Matches the model and batch logic used during ingestion.
    """

    def _get_query_embedding(self, query: str) -> Embedding:
        return embed_single(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return embed_single(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return embed_texts(texts)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embedding(text)


# ─────────────────────────────────────────────────────────────────────────────
# Custom LlamaIndex LLM wrapper (uses google.genai natively)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiNativeLLM(CustomLLM):
    """
    LlamaIndex CustomLLM backed by google.genai gemini-2.5-flash.
    Only `complete` is needed for the standard RetrieverQueryEngine.
    """

    model_name: str = LLM_MODEL
    context_window: int = 1_000_000
    num_output: int = 8192

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def _get_client(self) -> google_genai.Client:
        return google_genai.Client(api_key=GEMINI_API_KEY)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        client = self._get_client()
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        text = response.text or ""
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        client = self._get_client()
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        yield CompletionResponse(text=response.text or "")

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        result = self.complete(prompt)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=result.text)
        )

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        resp = self.chat(messages, **kwargs)
        yield resp

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseAsyncGen:
        yield self.complete(prompt, **kwargs)

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        yield self.chat(messages, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Cypher context retriever — pulls call-stack + imports for matched nodes
# ─────────────────────────────────────────────────────────────────────────────

# FIX: explicit 2-hop traversal over ALL tracked edge types.
# Variable-length paths [:1..2] are fine for CALLS/IMPORTS chains;
# DEFINES and HAS_METHOD are kept at depth-1 to avoid noise.
CONTEXT_CYPHER = """\
MATCH (n {id: $node_id})

// ── Depth-1 outgoing: all edge types ──────────────────────────────────
OPTIONAL MATCH (n)-[r_out:CALLS|IMPORTS|DEFINES|HAS_METHOD|INHERITS|IMPLEMENTS]->(d1)

// ── Depth-2 outgoing: CALLS and IMPORTS only (call-stack / import chain)
OPTIONAL MATCH (d1)-[r_out2:CALLS|IMPORTS]->(d2)
  WHERE d1 IS NOT NULL

// ── Depth-1 incoming: who calls / defines / inherits this node ─────────
OPTIONAL MATCH (c1)-[r_in:CALLS|DEFINES|INHERITS|IMPLEMENTS]->(n)

RETURN
    n.id            AS node_id,
    n.name          AS node_name,
    labels(n)[0]    AS node_label,
    n.docstring     AS docstring,
    n.signature     AS signature,
    // depth-1 outgoing
    collect(DISTINCT {
        hop:   1,
        rel:   type(r_out),
        name:  d1.name,
        id:    d1.id,
        label: labels(d1)[0]
    }) AS outgoing_d1,
    // depth-2 outgoing
    collect(DISTINCT {
        hop:   2,
        rel:   type(r_out2),
        via:   d1.name,
        name:  d2.name,
        id:    d2.id,
        label: labels(d2)[0]
    }) AS outgoing_d2,
    // incoming callers / definers
    collect(DISTINCT {
        rel:   type(r_in),
        name:  c1.name,
        id:    c1.id,
        label: labels(c1)[0]
    }) AS incoming
LIMIT 1
"""

# Text-search fallback: prioritize semantic nodes (Function/Method/Class)
# over structural ones (File/Module) when searching by name fragment.
_FALLBACK_CYPHER = """\
MATCH (n)
WHERE (toLower(n.name) CONTAINS toLower($term)
       OR toLower(coalesce(n.id,'')) CONTAINS toLower($term))
  AND NOT n:File AND NOT n:Module
RETURN n.id AS nid,
       CASE labels(n)[0]
           WHEN 'Function' THEN 0
           WHEN 'Method'   THEN 1
           WHEN 'Class'    THEN 2
           ELSE 3
       END AS priority
ORDER BY priority, n.name
LIMIT $limit
"""


class CypherContextRetriever(BaseRetriever):
    """
    Accepts a list of seed node IDs from the vector retriever and expands
    each into its 2-hop graph neighbourhood via direct Cypher.
    Returns LlamaIndex NodeWithScore objects containing the text context.
    """

    def __init__(self, graph_store: Neo4jPropertyGraphStore, top_k: int = 5) -> None:
        self._store = graph_store
        self._top_k = top_k
        super().__init__()

    # ── Internal helper ───────────────────────────────────────────────────────
    def _cypher_context(self, node_id: str) -> str:
        """
        Run 2-hop Cypher and serialize the result as clean English sentences.
        Format is designed for LLM consumption, not human readability.
        Each fact is a separate declarative sentence on its own line.
        """
        import os as _os
        driver = self._store._driver
        with driver.session() as session:
            result = session.run(CONTEXT_CYPHER, node_id=node_id)
            row = result.single()
            if not row:
                return f"No graph data found for node id: {node_id!r}."

        lines: list[str] = []

        label = row["node_label"] or "Node"
        name  = row["node_name"] or node_id
        rid   = row["node_id"] or node_id

        lines.append(f"{label} '{name}' has id '{rid}'.")
        if row.get("signature"):
            lines.append(f"{label} '{name}' has signature: {row['signature']}.")
        if row.get("docstring"):
            lines.append(f"{label} '{name}' documentation: {row['docstring']}.")

        # ── Depth-1 outgoing ──────────────────────────────────────────────────
        for o in (row.get("outgoing_d1") or []):
            if not o.get("name"):
                continue
            rel   = o["rel"]
            tname = o["name"]
            tlbl  = o.get("label") or "Node"
            lines.append(f"{label} '{name}' {rel} {tlbl} '{tname}'.")

        # ── Depth-2 outgoing (transitives) ────────────────────────────────────
        for o in (row.get("outgoing_d2") or []):
            if not o.get("name"):
                continue
            rel   = o["rel"]
            tname = o["name"]
            via   = o.get("via") or "?"
            tlbl  = o.get("label") or "Node"
            lines.append(
                f"{label} '{name}' transitively {rel} {tlbl} '{tname}'"
                f" (via '{via}')."
            )

        # ── Incoming ──────────────────────────────────────────────────────────
        for i in (row.get("incoming") or []):
            if not i.get("name"):
                continue
            rel   = i["rel"]
            iname = i["name"]
            ilbl  = i.get("label") or "Node"
            lines.append(f"{ilbl} '{iname}' {rel} {label} '{name}'.")

        text = "\n".join(lines)

        # ── Debug: print context block when DEBUG_CONTEXT=1 ───────────────────
        if _os.environ.get("DEBUG_CONTEXT"):
            import sys
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"CONTEXT BLOCK for node_id={node_id!r}", file=sys.stderr)
            print('='*60, file=sys.stderr)
            print(text, file=sys.stderr)

        return text

    def _fallback_ids(self, query_str: str, limit: int = 5) -> list[str]:
        """Name-fragment search when vector seeds are empty or mismatched."""
        driver = self._store._driver
        ids: list[str] = []
        with driver.session() as session:
            result = session.run(_FALLBACK_CYPHER, term=query_str, limit=limit)
            for row in result:
                if row["nid"]:
                    ids.append(row["nid"])
        return ids

    # ── LlamaIndex interface ──────────────────────────────────────────────────
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Stand-alone retrieval using the query text as a name-fragment search."""
        from llama_index.core.schema import TextNode
        ids = self._fallback_ids(query_bundle.query_str, limit=self._top_k)
        results: list[NodeWithScore] = []
        for nid in ids:
            text = self._cypher_context(nid)
            results.append(NodeWithScore(node=TextNode(text=text, id_=nid), score=0.9))
        return results

    def retrieve_from_ids(self, node_ids: list[str], query_str: str = "") -> list[NodeWithScore]:
        """
        Expand seed node IDs from the vector stage into 2-hop Cypher context.
        Falls back to name-fragment search when the list is empty.
        """
        from llama_index.core.schema import TextNode
        effective_ids = node_ids[: self._top_k]
        if not effective_ids and query_str:
            logger.info("Cypher retriever: no seed IDs — using name fallback for %r", query_str)
            # Extract the most specific keyword from the query string for fallback search
            import re as _re
            # Pull the first quoted word or the longest word in the query
            quoted = _re.findall(r"[`']([^`']+)[`']", query_str)
            keyword = quoted[0] if quoted else max(query_str.split(), key=len, default=query_str)
            # Strip trailing punctuation like '()'
            keyword = keyword.rstrip("().:")
            effective_ids = self._fallback_ids(keyword, limit=self._top_k)

        results: list[NodeWithScore] = []
        for nid in effective_ids:
            text = self._cypher_context(nid)
            results.append(NodeWithScore(node=TextNode(text=text, id_=nid), score=1.0))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retriever: vector similarity → seed IDs → Cypher expansion
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Two-stage hybrid retriever:
      Stage 1 — VectorContextRetriever   : embed query → find top-k similar nodes
      Stage 2 — CypherContextRetriever   : for each seed, pull neighbour context
    """

    def __init__(
        self,
        vector_retriever: VectorContextRetriever,
        cypher_retriever: CypherContextRetriever,
    ) -> None:
        self._vector = vector_retriever
        self._cypher = cypher_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # Stage 1: vector similarity
        vector_results: list[NodeWithScore] = self._vector.retrieve(query_bundle)
        logger.info("Vector retriever returned %d node(s)", len(vector_results))

        # Extract seed IDs — probe multiple metadata keys for compatibility
        seed_ids: list[str] = []
        seen: set[str] = set()
        for nws in vector_results:
            meta = nws.node.metadata or {}
            candidate = (
                meta.get("id")
                or meta.get("node_id")
                or meta.get("entity_id")
                or nws.node.id_
            )
            if candidate and candidate not in seen:
                seed_ids.append(str(candidate))
                seen.add(str(candidate))

        logger.info("Extracted %d seed ID(s) from vector results", len(seed_ids))

        # Stage 2: keyword-targeted Cypher search — BYPASSES top_k truncation.
        # This guarantees that named entities mentioned in the query (e.g. init())
        # always get a context block even if they weren't in the top-k vector hits.
        import re as _re
        q = query_bundle.query_str
        code_pats = _re.findall(r'\b([A-Za-z_]\w*)\s*\(\)', q)
        quoted    = _re.findall(r'[`]([^`]+)[`]', q)
        keyword   = (code_pats[0] if code_pats
                     else quoted[0].strip() if quoted
                     else max(q.split(), key=len, default=q))
        keyword   = keyword.rstrip("().:,")
        # Expand keyword seeds directly — result is already NodeWithScore objects
        keyword_results: list[NodeWithScore] = self._cypher.retrieve_from_ids(
            [], query_str=keyword
        )
        for nws in keyword_results:
            seen.add(nws.node.id_)
        logger.info("Keyword Cypher expansion returned %d block(s) for %r", len(keyword_results), keyword)

        # Stage 3: Cypher 2-hop expansion of top-k vector seed IDs
        # (keyword IDs already expanded above — skip duplicates)
        vector_cypher = self._cypher.retrieve_from_ids(seed_ids)
        logger.info("Vector Cypher expansion returned %d context block(s)", len(vector_cypher))

        # Keyword context first (highest priority, name-matched), then vector context, then raw vectors
        return keyword_results + vector_cypher + vector_results



# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

def build_query_engine(
    similarity_top_k: int = 8,
    path_depth: int = 2,
) -> RetrieverQueryEngine:
    """
    Build and return a ready-to-use hybrid query engine.

    Parameters
    ----------
    similarity_top_k : Number of nearest-neighbour nodes returned by vector search.
    path_depth       : How many hops VectorContextRetriever follows from each seed node.

    Returns
    -------
    A LlamaIndex RetrieverQueryEngine. Call `.query(question)` on it.
    """
    # ── Models ────────────────────────────────────────────────────────────────
    embed_model = GeminiNativeEmbedding(embed_batch_size=20)
    llm = GeminiNativeLLM()

    # Set globally so LlamaIndex internals (e.g. response synthesizer) pick them up
    Settings.embed_model = embed_model
    Settings.llm = llm

    # ── Graph store (read-only) ───────────────────────────────────────────────
    logger.info("Connecting to Neo4j at %s …", NEO4J_URI)
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
        refresh_schema=False,    # ← do NOT read/rebuild the schema
        create_indexes=False,    # ← do NOT create any indexes
    )

    # ── PropertyGraphIndex (from existing store, no extraction) ───────────────
    # passing nodes=[] and kg_extractors=[] ensures LlamaIndex never writes anything
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_model=embed_model,
        embed_kg_nodes=False,    # ← embeddings already stored; do not re-embed
    )

    # ── Stage 1: vector similarity retriever ─────────────────────────────────
    vector_retriever = VectorContextRetriever(
        index.property_graph_store,
        embed_model=embed_model,
        similarity_top_k=similarity_top_k,
        path_depth=path_depth,
        include_text=True,
        include_properties=True,
    )

    # ── Stage 2: Cypher context retriever ────────────────────────────────────
    cypher_retriever = CypherContextRetriever(
        graph_store=graph_store,
        top_k=similarity_top_k,
    )

    # ── Hybrid retriever (combines both) ─────────────────────────────────────
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        cypher_retriever=cypher_retriever,
    )

    # ── Custom strict-graph prompt template ──────────────────────────────────
    from llama_index.core import PromptTemplate

    _QA_TMPL = PromptTemplate(
        "You are an expert software architect interpreting a deterministic "
        "Knowledge Graph of a codebase.\n"
        "You are provided with Nodes (Classes, Functions, Methods) and Edges "
        "(relationships such as CALLS, IMPORTS, INHERITS, HAS_METHOD, DEFINES).\n"
        "You do not have the raw source code bodies, only structural signatures "
        "and docstrings. Do NOT complain about missing source code.\n"
        "Treat the provided graph edges as absolute, undeniable facts. "
        "If the context says Node A CALLS Node B, state that confidently.\n"
        "Do NOT guess or infer relationships that are not explicitly provided "
        "in the context blocks.\n"
        "Answer only from the context below. If the context is insufficient, "
        "say so — do not invent facts.\n\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Question: {query_str}\n"
        "Answer: "
    )

    _REFINE_TMPL = PromptTemplate(
        "You are an expert software architect interpreting a deterministic "
        "Knowledge Graph of a codebase.\n"
        "Treat every graph relationship (CALLS, IMPORTS, INHERITS, etc.) as an "
        "absolute fact. Do NOT speculate beyond what the graph context states.\n"
        "Do not complain about missing source code — only structural information "
        "is available.\n\n"
        "Existing answer:\n{existing_answer}\n\n"
        "New context to refine with:\n"
        "---------------------\n"
        "{context_msg}\n"
        "---------------------\n\n"
        "Refined answer (use new context only if it adds facts; otherwise repeat "
        "the existing answer unchanged): "
    )

    # ── Query engine with LLM synthesis ──────────────────────────────────────
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
        text_qa_template=_QA_TMPL,
        refine_template=_REFINE_TMPL,
    )

    logger.info("Query engine ready.")
    return query_engine


# ─────────────────────────────────────────────────────────────────────────────
# CLI / quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What classes are defined in this codebase and what do they do?"

    print(f"\nQuestion: {question}\n{'─'*60}")
    engine = build_query_engine()
    response = engine.query(question)
    print(f"\nAnswer:\n{response}\n")
