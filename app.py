"""
app.py  –  GraphRAG Codebase Explorer  (Streamlit UI)
══════════════════════════════════════════════════════
Panels
  Left Sidebar  — GitHub URL input → full ingestion pipeline with live progress
  Main / Chat   — Hybrid RAG chat over the ingested graph (session history kept)
  Main / Graph  — Pyvis interactive visualisation of the Neo4j graph
"""

from __future__ import annotations

import html as html_mod
import io
import logging
import textwrap
import threading
from typing import Optional

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="GraphRAG Codebase Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS (dark neo4j-inspired palette) ──────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #58a6ff;
}

/* ── headings ── */
h1, h2, h3 { color: #58a6ff; }

/* ── tabs ── */
[data-testid="stTabs"] [role="tab"] {
    color: #8b949e;
    font-weight: 600;
    border-radius: 6px 6px 0 0;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.45rem 1.2rem;
    transition: filter 0.2s;
}
.stButton > button:hover { filter: brightness(1.15); }

/* ── chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 10px;
    margin-bottom: 0.6rem;
}

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] { color: #58a6ff; font-size: 1.8rem; }

/* ── text input ── */
.stTextInput > div > div > input {
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging redirect: capture ingest logs into a StringIO buffer for display
# ─────────────────────────────────────────────────────────────────────────────
class _StreamlitLogHandler(logging.Handler):
    """Write log records into an in-memory buffer (thread-safe)."""
    def __init__(self):
        super().__init__()
        self._buf: list[str] = []
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        with self._lock:
            self._buf.append(self.format(record))

    def flush_lines(self) -> list[str]:
        with self._lock:
            lines, self._buf = self._buf[:], []
        return lines


_LOG_HANDLER = _StreamlitLogHandler()
_LOG_HANDLER.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
logging.getLogger().addHandler(_LOG_HANDLER)
logging.getLogger().setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],          # list of {"role": "user"|"assistant", "content": str}
        "query_engine": None,    # cached LlamaIndex query engine
        "ingested_repo": None,   # last ingested repo URL/path
        "graph_html": None,      # cached pyvis HTML string
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Neo4j helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_neo4j_driver():
    from neo4j import GraphDatabase
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _fetch_node_counts() -> dict[str, int]:
    try:
        driver = _get_neo4j_driver()
        with driver.session() as s:
            result = s.run("MATCH (n) WHERE NOT n:__Entity__ OR n:__Entity__ "
                           "RETURN labels(n)[0] AS lbl, count(*) AS cnt")
            counts = {r["lbl"]: r["cnt"] for r in result if r["lbl"]}
        driver.close()
        return counts
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Ingestion panel
# ─────────────────────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        st.markdown("## 🕸️ GraphRAG Explorer")
        st.caption("Ingest any GitHub repo, then chat with its codebase.")
        st.divider()

        # ── Ingestion form ────────────────────────────────────────────────────
        st.markdown("### ⚙️ Ingest Repository")
        repo_url = st.text_input(
            "GitHub URL or local path",
            placeholder="https://github.com/owner/repo",
            key="repo_url_input",
            label_visibility="collapsed",
        )

        ext_help = st.expander("📁 File type filters (optional)")
        with ext_help:
            col1, col2 = st.columns(2)
            with col1:
                do_py  = st.checkbox(".py",  value=True)
                do_js  = st.checkbox(".js",  value=True)
                do_ts  = st.checkbox(".ts",  value=True)
            with col2:
                do_go  = st.checkbox(".go",  value=True)
                do_cpp = st.checkbox(".cpp", value=True)
                do_h   = st.checkbox(".h",   value=True)

        ingest_clicked = st.button("🚀 Ingest", use_container_width=True, key="ingest_btn")

        if ingest_clicked:
            if not repo_url.strip():
                st.warning("Please enter a GitHub URL or local path.")
            else:
                _run_ingestion(
                    repo_url.strip(),
                    extensions={ext for ext, flag in [
                        (".py", do_py), (".js", do_js), (".ts", do_ts),
                        (".go", do_go), (".cpp", do_cpp), (".h", do_h),
                        (".hpp", do_cpp), (".cc", do_cpp),
                    ] if flag} or None,
                )

        # ── Graph stats ───────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📊 Graph Stats")
        counts = _fetch_node_counts()
        if counts:
            for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                if label and label != "__Entity__":
                    st.metric(label=label, value=cnt, label_visibility="visible")
        else:
            st.caption("No graph data yet — ingest a repo first.")

        st.divider()
        st.caption("v1.0 · Neo4j 5 · Gemini 2.5 Flash · LlamaIndex")


def _run_ingestion(source: str, extensions: set[str] | None):
    """Run the full ingestion pipeline with live status updates."""
    from ingest import clone_or_use_local, parse_repo, attach_embeddings, write_to_neo4j

    # Invalidate caches
    st.session_state.query_engine = None
    st.session_state.graph_html = None

    exts = frozenset(extensions) if extensions else None

    with st.sidebar:
        with st.status("🔄 Ingesting repository…", expanded=True) as status:
            try:
                st.write("📥 Cloning repo…")
                repo_path, is_temp = clone_or_use_local(source)
                st.write(f"✅ Cloned to `{repo_path}`")

                st.write("🔍 Parsing source files…")
                nodes, edges = parse_repo(repo_path, exts)
                st.write(f"✅ Extracted **{len(nodes)}** nodes, **{len(edges)}** edges")

                st.write("🧠 Generating embeddings…")
                attach_embeddings(nodes)
                st.write("✅ Embeddings attached")

                st.write("💾 Writing to Neo4j…")
                write_to_neo4j(nodes, edges)
                st.write("✅ Graph written (idempotent MERGE)")

                import shutil
                if is_temp:
                    shutil.rmtree(repo_path, ignore_errors=True)

                st.session_state.ingested_repo = source
                status.update(label="✅ Ingestion complete!", state="complete", expanded=False)
                st.success(f"Ingested **{len(nodes)}** nodes from `{source}`")

            except Exception as exc:
                status.update(label="❌ Ingestion failed", state="error", expanded=True)
                st.error(f"Error: {exc}")
                logging.exception("Ingestion failed")


# ─────────────────────────────────────────────────────────────────────────────
# Query engine — lazy cached loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_query_engine():
    from query_engine import build_query_engine
    return build_query_engine()


def _get_query_engine():
    if st.session_state.query_engine is None:
        with st.spinner("⚡ Initializing query engine…"):
            st.session_state.query_engine = _load_query_engine()
    return st.session_state.query_engine


# ─────────────────────────────────────────────────────────────────────────────
# Chat tab
# ─────────────────────────────────────────────────────────────────────────────
def _render_chat_tab():
    st.markdown("### 💬 Chat with your Codebase")

    # Render existing history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # New message input
    if prompt := st.chat_input("Ask anything about the ingested codebase…", key="chat_input"):
        # Show user bubble immediately
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get answer
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    engine = _get_query_engine()
                    response = engine.query(prompt)
                    answer = str(response)
                except Exception as exc:
                    answer = f"⚠️ Query failed: {exc}"
                    logging.exception("Query error")

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    # Clear history button
    if st.session_state.messages:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Graph visualisation tab (Pyvis)
# ─────────────────────────────────────────────────────────────────────────────
# Colour palette per node label
_LABEL_COLOURS = {
    "File":       "#388bfd",
    "Class":      "#56d364",
    "Function":   "#f2cc60",
    "Method":     "#db6d28",
    "Module":     "#bc8cff",
    "Struct":     "#2ea043",
    "Interface":  "#79c0ff",
    "TypeAlias":  "#ff7b72",
}
_DEFAULT_COLOUR = "#8b949e"

# Edge colour per relationship type
_REL_COLOURS = {
    "DEFINES":    "#30363d",
    "HAS_METHOD": "#21262d",
    "CALLS":      "#f2cc60",
    "IMPORTS":    "#388bfd",
    "INHERITS":   "#56d364",
    "IMPLEMENTS": "#2ea043",
    "CONTAINS":   "#21262d",
}


def _build_pyvis_html(limit: int = 300) -> str:
    """Query Neo4j and build interactive Pyvis network HTML."""
    from pyvis.network import Network
    from neo4j import GraphDatabase
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    net = Network(
        height="620px", width="100%",
        bgcolor="#0d1117", font_color="#e6edf3",
        directed=True,
        notebook=False,
    )
    net.set_options("""{
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 120,
                "springConstant": 0.04,
                "damping": 0.09
            },
            "stabilization": {"iterations": 150}
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
            "color": {"inherit": false},
            "width": 1.5
        },
        "nodes": {
            "shape": "dot",
            "size": 16,
            "font": {"size": 12, "face": "Inter"}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        }
    }""")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    added_nodes: set = set()
    added_edges: set = set()

    try:
        with driver.session() as s:
            # Fetch nodes (exclude bare __Entity__ label)
            node_result = s.run(
                "MATCH (n) WHERE NOT n:__Entity__ OR any(l IN labels(n) WHERE l <> '__Entity__') "
                "RETURN id(n) AS nid, labels(n) AS lbls, "
                "       coalesce(n.name, n.path, n.id) AS name, "
                "       coalesce(n.docstring, n.signature, '') AS tip "
                f"LIMIT {limit}"
            )
            for row in node_result:
                nid   = row["nid"]
                lbls  = [l for l in row["lbls"] if l != "__Entity__"]
                label = lbls[0] if lbls else "Unknown"
                name  = str(row["name"] or "").split("::")[-1] or str(nid)
                tip   = textwrap.shorten(str(row["tip"] or ""), width=120)
                colour = _LABEL_COLOURS.get(label, _DEFAULT_COLOUR)

                if nid not in added_nodes:
                    net.add_node(
                        nid, label=name, title=f"[{label}] {name}\n{tip}",
                        color=colour, group=label,
                    )
                    added_nodes.add(nid)

            # Fetch edges
            edge_result = s.run(
                "MATCH (a)-[r]->(b) "
                "WHERE NOT a:__Entity__ OR NOT b:__Entity__ "
                "    OR any(l IN labels(a) WHERE l <> '__Entity__') "
                "RETURN id(a) AS src, id(b) AS tgt, type(r) AS rel "
                f"LIMIT {limit * 3}"
            )
            for row in edge_result:
                src, tgt, rel = row["src"], row["tgt"], row["rel"]
                key = (src, tgt, rel)
                if key not in added_edges and src in added_nodes and tgt in added_nodes:
                    net.add_edge(
                        src, tgt, title=rel, label=rel,
                        color=_REL_COLOURS.get(rel, "#30363d"),
                        font={"size": 9, "color": "#8b949e"},
                    )
                    added_edges.add(key)

    finally:
        driver.close()

    return net.generate_html()


def _render_graph_tab():
    st.markdown("### 🌐 Graph Visualiser")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        node_limit = st.slider(
            "Max nodes to display", 50, 500, 300, 50, key="graph_limit"
        )
    with col_b:
        refresh = st.button("🔄 Refresh graph", key="refresh_graph", use_container_width=True)
    with col_c:
        st.markdown("")   # spacer

    # Legend
    legend_md = "  ".join(
        f'<span style="color:{c}">●</span> {l}'
        for l, c in _LABEL_COLOURS.items()
    )
    st.markdown(legend_md, unsafe_allow_html=True)
    st.divider()

    if refresh or st.session_state.graph_html is None:
        with st.spinner("Building graph visualisation…"):
            try:
                st.session_state.graph_html = _build_pyvis_html(limit=node_limit)
            except Exception as exc:
                st.error(f"Graph build error: {exc}")
                return

    if st.session_state.graph_html:
        import streamlit.components.v1 as components
        # Inject dark background into the pyvis iframe HTML
        graph_html = st.session_state.graph_html.replace(
            "<body>",
            '<body style="background:#0d1117; margin:0; padding:0;">'
        )
        components.html(graph_html, height=640, scrolling=False)
    else:
        st.info("No graph data to display. Ingest a repository first.")


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _render_sidebar()

    # Header
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0.2rem;'>"
        "🕸️ GraphRAG Codebase Explorer</h1>"
        "<p style='text-align:center; color:#8b949e; margin-top:0;'>"
        "Ingest any GitHub repository · Ask questions · Explore the knowledge graph</p>",
        unsafe_allow_html=True,
    )

    # Status banner when a repo has been ingested
    if st.session_state.ingested_repo:
        st.success(
            f"📦 Active repo: **{st.session_state.ingested_repo}**  "
            "— Query engine ready.",
            icon="✅",
        )

    st.divider()

    # Tabs
    tab_chat, tab_graph = st.tabs(["💬 Chat", "🌐 Graph"])

    with tab_chat:
        _render_chat_tab()

    with tab_graph:
        _render_graph_tab()


if __name__ == "__main__":
    main()
