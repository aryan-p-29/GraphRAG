# GraphRAG Codebase Explorer

GraphRAG Codebase Explorer is a comprehensive tool for exploring and querying codebases using a combination of **Graph Retrieval-Augmented Generation (GraphRAG)**, **Tree-sitter AST parsing**, and **Neo4j** graph databases.

It allows developers to deeply understand repositories by mapping out cross-file semantic relationships (e.g., function calls, class instantiations, and module imports) and providing a hybrid search interface powered by the Gemini Large Language Model.

---

## 🌟 Key Features

* **Multi-Language AST Parsing**: Uses `tree-sitter` to parse Python, JavaScript/TypeScript, Go, and C++ codebases, extracting semantic nodes (`Function`, `Class`, `Method`, `Module`) and their execution paths.
* **Intelligent Graph Resolution**: Handles global fuzzy resolution across files, correctly linking class instantiations to their `__init__` routines and tracing complex execution paths.
* **Hybrid Retrieval System**: Combines vector similarity search (using Gemini Embeddings) with multi-hop Cypher traversal through the Neo4j graph.
* **FastAPI Backend**: A lightweight, fast, and concurrent API server to handle repository ingestion via SSE streams and direct hybrid querying.
* **Modern React + Vite Frontend**: A rich web interface featuring:
  * **Interactive Force-Directed Graph**: Visualize the codebase architecture dynamically using `react-force-graph-2d`.
  * **Live Ingestion Logging**: Stream real-time logs directly to the sidebar as the repository is processed.
  * **Markdown Chat**: Chat with the LLM and receive highlighted code blocks and precise source citations.

---

## 🏗️ Architecture

1. **Ingestion (`ingest.py`)**: Walks through a target directory, selects files based on extensions, parses them into AST nodes/edges, generates vector embeddings for code blocks using Gemini, and pushes the data to Neo4j.
2. **Retrieval Engine (`query_engine.py`)**: Uses LlamaIndex to query Neo4j. It extracts keyword code patterns from user questions, queries the vector index, and fetches 2-3 hop graph neighborhoods via Cypher queries.
3. **API Layer (`api.py`)**: Fast HTTP/SSE endpoints bridging the python logic to the web frontend.
4. **Web UI (`frontend/`)**: React application containing the chat interface and canvas graph visualization.

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.10+**
* **Node.js** (for building/running the frontend)
* **Docker** (to run the Neo4j database)
* **Gemini API Key** (for embeddings and LLM)

### 1. Environment Setup

Copy `.env.example` to `.env` and fill in your Gemini API key:
```bash
cp .env.example .env
```
Ensure your `.env` looks like this:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
GEMINI_API_KEY=your_actual_api_key_here
```

### 2. Start the Neo4j Database

Run Neo4j locally using the provided Docker Compose file:
```bash
docker-compose up -d
```
*(Wait a moment for Neo4j to fully boot up)*

### 3. Start the Backend (FastAPI)

In a new terminal, activate your virtual environment (if you have one), install requirements, and run Uvicorn:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```
*The backend API will run on http://localhost:8000*

### 4. Start the Frontend (React / Vite)

In a separate terminal, navigate to the `frontend/` directory, install Node dependencies, and start the Vite dev server:
```bash
cd frontend
npm install
npm run dev
```
*The frontend will start. Check the terminal output for the local URL (typically http://localhost:5173).*

---

## 📖 Usage

1. Open the frontend in your browser.
2. **Ingest a Repository**: In the left sidebar, enter the absolute local path to a repository (e.g., `/home/user/my_project`), check the file extensions you want to parse, and click **Start Ingestion**. You will see live logs streaming as the graph builds.
3. **Graph Visualization**: Once ingestion completes, click the **Graph** tab to explore the structure and relationships of the parsed codebase.
4. **Chat & Query**: Go to the **Chat** tab and ask complex questions about execution paths. The engine will retrieve relevant graph contexts and source codes to answer accurately!
