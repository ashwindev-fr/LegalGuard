# LegalGuard — Project Handover & Detailed Progress Summary

**Project**: Explainable Indian Law Research Assistant using Neo4j GraphRAG + Local SLM (Qwen2.5-3B)  
**Date**: August 16, 2026  
**Status**: Fully Functional (Backend API + React Frontend + Neo4j Graph Integration + Ingestion Pipeline)

---

## 1. Executive Summary

LegalGuard is an explainable AI legal research assistant designed specifically for Indian Law. It uses a **Neo4j Knowledge Graph** to store legal documents (Acts, Sections, Articles, Cases, Precedents, Courts, Legal Principles), a hybrid vector + graph retrieval engine (`BAAI/bge-m3` embeddings), and a local Small Language Model (Qwen2.5-3B via Ollama) to generate evidence-grounded answers with deterministic legal citations.

---

## 2. Work Completed in This Session

### A. Environment & Module Resolution Fixes
- **Resolved Pyright/Pylance Import Error (`Cannot find module src.ingestion.validation.validator`)**:
  - **Root Cause**: Pyright auto-inferred `src/` as the primary import root, causing `from src.<module>` import statements to fail static analysis by looking for `src/src/...`. Additionally, `pyproject.toml` contained an obsolete build-backend string (`setuptools.backends._legacy:_Backend`).
  - **Fix Implemented**:
    1. Updated `pyproject.toml` build backend to `setuptools.build_meta`.
    2. Added `[tool.pyright]` configuration with `extraPaths = ["."]` in `pyproject.toml`.
    3. Re-installed the package in editable mode into `.venv` using `pip install --no-build-isolation -e .`.

### B. Backend API Enhancements (`src/api/main.py`)
- **Direct HTTP File Upload & Ingestion Endpoint (`POST /ingest/upload`)**:
  - Accepts `UploadFile` (`multipart/form-data`) uploads directly from web browsers/clients.
  - Parameters: `file`, `document_type` (`ACT`, `CONSTITUTION`, `JUDGMENT`, `NOTIFICATION`, `CIRCULAR`), `title` (optional), `source_id`, `authority_level`, `dry_run`.
  - Saves file temporarily to `data/temp_uploads/`, invokes `ManualImporter.ingest_file()`, and cleans up temp files automatically.
  - Returns detailed JSON response with counts of created Document nodes, text chunks, extracted entities, and merged relationships.
- **Graph Sampling Endpoint (`GET /graph/sample`)**:
  - Queries Neo4j for sample graph nodes and relationships (`MATCH (n)-[r]->(m)`).
  - Returns formatted `{nodes: [{id, label, title}], links: [{source, target, type}]}` for frontend visual rendering.

### C. Frontend UI & Component Development (`frontend/src/`)
- **Navigation Tabs (`App.tsx`)**:
  - Added a top navigation bar allowing users to switch between **Research & Q&A**, **Ingest Document**, and **Knowledge Graph**.
- **Drag-and-Drop Document Uploader (`frontend/src/components/DocumentUpload.tsx`)**:
  - Interactive dropzone for uploading `.pdf`, `.txt`, and `.md` files.
  - Form options for Document Type, custom Document Title, Authority Level slider, and Dry-Run simulation toggle.
  - Real-time loading indicator and detailed summary card showing document ID, chunks created, entities extracted, and relationships merged into Neo4j.
- **Interactive Knowledge Graph Visualizer (`frontend/src/components/GraphVisualizer.tsx`)**:
  - Real-time Neo4j entity label stats badges powered by `/graph/stats`.
  - Interactive SVG force-directed node-link graph with color-coded entity badges (`Document`, `Act`, `Section`, `Case`, `Article`, `Court`, `Judge`, `Chunk`, `LegalPrinciple`).
  - Interactive Node Inspector panel allowing users to select any node or search an entity ID to inspect its 2-hop graph neighborhood (`/graph/neighborhood/{entity_id}`).
- **CSS Design System (`frontend/src/index.css`)**:
  - Added modern glassmorphic styles for navbar tabs, dropzones, upload result cards, metric badges, and SVG graph canvas.

---

## 3. Key File Locations & Architecture

| Module / File | Description |
| :--- | :--- |
| **`src/api/main.py`** | FastAPI application with `/answer`, `/ingest/upload`, `/ingest/document`, `/retrieve`, `/graph/stats`, `/graph/sample`, and `/graph/neighborhood/{entity_id}`. |
| **`src/ingestion/sources/manual_import.py`** | Ingestion core logic: PDF/Text parsing, text normalization, structure-aware chunking, entity/relation extraction, and Neo4j MERGE. |
| **`src/graph/repository.py`** | Neo4j Cypher query abstractions for idempotent MERGE writes. |
| **`src/retrieval/`** | Hybrid graph + vector retrieval implementation using `bge-m3` embeddings. |
| **`frontend/src/App.tsx`** | Main React application with tab routing. |
| **`frontend/src/components/DocumentUpload.tsx`** | Drag-and-drop document upload UI component. |
| **`frontend/src/components/GraphVisualizer.tsx`** | Interactive knowledge graph SVG visualizer. |
| **`frontend/src/index.css`** | Complete CSS design system and tokens. |
| **`pyproject.toml`** | Python project dependencies, build system, mypy, ruff, and pyright configs. |

---

## 4. How to Run the Application

### 1. Prerequisites / Services
- **Ollama**: Running locally with `ollama serve` (serving `qwen2.5:3b`).
- **Neo4j**: Neo4j database running on `bolt://localhost:7687` (default user `neo4j`, password `password`).

### 2. Start Backend API
```bash
# In project root:
.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --port 8000
```

### 3. Start Frontend UI
```bash
# In frontend/ directory:
cd frontend
npm run dev
# App will run at http://localhost:5173
```

---

## 5. Verification Performed

1. **Python Imports**: Verified that `import src.ingestion.validation.validator` works cleanly in `.venv`.
2. **TypeScript Compilation**: Executed `npx tsc --noEmit` inside `frontend/` — passed with **0 errors**.
3. **API & Server Launch**: Verified FastAPI startup without errors.

---

## 6. Suggested Next Steps for the Next Agent / Developer

1. **Hot-Folder Auto-Ingestion Daemon (`scripts/watch_and_ingest.py`)**:
   - Create a background script monitoring `data/incoming/` using `watchfiles` or `watchdog` to automatically process and ingest any PDF dropped into that folder.
2. **Run Evaluation & Validation Benchmarks**:
   - Run `python scripts/validate_dataset.py` to check dataset graph connectivity.
   - Run `python scripts/run_evaluation.py` to evaluate retrieval precision/recall and citation faithfulness.
3. **Fine-Tuning Local SLM**:
   - Execute fine-tuning pipeline as documented in `FINE_TUNING.md` for domain adaptation of Qwen2.5 on Indian Law Q&A datasets.
