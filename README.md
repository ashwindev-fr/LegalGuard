# Explainable Small Language Model for Indian Law Using Temporal, Evidence-Grounded GraphRAG

**Short Name:** `IndianLaw-Explainable-GraphRAG`

An AI legal research assistant for Indian law built with a Neo4j legal knowledge graph, hybrid retrieval (vector + lexical + graph), evidence validation, deterministic claim-level citations, Qwen2.5-3B SLM, and a hallucination evaluation benchmark.

---

## 🏛️ Architecture Overview

```text
                  OFFICIAL LEGAL SOURCES
                             |
            +----------------+----------------+
            |                |                |
       Constitution        Acts          Judgments
            |                |                |
            +----------------+----------------+
                             |
                             v
                   DOCUMENT INGESTION
                             |
                 +-----------+-----------+
                 |                       |
          Legal structure          Text chunks
          extraction              + metadata
                 |                       |
                 v                       v
          Entity/relation          Embeddings
          extraction                    |
                 |                       |
                 +-----------+-----------+
                             |
                             v
                           NEO4J
                +------------+-------------+
                |                          |
         Knowledge Graph            Vector Index
                |                          |
                +------------+-------------+
                             |
                             v
                         USER QUERY
                             |
                             v
                      QUERY ANALYZER
                             |
               +-------------+-------------+
               |                           |
         VECTOR SEARCH                 GRAPH SEARCH
               |                           |
               +-------------+-------------+
                             |
                             v
                         MERGE & RERANK
                             |
                             v
                    EVIDENCE VALIDATOR
                             |
                             v
                    QWEN2.5-3B SLM
                             |
                             v
                   CLAIM & CITATION CHECK
                             |
               +-------------+-------------+
               |                           |
         EXPLAINED ANSWER              SOURCES
         + confidence               + exact links
         + caveats                  + page/para
```

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm
- Docker & Docker Compose
- Ollama (for serving Qwen2.5-3B locally)

### 1. Start Neo4j Database
```bash
docker compose up -d
```
Neo4j Browser will be available at `http://localhost:7474` (Credentials: `neo4j` / `change_me`).

### 2. Set Up Python Environment
```bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 3. Initialize Graph Schema & Ingest Demo Data
```bash
# Initialize graph constraints, vector index, and full-text indexes
python scripts/init_neo4j.py

# Ingest sample Indian legal corpus (Constitution Art 21, BNS Sec 103, Contract Act Sec 10, Maneka Gandhi judgment)
python scripts/ingest_batch.py --dir data/raw/demo --type ACT

# Compute embeddings for ingested chunks
python scripts/rebuild_embeddings.py
```

### 4. Start Local SLM (Ollama)
```bash
ollama run qwen2.5:3b
```

### 5. Start FastAPI Backend
```bash
uvicorn src.api.main:app --reload --port 8000
```
API Documentation: `http://localhost:8000/docs`

### 6. Start React Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## 🧪 Evaluation Benchmark

Run evaluation across the 4 systems (Base SLM, Vector RAG, GraphRAG, Fine-tuned GraphRAG):

```bash
python scripts/run_evaluation.py --system all
```

---

## ⚖️ Legal Disclaimer

This system provides AI-assisted legal information and research support based on retrieved sources. It is not a lawyer, does not provide guaranteed legal advice, and should not replace advice from a qualified legal professional. Always verify important legal claims against current authoritative sources.
