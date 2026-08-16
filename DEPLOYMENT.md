# Deployment Guide

## Services Containerization

- **Database**: Neo4j 5 Community Edition (`docker-compose.yml`)
- **Model Server**: Local Ollama serving `qwen2.5:3b`
- **Backend API**: FastAPI (`src/api/main.py`)
- **Frontend App**: React + Vite (`frontend/`)

## Launch Command

```bash
docker compose up -d
python scripts/init_neo4j.py
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
