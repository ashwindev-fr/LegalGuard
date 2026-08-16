# Troubleshooting Guide

## Common Issues & Solutions

### 1. Neo4j Connection Refused
- Ensure container is running: `docker compose ps`
- Check Bolt port 7687 and HTTP port 7474.
- Verify credentials in `.env` match `docker-compose.yml`.

### 2. Ollama Connection Error
- Ensure Ollama daemon is running: `ollama serve`
- Pull model: `ollama pull qwen2.5:3b`

### 3. Missing Vector Index Error
- Run initialization: `python scripts/init_neo4j.py`
- Rebuild embeddings: `python scripts/rebuild_embeddings.py`

### 4. Windows Path Issues
- Always run scripts from project root.
- Ensure virtual environment is activated (`.venv\Scripts\Activate.ps1`).
