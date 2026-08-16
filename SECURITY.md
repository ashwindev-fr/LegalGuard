# Security & Privacy Specification

## Data & Prompt Injection Defenses

1. **Retrieved Document Sanitization**: Retrieved legal texts are marked strictly as untrusted data, never instructions.
2. **Read-Only Database Operations**: Dynamic user query generation via Text2Cypher is sandboxed and limited to read-only Cypher.
3. **No Secret Leaks**: API credentials and secrets stored in `.env` and excluded via `.gitignore`.
