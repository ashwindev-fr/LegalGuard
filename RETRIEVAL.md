# Hybrid Retrieval Specification

## Multi-Channel Retrieval

1. **Vector Channel**: Cosine similarity search over Neo4j vector index (`chunk_embedding`) using `BAAI/bge-m3` query embeddings.
2. **Lexical Channel**: Neo4j full-text search index (`chunk_fulltext`, `case_fulltext`, `section_fulltext`) for exact section numbers, article numbers, case names, and citations.
3. **Graph Channel**: Entity traversal for section interpretation networks, precedent citation chains (`CITES`, `FOLLOWS`, `OVERRULES`), and temporal bounds.

## Weighted Candidate Scoring

Candidates are merged, deduplicated by `chunk_id`, and scored via:

$$Score = w_1 \cdot \text{Sim}_{\text{vector}} + w_2 \cdot \text{Score}_{\text{lexical}} + w_3 \cdot \text{Score}_{\text{graph}} + w_4 \cdot \text{Score}_{\text{reranker}} + w_5 \cdot \text{Auth} + w_6 \cdot \text{Temp}$$
