# Ingestion Pipeline Guide

## Document Ingestion Flow

1. **Extraction**: PDF parsed via PyMuPDF (`fitz`), text and page numbers extracted. SHA-256 file hash recorded.
2. **Normalization**: PDF hyphenation fixed, line breaks normalized, repeated headers/footers stripped. Verbatim legal text preserved.
3. **Chunking**: Structure-aware chunking based on document type (Acts split by Section/Subsection, Judgments split by Page/Paragraph). Neighboring chunk IDs linked.
4. **Extraction**: Layer A regex entity and relation extraction (`MENTIONS`, `DECIDED_BY`, `CITES`). Layer B model extraction flagged for review.
5. **Graph Store**: Idempotent `MERGE` statements execute on Neo4j.
6. **Embeddings**: Vector embeddings generated via `sentence-transformers` (`BAAI/bge-m3`) and stored on `Chunk.embedding`.
