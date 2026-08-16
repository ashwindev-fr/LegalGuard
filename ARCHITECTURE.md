# Architecture & Subsystem Specification

## System Modules

1. **Ingestion Engine (`src/ingestion/`)**
   - PDF Parsing (`fitz`/PyMuPDF)
   - Legal Text Normalization (preserves verbatim legal text while fixing PDF artifacts)
   - Legal-Aware Chunking (Act → Chapter → Section → Subsection, Case → Page → Paragraph)
   - Entity & Relation Extraction (Layer A regex rules + Layer B model extraction)

2. **Knowledge Graph Repository (`src/graph/`)**
   - Neo4j Graph Database
   - Node Labels: `Act`, `Article`, `Section`, `Case`, `Court`, `Judge`, `Chunk`, `Source`, `LegalPrinciple`, `Amendment`
   - Uniqueness Constraints & Indices
   - Neo4j Vector Index (cosine similarity on `Chunk.embedding`)
   - Neo4j Full-Text Indexes on sections, acts, cases, text

3. **Hybrid Retrieval Engine (`src/retrieval/`)**
   - Channel 1: Vector Search (dense semantics)
   - Channel 2: Lexical Search (full-text exact identifiers)
   - Channel 3: Graph Traversal (multi-hop case citations, section interpretations)
   - Candidate Merge, Deduplication, and Weighted Authority Scoring

4. **Evidence & Citation Validation (`src/citations/`)**
   - Pre-generation Evidence Packet Construction
   - Post-generation Claim Extraction & Verification
   - Rejection of invent-IDs and non-existent URLs

5. **Generation Module (`src/generation/`)**
   - Swappable LLM Provider (`OllamaProvider` for Qwen2.5-3B)
   - Answer Generator with Revision Loops
   - Structured JSON output format
   - Explainable Confidence Scoring & Uncertainty Disclosures
