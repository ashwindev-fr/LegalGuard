"""Hybrid retrieval pipeline — vector + lexical + graph with merge and reranking.

Implements the full retrieval architecture per spec Sections 27-32.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_settings
from src.embeddings import get_embedding_provider, EmbeddingProvider
from src.graph.driver import get_session, run_query
from src.retrieval.models import QueryAnalysis, RetrievalCandidate

logger = logging.getLogger(__name__)


# ── Channel 1: Vector retrieval ──────────────────────────────────────────


def vector_search(
    query_embedding: list[float],
    top_k: int | None = None,
) -> list[RetrievalCandidate]:
    """Search the Neo4j vector index for similar chunks."""
    settings = get_settings()
    k = top_k or settings.top_k_vector

    query = """
    CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
    YIELD node, score
    RETURN node.id AS chunk_id,
           node.document_id AS document_id,
           node.text AS text,
           score,
           node.authority_level AS authority_level,
           node.source_url AS source_url,
           node.page_start AS page_start,
           node.page_end AS page_end,
           node.paragraph_start AS paragraph_start,
           node.paragraph_end AS paragraph_end,
           node.section_number AS section_number,
           node.article_number AS article_number
    """
    try:
        results = run_query(query, {"top_k": k, "embedding": query_embedding})
    except Exception as e:
        logger.warning("Vector search failed (index may not exist yet): %s", e)
        return []

    candidates = []
    for i, r in enumerate(results):
        candidates.append(RetrievalCandidate(
            evidence_id=f"V{i+1:03d}",
            chunk_id=r.get("chunk_id", ""),
            document_id=r.get("document_id", ""),
            text=r.get("text", ""),
            score_vector=r.get("score", 0.0),
            authority_level=r.get("authority_level", 0),
            source_url=r.get("source_url", ""),
            page_start=r.get("page_start"),
            page_end=r.get("page_end"),
            paragraph_start=r.get("paragraph_start"),
            paragraph_end=r.get("paragraph_end"),
            section_number=r.get("section_number", ""),
            article_number=r.get("article_number", ""),
        ))

    logger.info("Vector search returned %d candidates", len(candidates))
    return candidates


# ── Channel 2: Graph retrieval ───────────────────────────────────────────


def graph_search(
    analysis: QueryAnalysis,
    top_k: int | None = None,
) -> list[RetrievalCandidate]:
    """Search the graph using extracted entities from query analysis."""
    settings = get_settings()
    k = top_k or settings.top_k_graph
    candidates: list[RetrievalCandidate] = []

    # Search by sections
    for section in analysis.sections:
        query = """
        MATCH (chunk:Chunk)-[:MENTIONS]->(s:Section)
        WHERE s.section_number = $section OR s.id CONTAINS $section
        OPTIONAL MATCH (chunk)<-[:HAS_CHUNK]-(doc:Document)
        RETURN chunk.id AS chunk_id,
               chunk.document_id AS document_id,
               chunk.text AS text,
               chunk.authority_level AS authority_level,
               chunk.source_url AS source_url,
               chunk.page_start AS page_start,
               chunk.section_number AS section_number
        LIMIT $limit
        """
        results = run_query(query, {"section": section, "limit": k})
        for i, r in enumerate(results):
            candidates.append(RetrievalCandidate(
                evidence_id=f"GS{len(candidates)+1:03d}",
                chunk_id=r.get("chunk_id", ""),
                document_id=r.get("document_id", ""),
                text=r.get("text", ""),
                score_graph=1.0,
                authority_level=r.get("authority_level", 0),
                source_url=r.get("source_url", ""),
                section_number=r.get("section_number", ""),
            ))

    # Search by articles
    for article in analysis.articles:
        query = """
        MATCH (chunk:Chunk)
        WHERE chunk.article_number = $article
        RETURN chunk.id AS chunk_id,
               chunk.document_id AS document_id,
               chunk.text AS text,
               chunk.authority_level AS authority_level,
               chunk.source_url AS source_url,
               chunk.article_number AS article_number
        LIMIT $limit
        """
        results = run_query(query, {"article": article, "limit": k})
        for r in results:
            candidates.append(RetrievalCandidate(
                evidence_id=f"GA{len(candidates)+1:03d}",
                chunk_id=r.get("chunk_id", ""),
                document_id=r.get("document_id", ""),
                text=r.get("text", ""),
                score_graph=1.0,
                authority_level=r.get("authority_level", 0),
                source_url=r.get("source_url", ""),
                article_number=r.get("article_number", ""),
            ))

    # Graph expansion for cases interpreting sections
    if analysis.sections:
        for section in analysis.sections:
            query = """
            MATCH (case:Case)-[:INTERPRETS|CONSIDERS|APPLIES]->(s:Section)
            WHERE s.section_number = $section OR s.id CONTAINS $section
            OPTIONAL MATCH (case)<-[:HAS_CHUNK]-(doc:Document)-[:HAS_CHUNK]->(chunk:Chunk)
            RETURN chunk.id AS chunk_id,
                   case.id AS document_id,
                   chunk.text AS text,
                   chunk.authority_level AS authority_level,
                   case.case_name AS case_name,
                   case.court AS court
            LIMIT $limit
            """
            results = run_query(query, {"section": section, "limit": k})
            for r in results:
                if r.get("text"):
                    candidates.append(RetrievalCandidate(
                        evidence_id=f"GE{len(candidates)+1:03d}",
                        chunk_id=r.get("chunk_id", ""),
                        document_id=r.get("document_id", ""),
                        text=r.get("text", ""),
                        score_graph=0.9,
                        authority_level=r.get("authority_level", 0),
                        case_name=r.get("case_name", ""),
                        court=r.get("court", ""),
                    ))

    logger.info("Graph search returned %d candidates", len(candidates))
    return candidates


# ── Channel 3: Lexical retrieval ─────────────────────────────────────────


def lexical_search(
    query: str,
    top_k: int | None = None,
) -> list[RetrievalCandidate]:
    """Full-text search on chunks, case names, section titles."""
    settings = get_settings()
    k = top_k or settings.top_k_lexical
    candidates: list[RetrievalCandidate] = []

    # Search chunks full-text
    try:
        ft_query = """
        CALL db.index.fulltext.queryNodes('chunk_fulltext', $query)
        YIELD node, score
        RETURN node.id AS chunk_id,
               node.document_id AS document_id,
               node.text AS text,
               score,
               node.authority_level AS authority_level,
               node.source_url AS source_url,
               node.section_number AS section_number,
               node.article_number AS article_number
        LIMIT $limit
        """
        results = run_query(ft_query, {"query": query, "limit": k})
        for i, r in enumerate(results):
            candidates.append(RetrievalCandidate(
                evidence_id=f"L{len(candidates)+1:03d}",
                chunk_id=r.get("chunk_id", ""),
                document_id=r.get("document_id", ""),
                text=r.get("text", ""),
                score_lexical=r.get("score", 0.0),
                authority_level=r.get("authority_level", 0),
                source_url=r.get("source_url", ""),
                section_number=r.get("section_number", ""),
                article_number=r.get("article_number", ""),
            ))
    except Exception as e:
        logger.warning("Lexical search failed: %s", e)

    logger.info("Lexical search returned %d candidates", len(candidates))
    return candidates


# ── Merge & Deduplicate ──────────────────────────────────────────────────


def merge_and_deduplicate(
    *candidate_lists: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """Merge candidates from all channels, deduplicating by chunk_id."""
    seen: dict[str, RetrievalCandidate] = {}

    for candidates in candidate_lists:
        for c in candidates:
            if c.chunk_id in seen:
                existing = seen[c.chunk_id]
                # Merge scores
                if c.score_vector and (not existing.score_vector or c.score_vector > existing.score_vector):
                    existing.score_vector = c.score_vector
                if c.score_lexical and (not existing.score_lexical or c.score_lexical > existing.score_lexical):
                    existing.score_lexical = c.score_lexical
                if c.score_graph and (not existing.score_graph or c.score_graph > existing.score_graph):
                    existing.score_graph = c.score_graph
            else:
                seen[c.chunk_id] = c

    return list(seen.values())


# ── Scoring ──────────────────────────────────────────────────────────────


def compute_final_scores(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    """Compute weighted final score per spec §32."""
    # Weights (configurable)
    W = {
        "semantic": 0.25,
        "lexical": 0.15,
        "graph": 0.20,
        "reranker": 0.25,
        "authority": 0.10,
        "temporal": 0.05,
    }

    for c in candidates:
        score = 0.0
        score += W["semantic"] * (c.score_vector or 0.0)
        score += W["lexical"] * min((c.score_lexical or 0.0) / 10.0, 1.0)  # Normalize
        score += W["graph"] * (c.score_graph or 0.0)
        score += W["reranker"] * (c.score_reranker or 0.0)
        score += W["authority"] * (c.authority_level / 5.0)
        c.score_final = score

    candidates.sort(key=lambda c: c.score_final, reverse=True)
    return candidates


# ── Full hybrid pipeline ─────────────────────────────────────────────────


def hybrid_retrieve(
    query: str,
    analysis: QueryAnalysis,
    embedding_provider: EmbeddingProvider | None = None,
    top_k: int | None = None,
) -> list[RetrievalCandidate]:
    """Execute the full hybrid retrieval pipeline."""
    settings = get_settings()
    final_k = top_k or settings.top_k_final

    # Embed query
    if embedding_provider is None:
        embedding_provider = get_embedding_provider()
    query_embedding = embedding_provider.embed_query(query)

    # Run all channels
    vector_results = vector_search(query_embedding)
    graph_results = graph_search(analysis)
    lexical_results = lexical_search(query)

    # Merge and deduplicate
    merged = merge_and_deduplicate(vector_results, graph_results, lexical_results)

    # Score
    scored = compute_final_scores(merged)

    # Return top-k
    final = scored[:final_k]
    logger.info(
        "Hybrid retrieval: %d vector + %d graph + %d lexical → %d merged → %d final",
        len(vector_results), len(graph_results), len(lexical_results),
        len(merged), len(final),
    )
    return final
