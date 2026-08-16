"""FastAPI application — Explainable Indian Law GraphRAG API (spec §54-55)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import configure_logging, get_settings
from src.graph.driver import close_driver, get_driver, health_check as neo4j_health
from src.graph.repository import GraphRepository

logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    configure_logging()
    settings = get_settings()
    logger.info("Starting IndianLaw-Explainable-GraphRAG API (%s)", settings.app_env.value)
    # Pre-connect to Neo4j
    try:
        get_driver()
    except Exception as e:
        logger.warning("Neo4j not available at startup: %s", e)
    yield
    close_driver()
    logger.info("API shutdown complete")


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IndianLaw-Explainable-GraphRAG",
    description=(
        "Explainable AI Legal Research Assistant for Indian Law. "
        "Uses Neo4j knowledge graph, hybrid GraphRAG retrieval, "
        "and evidence-grounded answer generation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ───────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    neo4j: dict[str, Any] = Field(default_factory=dict)


class AnswerRequest(BaseModel):
    question: str
    top_k: int | None = None
    use_reranker: bool = True


class ClaimResponse(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    support_status: str | None = None


class SourceResponse(BaseModel):
    evidence_id: str
    title: str | None = None
    case_name: str | None = None
    court: str | None = None
    page: int | None = None
    paragraph: int | None = None
    url: str | None = None
    section: str | None = None
    authority_level: int = 0


class ConfidenceResponse(BaseModel):
    label: str
    reasons: list[str] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    answer: str
    claims: list[ClaimResponse] = Field(default_factory=list)
    sources: list[SourceResponse] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    temporal_notes: list[str] = Field(default_factory=list)
    confidence: ConfidenceResponse = Field(
        default_factory=lambda: ConfidenceResponse(label="low", reasons=[])
    )
    disclaimer: str = (
        "This system provides AI-assisted legal information and research support "
        "based on retrieved sources. It is not a lawyer, does not provide guaranteed "
        "legal advice, and should not replace advice from a qualified legal professional."
    )


class IngestRequest(BaseModel):
    file_path: str
    document_type: str
    title: str
    source_id: str = "MANUAL"
    source_name: str = "Manual Import"
    source_url: str | None = None
    authority_level: int = 0
    short_title: str | None = None
    year: int | None = None
    case_name: str | None = None
    court: str | None = None
    dry_run: bool = False


class IngestBatchRequest(BaseModel):
    directory: str
    document_type: str
    source_id: str = "MANUAL"
    source_name: str = "Manual Import"
    authority_level: int = 0
    dry_run: bool = False


class GraphStatsResponse(BaseModel):
    node_counts: dict[str, int] = Field(default_factory=dict)


# ── Health endpoints ─────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", neo4j=neo4j_health())


@app.get("/ready", response_model=HealthResponse)
async def ready():
    """Readiness check — verifies Neo4j is reachable."""
    neo4j_status = neo4j_health()
    if neo4j_status.get("status") != "healthy":
        raise HTTPException(status_code=503, detail="Neo4j not ready")
    return HealthResponse(status="ready", neo4j=neo4j_status)


# ── Answer endpoint ──────────────────────────────────────────────────────


@app.post("/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest):
    """Answer a legal question using hybrid GraphRAG (spec §55)."""
    from src.retrieval.query_analyzer import analyze_query
    from src.retrieval import hybrid_retrieve
    from src.generation.answer_generator import build_evidence_packet, generate_answer
    from src.citations import CitationValidator, validate_pre_generation

    # 1. Analyze query
    analysis = analyze_query(request.question)

    # 2. Hybrid retrieval
    candidates = hybrid_retrieve(
        query=request.question,
        analysis=analysis,
        top_k=request.top_k,
    )

    # 3. Build evidence packet
    packet = build_evidence_packet(request.question, analysis, candidates)

    # 4. Pre-generation validation
    warnings = validate_pre_generation(packet)

    # 5. Check if we should abstain
    if not packet.evidence:
        return AnswerResponse(
            answer="I could not find sufficient authoritative evidence in the retrieved sources to provide a reliable answer to this question.",
            uncertainties=["No relevant evidence was retrieved from the knowledge graph."],
            confidence=ConfidenceResponse(label="low", reasons=["No evidence retrieved"]),
        )

    # 6. Generate answer
    legal_answer = generate_answer(packet)

    # 7. Validate citations
    validator = CitationValidator()
    legal_answer = validator.validate_answer(legal_answer)

    # 8. Build response
    return AnswerResponse(
        answer=legal_answer.answer,
        claims=[
            ClaimResponse(
                text=c.text,
                evidence_ids=c.evidence_ids,
                support_status=c.support_status,
            )
            for c in legal_answer.claims
        ],
        sources=[
            SourceResponse(
                evidence_id=s.evidence_id,
                title=s.title,
                case_name=s.case_name,
                court=s.court,
                page=s.page,
                paragraph=s.paragraph,
                url=s.source_url,
                section=s.section,
                authority_level=s.authority_level,
            )
            for s in legal_answer.sources
        ],
        uncertainties=legal_answer.uncertainties,
        temporal_notes=legal_answer.temporal_notes,
        confidence=ConfidenceResponse(
            label=legal_answer.confidence.label,
            reasons=legal_answer.confidence.reasons,
        ),
        disclaimer=legal_answer.disclaimer,
    )


# ── Ingestion endpoints ─────────────────────────────────────────────────


@app.post("/ingest/document")
async def ingest_document(request: IngestRequest):
    """Ingest a single legal document."""
    from src.ingestion.models import DocumentType
    from src.ingestion.sources.manual_import import ManualImporter

    try:
        doc_type = DocumentType(request.document_type)
    except ValueError:
        raise HTTPException(400, f"Invalid document_type: {request.document_type}")

    importer = ManualImporter()
    try:
        result = importer.ingest_file(
            file_path=request.file_path,
            document_type=doc_type,
            title=request.title,
            source_id=request.source_id,
            source_name=request.source_name,
            source_url=request.source_url,
            authority_level=request.authority_level,
            short_title=request.short_title,
            year=request.year,
            case_name=request.case_name,
            court=request.court,
            dry_run=request.dry_run,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest/batch")
async def ingest_batch(request: IngestBatchRequest):
    """Batch ingest documents from a directory."""
    from src.ingestion.models import DocumentType
    from src.ingestion.sources.manual_import import ingest_directory

    try:
        doc_type = DocumentType(request.document_type)
    except ValueError:
        raise HTTPException(400, f"Invalid document_type: {request.document_type}")

    try:
        results = ingest_directory(
            directory=request.directory,
            document_type=doc_type,
            source_id=request.source_id,
            source_name=request.source_name,
            authority_level=request.authority_level,
            dry_run=request.dry_run,
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    document_type: str = Form("ACT"),
    title: str | None = Form(None),
    source_id: str = Form("MANUAL"),
    authority_level: int = Form(5),
    dry_run: bool = Form(False),
):
    """Direct HTTP upload and ingestion for legal documents (PDF/TXT/MD)."""
    import tempfile
    from pathlib import Path
    from src.ingestion.models import DocumentType
    from src.ingestion.sources.manual_import import ManualImporter

    try:
        doc_type = DocumentType(document_type.upper())
    except ValueError:
        raise HTTPException(400, f"Invalid document_type: {document_type}")

    doc_title = title or Path(file.filename or "uploaded_document").stem.replace("_", " ").title()
    temp_dir = Path("data/temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / (file.filename or "uploaded_doc.pdf")

    try:
        content = await file.read()
        temp_path.write_bytes(content)

        importer = ManualImporter()
        result = importer.ingest_file(
            file_path=temp_path,
            document_type=doc_type,
            title=doc_title,
            source_id=source_id,
            authority_level=authority_level,
            dry_run=dry_run,
        )
        return result
    except Exception as e:
        logger.error("Upload ingestion error: %s", e)
        raise HTTPException(500, str(e))
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass



# ── Retrieval endpoint ───────────────────────────────────────────────────


class RetrieveRequest(BaseModel):
    query: str
    top_k: int | None = None


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """Retrieve relevant evidence without generating an answer."""
    from src.retrieval.query_analyzer import analyze_query
    from src.retrieval import hybrid_retrieve

    analysis = analyze_query(request.query)
    candidates = hybrid_retrieve(request.query, analysis, top_k=request.top_k)

    return {
        "query": request.query,
        "analysis": analysis.to_dict(),
        "candidates": [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text[:500],
                "score_final": c.score_final,
                "authority_level": c.authority_level,
                "section_number": c.section_number,
                "article_number": c.article_number,
            }
            for c in candidates
        ],
    }


# ── Graph exploration ────────────────────────────────────────────────────


@app.get("/graph/stats", response_model=GraphStatsResponse)
async def graph_stats():
    """Return node counts per label."""
    repo = GraphRepository()
    counts = repo.get_node_counts()
    return GraphStatsResponse(node_counts=counts)


@app.get("/graph/neighborhood/{entity_id}")
async def graph_neighborhood(entity_id: str, max_depth: int = 2):
    """Return the graph neighborhood of an entity."""
    from src.graph.driver import run_query

    query = """
    MATCH (n {id: $entity_id})
    CALL apoc.path.subgraphAll(n, {maxLevel: $max_depth})
    YIELD nodes, relationships
    RETURN nodes, relationships
    """
    # Fallback without APOC
    simple_query = """
    MATCH (n {id: $entity_id})-[r]-(m)
    RETURN n.id AS source_id, type(r) AS relation, m.id AS target_id,
           labels(n) AS source_labels, labels(m) AS target_labels
    LIMIT 50
    """
    try:
        results = run_query(simple_query, {"entity_id": entity_id, "max_depth": max_depth})
        return {"entity_id": entity_id, "neighbors": results}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/graph/sample")
async def graph_sample(limit: int = 60):
    """Return a sample of graph nodes and relationships for visual rendering."""
    from src.graph.driver import run_query

    query = """
    MATCH (n)-[r]->(m)
    RETURN n.id AS source_id, labels(n)[0] AS source_label, n.title AS source_title, n.name AS source_name,
           type(r) AS rel_type,
           m.id AS target_id, labels(m)[0] AS target_label, m.title AS target_title, m.name AS target_name
    LIMIT $limit
    """
    try:
        results = run_query(query, {"limit": limit})
        nodes_dict = {}
        links = []
        for row in results:
            s_id = str(row["source_id"] or "unknown")
            t_id = str(row["target_id"] or "unknown")
            if s_id not in nodes_dict:
                nodes_dict[s_id] = {
                    "id": s_id,
                    "label": str(row["source_label"] or "Node"),
                    "title": str(row["source_title"] or row["source_name"] or s_id),
                }
            if t_id not in nodes_dict:
                nodes_dict[t_id] = {
                    "id": t_id,
                    "label": str(row["target_label"] or "Node"),
                    "title": str(row["target_title"] or row["target_name"] or t_id),
                }
            links.append({
                "source": s_id,
                "target": t_id,
                "type": str(row["rel_type"] or "RELATED"),
            })
        return {"nodes": list(nodes_dict.values()), "links": links}
    except Exception as e:
        logger.error("Error fetching graph sample: %s", e)
        return {"nodes": [], "links": []}

