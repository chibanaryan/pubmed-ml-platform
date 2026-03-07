"""
FastAPI serving layer for PubMed semantic search.

Endpoints:
    POST /search       — semantic search over abstracts
    GET  /paper/{pmid} — fetch a specific paper
    GET  /similar/{pmid} — find similar papers
    GET  /health       — health check
    GET  /metrics      — Prometheus metrics
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# --- Config ---

DB_URL = os.environ.get("DATABASE_URL", "postgresql://pubmed:pubmed@localhost:5432/pubmed")
MODEL_NAME = "all-MiniLM-L6-v2"  # Switch after MLflow comparison

# --- Models ---


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    min_date: date | None = None
    max_date: date | None = None
    mesh_filter: list[str] | None = None
    model_name: str = MODEL_NAME


class PaperResult(BaseModel):
    pmid: int
    title: str
    abstract: str | None
    authors: list[str]
    journal: str | None
    pub_date: date | None
    mesh_terms: list[str]
    similarity: float | None = None


class SearchResponse(BaseModel):
    results: list[PaperResult]
    query: str
    total: int
    latency_ms: float


# --- App ---

_model: SentenceTransformer | None = None
_conn = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _conn
    logger.info(f"Loading model {MODEL_NAME}...")
    _model = SentenceTransformer(MODEL_NAME)
    _conn = psycopg2.connect(DB_URL)
    logger.info("Ready.")
    yield
    if _conn:
        _conn.close()


app = FastAPI(
    title="PubMed Semantic Search",
    description="Semantic search over PubMed biomedical abstracts",
    version="0.1.0",
    lifespan=lifespan,
)


def _parse_json_field(val) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return json.loads(val)
    return list(val)


# --- Endpoints ---


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    start = time.time()

    query_embedding = _model.encode(req.query, normalize_embeddings=True).tolist()

    # Build query with optional filters
    conditions = ["e.model_name = %s"]
    params: list = [req.model_name]

    if req.min_date:
        conditions.append("p.pub_date >= %s")
        params.append(req.min_date)
    if req.max_date:
        conditions.append("p.pub_date <= %s")
        params.append(req.max_date)
    if req.mesh_filter:
        # Match papers that have ANY of the specified MeSH terms
        conditions.append("p.mesh_terms ?| %s")
        params.append(req.mesh_filter)

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT p.pmid, p.title, p.abstract, p.authors, p.journal,
               p.pub_date, p.mesh_terms,
               1 - (e.embedding <=> %s::vector) as similarity
        FROM embeddings e
        JOIN papers p ON e.pmid = p.pmid
        WHERE {where_clause}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    params = [query_embedding] + params + [query_embedding, req.top_k]

    with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    results = [
        PaperResult(
            pmid=r["pmid"],
            title=r["title"],
            abstract=r["abstract"],
            authors=_parse_json_field(r["authors"]),
            journal=r["journal"],
            pub_date=r["pub_date"],
            mesh_terms=_parse_json_field(r["mesh_terms"]),
            similarity=round(float(r["similarity"]), 4),
        )
        for r in rows
    ]

    latency_ms = (time.time() - start) * 1000

    return SearchResponse(
        results=results,
        query=req.query,
        total=len(results),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/paper/{pmid}", response_model=PaperResult)
async def get_paper(pmid: int):
    with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT pmid, title, abstract, authors, journal, pub_date, mesh_terms FROM papers WHERE pmid = %s",
            (pmid,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Paper {pmid} not found")

    return PaperResult(
        pmid=row["pmid"],
        title=row["title"],
        abstract=row["abstract"],
        authors=_parse_json_field(row["authors"]),
        journal=row["journal"],
        pub_date=row["pub_date"],
        mesh_terms=_parse_json_field(row["mesh_terms"]),
    )


@app.get("/similar/{pmid}", response_model=SearchResponse)
async def find_similar(
    pmid: int,
    top_k: int = Query(default=10, ge=1, le=100),
    model_name: str = Query(default=MODEL_NAME),
):
    start = time.time()

    # Get the embedding for this paper
    with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT embedding FROM embeddings WHERE pmid = %s AND model_name = %s",
            (pmid, model_name),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"No embedding for PMID {pmid}")

    embedding = row["embedding"]

    # Find similar papers (exclude self)
    with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT p.pmid, p.title, p.abstract, p.authors, p.journal,
                   p.pub_date, p.mesh_terms,
                   1 - (e.embedding <=> %s::vector) as similarity
            FROM embeddings e
            JOIN papers p ON e.pmid = p.pmid
            WHERE e.model_name = %s AND e.pmid != %s
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, model_name, pmid, embedding, top_k),
        )
        rows = cur.fetchall()

    results = [
        PaperResult(
            pmid=r["pmid"],
            title=r["title"],
            abstract=r["abstract"],
            authors=_parse_json_field(r["authors"]),
            journal=r["journal"],
            pub_date=r["pub_date"],
            mesh_terms=_parse_json_field(r["mesh_terms"]),
            similarity=round(float(r["similarity"]), 4),
        )
        for r in rows
    ]

    latency_ms = (time.time() - start) * 1000

    return SearchResponse(
        results=results,
        query=f"similar to PMID:{pmid}",
        total=len(results),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
async def health():
    try:
        with _conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM papers")
            count = cur.fetchone()[0]
        return {"status": "healthy", "papers_count": count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
