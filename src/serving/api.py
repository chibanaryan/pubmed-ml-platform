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
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# --- Config ---

DB_URL = os.environ.get("DATABASE_URL", "postgresql://pubmed:pubmed@localhost:5432/pubmed")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
MODEL_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb": 768,
}

# --- Metrics ---

_metrics = {
    "requests_total": 0,
    "requests_by_endpoint": {},
    "latency_sum_ms": 0.0,
    "latency_count": 0,
    "errors_total": 0,
}

# --- Models ---


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    min_date: date | None = None
    max_date: date | None = None
    mesh_filter: list[str] | None = None
    model_name: str = DEFAULT_MODEL


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

_models: dict[str, SentenceTransformer] = {}
_conn = None


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in MODEL_DIMS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    if model_name not in _models:
        logger.info(f"Loading model {model_name} (on-demand)...")
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _conn
    logger.info(f"Loading default model {DEFAULT_MODEL}...")
    _models[DEFAULT_MODEL] = SentenceTransformer(DEFAULT_MODEL)
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
    _metrics["requests_total"] += 1
    _metrics["requests_by_endpoint"]["search"] = _metrics["requests_by_endpoint"].get("search", 0) + 1
    start = time.time()

    model = _get_model(req.model_name)
    query_embedding = model.encode(req.query, normalize_embeddings=True).tolist()

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

    dim = MODEL_DIMS.get(req.model_name, 384)
    vec_cast = f"::vector({dim})"

    sql = f"""
        SELECT p.pmid, p.title, p.abstract, p.authors, p.journal,
               p.pub_date, p.mesh_terms,
               1 - (e.embedding{vec_cast} <=> %s::vector({dim})) as similarity
        FROM embeddings e
        JOIN papers p ON e.pmid = p.pmid
        WHERE {where_clause}
        ORDER BY e.embedding{vec_cast} <=> %s::vector({dim})
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
    _metrics["latency_sum_ms"] += latency_ms
    _metrics["latency_count"] += 1

    return SearchResponse(
        results=results,
        query=req.query,
        total=len(results),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/paper/{pmid}", response_model=PaperResult)
async def get_paper(pmid: int):
    _metrics["requests_total"] += 1
    _metrics["requests_by_endpoint"]["paper"] = _metrics["requests_by_endpoint"].get("paper", 0) + 1
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
    model_name: str = Query(default=DEFAULT_MODEL),
):
    _metrics["requests_total"] += 1
    _metrics["requests_by_endpoint"]["similar"] = _metrics["requests_by_endpoint"].get("similar", 0) + 1
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
    dim = MODEL_DIMS.get(model_name, 384)
    with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT p.pmid, p.title, p.abstract, p.authors, p.journal,
                   p.pub_date, p.mesh_terms,
                   1 - (e.embedding::vector({dim}) <=> %s::vector({dim})) as similarity
            FROM embeddings e
            JOIN papers p ON e.pmid = p.pmid
            WHERE e.model_name = %s AND e.pmid != %s
            ORDER BY e.embedding::vector({dim}) <=> %s::vector({dim})
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
    _metrics["latency_sum_ms"] += latency_ms
    _metrics["latency_count"] += 1

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
        return {"status": "healthy", "papers_count": count, "models_loaded": list(_models.keys())}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/metrics")
async def metrics():
    avg_latency = (
        _metrics["latency_sum_ms"] / _metrics["latency_count"]
        if _metrics["latency_count"] > 0
        else 0
    )
    lines = [
        "# HELP pubmed_requests_total Total number of API requests.",
        "# TYPE pubmed_requests_total counter",
        f'pubmed_requests_total {_metrics["requests_total"]}',
        "",
        "# HELP pubmed_errors_total Total number of errors.",
        "# TYPE pubmed_errors_total counter",
        f'pubmed_errors_total {_metrics["errors_total"]}',
        "",
        "# HELP pubmed_search_latency_ms_avg Average search latency in milliseconds.",
        "# TYPE pubmed_search_latency_ms_avg gauge",
        f"pubmed_search_latency_ms_avg {avg_latency:.2f}",
        "",
        "# HELP pubmed_search_latency_ms_sum Total search latency in milliseconds.",
        "# TYPE pubmed_search_latency_ms_sum counter",
        f'pubmed_search_latency_ms_sum {_metrics["latency_sum_ms"]:.2f}',
        "",
        "# HELP pubmed_search_latency_count Total number of search requests.",
        "# TYPE pubmed_search_latency_count counter",
        f'pubmed_search_latency_count {_metrics["latency_count"]}',
        "",
        "# HELP pubmed_models_loaded Number of models loaded in memory.",
        "# TYPE pubmed_models_loaded gauge",
        f"pubmed_models_loaded {len(_models)}",
    ]
    for endpoint, count in _metrics["requests_by_endpoint"].items():
        lines.extend([
            "",
            f'pubmed_requests_by_endpoint{{endpoint="{endpoint}"}} {count}',
        ])
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")
