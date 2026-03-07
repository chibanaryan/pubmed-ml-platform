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

import asyncpg
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# --- Config ---

DB_URL = os.environ.get("DATABASE_URL", "postgresql://pubmed:pubmed@localhost:5432/pubmed")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
MODEL_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb": 768,
}
MLFLOW_REGISTRY_NAMES = {
    "all-MiniLM-L6-v2": "pubmed-minilm",
    "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb": "pubmed-pubmedbert",
}
POOL_MIN = int(os.environ.get("DB_POOL_MIN", "2"))
POOL_MAX = int(os.environ.get("DB_POOL_MAX", "10"))

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
    model_config = {"json_schema_extra": {"examples": [{"query": "does creatine help with muscle recovery", "top_k": 5}]}}

    query: str = Field(..., min_length=1, max_length=1000, description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    min_date: date | None = Field(default=None, description="Filter: earliest publication date (optional)")
    max_date: date | None = Field(default=None, description="Filter: latest publication date (optional)")
    mesh_filter: list[str] | None = Field(default=None, description="Filter: MeSH terms to match (optional)")
    model_name: str = Field(default=DEFAULT_MODEL, description="Embedding model to use (optional)")


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
_pool: asyncpg.Pool | None = None


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in MODEL_DIMS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    if model_name not in _models:
        # Try loading from MLflow registry first (production alias)
        registry_name = MLFLOW_REGISTRY_NAMES.get(model_name)
        if MLFLOW_URI and registry_name:
            try:
                import mlflow
                mlflow.set_tracking_uri(MLFLOW_URI)
                model_uri = f"models:/{registry_name}@production"
                logger.info(f"Loading model from MLflow registry: {model_uri}")
                _models[model_name] = mlflow.sentence_transformers.load_model(model_uri)
                return _models[model_name]
            except Exception as e:
                logger.warning(f"Failed to load from registry, falling back to HuggingFace: {e}")
        logger.info(f"Loading model {model_name} from HuggingFace (on-demand)...")
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    _pool = await asyncpg.create_pool(DB_URL, min_size=POOL_MIN, max_size=POOL_MAX)
    logger.info(f"DB pool created (min={POOL_MIN}, max={POOL_MAX}). Model will load on first request.")
    yield
    if _pool:
        await _pool.close()


app = FastAPI(
    title="PubMed Semantic Search",
    description="Semantic search over PubMed biomedical abstracts",
    version="0.2.0",
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
    conditions = ["e.model_name = $2"]
    params: list = [query_embedding, req.model_name]
    param_idx = 3

    if req.min_date:
        conditions.append(f"p.pub_date >= ${param_idx}")
        params.append(req.min_date)
        param_idx += 1
    if req.max_date:
        conditions.append(f"p.pub_date <= ${param_idx}")
        params.append(req.max_date)
        param_idx += 1
    if req.mesh_filter:
        conditions.append(f"p.mesh_terms ?| ${param_idx}")
        params.append(req.mesh_filter)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    dim = MODEL_DIMS.get(req.model_name, 384)
    vec_cast = f"::vector({dim})"

    sql = f"""
        SELECT p.pmid, p.title, p.abstract, p.authors, p.journal,
               p.pub_date, p.mesh_terms,
               1 - (e.embedding{vec_cast} <=> $1::vector({dim})) as similarity
        FROM embeddings e
        JOIN papers p ON e.pmid = p.pmid
        WHERE {where_clause}
        ORDER BY e.embedding{vec_cast} <=> $1::vector({dim})
        LIMIT ${param_idx}
    """
    params.append(req.top_k)

    async with _pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

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

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT pmid, title, abstract, authors, journal, pub_date, mesh_terms FROM papers WHERE pmid = $1",
            pmid,
        )

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

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT embedding FROM embeddings WHERE pmid = $1 AND model_name = $2",
            pmid, model_name,
        )

    if not row:
        raise HTTPException(status_code=404, detail=f"No embedding for PMID {pmid}")

    embedding = row["embedding"]

    dim = MODEL_DIMS.get(model_name, 384)
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT p.pmid, p.title, p.abstract, p.authors, p.journal,
                   p.pub_date, p.mesh_terms,
                   1 - (e.embedding::vector({dim}) <=> $1::vector({dim})) as similarity
            FROM embeddings e
            JOIN papers p ON e.pmid = p.pmid
            WHERE e.model_name = $2 AND e.pmid != $3
            ORDER BY e.embedding::vector({dim}) <=> $1::vector({dim})
            LIMIT $4
            """,
            embedding, model_name, pmid, top_k,
        )

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
        async with _pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM papers")
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
        "# HELP pubmed_search_latency_seconds_sum Total search latency in seconds.",
        "# TYPE pubmed_search_latency_seconds_sum counter",
        f'pubmed_search_latency_seconds_sum {_metrics["latency_sum_ms"] / 1000:.6f}',
        "",
        "# HELP pubmed_search_latency_seconds_count Total number of search requests.",
        "# TYPE pubmed_search_latency_seconds_count counter",
        f'pubmed_search_latency_seconds_count {_metrics["latency_count"]}',
        "",
        "# HELP pubmed_models_loaded Number of models loaded in memory.",
        "# TYPE pubmed_models_loaded gauge",
        f"pubmed_models_loaded {len(_models)}",
    ]
    for endpoint, count in _metrics["requests_by_endpoint"].items():
        lines.extend([
            "",
            "# HELP pubmed_endpoint_requests_total Requests by endpoint.",
            "# TYPE pubmed_endpoint_requests_total counter",
            f'pubmed_endpoint_requests_total{{endpoint="{endpoint}"}} {count}',
        ])
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")
