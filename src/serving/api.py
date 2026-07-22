"""
FastAPI serving layer for PubMed semantic search.

Endpoints:
    POST /search        — semantic search over abstracts
    GET  /paper/{pmid}  — fetch a specific paper
    GET  /similar/{pmid} — find similar papers
    GET  /health        — health check
    GET  /metrics       — Prometheus metrics
    GET  /ab-results    — A/B test comparison
"""

import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from datetime import date

import asyncpg
from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

# "onnx" serves the INT8 MiniLM via onnxruntime without ever importing torch —
# required to fit free-tier memory limits (see src/serving/onnx_embedder.py)
SERVING_BACKEND = os.environ.get("SERVING_BACKEND", "torch")
if SERVING_BACKEND == "torch":
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = None  # type: ignore[assignment,misc]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    if os.environ.get("LOG_FORMAT", "json") == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


_configure_logging()
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

# A/B testing config (set AB_TEST_MODEL to enable)
AB_TEST_MODEL = os.environ.get("AB_TEST_MODEL", "")  # treatment model name
AB_TEST_TRAFFIC = float(os.environ.get("AB_TEST_TRAFFIC", "0.0"))  # fraction routed to treatment (0.0-1.0)

# --- Metrics ---

REQUESTS = Counter("pubmed_requests", "API requests by endpoint", ["endpoint"])
ERRORS = Counter("pubmed_errors", "Unhandled errors / HTTP 5xx by endpoint", ["endpoint"])
SEARCH_LATENCY = Histogram(
    "pubmed_search_latency_seconds",
    "Search latency by serving model",
    ["model"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
MODELS_LOADED = Gauge("pubmed_models_loaded", "Number of models loaded in memory")
AB_REQUESTS = Counter("pubmed_ab_requests", "Search requests by serving model", ["model"])

# In-memory per-model stats backing the /ab-results JSON endpoint (Prometheus
# counters can't be read back cleanly for a response payload).
_ab_stats: dict[str, dict[str, float]] = {}

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
    model_used: str | None = None


# --- App ---

# SentenceTransformer or OnnxEmbedder — both expose .encode(text, normalize_embeddings=)
_models: dict[str, object] = {}
_pool: asyncpg.Pool | None = None

MODELS_LOADED.set_function(lambda: float(len(_models)))


def _db_pool() -> asyncpg.Pool:
    assert _pool is not None, "DB pool not initialized (lifespan has not run)"
    return _pool


def _get_model(model_name: str):
    if model_name not in MODEL_DIMS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    if model_name not in _models:
        if SERVING_BACKEND == "onnx":
            if model_name != DEFAULT_MODEL:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {model_name} is not available on the onnx serving backend",
                )
            from src.serving.onnx_embedder import load_onnx_embedder

            _models[model_name] = load_onnx_embedder()
            return _models[model_name]
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


@app.middleware("http")
async def track_requests(request: Request, call_next):
    def endpoint_label() -> str:
        route = request.scope.get("route")
        return route.path if route else request.url.path

    try:
        response = await call_next(request)
    except Exception:
        ERRORS.labels(endpoint=endpoint_label()).inc()
        raise
    label = endpoint_label()
    REQUESTS.labels(endpoint=label).inc()
    if response.status_code >= 500:
        ERRORS.labels(endpoint=label).inc()
    return response


# --- Endpoints ---


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    start = time.time()

    # A/B test: route to treatment model if enabled and user didn't explicitly choose
    actual_model_name = req.model_name
    if AB_TEST_MODEL and req.model_name == DEFAULT_MODEL and random.random() < AB_TEST_TRAFFIC:
        actual_model_name = AB_TEST_MODEL

    model = _get_model(actual_model_name)
    # pgvector text format — asyncpg has no codec for the vector type, so the
    # parameter must be passed as a string, not a list
    query_embedding = "[" + ",".join(map(str, model.encode(req.query, normalize_embeddings=True).tolist())) + "]"

    # Build query with optional filters
    conditions = ["e.model_name = $2"]
    params: list = [query_embedding, actual_model_name]
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

    dim = MODEL_DIMS.get(actual_model_name, 384)
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

    async with _db_pool().acquire() as conn:
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
    SEARCH_LATENCY.labels(model=actual_model_name).observe(latency_ms / 1000)
    AB_REQUESTS.labels(model=actual_model_name).inc()
    stats = _ab_stats.setdefault(actual_model_name, {"requests": 0, "latency_sum_ms": 0.0})
    stats["requests"] += 1
    stats["latency_sum_ms"] += latency_ms

    return SearchResponse(
        results=results,
        query=req.query,
        total=len(results),
        latency_ms=round(latency_ms, 2),
        model_used=actual_model_name,
    )


@app.get("/paper/{pmid}", response_model=PaperResult)
async def get_paper(pmid: int):
    async with _db_pool().acquire() as conn:
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
    start = time.time()

    async with _db_pool().acquire() as conn:
        row = await conn.fetchrow(
            "SELECT embedding FROM embeddings WHERE pmid = $1 AND model_name = $2",
            pmid, model_name,
        )

    if not row:
        raise HTTPException(status_code=404, detail=f"No embedding for PMID {pmid}")

    embedding = row["embedding"]

    dim = MODEL_DIMS.get(model_name, 384)
    async with _db_pool().acquire() as conn:
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
    SEARCH_LATENCY.labels(model=model_name).observe(latency_ms / 1000)

    return SearchResponse(
        results=results,
        query=f"similar to PMID:{pmid}",
        total=len(results),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
async def health():
    try:
        async with _db_pool().acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM papers")
        return {"status": "healthy", "papers_count": count, "models_loaded": list(_models.keys())}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/ab-results")
async def ab_results():
    """Compare A/B test metrics between models."""
    if not _ab_stats:
        return {"status": "no_data", "message": "No A/B test data collected yet."}

    total_requests = sum(s["requests"] for s in _ab_stats.values())
    results = {}
    for model_name, stats in _ab_stats.items():
        count = stats["requests"]
        results[model_name] = {
            "requests": int(count),
            "avg_latency_ms": round(stats["latency_sum_ms"] / max(count, 1), 2),
            "traffic_share": round(count / max(total_requests, 1), 3),
        }

    return {
        "status": "active" if AB_TEST_MODEL else "inactive",
        "control": DEFAULT_MODEL,
        "treatment": AB_TEST_MODEL or None,
        "traffic_split": AB_TEST_TRAFFIC,
        "models": results,
    }
