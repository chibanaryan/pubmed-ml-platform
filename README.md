# PubMed ML Platform

An end-to-end ML infrastructure project that builds a semantic search engine over PubMed biomedical abstracts, covering data ingestion, embedding generation, model evaluation, serving, and LLM tool integration via MCP.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   PubMed    │     │   Airflow    │     │   Postgres    │     │   FastAPI    │
│  E-utils API│────▶│  Ingestion   │────▶│  + pgvector   │────▶│  Serving     │
│             │     │  DAG         │     │               │     │  Layer       │
└─────────────┘     └──────┬───────┘     └───────────────┘     └──────┬───────┘
                           │                    ▲                      │
                    ┌──────▼───────┐            │               ┌──────▼───────┐
                    │  Embedding   │            │               │   MCP        │
                    │  Pipeline    │────────────┘               │   Server     │
                    │  (HF + MLflow)                            │              │
                    └──────────────┘                            └──────────────┘
```

## Quick Start

```bash
# Start all services
docker compose up -d

# API at http://localhost:8000
# MLflow UI at http://localhost:5001
# Airflow UI at http://localhost:8080 (admin/admin)
# Grafana at http://localhost:3000 (admin/admin)
# Prometheus at http://localhost:9090

# Run embedding model comparison
docker compose exec api python -m src.embeddings.embed_pipeline --compare \
  --db-url postgresql://pubmed:pubmed@postgres:5432/pubmed \
  --mlflow-uri http://mlflow:5000

# Run tests
python -m pytest tests/ -v
```

## Components

### 1. Data Ingestion (Airflow)
- Airflow DAG queries PubMed's E-utilities API for abstracts across five MeSH categories: nutrition, exercise physiology, psychology, behavioral science, and bioethics
- Incremental ingestion tracks the last-fetched date per category and pulls new publications daily
- Rate-limited client with exponential backoff on 429s
- Handles PubMed's structured XML responses, including multi-part abstracts and text-format month fields

### 2. Embedding Pipeline (PyTorch + HuggingFace + MLflow)
- Generates vector embeddings for ingested abstracts using HuggingFace sentence transformers
- MLflow experiment tracking compares two models:
  - `all-MiniLM-L6-v2` (384-dim, general-purpose, fast)
  - `PubMedBERT` (768-dim, domain-specific, trained on biomedical NLI tasks)
- **Contrastive fine-tuning** on 100K MeSH-based pairs using MultipleNegativesRankingLoss (NDCG@5: 0.83 → 0.86)
- **Cross-encoder re-ranker** for two-stage retrieval: bi-encoder top-50 → cross-encoder top-10 (NDCG@5: 0.83 → 0.92)
- **Knowledge distillation** from PubMedBERT (teacher) into MiniLM (student) via KL divergence on similarity distributions
- **ONNX export + INT8 quantization**: 5.3x inference speedup (4.4ms → 0.84ms) with negligible quality loss
- Evaluation uses MeSH term overlap between query intent and top-k results as a relevance proxy (8 queries, graded 0-3)
- Batch processing with progress tracking; only embeds papers that don't already have embeddings for the target model
- **MLflow model registry**: models registered as versioned artifacts after training, with alias-based promotion (`@production`). Registry management CLI for listing, promoting, and loading models

### 3. Serving Layer (FastAPI + asyncpg)
- `POST /search` — semantic search with optional date range and MeSH term filters; supports model selection at query time
- `GET /paper/{pmid}` — retrieve a specific paper's metadata and abstract
- `GET /similar/{pmid}` — find semantically similar papers using an existing paper's embedding
- `GET /health` — health check with paper count and loaded models
- `GET /metrics` — Prometheus-compatible metrics (request counts, latency, per-endpoint breakdown)
- Lazy model loading: tries MLflow registry first (if configured), falls back to HuggingFace. Models cached in memory
- Async DB layer with asyncpg connection pool (configurable min/max connections)
- **A/B testing**: route a configurable percentage of traffic to a treatment model via env vars. Per-model metrics at `/metrics` and `/ab-results`

### 4. MCP Server
- Wraps the FastAPI endpoints as MCP tools for LLM integration
- Three tools: `search_papers`, `get_paper`, `find_similar`
- Configured via `.mcp.json` for Claude Code; runs over stdio transport
- Formats results for LLM consumption with truncated abstracts, author lists, and relevance scores

### 5. Infrastructure
- **Docker Compose** for local development (Postgres+pgvector, MLflow, Airflow with separate metadata DB, FastAPI, Prometheus, Grafana)
- **Kubernetes manifests** for deployment (namespace, PVCs, deployments with health probes, services)
- **pgvector** with HNSW expression indexes for vector similarity search
- **CPU-only Docker image** with PyTorch installed from the CPU index (641MB vs 3.5GB with CUDA)
- **Render free tier** for production API hosting (`render.yaml`) — live at https://pubmed-search-683d.onrender.com (first request after idle takes ~1min to wake). Serves the **INT8 ONNX model** (`SERVING_BACKEND=onnx`, artifacts on the [HF Hub](https://huggingface.co/chibanaryan/minilm-pubmed-onnx)) because torch doesn't fit in 512MB; **Neon** for managed Postgres with pgvector
- **GitHub Actions CI**: ruff lint, mypy type check, pytest with coverage, Docker image build (pushed to GHCR on main)
- **Eval gate**: `make eval-gate` (or the on-demand `Eval Gate` workflow) fails if mean NDCG@5 drops below a threshold — a regression gate for model changes
- **Observability**: `prometheus_client` metrics (labeled request/error counters, per-model latency histograms), JSON structured logs (`LOG_FORMAT=json`), Prometheus alert rules (`monitoring/alerts.yml`: APIDown, HighErrorRate, HighSearchLatencyP95)
- **Load testing**: `make loadtest` runs locust headless (20 users, 60s) against the local API; warm the model with one search first. Measured locally (40K papers, Apple Silicon, Docker): 2,019 requests, 0 failures, 34 req/s sustained — `/search` p50 15ms / p95 71ms / p99 460ms, `/paper` p50 4ms / p95 29ms, `/similar` p50 7ms / p95 46ms

## Design Decisions

**pgvector over a dedicated vector DB.** Pinecone or Weaviate would add a managed service dependency for marginal benefit at this scale. pgvector keeps everything in one database, simplifies joins between paper metadata and embeddings, and avoids the operational overhead of syncing two data stores. HNSW indexes achieve 3.9ms search latency at 40K vectors.

**Untyped vector column.** The embeddings table uses `vector` without a dimension constraint, which lets MiniLM (384-dim) and PubMedBERT (768-dim) coexist in the same table. The alternative was separate tables per model, but a single table with a `model_name` discriminator is simpler and makes the comparison pipeline cleaner. The cost is that pgvector can't enforce dimension consistency at the schema level.

**Graded relevance evaluation via MeSH terms.** Without hand-labeled relevance judgments, MeSH terms serve as a structured proxy for topical relevance. The evaluation harness defines high/medium/low relevance MeSH terms per query and computes NDCG@5/10 with a 0-3 graded relevance scale. At 40K papers, MiniLM achieves mean NDCG@5 of 0.83 and NDCG@10 of 0.91.

**Airflow for orchestration.** For a personal project this is arguably overkill. A cron job calling a Python script would work fine. But the point is demonstrating familiarity with production orchestration patterns: incremental state tracking, task dependencies, retry policies, and monitoring via the Airflow UI. The DAG is structured so each category runs independently, which would parallelize naturally at higher scale.

**MCP over a standalone chatbot.** Rather than building a chat UI, the MCP server lets any MCP-compatible LLM use the search engine as a tool. This is more composable and avoids reinventing the conversation layer. It also extends naturally: an LLM can chain `search_papers` with `find_similar` to explore citation-adjacent research without any custom orchestration code.

## Tech Stack

| Layer         | Technology                              |
|---------------|----------------------------------------|
| Orchestration | Apache Airflow                         |
| Embeddings    | PyTorch, HuggingFace Transformers      |
| Experiment    | MLflow                                 |
| Storage       | PostgreSQL + pgvector                  |
| Serving       | FastAPI, Uvicorn                       |
| LLM Tools     | Model Context Protocol (MCP)           |
| Deployment    | Docker, Render, Neon Postgres, ONNX Runtime |
| Language      | Python 3.11+                           |

## Project Structure

```
pubmed-ml-platform/
├── dags/
│   └── pubmed_ingest.py          # Airflow DAG for PubMed ingestion
├── src/
│   ├── ingestion/
│   │   └── pubmed_client.py      # PubMed E-utilities API client
│   ├── embeddings/
│   │   ├── embed_pipeline.py     # Embedding generation + MLflow tracking
│   │   ├── evaluate.py           # NDCG evaluation harness
│   │   ├── finetune.py           # Contrastive fine-tuning on MeSH pairs
│   │   ├── cross_encoder.py      # Cross-encoder re-ranker (two-stage pipeline)
│   │   ├── distill.py            # Knowledge distillation (PubMedBERT → MiniLM)
│   │   ├── onnx_export.py        # ONNX export + INT8 quantization
│   │   └── registry.py           # MLflow model registry management
│   ├── serving/
│   │   └── api.py                # FastAPI application
│   └── mcp/
│       └── server.py             # MCP server wrapping the search API
├── db/
│   └── init.sql                  # Schema + pgvector setup
├── k8s/
│   ├── namespace.yaml
│   ├── postgres.yaml
│   ├── api.yaml
│   └── mlflow.yaml
├── tests/
│   ├── test_pubmed_client.py
│   ├── test_api.py
│   └── test_evaluate.py
├── .github/workflows/ci.yml
├── docker-compose.yml
├── Dockerfile
├── fly.toml
├── Makefile
├── pyproject.toml
├── DEVLOG.md
└── TODO.md
```
