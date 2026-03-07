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
- Evaluation uses MeSH term overlap between query intent and top-k results as a relevance proxy
- Batch processing with progress tracking; only embeds papers that don't already have embeddings for the target model

### 3. Serving Layer (FastAPI)
- `POST /search` — semantic search with optional date range and MeSH term filters
- `GET /paper/{pmid}` — retrieve a specific paper's metadata and abstract
- `GET /similar/{pmid}` — find semantically similar papers using an existing paper's embedding
- `GET /health` — health check with paper count

### 4. MCP Server
- Wraps the FastAPI endpoints as MCP tools for LLM integration
- Three tools: `search_papers`, `get_paper`, `find_similar`
- Configured via `.mcp.json` for Claude Code; runs over stdio transport
- Formats results for LLM consumption with truncated abstracts, author lists, and relevance scores

### 5. Infrastructure
- **Docker Compose** for local development (Postgres+pgvector, MLflow, Airflow, FastAPI)
- **Kubernetes manifests** for deployment (namespace, PVCs, deployments with health probes, services)
- **pgvector** for vector similarity search with IVFFlat indexing

## Design Decisions

**pgvector over a dedicated vector DB.** Pinecone or Weaviate would add a managed service dependency for marginal benefit at this scale. pgvector keeps everything in one database, simplifies joins between paper metadata and embeddings, and avoids the operational overhead of syncing two data stores. The tradeoff is that pgvector's IVFFlat index is less sophisticated than HNSW-based alternatives, but for sub-100K vectors the performance difference is negligible.

**Untyped vector column.** The embeddings table uses `vector` without a dimension constraint, which lets MiniLM (384-dim) and PubMedBERT (768-dim) coexist in the same table. The alternative was separate tables per model, but a single table with a `model_name` discriminator is simpler and makes the comparison pipeline cleaner. The cost is that pgvector can't enforce dimension consistency at the schema level.

**MeSH overlap as an evaluation metric.** Without hand-labeled relevance judgments, MeSH terms serve as a structured proxy for topical relevance. If a query about "creatine supplementation" retrieves papers tagged with the Creatine and Dietary Supplements MeSH headings, that's a reasonable signal. It's not a substitute for proper NDCG evaluation, but it's something you can compute automatically and it differentiates between models that retrieve topically relevant papers and those that just return high-similarity noise.

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
| Deployment    | Docker, Kubernetes                     |
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
│   │   └── embed_pipeline.py     # Embedding generation + MLflow tracking
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
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── DEVLOG.md
└── TODO.md
```
