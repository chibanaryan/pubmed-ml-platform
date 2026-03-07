# PubMed ML Platform

An end-to-end ML infrastructure project that builds a semantic search engine over PubMed biomedical abstracts — from data ingestion through embedding generation, model evaluation, serving, and LLM tool integration via MCP.

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

## Components

### 1. Data Ingestion (Airflow)
- DAG that queries PubMed E-utilities API for abstracts in target MeSH categories
- Incremental ingestion: tracks last-fetched date, pulls new publications daily
- Categories: nutrition, exercise physiology, psychology, behavioral science, bioethics

### 2. Embedding Pipeline (PyTorch + HuggingFace + MLflow)
- Generates embeddings for all ingested abstracts
- MLflow experiment comparing:
  - `all-MiniLM-L6-v2` (general-purpose sentence transformer)
  - `pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb` (domain-specific)
- Evaluation: manual relevance judgments on a test query set
- Batch processing with checkpointing for large corpus

### 3. Serving Layer (FastAPI)
- `POST /search` — semantic search with optional MeSH/date filters
- `GET /paper/{pmid}` — retrieve a specific abstract and metadata
- `GET /similar/{pmid}` — find semantically similar papers
- Health checks, request logging, Prometheus metrics

### 4. MCP Server
- Exposes search and retrieval as MCP tools
- Enables LLM-driven queries: "What does recent research say about creatine and muscle recovery?"
- Tool definitions for `search_papers`, `get_paper`, `find_similar`

### 5. Infrastructure
- Docker Compose for local development
- Kubernetes manifests for deployment
- PostgreSQL + pgvector for storage and vector search

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

## Quick Start

```bash
# Start all services
docker compose up -d

# Run initial ingestion (backfill)
docker compose exec airflow airflow dags trigger pubmed_ingest --conf '{"backfill": true}'

# Generate embeddings
docker compose exec app python -m src.embeddings.embed_pipeline

# API is available at http://localhost:8000
# MLflow UI at http://localhost:5000
# Airflow UI at http://localhost:8080
```

## Project Structure

```
pubmed-ml-platform/
├── dags/
│   └── pubmed_ingest.py          # Airflow DAG for PubMed ingestion
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pubmed_client.py      # PubMed E-utilities API client
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embed_pipeline.py     # Embedding generation + MLflow tracking
│   ├── serving/
│   │   ├── __init__.py
│   │   └── api.py                # FastAPI application
│   └── mcp/
│       ├── __init__.py
│       └── server.py             # MCP server wrapping the search API
├── db/
│   └── init.sql                  # Schema + pgvector setup
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── tests/
│   ├── test_ingestion.py
│   ├── test_embeddings.py
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```
