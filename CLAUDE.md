# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Semantic search engine over PubMed biomedical abstracts: Airflow ingestion → Postgres+pgvector storage → embedding pipeline (HuggingFace + MLflow) → FastAPI serving → MCP server for LLM tool use. `DEVLOG.md` records what was tried, results (NDCG numbers), and gotchas encountered — read it before redoing an experiment.

## Commands

```bash
# Setup (Python 3.11+)
pip install -e ".[all]"          # installs mlflow, mcp, and dev extras

# Test / lint (also what CI runs)
make test                        # python -m pytest tests/ -v
python -m pytest tests/test_api.py -v                 # single file
python -m pytest tests/test_api.py::test_name -v      # single test
make lint                        # ruff check src/ tests/ dags/

# Local stack
make up / make down              # docker compose up -d / down
make logs                        # tail all service logs
make ingest                      # load papers from PubMed into Postgres
make embed / make compare / make evaluate   # embedding pipeline (runs inside api container)
```

Service URLs after `make up`: API :8000, MLflow :5001 (remapped — port 5000 conflicts with macOS AirPlay), Airflow :8080 (admin/admin), Grafana :3000 (admin/admin), Prometheus :9090.

Training/experiment modules are CLIs run as `python -m src.embeddings.<module>` (finetune, cross_encoder, distill, onnx_export, train_from_scratch, reembed_ft, compare_ft, registry). Most require `--db-url`. Model registry: `python -m src.embeddings.registry list|promote|load --mlflow-uri http://localhost:5001`.

DB URLs differ by context: `postgresql://pubmed:pubmed@localhost:5432/pubmed` from the host, `...@postgres:5432/pubmed` inside Docker. A locally running Postgres will conflict with the Docker one on 5432.

## Architecture

**Data flow:** the Airflow DAG (`dags/pubmed_ingest.py`) pulls abstracts from PubMed E-utilities via `src/ingestion/pubmed_client.py` (rate-limited, exponential backoff on 429s) into the `papers` table, tracking incremental state per MeSH category in `ingestion_state`. `src/embeddings/embed_pipeline.py` generates embeddings into the `embeddings` table and logs runs/models to MLflow. `src/serving/api.py` serves search; `src/mcp/server.py` wraps the API as MCP tools (stdio transport, configured in `.mcp.json`).

**Untyped vector column (load-bearing design decision).** `embeddings.embedding` is `vector` with no dimension so MiniLM (384-dim) and PubMedBERT (768-dim) coexist in one table, discriminated by `model_name`. Consequences:
- Plain vector indexes don't work; HNSW *expression* indexes on casts are used instead, created manually after data load (see comments in `db/init.sql`).
- Every similarity query must cast **both sides** to the model's dimension: `e.embedding::vector(384) <=> $1::vector(384)`. Missing the cast silently falls back to a sequential scan (~80ms vs ~4ms).
- Adding a model to the API means updating `MODEL_DIMS` (and `MLFLOW_REGISTRY_NAMES`) in `src/serving/api.py`.

**API (`src/serving/api.py`).** Single-file FastAPI app. asyncpg pool (not psycopg2 — parameters are `$1, $2`, not `%s`) created in the lifespan handler. Models lazy-load on first request: MLflow registry `@production` alias first (when `MLFLOW_TRACKING_URI` is set), HuggingFace fallback, then cached in `_models`. A/B testing via `AB_TEST_MODEL` + `AB_TEST_TRAFFIC` env vars reroutes default-model traffic only; responses carry `model_used`. Metrics are hand-rolled Prometheus text format in the module-level `_metrics` dict.

**Two Postgres instances in docker-compose:** `postgres` (app data, pgvector) and `airflow-db` (Airflow metadata). Don't point Airflow's metadata at the app DB.

**Tests** (`tests/`) mock the DB with `AsyncMock` against the asyncpg pool pattern and patch `SentenceTransformer` — they don't need Docker or a real DB. `asyncio_mode = "auto"` is set, so async tests need no marker.

**Model artifacts** under `models/` (fine-tuned, distilled, ONNX variants) are gitignored and exist only locally; embeddings for variant models are stored under distinct `model_name` values (e.g. `minilm-pubmed-ft`).

**Production:** API on Fly.io (`fly.toml`), Postgres on Neon. Neon's free tier is 512MB — storing multiple full-corpus embedding sets exceeds it, and deleted rows don't free space until auto-vacuum catches up (see DEVLOG).
