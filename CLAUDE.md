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
make lint                        # ruff check src/ tests/ dags/ loadtest/
mypy src/                        # type check (training CLIs exempt via pyproject overrides)

# Local stack
make up / make down              # docker compose up -d / down
make logs                        # tail all service logs
make ingest                      # load papers from PubMed into Postgres
make embed / make compare / make evaluate   # embedding pipeline (runs inside api container)
make eval-gate                   # fail if NDCG@5 < MIN_NDCG (default 0.80)
make loadtest                    # locust headless run against localhost:8000 (stack must be up)
```

Service URLs after `make up`: API :8000, MLflow :5001 (remapped — port 5000 conflicts with macOS AirPlay), Airflow :8080 (admin/admin), Grafana :3000 (admin/admin), Prometheus :9090.

Training/experiment modules are CLIs run as `python -m src.embeddings.<module>` (finetune, cross_encoder, distill, onnx_export, train_from_scratch, reembed_ft, compare_ft, registry). Most require `--db-url`. Model registry: `python -m src.embeddings.registry list|promote|load --mlflow-uri http://localhost:5001`.

DB URLs differ by context: `postgresql://pubmed:pubmed@localhost:5432/pubmed` from the host, `...@postgres:5432/pubmed` inside Docker. A locally running Postgres will conflict with the Docker one on 5432.

## Architecture

**Data flow:** the Airflow DAG (`dags/pubmed_ingest.py`) pulls abstracts from PubMed E-utilities via `src/ingestion/pubmed_client.py` into the `papers` table, tracking incremental state per MeSH category in `ingestion_state`, then its final task `embed_new_papers` writes vectors into `embeddings` so ingested papers are searchable without a manual step. `src/serving/api.py` serves search; `src/mcp/server.py` wraps the API as MCP tools (stdio, plus streamable HTTP mounted at `/mcp` when `MCP_HTTP=1`).

`src/embeddings/embed_pipeline.py` is the standalone (torch) embedder, used when embedding a *specific* model on demand — model comparisons, re-embedding after fine-tuning — and it logs runs/models to MLflow. The DAG task deliberately does not use it: it calls `src/serving/onnx_embedder.py` instead so the Airflow image needs no torch, and so documents and queries are encoded by identical weights.

**DAG gotchas.**
- `fetch_abstracts` is pinned to `max_active_tis_per_dag=1`. The client's rate limiter is per-instance, so parallel category tasks each throttle correctly and collectively exceed PubMed's limit.
- Pagination must stop at `retstart > 9998` (ESearch's hard ceiling). Past it PubMed returns HTTP 200 with an error payload containing raw newlines, which strict JSON parsing reports as "invalid control character" rather than the real message.
- Airflow Variables: `max_embed_per_run` (default 5,000), `max_db_bytes` (0 = guard off; set it only for quota-limited targets like Neon).
- The embedding task is idempotent — it selects on absence of a vector, so re-running is always safe.

**Untyped vector column (load-bearing design decision).** `embeddings.embedding` is `vector` with no dimension so MiniLM (384-dim) and PubMedBERT (768-dim) coexist in one table, discriminated by `model_name`. Consequences:
- Plain vector indexes don't work; HNSW *expression* indexes on casts are used instead, created manually after data load (see comments in `db/init.sql`).
- Every similarity query must cast **both sides** to the model's dimension: `e.embedding::vector(384) <=> $1::vector(384)`. Missing the cast silently falls back to a sequential scan (~80ms vs ~4ms).
- Adding a model to the API means updating `MODEL_DIMS` (and `MLFLOW_REGISTRY_NAMES`) in `src/serving/api.py`.

**API (`src/serving/api.py`).** Single-file FastAPI app. asyncpg pool (not psycopg2 — parameters are `$1, $2`, not `%s`) created in the lifespan handler. Models lazy-load on first request: MLflow registry `@production` alias first (when `MLFLOW_TRACKING_URI` is set), HuggingFace fallback, then cached in `_models`. A/B testing via `AB_TEST_MODEL` + `AB_TEST_TRAFFIC` env vars reroutes default-model traffic only; responses carry `model_used`. Metrics are hand-rolled Prometheus text format in the module-level `_metrics` dict.

**Two Postgres instances in docker-compose:** `postgres` (app data, pgvector) and `airflow-db` (Airflow metadata). Don't point Airflow's metadata at the app DB.

**Tests** (`tests/`) mock the DB with `AsyncMock` against the asyncpg pool pattern and patch `SentenceTransformer` — they don't need Docker or a real DB. `asyncio_mode = "auto"` is set, so async tests need no marker.

**Model artifacts** under `models/` (fine-tuned, distilled, ONNX variants) are gitignored and exist only locally; embeddings for variant models are stored under distinct `model_name` values (e.g. `minilm-pubmed-ft`).

**Production:** API on Render free tier (`render.yaml`, https://pubmed-search-683d.onrender.com) serving the INT8 ONNX model (`SERVING_BACKEND=onnx`, artifacts at HF Hub `chibanaryan/minilm-pubmed-onnx`) — torch doesn't fit the 512MB instance. Postgres on Neon. The old Fly.io deploy is dead (trial ended); `fly.toml` removed. Neon's free tier is 512MB — storing multiple full-corpus embedding sets exceeds it, and deleted rows don't free space until auto-vacuum catches up (see DEVLOG).
