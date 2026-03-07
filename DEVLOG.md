# Dev Log

## 2026-03-06 — Initial build

### What got built

Scaffolded and got the full pipeline running in one session. Starting from generated code files, organized into a proper repo structure, fixed several bugs in the scaffolding, and got everything working end-to-end.

**Bugs fixed from the generated code:**
- `pyproject.toml` had a nonexistent setuptools build backend (`setuptools.backends._legacy:_Backend` → `setuptools.build_meta`)
- Airflow DAG's `load_to_postgres` was inserting every row twice (both `hook.insert_rows` and a raw `cur.execute` loop)
- `init.sql` hardcoded `vector(384)` which would reject PubMedBERT's 768-dim embeddings. Changed to untyped `vector`
- IVFFlat index was created at table init time, but IVFFlat needs existing rows to build clusters. Deferred to post-data-load
- `api.py` ignored the `DATABASE_URL` env var from docker-compose, had a hardcoded localhost string
- MLflow port 5000 conflicts with macOS AirPlay Receiver. Remapped to 5001

**Data loaded:**
- 9,693 PubMed papers across 5 MeSH categories (nutrition, exercise, psychology, habits, ethics)
- Papers from the last year, ~2000 per category
- 9,693 MiniLM embeddings (384-dim), 974 PubMedBERT embeddings (768-dim)

**Model comparison results (974 papers, logged in MLflow):**

| Metric | MiniLM | PubMedBERT |
|--------|--------|------------|
| Mean similarity | 0.476 | 0.516 |
| MeSH overlap | **0.240** | 0.157 |
| Latency | **4.1ms** | 7.5ms |

MiniLM retrieves papers with better MeSH term overlap and is faster. PubMedBERT has higher raw similarity scores but that's a property of the embedding space, not necessarily better retrieval. With only 974 papers this isn't conclusive. Worth re-running at full corpus size.

**Search quality at 9.7K papers:**
- "effects of creatine on muscle recovery" → 0.654 similarity, relevant results
- "ethics of AI in clinical decision making" → 0.812 similarity, highly relevant results
- Latency ~60-80ms without IVFFlat index (sequential scan)

**Tests:** 14 passing — PubMed client XML parsing, date handling, rate limiting, API validation, endpoint behavior.

**Infrastructure:** Docker Compose running Postgres+pgvector, MLflow, Airflow, FastAPI. K8s manifests written but not deployed. MCP server tested inside Docker, `.mcp.json` config created for Claude Code integration.

## 2026-03-06 — Scaling and indexing

### What changed

**Scaled to full corpus:** Embedded all 9,693 papers with both MiniLM and PubMedBERT. PubMedBERT took ~10.5 minutes (13.7 papers/sec on CPU).

**HNSW indexes:** IVFFlat doesn't work on untyped vector columns. Neither does HNSW directly. Solved with expression indexes using casts: `USING hnsw ((embedding::vector(384)) vector_cosine_ops)`. Required updating all API queries to cast both sides: `e.embedding::vector(384) <=> %s::vector(384)`. Latency dropped from ~80ms to ~7-22ms.

**Full-corpus model comparison (9,693 papers):**

| Metric | MiniLM | PubMedBERT |
|--------|--------|------------|
| Mean similarity | 0.584 | 0.600 |
| MeSH overlap | **0.357** | 0.340 |
| Latency | **7.0ms** | 10.1ms |

MiniLM still wins on retrieval quality (MeSH overlap) and speed. The gap narrowed compared to the 974-paper run. PubMedBERT's top hits are sometimes more precisely targeted (e.g., it returns the creatine cognition review directly for the creatine query) but its overall top-10 is less consistently on-topic.

**Other fixes:**
- Added exponential backoff retry logic to `pubmed_client._get()` for 429 rate limit responses
- Rewrote README with Design Decisions section explaining pgvector, untyped vectors, MeSH evaluation, Airflow, and MCP choices
- Updated project structure in README to match actual files

## 2026-03-07 — Scale-up and NDCG evaluation

### What changed

**Scaled to ~40K papers.** Pulled 3 years of data, ~10K per category. Final count: 39,731 papers with MiniLM embeddings. PubMedBERT stays at 9,693 (embedding 30K more at ~14 papers/sec would take 35+ minutes on CPU, not worth it for a portfolio piece).

**NDCG evaluation harness.** Built a proper evaluation system with graded relevance scoring (0-3 scale based on MeSH term overlap tiers: high/medium/low relevance terms per query). 8 evaluation queries covering all 5 categories plus cross-domain topics (sleep/cognition, gut/brain, resistance training for elderly).

**Results at 39,731 papers (MiniLM):**

| Query | NDCG@5 |
|-------|--------|
| Creatine + muscle recovery | 1.00 |
| Psychological effects of quitting alcohol | 0.59 |
| HIIT benefits | 0.59 |
| Vegetarian protein | 0.97 |
| AI ethics in healthcare | 0.92 |
| Sleep deprivation + cognition | 0.67 |
| Gut microbiome + mental health | 0.87 |
| Resistance training for elderly | 1.00 |
| **Mean NDCG@5** | **0.83** |
| **Mean NDCG@10** | **0.91** |

Mean latency: 3.9ms with HNSW index.

The weakest queries (alcohol psychology, HIIT) suffer because the corpus has fewer papers with exact MeSH matches for those topics. The strongest queries (creatine, resistance training) achieve perfect NDCG because the corpus is dense with directly relevant papers.

**Other additions:**
- GitHub Actions CI (passing)
- Makefile with targets for all common operations
- 25 tests total (14 original + 11 evaluation tests)

## 2026-03-07 — API improvements and infra cleanup

### What changed

**On-demand model loading.** The API now supports switching between MiniLM and PubMedBERT at query time via the `model_name` parameter. The default model (MiniLM) loads at startup. PubMedBERT loads on first request and stays cached in memory. Added validation for unknown model names.

**Prometheus metrics endpoint.** `GET /metrics` returns Prometheus-compatible text format with:
- Total request count and per-endpoint breakdown
- Search latency (sum, count, average)
- Number of models loaded in memory
- Error count

**Multi-stage Docker build.** Split into builder and runtime stages. The builder installs all dependencies (including build-essential for native extensions), then only the installed packages are copied to the slim runtime image. Build tools don't ship in production.

**Separated Airflow DB.** Added a dedicated `airflow-db` Postgres service in docker-compose. Airflow's 50+ metadata tables no longer clutter the application database. Airflow still connects to the app DB for DAG operations via the `pubmed_postgres` connection. Cleaned up the 48 leftover Airflow tables from the app DB.

**Airflow DAG verified.** Triggered the DAG through Airflow CLI. Task execution works correctly: `get_ingestion_state` reads from app DB, `fetch_abstracts` calls PubMed API with retry logic, `load_to_postgres` upserts results. All 5 category tasks run in parallel. Rate limiting kicks in when both the scheduled and manual runs fire simultaneously (10 concurrent PubMed API calls), but the retry mechanism handles it.

**Tests expanded to 28.** Added tests for the `/metrics` endpoint (Prometheus format validation, request tracking) and unknown model rejection (400 response). All passing in CI.

**Other cleanup:**
- Added `.dockerignore` to exclude `.venv`, `.git`, `__pycache__`, caches from Docker build context
- Updated README: HNSW latency numbers, NDCG results, metrics endpoint, on-demand model loading, multi-stage build, CI, full project structure
- Docker multi-stage image built and verified running against live DB
