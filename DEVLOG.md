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
