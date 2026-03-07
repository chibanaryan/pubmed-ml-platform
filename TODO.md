# TODO

## Completed

- [x] **Re-run model comparison at full corpus size.** Both models at 9,693 papers. MiniLM wins on MeSH overlap (0.357 vs 0.340) and latency (7ms vs 10ms).
- [x] **Create vector indexes.** HNSW expression indexes on casted vectors. Latency dropped from ~80ms to ~3.9ms.
- [x] **Get MCP server working from Claude Code.** Venv + `.mcp.json` configured, tested locally against running API.
- [x] **Architecture blog post / README writeup.** Design Decisions section covering pgvector, untyped vectors, MeSH evaluation, Airflow, and MCP.
- [x] **GitHub Actions CI.** Lint with ruff, run tests on push.
- [x] **Add Makefile.** Targets: up, down, test, lint, ingest, embed, compare, evaluate, logs.
- [x] **Retry logic in pubmed_client.** Exponential backoff on 429s.
- [x] **Scale to ~40K papers.** 39,731 papers with MiniLM embeddings.
- [x] **NDCG evaluation harness.** 8 queries with graded relevance scoring (0-3). Mean NDCG@5: 0.83, NDCG@10: 0.91.

- [x] **PubMedBERT as selectable model in the API.** On-demand model loading. Default model loads at startup, PubMedBERT loads on first request.
- [x] **Prometheus metrics endpoint.** `/metrics` endpoint with request counts, latency stats, models loaded, per-endpoint breakdown.
- [x] **Docker image optimization.** Multi-stage build separating build deps from runtime.
- [x] **Separate Airflow DB from application DB.** Dedicated `airflow-db` service in docker-compose, Airflow tables no longer clutter the app schema.

- [x] **Airflow DAG testing.** Triggered via CLI. Task pipeline works end to end. Rate limiting from concurrent runs handled by retry mechanism.

## Remaining

- [ ] **Add `.dockerignore`.** Exclude `.venv`, `.git`, `__pycache__`, etc. from Docker build context.
