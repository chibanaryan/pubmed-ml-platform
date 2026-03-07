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

## Remaining

- [ ] **PubMedBERT as selectable model in the API.** The API already accepts `model_name` as a param and handles dimension lookup. Just needs the model loaded at startup or loaded on-demand.
- [ ] **Airflow DAG testing.** The DAG imports work and it shows up in the Airflow UI, but it hasn't been triggered through Airflow itself.
- [ ] **Prometheus metrics endpoint.** Add request count, latency histogram.
- [ ] **Docker image optimization.** Multi-stage build to cut image size.
- [ ] **Separate Airflow DB from application DB.** Both share the same Postgres instance, cluttering the schema with 50+ Airflow tables.
