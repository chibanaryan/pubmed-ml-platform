# TODO

## High priority — completes the portfolio story

- [x] **Re-run model comparison at full corpus size.** Both models at 9,693 papers. MiniLM wins on MeSH overlap (0.357 vs 0.340) and latency (7ms vs 10ms).
- [x] **Create vector indexes.** HNSW expression indexes on casted vectors. Latency dropped from ~80ms to ~7-22ms.
- [x] **Get MCP server working from Claude Code.** Venv + `.mcp.json` configured, tested locally against running API.
- [x] **Architecture blog post / README writeup.** Design Decisions section added covering pgvector, untyped vectors, MeSH evaluation, Airflow, and MCP.

## Medium priority — makes it more credible

- [x] **GitHub Actions CI.** Lint with ruff, run tests on push.
- [x] **Add Makefile.** Targets: up, down, test, lint, ingest, embed, compare, logs.
- [x] **Retry logic in pubmed_client.** Exponential backoff on 429s added.
- [ ] **Scale to 50K+ papers.** 9.7K is fine for a demo but doesn't stress the infrastructure. Pulling 10K per category over a longer time range would make the "this needs real infrastructure" argument more honest.
- [ ] **Evaluation harness improvements.** Current eval uses MeSH term overlap as a proxy for relevance. Add NDCG with hand-labeled relevance judgments for 5-10 queries.

## Lower priority — nice to have

- [ ] **PubMedBERT as selectable model in the API.** The API already accepts `model_name` as a param and handles dimension lookup. Just needs the model loaded at startup or loaded on-demand.
- [ ] **Airflow DAG testing.** The DAG imports work and it shows up in the Airflow UI, but it hasn't been triggered through Airflow itself.
- [ ] **Prometheus metrics endpoint.** The README mentioned it initially but it doesn't exist. Add request count, latency histogram.
- [ ] **Docker image optimization.** Multi-stage build to cut image size.
- [ ] **Separate Airflow DB from application DB.** Both share the same Postgres instance, cluttering the schema with 50+ Airflow tables.
