# TODO

## High priority — completes the portfolio story

- [ ] **Re-run model comparison at full corpus size.** Only 974 papers were embedded with PubMedBERT. Embed all 9.7K and re-run evaluation to get a real comparison. The current numbers aren't apples-to-apples.
- [ ] **Create IVFFlat indexes.** Latency is ~60-80ms with sequential scan. After indexing it should drop to single-digit ms. Need to tune `lists` parameter based on corpus size (rule of thumb: sqrt(n_rows), so ~100 for 10K).
- [ ] **Get MCP server working from Claude Code.** Venv is set up, `.mcp.json` exists, but hasn't been tested live from a Claude Code session in the project directory. Need to verify stdio transport works.
- [ ] **Architecture blog post / README writeup.** The README has the basics but needs a "Design Decisions" section explaining why each tech choice was made. This is what interviewers will read.

## Medium priority — makes it more credible

- [ ] **GitHub Actions CI.** Lint with ruff, run tests on push. Simple workflow, ~20 min to set up.
- [ ] **Scale to 50K+ papers.** 9.7K is fine for a demo but doesn't stress the infrastructure. Pulling 10K per category over a longer time range would make the "this needs real infrastructure" argument more honest.
- [ ] **Add Makefile or task runner.** Common operations (ingest, embed, compare, test) currently require long docker compose exec commands. A Makefile would clean this up.
- [ ] **Evaluation harness improvements.** Current eval uses MeSH term overlap as a proxy for relevance. Add NDCG with hand-labeled relevance judgments for 5-10 queries. Small effort, big credibility signal.
- [ ] **Retry logic in pubmed_client.** The client has rate limiting but no retry on 429s. Hit this during the 2K/category ingestion. Add exponential backoff.

## Lower priority — nice to have

- [ ] **PubMedBERT as selectable model in the API.** Currently hardcoded to MiniLM. The embeddings table supports multiple models, but the API only queries one. Add a query param or config.
- [ ] **Airflow DAG testing.** The DAG imports work and it shows up in the Airflow UI, but it hasn't been triggered through Airflow itself. Test the full DAG execution path.
- [ ] **Prometheus metrics endpoint.** The README mentions it but it doesn't exist. Add request count, latency histogram, embedding cache hit rate.
- [ ] **Docker image optimization.** Current image installs everything including dev deps. Multi-stage build would cut the image size significantly.
- [ ] **Separate Airflow DB from application DB.** Both currently share the same Postgres instance, which clutters the schema (51 tables, most are Airflow's). Use a separate DB or at minimum a separate schema.
