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

## PyTorch / Model Training

- [x] **Fine-tune MiniLM on PubMed abstracts.** Contrastive learning with 100K MeSH-based pairs using `MultipleNegativesRankingLoss`. NDCG@5 improved from 0.83 to 0.86, with biggest gains on previously weak queries (sleep deprivation +0.19, HIIT +0.13).
- [x] **Train a cross-encoder re-ranker.** Two-stage pipeline: bi-encoder retrieves top-50, cross-encoder re-ranks to top-10. Trained on 20K MeSH-derived examples. NDCG@5 improved from 0.83 to 0.92 (+0.09). HIIT query went from 0.59 to 1.00. Adds ~272ms latency per query.
- [x] **ONNX export + quantization.** Exported MiniLM to ONNX, quantized to INT8. 5.3x speedup (4.4ms → 0.84ms per query) with only 0.017 NDCG@5 degradation. ONNX FP32 is lossless.
- [x] **Distill PubMedBERT into a smaller model.** KL divergence on pairwise similarity distributions, temperature=2.0, 3 epochs on 40K texts. Net result: NDCG@5 0.812 vs 0.828 baseline (-0.016). Per-query gains on HIIT (+0.19) and AI ethics (+0.08) but regression on sleep deprivation (-0.32). Transferred some domain knowledge but not enough to beat contrastive fine-tuning.
- [x] **Custom embedding model from scratch.** Initialized from `bert-base-uncased` with mean pooling, trained on 10K MeSH pairs using MultipleNegativesRankingLoss. 1 epoch, ~8 min. Training converged (loss 1.17) but NDCG evaluation blocked by Neon 512MB storage limit (768-dim embeddings too large to store alongside existing 40K MiniLM embeddings).

## Infrastructure

- [x] **Grafana dashboard.** Prometheus scrapes `/metrics` every 15s, Grafana auto-provisions with a pre-built dashboard showing request rate, latency, errors, and model status. Grafana at :3000, Prometheus at :9090.
- [ ] **Model registry with MLflow.** Register embedding models as versioned artifacts. Add a promotion workflow (staging → production) so model swaps don't require a code deploy.
- [ ] **Async DB connections.** Replace psycopg2 with asyncpg to match FastAPI's async model. Add connection pooling.
- [ ] **A/B testing for models.** Route a percentage of traffic to a new model, compare NDCG and latency in production. Log which model served each request.
