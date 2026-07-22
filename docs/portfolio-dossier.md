# PubMed ML Platform — Portfolio Dossier

Source material for a blog post and design diagrams. Everything here is verified against the repo, DEVLOG.md, git history, and live systems as of 2026-07-22.

---

## 1. Elevator pitch

An end-to-end semantic search engine over ~40,000 PubMed biomedical abstracts, built solo as an ML-infrastructure learning project: Airflow ingestion → Postgres+pgvector storage → embedding pipeline with MLflow experiment tracking → FastAPI serving with A/B testing and Prometheus/Grafana observability → an MCP server exposing search as LLM tools. Five model-training experiments (contrastive fine-tuning, cross-encoder re-ranking, knowledge distillation, from-scratch training, ONNX INT8 quantization) are evaluated with a graded-relevance NDCG harness. The system runs in production today at **$0/month** — the INT8 quantization experiment became the production serving path when free-tier constraints made torch impossible.

Live: https://pubmed-search-683d.onrender.com (docs at `/docs`). Repo: https://github.com/chibanaryan/pubmed-ml-platform

## 2. Motivation

- **Primary goal:** build demonstrable, hands-on breadth for an ML infrastructure job search — not a toy notebook, but a system with ingestion, storage, training, serving, monitoring, CI/CD, and a real deploy.
- **Why semantic search over PubMed:** free high-quality data (NCBI E-utilities), a natural embedding use case, and MeSH terms provide a *structured relevance signal* enabling quantitative evaluation without hand-labeling.
- **Working style:** AI-assisted development (Claude Code) from an initial generated scaffold, with the developer directing architecture, catching bugs, running experiments, and recording every result and gotcha in `DEVLOG.md`. The DEVLOG-as-lab-notebook discipline is itself part of the portfolio story.

## 3. Chronology

### Era 1 — Build sprint (2026-03-06 → 03-07, ~27 commits in two days)

**Day 1 (03-06): scaffold → working system**
- Organized generated scaffold into a real repo; fixed scaffold bugs: nonexistent setuptools backend, Airflow DAG double-inserting every row, `init.sql` hardcoding `vector(384)` (would reject 768-dim), IVFFlat index created before data existed (IVFFlat needs rows to build clusters), API ignoring `DATABASE_URL`, MLflow port 5000 vs macOS AirPlay conflict (→ 5001).
- Ingested 9,693 papers across 5 MeSH categories (nutrition, exercise, psychology, habits, ethics); embedded with MiniLM (384-d) and PubMedBERT (768-d).
- First model comparison at 974 papers: MiniLM won MeSH overlap 0.240 vs 0.157 — **later shown to be misleading** (see §9.2).
- HNSW *expression* indexes on casts (plain indexes don't work on the untyped vector column): search latency 80ms → ~4ms.
- 14 tests, K8s manifests written (not deployed), MCP server + `.mcp.json`.

**Day 2 (03-07): scale, evaluate, train, deploy**
- Scaled to 39,731 papers (3 years of data). PubMedBERT embedding at 13.7 papers/sec on CPU.
- **NDCG evaluation harness**: 8 queries, graded relevance 0–3 from tiered MeSH overlap. Baseline MiniLM: NDCG@5 0.83, NDCG@10 0.91, 3.9ms mean latency.
- CI (GitHub Actions), Makefile, Prometheus `/metrics`, multi-stage Docker build, separate Airflow metadata DB, on-demand model loading. Deployed to Fly.io (CPU-only torch image: 641MB vs 3.5GB CUDA).
- **Five training experiments** (see §5) — all in one day, each properly evaluated.
- Prometheus+Grafana stack, psycopg2→asyncpg migration (see §9.5 for the bug this silently introduced), MLflow model registry with `@production` alias promotion, A/B traffic splitting.
- Full-corpus PubMedBERT run reversed the early comparison: **0.90 vs 0.83 NDCG@5** at 40K papers.

### Era 2 — Production hardening (2026-07-20)

Assessed gaps against ML-infra job expectations; executed "Tier 1":
- **CI-gated evaluation**: `--min-ndcg` flag on the eval harness; `make eval-gate`; on-demand GitHub workflow running NDCG against the production DB (`EVAL_DATABASE_URL` secret). Verified in CI: 0.8275 ≥ 0.80 threshold, gate passed.
- **CI/CD**: mypy (tiered strictness), pytest coverage artifact, Docker build + push to GHCR with layer caching.
- **Observability rewrite**: hand-rolled metrics dict → `prometheus_client` (labeled counters via middleware, per-model latency histograms enabling `histogram_quantile`); JSON structured logging; Prometheus alert rules (APIDown, error-rate >5%, search p95 >500ms). Notably: the old `errors_total` metric had *never been incremented* — middleware wired it for real.
- **Load testing**: locust harness; 20 users / 60s against 40K papers locally: 2,019 requests, 0 failures, 34 req/s; `/search` p50 15ms / p95 71ms / p99 460ms.
- **Tests 28 → 49**: embed pipeline, MLflow registry, MCP server (httpx MockTransport).
- **Found a production-breaking bug** during live verification (§9.5).

### Era 3 — The $0 redeploy (2026-07-22)

- Fly.io trial had ended (app suspended, card required). Hugging Face had made Docker Spaces PRO-only. The only free, no-card host left: Render free tier — 512MB RAM, 0.1 vCPU.
- **Torch cannot fit 512MB → the INT8 ONNX experiment became production serving.** New torch-free code path (`SERVING_BACKEND=onnx`): onnxruntime + bare `tokenizers`, manual mean-pooling/L2-norm. Verified in a clean venv: **211MB RSS**. Model artifacts published to HF Hub (`chibanaryan/minilm-pubmed-onnx`), downloaded at first request.
- Deployed via `render.yaml` + Render API. Two production incidents diagnosed and fixed same-day (§9.6, §9.7). Final warm `/search`: **75–99ms** (encode 30–60ms + DB 37–50ms), measured via the new stage-timing structured logs.

## 4. Architecture

**Data flow:** Airflow DAG (daily, per-category tasks in parallel, incremental state in `ingestion_state`) → PubMed E-utilities client (rate-limited, exponential backoff on 429) → `papers` table (JSONB authors/mesh) → embed pipeline (batched, MLflow-tracked) → `embeddings` table (untyped `vector` column, multi-model) → FastAPI (asyncpg pool, lazy model load: MLflow registry `@production` → HF fallback → cache) → clients, including an MCP server (stdio + SSE) exposing `search_papers` / `get_paper` / `find_similar` to LLMs.

**Two serving backends:** `torch` (SentenceTransformer, supports MiniLM + PubMedBERT, used locally/docker) and `onnx` (INT8, MiniLM only, 211MB RSS, used in production).

**Suggested diagrams:**
1. System architecture (components + data flow above)
2. Two-stage retrieval: bi-encoder top-50 → cross-encoder re-rank top-10
3. Experiment tree: baseline 0.83 → {fine-tune 0.86, cross-encoder 0.92, distill 0.81, PubMedBERT 0.90} with latency annotations
4. Deployment evolution: local Docker Compose stack (7 services) → Fly.io (died) → Render+Neon+HF Hub ($0)
5. CI/CD: push → lint/mypy/test+coverage → Docker→GHCR; on-demand eval gate → NDCG vs threshold
6. The untyped-vector design: one table, two dimensions, expression indexes, cast-both-sides queries

## 5. Experiments & results (all logged in MLflow / DEVLOG)

| Experiment | Method | Result (NDCG@5) | Notes |
|---|---|---|---|
| Baseline | MiniLM (all-MiniLM-L6-v2), 384-d | 0.83 (NDCG@10 0.91) | 3.9ms search w/ HNSW |
| Contrastive fine-tune | 100K MeSH-derived positive pairs, MultipleNegativesRankingLoss, 1 epoch, bs 64, ~20min on Apple MPS | **0.86** | Gains on weak queries: sleep +0.19, HIIT +0.13; re-embedded 40K corpus for honest eval |
| Cross-encoder re-rank | ms-marco-MiniLM-L-6-v2 fine-tuned on 20K graded triples; retrieve 50 → re-rank 10 | **0.92** | +272ms/query; HIIT 0.59→1.00 |
| Knowledge distillation | PubMedBERT→MiniLM, KL on pairwise similarity dists, T=2.0, 3 epochs | 0.81 (**negative result**) | HIIT +0.19, AI ethics +0.08 but sleep −0.32; lesson: similarity distillation lacks task signal vs contrastive pairs |
| From-scratch | bert-base-uncased + mean pooling, 10K pairs | n/a — eval blocked | Neon 512MB couldn't hold a third 40K×768-d embedding set; lesson about model size vs infra budget |
| PubMedBERT full corpus | 768-d, domain-specific | **0.90** (NDCG@10 0.96) | Reversed the small-corpus conclusion |
| ONNX INT8 | torch.onnx.export + dynamic quantization | −0.017 vs baseline | 5.3× faster: 4.41ms → 0.84ms (p50); FP32 export lossless at 1.46ms |

Load/serving numbers: local stack 34 req/s @ 20 users, 0 failures, `/search` p50 15ms / p95 71ms / p99 460ms (p99 = shared in-process encoder contention). Production (Render 0.1 vCPU): warm 75–99ms.

## 6. Technology inventory

| Layer | Tech | Used for |
|---|---|---|
| Orchestration | Apache Airflow 2.10 (TaskFlow API, separate metadata DB) | Daily incremental ingestion, per-category parallel tasks, retries |
| Data | PostgreSQL + pgvector (HNSW expression indexes), Neon (managed, prod), JSONB columns | Papers + multi-model embeddings in one schema |
| Training | PyTorch (MPS), sentence-transformers, HuggingFace Transformers | Fine-tune, cross-encoder, distillation, from-scratch |
| Experiment mgmt | MLflow (tracking + model registry, alias-based promotion) | Every run logged; API loads `models:/<name>@production` |
| Inference opt. | ONNX Runtime, dynamic INT8 quantization, single-thread pinning | Production serving in 211MB / 0.1 vCPU |
| Serving | FastAPI, asyncpg pool, Pydantic v2, uvicorn | Async API, A/B testing, lazy model loading |
| Observability | prometheus_client, Prometheus (+alert rules), Grafana (provisioned dashboard), JSON structured logging | Labeled counters, latency histograms, stage timings |
| Load testing | Locust | p50/p95/p99 under concurrency |
| CI/CD | GitHub Actions (ruff, mypy, pytest+coverage, Docker→GHCR, on-demand NDCG eval gate) | Quality gates incl. model quality |
| Packaging | Docker multi-stage (CPU-only torch), docker-compose (7 services), K8s manifests (written, undeployed) | Local stack + deploy |
| Hosting | Render (free tier), formerly Fly.io; HF Hub for model artifacts | $0/month production |
| LLM integration | MCP (stdio + SSE), httpx | Search as LLM tools |
| Testing | pytest, AsyncMock, httpx MockTransport, 52 tests | No-Docker unit suite |

## 7. Skills → ML-infra role mapping

- **Model training breadth**: contrastive learning, re-ranking, distillation, from-scratch training, quantization — with honest evaluation of each (including a documented negative result).
- **Evaluation engineering**: proxy-label harness (MeSH → graded relevance), NDCG@k, apples-to-apples re-embedding, eval-as-CI-gate.
- **MLOps**: experiment tracking, model registry with alias promotion (model swap without deploy), A/B traffic splitting with per-model metrics, model artifacts on a hub rather than in git.
- **Vector search internals**: multi-dimension schema design, expression indexes, the cast-both-sides sequential-scan trap, index lifecycle (build after load).
- **Inference optimization**: ONNX export pitfalls (pooling not in graph, MPS→CPU, dynamic axes), INT8 trade-off measurement, thread-pool tuning for fractional-vCPU containers.
- **Serving/backend**: async Python, connection pooling, lazy loading, middleware instrumentation, structured logging, memory budgeting (211MB proof).
- **Data engineering**: incremental ingestion state, rate limiting/backoff, idempotent upserts, orchestration.
- **Operations**: root-cause debugging in production (five distinct incidents, §9), load testing, alert rules, deploy pipelines across three hosting providers' constraints.

## 8. Design decisions & trade-offs

1. **pgvector over a dedicated vector DB.** One database for metadata + vectors; joins are trivial; no sync between stores. Cost: fewer vector-native features. At 40K vectors HNSW gives ~4ms — dedicated infra unjustified.
2. **Untyped `vector` column (load-bearing).** Lets 384-d and 768-d coexist in one table discriminated by `model_name`, keeping comparisons clean. Costs: no schema-level dimension enforcement; plain indexes don't work (expression indexes on casts required); *every query must cast both sides* or silently seq-scans (80ms vs 4ms).
3. **MeSH terms as relevance proxy.** No hand-labeled judgments; tiered MeSH overlap → graded 0–3 relevance. Enables NDCG without annotation budget. Cost: bounded by MeSH quality; weak-topic queries score low partly due to corpus sparsity.
4. **Airflow, knowingly overkill.** A cron job would suffice at this scale; chosen to demonstrate production orchestration patterns (state, retries, parallelism, monitoring). Cost honestly documented.
5. **asyncpg over psycopg2 for serving.** Non-blocking event loop, real pooling. Cost: different param style, and a silent production breakage (§9.5) — the migration's true price was found only by live testing.
6. **Serve the quantized model, not the best model.** Production runs INT8 MiniLM (NDCG −0.017) instead of PubMedBERT (+0.07) because the free tier dictates 512MB/0.1 vCPU. An explicit cost/quality/latency triangle decision — and the quantization experiment paying rent.
7. **Tiered mypy strictness.** Serving/ingestion/MCP held to `check_untyped_defs`; torch-heavy training CLIs exempted — annotating around torch stubs has negative ROI. Type-checking budget spent where correctness matters most.
8. **Eval gate as on-demand workflow, not push CI.** The gate needs a populated DB; push CI has none. Honest split: fast checks on every push, model-quality gate on demand against production data.
9. **MCP server instead of a chat UI.** Composable LLM tool-use beats reinventing a conversation layer; extends naturally (chain `search_papers` → `find_similar`).

## 9. War stories (interview / blog material)

1. **The scaffold lied.** AI-generated scaffold had five real bugs (double-insert DAG, premature IVFFlat, dimension-hardcoded schema, ignored env var, broken build backend). Lesson: generated code needs the same review as human code; finding these is the skill.
2. **The corpus-size reversal.** At ≤10K papers MiniLM beat PubMedBERT on MeSH overlap; at 40K, PubMedBERT won decisively (0.90 vs 0.83 NDCG@5) — denser topic clusters let domain knowledge show. Lesson: benchmark conclusions are scale-dependent; the early "honest caveat" in the writeup was later replaced with data.
3. **Distillation that made things worse.** Similarity-distribution KL transfer moved knowledge (HIIT +0.19) but hurt elsewhere (sleep −0.32), net −0.02. Lesson: distillation objectives must encode *what matters for the task*; contrastive pairs did.
4. **Neon's 512MB reality.** Deleted embedding rows don't free space until auto-vacuum catches up; a third full-corpus embedding set was impossible. Blocked the from-scratch eval; forced storage discipline. Lesson: managed-service quotas are architectural constraints, not billing details.
5. **Search silently broken in production for four months.** The psycopg2→asyncpg migration passed all 28 tests — which mock `conn.fetch`. First live search: `DataError: expected str, got list`. asyncpg has no pgvector codec; embeddings must be passed in text form. Lesson: mocked tests validate shapes, not DB contracts; one real-Postgres smoke test would have caught it instantly. (Fixed with a regression test pinning the wire format.)
6. **The fix that "didn't work" — because it never deployed.** Render services created via API from public repos get no GitHub webhook; `autoDeploy` silently no-ops. Two pushes never reached production, making a correct fix look ineffective and a commit-keyed deploy-watch loop hang forever. Lesson: verify the running build contains your fix before judging it.
7. **Thread thrash on 0.1 vCPU.** onnxruntime's default per-core thread pool fought the cgroup throttle: 1–2.2s searches. `intra_op_num_threads=1` → 75–99ms. Diagnosed by endpoint triangulation (`/similar` fast ⇒ encode slow) + stage-timing logs. Lesson: library defaults assume dedicated cores.
8. **The free-tier landscape shifts under you.** Fly trial expired (card wall); HF made Docker Spaces PRO-only mid-project. Each pivot documented; final answer combined three providers' free tiers (Render compute, Neon storage, HF Hub artifacts).
9. **Assorted paper cuts** (DEVLOG "issues encountered"): CrossEncoder.fit() not persisting models in sentence-transformers v4; ONNX export needing CPU tensors while the model sat on MPS; sentence-transformers pooling absent from the ONNX graph (reimplemented in numpy); `encode()` returning detached tensors breaking distillation gradients (used `auto_model` directly); MLflow port 5000 vs macOS AirPlay; local Postgres vs Docker on 5432; mypy version drift between local and CI breaking on an ignore comment.

## 10. Production state (2026-07-22)

- **API**: Render free tier (Ohio), `SERVING_BACKEND=onnx`, 211MB RSS, warm `/search` 75–99ms, sleeps after ~15min idle (~1min wake). Deploys triggered via Render API (webhook gap documented).
- **DB**: Neon free tier (us-east-1), 39,731 papers + MiniLM embeddings, HNSW-indexed.
- **Model artifacts**: HF Hub `chibanaryan/minilm-pubmed-onnx` (INT8 + tokenizer).
- **CI**: green — ruff, mypy, 52 tests + coverage, Docker→GHCR; eval gate on demand (last run: 0.8275 ≥ 0.80).
- **Pipeline**: DAG ingests *and* embeds, so ingested papers are searchable without a manual step. Verified: full run, zero task failures, 40,007 papers across five categories; backlog drained to 0 unembedded; NDCG@5 0.8267 on the mixed fp32/INT8 corpus.
- **MCP**: hosted at `/mcp` (streamable HTTP, stateless); browsers get a setup page instead of a protocol error.
- **Cost: $0/month** across Render + Neon + HF Hub + GitHub.

## 11. Honest gaps & roadmap (Tier 2/3)

Not yet done — and worth saying so in the blog post, because the gap analysis is itself an infra skill:
- K8s manifests exist but have never been applied to a real cluster (next: kind/k3s + HPA + Ingress + Helm).
- No IaC (Terraform) — hosting is click/API-provisioned.
- All training on laptop MPS/CPU; no cloud-GPU or distributed (DDP/Ray) run.
- No data versioning (DVC); non-serving model artifacts live only on one laptop.
- Secrets hygiene: k8s manifests carry plaintext dev credentials; `pubmed:pubmed` defaults in compose.
- No tracing (OpenTelemetry), no streaming ingestion, no drift monitoring/retraining triggers.

## 12. Blog post angles

1. **"My quantization experiment became my production server"** — the $0 deploy story (Era 3), strongest single narrative arc: constraint → experiment reuse → measured trade-off → incident debugging.
2. **"What 40K papers taught me that 10K couldn't"** — evaluation rigor: proxy labels, NDCG, the scale reversal, the negative distillation result.
3. **"Five bugs my mocked tests couldn't see"** — testing philosophy via the asyncpg wire-format bug, the undeployed fix, and friends.
4. **"An ML platform on three free tiers"** — architecture tour with the cost/constraint lens throughout.
5. Combined long-form: chronological build log (the DEVLOG practically is one) with the diagrams from §4.
