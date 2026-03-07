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

## 2026-03-07 — Fine-tuning, cross-encoder, ONNX

### Contrastive fine-tuning

Fine-tuned MiniLM on PubMed abstracts using `MultipleNegativesRankingLoss`. Training data: ~100K positive pairs from papers sharing 2+ topic-relevant MeSH terms (after filtering out demographics, study design, and geography MeSH terms). 1 epoch, batch size 64, ~20 minutes on Apple Silicon MPS.

**Results:** NDCG@5 improved from 0.83 to 0.86. The gains were concentrated on previously weak queries: sleep deprivation +0.19, HIIT +0.13. Strong queries held steady. Re-embedded all 40K papers with the fine-tuned model and stored under `model_name="minilm-pubmed-ft"` for proper apples-to-apples evaluation.

### Cross-encoder re-ranker

Built a two-stage search pipeline: bi-encoder retrieves top-50, cross-encoder re-ranks to top-10. Fine-tuned `cross-encoder/ms-marco-MiniLM-L-6-v2` on ~20K (query, document, score) triples. Queries are MeSH terms converted to natural language, with graded relevance scoring (0.0 for negatives, 0.5-1.0 for positives based on MeSH overlap depth).

**Results:** NDCG@5 improved from 0.83 to 0.92 (+0.09). HIIT query went from 0.59 to 1.00. Adds ~272ms latency per query for re-ranking 50 candidates.

**Issues encountered:**
- sentence-transformers v4 `CrossEncoder.fit()` uses the Trainer API internally and didn't always persist the model to `output_dir`. Fixed by adding explicit `model.save(output_dir)` after `.fit()`.
- `CrossEncoder()` rejects relative paths as model identifiers. Had to use `Path.resolve()` for absolute paths.
- DB connection timed out during 6-minute training. Split training and evaluation into separate commands with fresh connections.

### ONNX export + INT8 quantization

Exported MiniLM to ONNX via `torch.onnx.export` with dynamic axes for variable batch size and sequence length. Quantized to INT8 using `onnxruntime.quantization.quantize_dynamic`.

Had to reimplement mean pooling manually since sentence-transformers' pooling layer isn't part of the exported ONNX graph. The `OnnxEmbedder` class wraps ONNX Runtime inference with numpy-based mean pooling and L2 normalization.

**Latency results (100 iterations):**

| Model | Mean | P50 | P95 |
|-------|------|-----|-----|
| PyTorch | 4.41ms | 4.25ms | 5.12ms |
| ONNX FP32 | 1.46ms | 1.39ms | 1.68ms |
| ONNX INT8 | 0.84ms | 0.82ms | 0.97ms |

INT8 is 5.3x faster than PyTorch. NDCG@5 degradation: -0.017 for INT8, 0.000 for FP32.

**Issues encountered:**
- `onnxscript` not installed (required by newer torch.onnx). Pip installed it.
- `AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")` fails; needs full HuggingFace path `sentence-transformers/all-MiniLM-L6-v2`. Added `DB_MODEL_NAME` as separate constant.
- MPS device conflict during export: the transformer model was on MPS but ONNX export needs CPU tensors. Fixed with `transformer.auto_model.cpu()`.

## 2026-03-07 — Knowledge distillation

### What changed

**Distilled PubMedBERT into MiniLM.** Used KL divergence on pairwise similarity distributions to transfer PubMedBERT's domain knowledge into MiniLM's smaller architecture. For each batch of 32 texts, both models compute pairwise cosine similarity matrices. The teacher's distribution (after softmax with temperature=2.0) becomes the target, and the student is trained to match it.

Training: 3 epochs on 40K paper texts, ~20 min per epoch on MPS. Loss converged from 0.0008 to 0.0004.

**Results:**

| Query | Base | Distilled | Delta |
|-------|------|-----------|-------|
| Creatine | 1.00 | 1.00 | 0.00 |
| Quitting alcohol | 0.59 | 0.55 | -0.04 |
| HIIT | 0.59 | 0.78 | +0.19 |
| Vegetarian protein | 0.97 | 0.92 | -0.05 |
| AI ethics | 0.92 | 1.00 | +0.08 |
| Sleep deprivation | 0.68 | 0.36 | -0.32 |
| Gut microbiome | 0.87 | 0.89 | +0.02 |
| Resistance training | 1.00 | 1.00 | 0.00 |
| **Mean NDCG@5** | **0.83** | **0.81** | **-0.02** |

The distillation transferred some domain-specific knowledge (HIIT +0.19, AI ethics +0.08) but hurt others (sleep deprivation -0.32). Net effect was a slight regression. The contrastive fine-tuning approach (NDCG@5 0.86) remains the better method for this use case.

**Issues encountered:**
- `SentenceTransformer.encode()` returns detached tensors with no `grad_fn`. Had to use the underlying `auto_model` directly for the student's forward pass with manual mean pooling.
- Neon free tier hit 512MB storage limit after storing 30K distilled embeddings (already had 40K base + 40K fine-tuned). Deleted the fine-tuned embeddings to make room, but Neon's auto-vacuum hadn't reclaimed physical space yet. Evaluated with 30K embeddings (still a representative sample).

**Takeaway:** Similarity-based distillation is a blunt tool. The teacher's pairwise similarity distribution captures document relationships, but it doesn't encode which relationships matter for retrieval. Contrastive fine-tuning with task-specific pairs (MeSH overlap) gives the model more directed signal about what "relevant" means in this domain.

## 2026-03-07 — Custom embedding model from scratch

### What changed

**Built a sentence embedding model from bert-base-uncased.** Architecture: Transformer + mean pooling (768-dim output). Trained on 10K MeSH-based pairs with `MultipleNegativesRankingLoss`, 1 epoch, batch size 32, max_seq_length 128. Training took ~8 min on MPS, loss converged to 1.17.

**Blocked by Neon storage limits.** The 768-dim BERT embeddings are 2x larger per row than MiniLM's 384-dim. With 40K base MiniLM embeddings already stored, there wasn't room for another 40K at 768-dim within Neon's 512MB free tier. The HNSW index also rejects 768-dim inserts because only the 384-dim expression index exists (the 768-dim index was dropped earlier).

A query-only comparison (encoding queries with BERT and searching against MiniLM embeddings) is meaningless here because the embedding spaces are completely incompatible. Proper evaluation requires storing the BERT embeddings and querying against them.

**Takeaway:** bert-base-uncased is ~4x slower than MiniLM (110M vs 22M params) and produces 768-dim embeddings that are harder to store and index. For a production search system, MiniLM's architecture (small, fast, 384-dim) is clearly the better starting point. Training from a larger base model only makes sense if you need the extra capacity and have the infrastructure to support it.

**Neon storage issue:** Deleted rows don't immediately free disk space. Neon's auto-vacuum reclaims space asynchronously, but in practice it took longer than expected. Multiple DELETEs of 30K-40K embedding rows didn't free enough space for new inserts within the session.
