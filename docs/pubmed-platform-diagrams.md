# PubMed ML Platform — Design Diagrams

Companion to the [portfolio dossier](portfolio-dossier.md). All numbers are measured (DEVLOG.md); Mermaid sources are blog-ready.

## 1. System architecture

```mermaid
flowchart LR
    subgraph Ingestion
        EU[PubMed E-utilities] -->|"rate-limited,<br/>backoff on 429"| PC[pubmed_client.py]
        AF[Airflow DAG<br/>daily · fetch serialized<br/>to respect rate limit] --> PC
    end

    subgraph Storage["Postgres + pgvector (Neon in prod)"]
        P[(papers<br/>39,731 rows)]
        E[(embeddings<br/>untyped vector,<br/>multi-model)]
        ST[(ingestion_state<br/>incremental cursor)]
    end

    PC -->|upsert| P
    AF <-->|track progress| ST

    subgraph ML["Embedding & training"]
        EP[embed_new_papers task<br/>INT8 ONNX · batched<br/>idempotent · capped]
        ML1[MLflow<br/>tracking + registry<br/>'@production' alias]
        EP --- ML1
    end

    P --> EP -->|"384-d / 768-d"| E

    subgraph Serving
        API[FastAPI<br/>asyncpg pool, A/B split]
        BE1[torch backend<br/>MiniLM + PubMedBERT]
        BE2[onnx backend<br/>INT8 MiniLM, 211MB]
        API -.->|local / docker| BE1
        API -->|production| BE2
    end

    E -->|"HNSW ~4ms"| API
    ML1 -->|registry-first<br/>model load| API

    subgraph Consumers
        MCP[MCP server<br/>search / get / similar] --> LLM[LLM clients]
        PR[Prometheus] --> GF[Grafana]
    end

    API --> MCP
    API -->|/metrics| PR

    classDef prod stroke:#2f855a,stroke-width:2.5px
    class BE2,API prod
```

## 2. Two-stage retrieval

```mermaid
flowchart LR
    Q["query:<br/>'benefits of HIIT'"] --> BI[bi-encoder<br/>MiniLM 384-d]
    BI -->|"query vector"| VS["pgvector HNSW<br/>cosine top-50<br/>~4ms"]
    VS --> CE["cross-encoder<br/>ms-marco-MiniLM<br/>scores all 50 pairs<br/>+272ms"]
    CE --> R["top-10<br/>NDCG@5: 0.83 → 0.92"]

    classDef stage stroke-width:2px
    class BI,VS,CE stage
    classDef win stroke:#2f855a,stroke-width:2.5px
    class R win
```

## 3. Experiment tree (NDCG@5 on the 8-query MeSH-graded eval set)

```mermaid
flowchart TD
    B["baseline<br/>MiniLM<br/>0.83 · 3.9ms search"]

    B -->|"contrastive fine-tune<br/>100K MeSH pairs, MNRL"| FT["minilm-pubmed-ft<br/>0.86"]
    B -->|"cross-encoder re-rank<br/>20K graded triples"| CE["two-stage<br/>0.92 · +272ms"]
    B -->|"KL distillation<br/>from PubMedBERT, T=2.0"| D["minilm-distilled<br/>0.81 · net regression"]
    B -->|"INT8 ONNX<br/>dynamic quantization"| O["serving model<br/>0.81 · 5.3× faster<br/>(0.84ms encode)"]
    B -->|"swap model<br/>domain-specific 768-d"| PB["PubMedBERT<br/>0.90 · reversed the<br/>10K-corpus verdict"]
    B -->|"from-scratch<br/>bert-base + 10K pairs"| FS["eval blocked:<br/>Neon 512MB limit"]

    classDef win stroke:#2f855a,stroke-width:2.5px
    class CE,PB win
    classDef neg stroke:#c53030,stroke-dasharray:5 4
    class D,FS neg
    classDef prod stroke:#2f855a,stroke-width:2.5px,stroke-dasharray:2 3
    class O prod
```

Solid green = best quality. Dotted green = shipped to production (quality/memory trade-off). Red dashed = negative result / blocked — both documented, not hidden.

## 4. Deployment evolution

```mermaid
flowchart LR
    subgraph Mar["March 2026 — v1"]
        FLY["Fly.io<br/>4GB VM, torch backend<br/>auto-stop machines"]
    end

    subgraph Jul20["July 2026 — reality check"]
        DEAD["Fly trial expired<br/>app suspended"]
        HFS["HF Docker Spaces<br/>now PRO-only (402)"]
    end

    subgraph Jul22["July 2026 — v2: $0/month"]
        R["Render free tier<br/>512MB · 0.1 vCPU<br/>onnx backend, 211MB RSS"]
        N["Neon Postgres<br/>40K papers + pgvector"]
        HF["HF Hub<br/>INT8 model artifacts"]
        R --> N
        R -->|"download on<br/>first request"| HF
    end

    FLY --> DEAD --> HFS -->|"torch can't fit 512MB →<br/>quantization experiment<br/>becomes the serving path"| R

    classDef dead stroke:#c53030,stroke-dasharray:5 4
    class FLY,DEAD,HFS dead
    classDef live stroke:#2f855a,stroke-width:2.5px
    class R,N,HF live
```

## 5. CI/CD with model-quality gate

```mermaid
flowchart LR
    PUSH([git push main]) --> CI["CI job<br/>ruff · mypy (tiered)<br/>52 tests + coverage"]
    CI --> DK["Docker build<br/>GHA layer cache"]
    DK -->|on main| GHCR[(ghcr.io image<br/>:latest + :sha)]

    DISPATCH([workflow_dispatch]) --> EG["eval gate<br/>NDCG@5 vs live Neon DB"]
    EG -->|"≥ 0.80<br/>(last: 0.8275)"| PASS([gate passed])
    EG -->|"< 0.80"| FAIL([exit 1])

    PUSH -.->|"no webhook —<br/>deploy is explicit"| RAPI["Render API<br/>POST /deploys"]
    RAPI --> LIVE([live: 75–99ms<br/>warm /search])

    classDef ok stroke:#2f855a,stroke-width:2.5px
    class PASS,LIVE ok
    classDef bad stroke:#c53030,stroke-dasharray:5 4
    class FAIL bad
```

## 6. The untyped-vector design (load-bearing schema decision)

```mermaid
flowchart TD
    subgraph T["embeddings table — one column, no dimension: 'embedding vector'"]
        R1["rows: model_name = 'all-MiniLM-L6-v2' → 384-d"]
        R2["rows: model_name = 'PubMedBERT…' → 768-d"]
    end

    T --> I1["HNSW expression index<br/>ON ((embedding::vector(384)))<br/>built after data load"]
    T --> I2["HNSW expression index<br/>ON ((embedding::vector(768)))"]

    Q["similarity query"] --> C{"cast BOTH sides?<br/>embedding::vector(384)<br/><=> $1::vector(384)"}
    C -->|yes| FAST["index scan<br/>~4ms"]
    C -->|"no — silent!"| SLOW["sequential scan<br/>~80ms, no error raised"]

    I1 -.-> FAST

    classDef ok stroke:#2f855a,stroke-width:2.5px
    class FAST ok
    classDef bad stroke:#c53030,stroke-dasharray:5 4
    class SLOW bad
```

**Why it matters:** one table serves every model comparison cleanly, at the cost of schema-level dimension safety — the 20× cast trap is invisible until you profile.
