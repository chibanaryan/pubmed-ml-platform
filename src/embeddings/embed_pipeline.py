"""
Embedding pipeline for PubMed abstracts.

Generates embeddings using HuggingFace sentence transformers,
stores them in pgvector, and tracks experiments in MLflow.

Usage:
    # Compare models
    python -m src.embeddings.embed_pipeline --compare

    # Generate embeddings with chosen model
    python -m src.embeddings.embed_pipeline --model all-MiniLM-L6-v2 --batch-size 256
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass

import mlflow
import numpy as np
import psycopg2
import psycopg2.extras
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Models to evaluate
MODELS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Fast general-purpose sentence transformer",
    },
    "pubmedbert": {
        "name": "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "dim": 768,
        "description": "Domain-specific biomedical sentence transformer",
    },
}

# Hand-curated evaluation queries and expected relevant PMIDs.
# Fill these in with real PMIDs once you've ingested data.
EVAL_QUERIES = [
    {
        "query": "effects of creatine supplementation on muscle recovery",
        "category": "nutrition",
        "expected_mesh": ["Creatine", "Muscle, Skeletal", "Dietary Supplements"],
    },
    {
        "query": "psychological effects of quitting alcohol",
        "category": "habits",
        "expected_mesh": ["Alcohol Drinking", "Temperance", "Mental Health"],
    },
    {
        "query": "benefits of high intensity interval training",
        "category": "exercise",
        "expected_mesh": ["High-Intensity Interval Training", "Physical Fitness"],
    },
    {
        "query": "vegetarian diet and protein intake",
        "category": "nutrition",
        "expected_mesh": ["Diet, Vegetarian", "Dietary Proteins"],
    },
    {
        "query": "ethics of artificial intelligence in healthcare",
        "category": "ethics",
        "expected_mesh": ["Artificial Intelligence", "Bioethics", "Ethics"],
    },
]


@dataclass
class EmbeddingConfig:
    db_url: str = "postgresql://pubmed:pubmed@localhost:5432/pubmed"
    mlflow_uri: str = "http://localhost:5000"
    batch_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_connection(db_url: str):
    return psycopg2.connect(db_url)


def get_unembedded_abstracts(conn, model_name: str, limit: int | None = None) -> list[dict]:
    """Fetch abstracts that don't yet have embeddings for this model."""
    sql = """
        SELECT p.pmid, p.title, p.abstract, p.mesh_terms
        FROM papers p
        LEFT JOIN embeddings e ON p.pmid = e.pmid AND e.model_name = %s
        WHERE e.id IS NULL AND p.abstract IS NOT NULL
        ORDER BY p.pub_date DESC
    """
    if limit:
        sql += f" LIMIT {limit}"

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (model_name,))
        return cur.fetchall()


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """Generate embeddings in batches."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # for cosine similarity via dot product
        )
        all_embeddings.append(embeddings)

        if show_progress and (i // batch_size) % 10 == 0:
            logger.info(f"  Embedded {i + len(batch)}/{len(texts)}")

    return np.vstack(all_embeddings)


def store_embeddings(conn, pmids: list[int], embeddings: np.ndarray, model_name: str):
    """Store embeddings in pgvector."""
    sql = """
        INSERT INTO embeddings (pmid, model_name, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (pmid, model_name) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            created_at = NOW()
    """
    with conn.cursor() as cur:
        for pmid, emb in zip(pmids, embeddings):
            cur.execute(sql, (pmid, model_name, emb.tolist()))
    conn.commit()
    logger.info(f"Stored {len(pmids)} embeddings for model {model_name}")


def evaluate_model(
    conn, model: SentenceTransformer, model_name: str, model_key: str
) -> dict:
    """
    Evaluate embedding quality using the hand-curated test queries.

    Metrics:
    - MeSH overlap: do top results share MeSH terms with the query intent?
    - Retrieval time: latency for vector search
    """
    results = []

    for eval_q in EVAL_QUERIES:
        query_embedding = model.encode(
            eval_q["query"], normalize_embeddings=True
        ).tolist()

        start = time.time()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.pmid, p.title, p.mesh_terms,
                       1 - (e.embedding <=> %s::vector) as similarity
                FROM embeddings e
                JOIN papers p ON e.pmid = p.pmid
                WHERE e.model_name = %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT 10
                """,
                (query_embedding, model_name, query_embedding),
            )
            hits = cur.fetchall()
        latency = time.time() - start

        # Calculate MeSH overlap for top-k results
        expected_mesh = set(eval_q["expected_mesh"])
        mesh_overlaps = []
        for hit in hits:
            hit_mesh = set(json.loads(hit["mesh_terms"]) if isinstance(hit["mesh_terms"], str) else hit["mesh_terms"])
            overlap = len(expected_mesh & hit_mesh) / max(len(expected_mesh), 1)
            mesh_overlaps.append(overlap)

        avg_similarity = np.mean([h["similarity"] for h in hits]) if hits else 0
        avg_mesh_overlap = np.mean(mesh_overlaps) if mesh_overlaps else 0

        results.append({
            "query": eval_q["query"],
            "avg_similarity": float(avg_similarity),
            "avg_mesh_overlap": float(avg_mesh_overlap),
            "latency_ms": latency * 1000,
            "top_hit": hits[0]["title"] if hits else None,
        })

    # Aggregate metrics
    metrics = {
        "mean_similarity": np.mean([r["avg_similarity"] for r in results]),
        "mean_mesh_overlap": np.mean([r["avg_mesh_overlap"] for r in results]),
        "mean_latency_ms": np.mean([r["latency_ms"] for r in results]),
        "p95_latency_ms": np.percentile([r["latency_ms"] for r in results], 95),
    }

    return {"metrics": metrics, "details": results}


def run_comparison(config: EmbeddingConfig):
    """Run a side-by-side comparison of embedding models, tracked in MLflow."""
    mlflow.set_tracking_uri(config.mlflow_uri)
    mlflow.set_experiment("pubmed-embedding-comparison")

    conn = get_connection(config.db_url)

    for model_key, model_info in MODELS.items():
        model_name = model_info["name"]
        logger.info(f"\n{'='*60}\nEvaluating: {model_name}\n{'='*60}")

        with mlflow.start_run(run_name=f"eval-{model_key}"):
            mlflow.log_params({
                "model_name": model_name,
                "model_key": model_key,
                "embedding_dim": model_info["dim"],
                "device": config.device,
                "batch_size": config.batch_size,
            })

            # Load model
            logger.info(f"Loading model {model_name}...")
            model = SentenceTransformer(model_name, device=config.device)

            # Embed a sample of abstracts
            abstracts = get_unembedded_abstracts(conn, model_name, limit=5000)
            if not abstracts:
                logger.warning(f"No unembedded abstracts for {model_name}. Run ingestion first.")
                continue

            texts = [
                f"{a['title']}. {a['abstract']}" for a in abstracts
            ]
            pmids = [a["pmid"] for a in abstracts]

            logger.info(f"Generating embeddings for {len(texts)} abstracts...")
            start = time.time()
            embeddings = generate_embeddings(model, texts, config.batch_size)
            embed_time = time.time() - start

            mlflow.log_metric("embedding_time_seconds", embed_time)
            mlflow.log_metric("abstracts_per_second", len(texts) / embed_time)

            # Store
            store_embeddings(conn, pmids, embeddings, model_name)

            # Evaluate
            logger.info("Running evaluation queries...")
            eval_results = evaluate_model(conn, model, model_name, model_key)

            for metric_name, value in eval_results["metrics"].items():
                mlflow.log_metric(metric_name, value)

            mlflow.log_dict(eval_results["details"], "eval_details.json")

            # Register model in MLflow model registry
            logger.info(f"Registering model {model_name} in MLflow registry...")
            model_info_mlflow = mlflow.sentence_transformers.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"pubmed-{model_key}",
            )
            logger.info(f"Model registered: {model_info_mlflow.model_uri}")

            logger.info(f"Results for {model_key}: {eval_results['metrics']}")

    conn.close()
    logger.info("\nComparison complete. View results at MLflow UI.")


def run_embedding(config: EmbeddingConfig, model_key: str):
    """Generate embeddings for all unembedded abstracts with a specific model."""
    model_info = MODELS[model_key]
    model_name = model_info["name"]

    mlflow.set_tracking_uri(config.mlflow_uri)
    mlflow.set_experiment("pubmed-embedding-production")

    conn = get_connection(config.db_url)
    abstracts = get_unembedded_abstracts(conn, model_name)

    if not abstracts:
        logger.info("All abstracts are already embedded.")
        return

    logger.info(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name, device=config.device)

    with mlflow.start_run(run_name=f"embed-{model_key}-{len(abstracts)}"):
        mlflow.log_params({
            "model_name": model_name,
            "num_abstracts": len(abstracts),
            "batch_size": config.batch_size,
        })

        texts = [f"{a['title']}. {a['abstract']}" for a in abstracts]
        pmids = [a["pmid"] for a in abstracts]

        logger.info(f"Generating embeddings for {len(texts)} abstracts...")
        start = time.time()
        embeddings = generate_embeddings(model, texts, config.batch_size)
        embed_time = time.time() - start

        store_embeddings(conn, pmids, embeddings, model_name)

        mlflow.log_metrics({
            "embedding_time_seconds": embed_time,
            "abstracts_per_second": len(texts) / embed_time,
            "total_abstracts": len(texts),
        })

        # Register model in MLflow model registry
        mlflow.sentence_transformers.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"pubmed-{model_key}",
        )

    conn.close()
    logger.info(f"Done. Embedded {len(texts)} abstracts in {embed_time:.1f}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="PubMed embedding pipeline")
    parser.add_argument("--compare", action="store_true", help="Run model comparison experiment")
    parser.add_argument("--model", default="minilm", choices=MODELS.keys(), help="Model to use for embedding")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--db-url", default="postgresql://pubmed:pubmed@localhost:5432/pubmed")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    args = parser.parse_args()

    config = EmbeddingConfig(
        db_url=args.db_url,
        mlflow_uri=args.mlflow_uri,
        batch_size=args.batch_size,
    )

    if args.compare:
        run_comparison(config)
    else:
        run_embedding(config, args.model)
