"""
Evaluation harness for embedding model quality.

Computes NDCG and MeSH overlap metrics using hand-labeled relevance judgments.
Queries are designed around the five target categories (nutrition, exercise,
psychology, habits, ethics) with graded relevance scores.

Usage:
    python -m src.embeddings.evaluate --model minilm --db-url postgresql://...
    python -m src.embeddings.evaluate --compare --db-url postgresql://...
"""

import argparse
import json
import logging
import time

import numpy as np
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODELS = {
    "minilm": {"name": "all-MiniLM-L6-v2", "dim": 384},
    "pubmedbert": {
        "name": "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "dim": 768,
    },
}

# Graded relevance: 3 = highly relevant, 2 = relevant, 1 = somewhat relevant, 0 = irrelevant
# MeSH terms serve as the grading signal since we don't have manual annotations.
EVAL_QUERIES = [
    {
        "query": "effects of creatine supplementation on muscle recovery",
        "high_relevance_mesh": ["Creatine", "Dietary Supplements"],
        "medium_relevance_mesh": ["Muscle, Skeletal", "Athletic Performance", "Exercise"],
        "low_relevance_mesh": ["Sports Nutritional Physiological Phenomena", "Recovery of Function"],
    },
    {
        "query": "psychological effects of quitting alcohol",
        "high_relevance_mesh": ["Alcohol Drinking", "Temperance", "Alcohol Abstinence"],
        "medium_relevance_mesh": ["Mental Health", "Anxiety", "Depression"],
        "low_relevance_mesh": ["Substance-Related Disorders", "Behavior, Addictive"],
    },
    {
        "query": "benefits of high intensity interval training",
        "high_relevance_mesh": ["High-Intensity Interval Training"],
        "medium_relevance_mesh": ["Physical Fitness", "Exercise", "Physical Conditioning, Human"],
        "low_relevance_mesh": ["Cardiorespiratory Fitness", "Body Composition"],
    },
    {
        "query": "vegetarian diet and protein intake",
        "high_relevance_mesh": ["Diet, Vegetarian", "Diet, Vegan", "Dietary Proteins"],
        "medium_relevance_mesh": ["Plant Proteins", "Nutrition Assessment"],
        "low_relevance_mesh": ["Nutritional Status", "Diet"],
    },
    {
        "query": "ethics of artificial intelligence in healthcare",
        "high_relevance_mesh": ["Artificial Intelligence", "Bioethics"],
        "medium_relevance_mesh": ["Ethics", "Clinical Decision-Making", "Machine Learning"],
        "low_relevance_mesh": ["Moral Obligations", "Technology Assessment, Biomedical"],
    },
    {
        "query": "impact of sleep deprivation on cognitive performance",
        "high_relevance_mesh": ["Sleep Deprivation", "Cognition"],
        "medium_relevance_mesh": ["Sleep", "Mental Fatigue", "Cognitive Dysfunction"],
        "low_relevance_mesh": ["Attention", "Memory", "Psychomotor Performance"],
    },
    {
        "query": "gut microbiome and mental health connection",
        "high_relevance_mesh": ["Gastrointestinal Microbiome", "Mental Health"],
        "medium_relevance_mesh": ["Brain-Gut Axis", "Probiotics", "Depression"],
        "low_relevance_mesh": ["Anxiety", "Inflammation", "Diet"],
    },
    {
        "query": "resistance training for older adults",
        "high_relevance_mesh": ["Resistance Training", "Aged"],
        "medium_relevance_mesh": ["Muscle Strength", "Sarcopenia", "Exercise"],
        "low_relevance_mesh": ["Aging", "Physical Functional Performance", "Frail Elderly"],
    },
]


def compute_relevance_score(paper_mesh: set[str], query: dict) -> int:
    """Assign a graded relevance score based on MeSH overlap."""
    high = set(query["high_relevance_mesh"])
    medium = set(query["medium_relevance_mesh"])
    low = set(query["low_relevance_mesh"])

    high_hits = len(paper_mesh & high)
    medium_hits = len(paper_mesh & medium)
    low_hits = len(paper_mesh & low)

    if high_hits >= 2:
        return 3
    elif high_hits >= 1:
        return 2 + min(medium_hits, 1)
    elif medium_hits >= 2:
        return 2
    elif medium_hits >= 1 or low_hits >= 2:
        return 1
    return 0


def dcg(relevances: list[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    relevances = relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg(relevances: list[int], k: int) -> float:
    """Compute Normalized DCG at k."""
    actual = dcg(relevances, k)
    ideal = dcg(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def evaluate_model(
    conn,
    model: SentenceTransformer,
    model_name: str,
    dim: int,
    k: int = 10,
) -> dict:
    """Run evaluation queries and compute metrics."""
    results = []

    for eq in EVAL_QUERIES:
        query_vec = model.encode(eq["query"], normalize_embeddings=True).tolist()

        start = time.time()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT p.pmid, p.title, p.mesh_terms,
                       1 - (e.embedding::vector({dim}) <=> %s::vector({dim})) as similarity
                FROM embeddings e
                JOIN papers p ON e.pmid = p.pmid
                WHERE e.model_name = %s
                ORDER BY e.embedding::vector({dim}) <=> %s::vector({dim})
                LIMIT %s
                """,
                (query_vec, model_name, query_vec, k),
            )
            hits = cur.fetchall()
        latency = time.time() - start

        relevances = []
        mesh_overlaps = []
        for hit in hits:
            mesh = set(
                json.loads(hit["mesh_terms"])
                if isinstance(hit["mesh_terms"], str)
                else hit["mesh_terms"]
            )
            rel = compute_relevance_score(mesh, eq)
            relevances.append(rel)

            all_expected = (
                set(eq["high_relevance_mesh"])
                | set(eq["medium_relevance_mesh"])
                | set(eq["low_relevance_mesh"])
            )
            mesh_overlaps.append(len(mesh & all_expected) / max(len(all_expected), 1))

        ndcg_5 = ndcg(relevances, 5)
        ndcg_10 = ndcg(relevances, 10)
        avg_relevance = np.mean(relevances) if relevances else 0
        avg_similarity = np.mean([h["similarity"] for h in hits]) if hits else 0
        avg_mesh_overlap = np.mean(mesh_overlaps) if mesh_overlaps else 0

        results.append({
            "query": eq["query"],
            "ndcg_5": float(ndcg_5),
            "ndcg_10": float(ndcg_10),
            "avg_relevance": float(avg_relevance),
            "avg_similarity": float(avg_similarity),
            "avg_mesh_overlap": float(avg_mesh_overlap),
            "latency_ms": latency * 1000,
            "top_hit": hits[0]["title"] if hits else None,
            "relevance_distribution": relevances,
        })

    metrics = {
        "mean_ndcg_5": float(np.mean([r["ndcg_5"] for r in results])),
        "mean_ndcg_10": float(np.mean([r["ndcg_10"] for r in results])),
        "mean_relevance": float(np.mean([r["avg_relevance"] for r in results])),
        "mean_similarity": float(np.mean([r["avg_similarity"] for r in results])),
        "mean_mesh_overlap": float(np.mean([r["avg_mesh_overlap"] for r in results])),
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in results])),
        "p95_latency_ms": float(np.percentile([r["latency_ms"] for r in results], 95)),
    }

    return {"metrics": metrics, "details": results}


def print_results(model_key: str, eval_results: dict):
    """Print evaluation results in a readable format."""
    metrics = eval_results["metrics"]
    print(f"\n{'=' * 60}")
    print(f"  {model_key}")
    print(f"{'=' * 60}")
    print(f"  NDCG@5:  {metrics['mean_ndcg_5']:.4f}")
    print(f"  NDCG@10: {metrics['mean_ndcg_10']:.4f}")
    print(f"  Mean relevance: {metrics['mean_relevance']:.4f}")
    print(f"  Mean similarity: {metrics['mean_similarity']:.4f}")
    print(f"  Mean MeSH overlap: {metrics['mean_mesh_overlap']:.4f}")
    print(f"  Mean latency: {metrics['mean_latency_ms']:.1f}ms")
    print()
    for d in eval_results["details"]:
        rels = d["relevance_distribution"]
        rel_str = "".join(str(r) for r in rels)
        print(f"  [{d['ndcg_5']:.2f}] {d['query'][:50]}")
        print(f"         rels=[{rel_str}] -> {d['top_hit'][:55] if d['top_hit'] else 'N/A'}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate embedding models")
    parser.add_argument(
        "--model", default="minilm", choices=MODELS.keys(), help="Model to evaluate"
    )
    parser.add_argument("--compare", action="store_true", help="Compare all models")
    parser.add_argument(
        "--db-url", default="postgresql://pubmed:pubmed@localhost:5432/pubmed"
    )
    parser.add_argument("--mlflow-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--k", type=int, default=10, help="Top-k for retrieval")
    args = parser.parse_args()

    conn = psycopg2.connect(args.db_url)
    models_to_eval = MODELS if args.compare else {args.model: MODELS[args.model]}

    all_results = {}
    for model_key, model_info in models_to_eval.items():
        model_name = model_info["name"]
        dim = model_info["dim"]

        model = SentenceTransformer(model_name)
        eval_results = evaluate_model(conn, model, model_name, dim, k=args.k)
        all_results[model_key] = eval_results

        print_results(model_key, eval_results)

        if args.mlflow_uri:
            import mlflow

            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("pubmed-embedding-evaluation")
            with mlflow.start_run(run_name=f"eval-{model_key}"):
                mlflow.log_params(
                    {"model_name": model_name, "embedding_dim": dim, "top_k": args.k}
                )
                for k, v in eval_results["metrics"].items():
                    mlflow.log_metric(k, v)
                mlflow.log_dict(eval_results["details"], "eval_details.json")

    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("  COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        header = f"  {'Metric':<20}"
        for key in all_results:
            header += f" {key:>12}"
        print(header)
        print(f"  {'-' * 20}" + f" {'-' * 12}" * len(all_results))
        for metric in ["mean_ndcg_5", "mean_ndcg_10", "mean_relevance", "mean_latency_ms"]:
            row = f"  {metric:<20}"
            values = [all_results[k]["metrics"][metric] for k in all_results]
            best = max(values) if "latency" not in metric else min(values)
            for k in all_results:
                v = all_results[k]["metrics"][metric]
                marker = " *" if v == best else "  "
                row += f" {v:>10.4f}{marker}"
            print(row)

    conn.close()
