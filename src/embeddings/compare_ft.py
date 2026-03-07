"""
Quick comparison: base MiniLM vs fine-tuned MiniLM on eval queries.

Encodes queries with both models and searches against the existing
base MiniLM embeddings in the database. This tests whether the
fine-tuned query encoder retrieves more relevant results.

For a full comparison, re-embed all papers with the fine-tuned model.

Usage:
    python -m src.embeddings.compare_ft \
        --db-url postgresql://... \
        --ft-model-path models/minilm-pubmed-ft
"""

import argparse
import json
import logging
import time

import numpy as np
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

from src.embeddings.evaluate import EVAL_QUERIES, compute_relevance_score, ndcg

logger = logging.getLogger(__name__)

BASE_MODEL = "all-MiniLM-L6-v2"
DIM = 384


def evaluate_queries(conn, model, model_label: str, k: int = 10):
    """Run eval queries and compute NDCG."""
    results = []

    for eq in EVAL_QUERIES:
        query_vec = model.encode(eq["query"], normalize_embeddings=True).tolist()

        start = time.time()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT p.pmid, p.title, p.mesh_terms,
                       1 - (e.embedding::vector({DIM}) <=> %s::vector({DIM})) as similarity
                FROM embeddings e
                JOIN papers p ON e.pmid = p.pmid
                WHERE e.model_name = %s
                ORDER BY e.embedding::vector({DIM}) <=> %s::vector({DIM})
                LIMIT %s
                """,
                (query_vec, BASE_MODEL, query_vec, k),
            )
            hits = cur.fetchall()
        latency = time.time() - start

        relevances = []
        for hit in hits:
            mesh = set(
                json.loads(hit["mesh_terms"])
                if isinstance(hit["mesh_terms"], str)
                else hit["mesh_terms"]
            )
            relevances.append(compute_relevance_score(mesh, eq))

        results.append({
            "query": eq["query"],
            "ndcg_5": ndcg(relevances, 5),
            "ndcg_10": ndcg(relevances, 10),
            "relevances": relevances,
            "latency_ms": latency * 1000,
            "top_hit": hits[0]["title"] if hits else None,
        })

    return results


def print_comparison(base_results, ft_results):
    print(f"\n{'='*70}")
    print(f"  {'Query':<45} {'Base':>8} {'FT':>8} {'Δ':>8}")
    print(f"{'='*70}")

    for b, f in zip(base_results, ft_results):
        query = b["query"][:43]
        delta = f["ndcg_5"] - b["ndcg_5"]
        marker = "+" if delta > 0 else ""
        print(f"  {query:<45} {b['ndcg_5']:>7.3f} {f['ndcg_5']:>7.3f} {marker}{delta:>7.3f}")

    base_mean = np.mean([r["ndcg_5"] for r in base_results])
    ft_mean = np.mean([r["ndcg_5"] for r in ft_results])
    delta = ft_mean - base_mean
    marker = "+" if delta > 0 else ""
    print(f"{'─'*70}")
    print(f"  {'Mean NDCG@5':<45} {base_mean:>7.3f} {ft_mean:>7.3f} {marker}{delta:>7.3f}")

    base_mean_10 = np.mean([r["ndcg_10"] for r in base_results])
    ft_mean_10 = np.mean([r["ndcg_10"] for r in ft_results])
    delta_10 = ft_mean_10 - base_mean_10
    marker_10 = "+" if delta_10 > 0 else ""
    print(f"  {'Mean NDCG@10':<45} {base_mean_10:>7.3f} {ft_mean_10:>7.3f} {marker_10}{delta_10:>7.3f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--ft-model-path", default="models/minilm-pubmed-ft")
    args = parser.parse_args()

    conn = psycopg2.connect(args.db_url)

    logger.info("Loading base model...")
    base_model = SentenceTransformer(BASE_MODEL)
    logger.info("Loading fine-tuned model...")
    ft_model = SentenceTransformer(args.ft_model_path)

    logger.info("Evaluating base model...")
    base_results = evaluate_queries(conn, base_model, "base")
    logger.info("Evaluating fine-tuned model...")
    ft_results = evaluate_queries(conn, ft_model, "fine-tuned")

    print_comparison(base_results, ft_results)
    conn.close()
