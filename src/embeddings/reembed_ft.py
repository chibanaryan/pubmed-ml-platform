"""
Re-embed all papers with the fine-tuned MiniLM model and evaluate.

Stores embeddings under model_name="minilm-pubmed-ft" so they don't
conflict with the base MiniLM embeddings.

Usage:
    python -m src.embeddings.reembed_ft --db-url postgresql://...
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

FT_MODEL_NAME = "minilm-pubmed-ft"
BASE_MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384


def get_all_papers(conn):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT pmid, title, abstract
            FROM papers
            WHERE abstract IS NOT NULL
            ORDER BY pmid
        """)
        return cur.fetchall()


def embed_and_store(conn, model, papers, batch_size=256):
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    pmids = [p["pmid"] for p in papers]

    logger.info(f"Embedding {len(texts)} papers...")
    start = time.time()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embs)
        if (i // batch_size) % 10 == 0:
            logger.info(f"  {i + len(batch)}/{len(texts)}")

    embeddings = np.vstack(all_embeddings)
    embed_time = time.time() - start
    logger.info(f"Embedding took {embed_time:.1f}s ({len(texts)/embed_time:.0f} papers/sec)")

    logger.info("Storing embeddings...")
    sql = """
        INSERT INTO embeddings (pmid, model_name, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (pmid, model_name) DO UPDATE SET
            embedding = EXCLUDED.embedding, created_at = NOW()
    """
    with conn.cursor() as cur:
        for pmid, emb in zip(pmids, embeddings):
            cur.execute(sql, (pmid, FT_MODEL_NAME, emb.tolist()))
    conn.commit()
    logger.info(f"Stored {len(pmids)} embeddings as '{FT_MODEL_NAME}'")


def evaluate_both(conn, base_model, ft_model):
    """Evaluate both models against their own embeddings."""
    models = [
        ("base", base_model, BASE_MODEL_NAME),
        ("fine-tuned", ft_model, FT_MODEL_NAME),
    ]

    all_results = {}
    for label, model, db_model_name in models:
        results = []
        for eq in EVAL_QUERIES:
            query_vec = model.encode(eq["query"], normalize_embeddings=True).tolist()

            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT p.pmid, p.title, p.mesh_terms,
                           1 - (e.embedding::vector({DIM}) <=> %s::vector({DIM})) as similarity
                    FROM embeddings e
                    JOIN papers p ON e.pmid = p.pmid
                    WHERE e.model_name = %s
                    ORDER BY e.embedding::vector({DIM}) <=> %s::vector({DIM})
                    LIMIT 10
                    """,
                    (query_vec, db_model_name, query_vec, ),
                )
                hits = cur.fetchall()

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
            })

        all_results[label] = results

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  {'Query':<45} {'Base':>8} {'FT':>8} {'Δ':>8}")
    print(f"{'='*70}")
    for b, f in zip(all_results["base"], all_results["fine-tuned"]):
        query = b["query"][:43]
        delta = f["ndcg_5"] - b["ndcg_5"]
        marker = "+" if delta > 0 else ""
        print(f"  {query:<45} {b['ndcg_5']:>7.3f} {f['ndcg_5']:>7.3f} {marker}{delta:>7.3f}")

    base_5 = np.mean([r["ndcg_5"] for r in all_results["base"]])
    ft_5 = np.mean([r["ndcg_5"] for r in all_results["fine-tuned"]])
    base_10 = np.mean([r["ndcg_10"] for r in all_results["base"]])
    ft_10 = np.mean([r["ndcg_10"] for r in all_results["fine-tuned"]])

    print(f"{'─'*70}")
    d5 = ft_5 - base_5
    d10 = ft_10 - base_10
    print(f"  {'Mean NDCG@5':<45} {base_5:>7.3f} {ft_5:>7.3f} {'+'if d5>0 else ''}{d5:>7.3f}")
    print(f"  {'Mean NDCG@10':<45} {base_10:>7.3f} {ft_10:>7.3f} {'+'if d10>0 else ''}{d10:>7.3f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--ft-model-path", default="models/minilm-pubmed-ft")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-only", action="store_true", help="Skip embedding, just evaluate")
    args = parser.parse_args()

    conn = psycopg2.connect(args.db_url)

    logger.info("Loading models...")
    base_model = SentenceTransformer(BASE_MODEL_NAME)
    ft_model = SentenceTransformer(args.ft_model_path)

    if not args.eval_only:
        papers = get_all_papers(conn)
        embed_and_store(conn, ft_model, papers, batch_size=args.batch_size)

    evaluate_both(conn, base_model, ft_model)
    conn.close()
