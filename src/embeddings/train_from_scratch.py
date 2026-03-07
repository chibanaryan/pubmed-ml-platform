"""
Train a sentence embedding model from scratch on PubMed data.

Initializes from bert-base-uncased and trains a sentence embedding model
using in-batch negatives (MultipleNegativesRankingLoss). This gives full
control over the base model, pooling strategy, and training procedure.

The goal is to see what domain-specific training buys you when starting
from a general-purpose language model vs. using MiniLM (which was
pre-trained specifically for sentence embeddings).

Usage:
    python -m src.embeddings.train_from_scratch --db-url postgresql://...
    python -m src.embeddings.train_from_scratch --db-url postgresql://... --eval-only
"""

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader

from src.embeddings.evaluate import EVAL_QUERIES, compute_relevance_score, ndcg

logger = logging.getLogger(__name__)

BASE_MODEL = "bert-base-uncased"
OUTPUT_DIR = Path("models/bert-pubmed-scratch")
DB_MODEL_NAME = "bert-pubmed-scratch"
BASE_DB_MODEL = "all-MiniLM-L6-v2"
DIM = 768  # bert-base hidden size

# MeSH terms that don't indicate topical similarity
SKIP_MESH = {
    "Humans", "Male", "Female", "Adult", "Middle Aged", "Young Adult",
    "Aged", "Aged, 80 and over", "Adolescent", "Child", "Child, Preschool",
    "Infant", "Infant, Newborn", "Pregnancy",
    "Animals", "Mice", "Rats", "Mice, Inbred C57BL",
    "Cross-Sectional Studies", "Retrospective Studies", "Prospective Studies",
    "Surveys and Questionnaires", "Qualitative Research",
    "Randomized Controlled Trials as Topic", "Treatment Outcome",
    "Risk Factors", "Cohort Studies", "Follow-Up Studies",
    "United States", "China", "Japan", "Brazil", "Europe",
    "Time Factors", "Reproducibility of Results", "Pilot Projects",
    "Patient Satisfaction", "Patient Acceptance of Health Care",
    "Health Knowledge, Attitudes, Practice",
}

MIN_SHARED_MESH = 2


def build_model(base_model: str = BASE_MODEL, max_seq_length: int = 256) -> SentenceTransformer:
    """Build a sentence embedding model from a base transformer.

    Architecture:
      1. Transformer (bert-base-uncased)
      2. Mean pooling over token embeddings
    """
    word_embedding = models.Transformer(base_model, max_seq_length=max_seq_length)
    pooling = models.Pooling(
        word_embedding.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(modules=[word_embedding, pooling])


def load_papers(conn) -> list[dict]:
    """Load all papers with abstracts and MeSH terms."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT pmid, title, abstract, mesh_terms
            FROM papers
            WHERE abstract IS NOT NULL
              AND LENGTH(abstract) > 100
              AND mesh_terms IS NOT NULL
              AND mesh_terms != '[]'
        """)
        rows = cur.fetchall()

    papers = []
    for r in rows:
        mesh = json.loads(r["mesh_terms"]) if isinstance(r["mesh_terms"], str) else r["mesh_terms"]
        topic_mesh = set(mesh) - SKIP_MESH
        if len(topic_mesh) >= 2:
            papers.append({
                "pmid": r["pmid"],
                "text": f"{r['title']}. {r['abstract'][:512]}",
                "topic_mesh": topic_mesh,
            })

    logger.info(f"Loaded {len(papers)} papers with topic-relevant MeSH terms")
    return papers


def generate_pairs(
    papers: list[dict],
    max_pairs: int = 100_000,
    seed: int = 42,
) -> list[InputExample]:
    """Generate positive pairs from papers sharing topic MeSH terms."""
    rng = random.Random(seed)

    # Build MeSH index
    mesh_index = defaultdict(list)
    for i, p in enumerate(papers):
        for term in p["topic_mesh"]:
            mesh_index[term].append(i)

    seen = set()
    pairs = []

    terms = list(mesh_index.keys())
    rng.shuffle(terms)

    for term in terms:
        paper_indices = mesh_index[term]
        if len(paper_indices) < 2:
            continue

        sampled = rng.sample(paper_indices, min(len(paper_indices), 50))
        for i in range(len(sampled)):
            for j in range(i + 1, min(i + 5, len(sampled))):
                a, b = sampled[i], sampled[j]
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue

                shared = papers[a]["topic_mesh"] & papers[b]["topic_mesh"]
                if len(shared) >= MIN_SHARED_MESH:
                    seen.add(key)
                    pairs.append(InputExample(
                        texts=[papers[a]["text"], papers[b]["text"]],
                    ))

                    if len(pairs) >= max_pairs:
                        logger.info(f"Generated {len(pairs)} pairs (hit max)")
                        return pairs

    logger.info(f"Generated {len(pairs)} pairs from {len(seen)} unique combinations")
    return pairs


def train(
    model: SentenceTransformer,
    pairs: list[InputExample],
    epochs: int = 2,
    batch_size: int = 32,
    warmup_ratio: float = 0.1,
    output_dir: str | None = None,
) -> SentenceTransformer:
    """Train using MultipleNegativesRankingLoss (in-batch negatives)."""
    if output_dir is None:
        output_dir = str(OUTPUT_DIR)
    output_dir = str(Path(output_dir).resolve())

    dataloader = DataLoader(pairs, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(dataloader) * epochs * warmup_ratio)

    logger.info(f"Training: {len(pairs)} pairs, {epochs} epochs, batch_size={batch_size}")
    logger.info(f"Warmup steps: {warmup_steps}, output: {output_dir}")

    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
    )

    logger.info(f"Model saved to {output_dir}")
    return model


def embed_and_store(conn, model: SentenceTransformer, batch_size: int = 128):
    """Re-embed all papers with the trained model."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT pmid, title, abstract
            FROM papers
            WHERE abstract IS NOT NULL
            ORDER BY pmid
        """)
        papers = cur.fetchall()

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
    elapsed = time.time() - start
    logger.info(f"Embedding took {elapsed:.1f}s ({len(texts)/elapsed:.0f} papers/sec)")

    logger.info("Storing embeddings...")
    sql = """
        INSERT INTO embeddings (pmid, model_name, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (pmid, model_name) DO UPDATE SET
            embedding = EXCLUDED.embedding, created_at = NOW()
    """
    with conn.cursor() as cur:
        for i, (pmid, emb) in enumerate(zip(pmids, embeddings)):
            cur.execute(sql, (pmid, DB_MODEL_NAME, emb.tolist()))
            if i % 5000 == 0:
                conn.commit()
                logger.info(f"  Stored {i}/{len(pmids)}")
    conn.commit()
    logger.info(f"Stored {len(pmids)} embeddings as '{DB_MODEL_NAME}'")


def evaluate(conn, model: SentenceTransformer):
    """Evaluate trained model against base MiniLM."""
    base_model = SentenceTransformer("all-MiniLM-L6-v2")
    base_dim = 384

    models_to_eval = [
        ("base-minilm", base_model, BASE_DB_MODEL, base_dim),
        ("bert-scratch", model, DB_MODEL_NAME, DIM),
    ]

    all_results = {}
    for label, m, db_name, dim in models_to_eval:
        results = []
        for eq in EVAL_QUERIES:
            query_vec = m.encode(eq["query"], normalize_embeddings=True).tolist()

            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT p.pmid, p.title, p.mesh_terms,
                           1 - (e.embedding::vector({dim}) <=> %s::vector({dim})) as similarity
                    FROM embeddings e
                    JOIN papers p ON e.pmid = p.pmid
                    WHERE e.model_name = %s
                    ORDER BY e.embedding::vector({dim}) <=> %s::vector({dim})
                    LIMIT 10
                    """,
                    (query_vec, db_name, query_vec),
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
            })

        all_results[label] = results

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  {'Query':<45} {'MiniLM':>8} {'BERT':>8} {'D':>8}")
    print(f"{'='*70}")
    for b, s in zip(all_results["base-minilm"], all_results["bert-scratch"]):
        query = b["query"][:43]
        delta = s["ndcg_5"] - b["ndcg_5"]
        marker = "+" if delta > 0 else ""
        print(f"  {query:<45} {b['ndcg_5']:>7.3f} {s['ndcg_5']:>7.3f} {marker}{delta:>7.3f}")

    base_5 = np.mean([r["ndcg_5"] for r in all_results["base-minilm"]])
    scratch_5 = np.mean([r["ndcg_5"] for r in all_results["bert-scratch"]])
    base_10 = np.mean([r["ndcg_10"] for r in all_results["base-minilm"]])
    scratch_10 = np.mean([r["ndcg_10"] for r in all_results["bert-scratch"]])

    print(f"{'─'*70}")
    d5 = scratch_5 - base_5
    d10 = scratch_10 - base_10
    print(f"  {'Mean NDCG@5':<45} {base_5:>7.3f} {scratch_5:>7.3f} {'+'if d5>0 else ''}{d5:>7.3f}")
    print(f"  {'Mean NDCG@10':<45} {base_10:>7.3f} {scratch_10:>7.3f} {'+'if d10>0 else ''}{d10:>7.3f}")

    return {"base_ndcg5": base_5, "scratch_ndcg5": scratch_5, "delta": d5}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train sentence embeddings from bert-base-uncased")
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=100_000)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate")
    parser.add_argument("--skip-embed", action="store_true", help="Skip re-embedding")
    args = parser.parse_args()

    conn = psycopg2.connect(args.db_url)

    if args.eval_only:
        logger.info("Loading trained model for evaluation...")
        model = SentenceTransformer(str(Path(args.output_dir).resolve()))
        evaluate(conn, model)
    else:
        papers = load_papers(conn)
        pairs = generate_pairs(papers, max_pairs=args.max_pairs)
        conn.close()

        model = build_model(max_seq_length=args.max_seq_length)
        logger.info(f"Model architecture: {model}")
        logger.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

        model = train(
            model, pairs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        # Reconnect for embedding + evaluation
        conn = psycopg2.connect(args.db_url)

        if not args.skip_embed:
            embed_and_store(conn, model)

        evaluate(conn, model)

    conn.close()
