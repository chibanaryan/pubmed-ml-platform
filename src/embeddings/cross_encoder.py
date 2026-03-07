"""
Cross-encoder re-ranker for PubMed search.

Bi-encoders are fast but approximate: they encode query and document
independently, so they can't model token-level interactions. A cross-encoder
takes the (query, document) pair as a single input and scores it jointly,
which is more accurate but O(n) per query instead of O(1).

The two-stage pipeline: bi-encoder retrieves top-50 candidates, cross-encoder
re-ranks to top-10.

Training data comes from MeSH-derived relevance labels, the same grading
system used in evaluate.py.

Usage:
    # Generate training data and train
    python -m src.embeddings.cross_encoder --db-url postgresql://... --train

    # Re-rank search results (requires trained model)
    python -m src.embeddings.cross_encoder --db-url postgresql://... --evaluate
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader, Dataset

from src.embeddings.evaluate import EVAL_QUERIES, compute_relevance_score, ndcg

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/cross-encoder-pubmed")
BASE_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
BI_ENCODER_DIM = 384

# MeSH terms that are too generic to indicate topical relevance
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
}


class RelevanceDataset(Dataset):
    """Dataset of (query, document, relevance_score) triples."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex["query"], ex["document"], ex["score"]


def generate_training_queries(conn, n_queries: int = 200, seed: int = 42) -> list[dict]:
    """
    Generate synthetic queries from MeSH term clusters.

    For each query, sample a MeSH term cluster, use the term as the query,
    and grade papers by how many related MeSH terms they share.
    """
    rng = random.Random(seed)

    # Get all papers with MeSH terms
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT pmid, title, abstract, mesh_terms
            FROM papers
            WHERE abstract IS NOT NULL
              AND LENGTH(abstract) > 100
              AND mesh_terms IS NOT NULL
              AND mesh_terms != '[]'
        """)
        papers = cur.fetchall()

    # Build MeSH index
    mesh_to_papers = {}
    for p in papers:
        mesh = json.loads(p["mesh_terms"]) if isinstance(p["mesh_terms"], str) else p["mesh_terms"]
        topic_mesh = set(mesh) - SKIP_MESH
        p["_topic_mesh"] = topic_mesh
        for term in topic_mesh:
            mesh_to_papers.setdefault(term, []).append(p)

    # Filter to MeSH terms with enough papers for meaningful training
    viable_terms = [t for t, ps in mesh_to_papers.items() if 10 <= len(ps) <= 500]
    rng.shuffle(viable_terms)

    queries = []
    for term in viable_terms[:n_queries]:
        # Use the MeSH term as a natural language query
        query_text = term.lower().replace(",", "")

        # Papers with this term are positives, sample negatives from other papers
        positive_papers = mesh_to_papers[term]
        positive_pmids = {p["pmid"] for p in positive_papers}

        # Sample negatives: papers that don't have this MeSH term
        negatives = [p for p in rng.sample(papers, min(len(papers), 500))
                     if p["pmid"] not in positive_pmids]

        queries.append({
            "query": query_text,
            "anchor_mesh": term,
            "positives": positive_papers,
            "negatives": negatives[:len(positive_papers)],  # balance pos/neg
        })

    logger.info(f"Generated {len(queries)} training queries from MeSH clusters")
    return queries


def build_training_examples(queries: list[dict], max_examples: int = 50_000, seed: int = 42) -> list[dict]:
    """
    Convert queries into (query, document, score) training examples.

    Scoring:
    - 1.0: paper has the query's anchor MeSH term + shares additional MeSH terms
    - 0.5: paper has the query's anchor MeSH term
    - 0.0: paper doesn't have the anchor MeSH term (negative)
    """
    rng = random.Random(seed)
    examples = []

    for q in queries:
        anchor = q["anchor_mesh"]

        # Positive examples
        for p in q["positives"]:
            text = f"{p['title']}. {p['abstract'][:512]}"
            # Higher score if they share more topic MeSH terms beyond the anchor
            shared = len(p["_topic_mesh"] - {anchor})
            score = min(1.0, 0.5 + shared * 0.1)
            examples.append({"query": q["query"], "document": text, "score": score})

        # Negative examples
        for p in q["negatives"]:
            text = f"{p['title']}. {p['abstract'][:512]}"
            examples.append({"query": q["query"], "document": text, "score": 0.0})

    rng.shuffle(examples)
    examples = examples[:max_examples]
    logger.info(f"Built {len(examples)} training examples")

    pos = sum(1 for e in examples if e["score"] > 0)
    logger.info(f"  Positive: {pos}, Negative: {len(examples) - pos}")
    return examples


def train_cross_encoder(
    examples: list[dict],
    epochs: int = 2,
    batch_size: int = 32,
    warmup_ratio: float = 0.1,
    output_dir: str | None = None,
) -> CrossEncoder:
    """Fine-tune a cross-encoder on relevance examples."""
    if output_dir is None:
        output_dir = str(MODEL_DIR)
    output_dir = str(Path(output_dir).resolve())

    model = CrossEncoder(BASE_CROSS_ENCODER, max_length=512)

    train_samples = [
        InputExample(texts=[ex["query"], ex["document"]], label=ex["score"])
        for ex in examples
    ]

    logger.info(f"Training cross-encoder: {len(train_samples)} examples, {epochs} epochs")

    model.fit(
        train_dataloader=DataLoader(train_samples, batch_size=batch_size, shuffle=True),
        epochs=epochs,
        warmup_steps=int(len(train_samples) / batch_size * epochs * warmup_ratio),
        output_path=output_dir,
        show_progress_bar=True,
    )

    # Explicitly save in case Trainer didn't persist to output_dir
    model.save(output_dir)
    logger.info(f"Model saved to {output_dir}")
    return model


def rerank(
    cross_encoder: CrossEncoder,
    query: str,
    candidates: list[dict],
) -> list[dict]:
    """Re-rank candidate documents using the cross-encoder."""
    if not candidates:
        return candidates

    texts = [f"{c['title']}. {c.get('abstract', '')[:512]}" for c in candidates]
    pairs = [[query, t] for t in texts]

    scores = cross_encoder.predict(pairs, show_progress_bar=False)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


def evaluate_reranking(conn, bi_encoder, cross_encoder, k_retrieve: int = 50, k_final: int = 10):
    """
    Evaluate the two-stage pipeline: bi-encoder retrieves top-k_retrieve,
    cross-encoder re-ranks to top-k_final.
    """
    bi_only_results = []
    reranked_results = []

    for eq in EVAL_QUERIES:
        query_vec = bi_encoder.encode(eq["query"], normalize_embeddings=True).tolist()

        # Stage 1: bi-encoder retrieval
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT p.pmid, p.title, p.abstract, p.mesh_terms,
                       1 - (e.embedding::vector({BI_ENCODER_DIM}) <=> %s::vector({BI_ENCODER_DIM})) as similarity
                FROM embeddings e
                JOIN papers p ON e.pmid = p.pmid
                WHERE e.model_name = %s
                ORDER BY e.embedding::vector({BI_ENCODER_DIM}) <=> %s::vector({BI_ENCODER_DIM})
                LIMIT %s
                """,
                (query_vec, BI_ENCODER_MODEL, query_vec, k_retrieve),
            )
            candidates = cur.fetchall()

        # Compute relevances for bi-encoder top-k_final
        bi_relevances = []
        for hit in candidates[:k_final]:
            mesh = set(
                json.loads(hit["mesh_terms"])
                if isinstance(hit["mesh_terms"], str)
                else hit["mesh_terms"]
            )
            bi_relevances.append(compute_relevance_score(mesh, eq))

        # Stage 2: cross-encoder re-ranking
        start = time.time()
        reranked = rerank(cross_encoder, eq["query"], list(candidates))
        rerank_time = time.time() - start

        rerank_relevances = []
        for hit in reranked[:k_final]:
            mesh = set(
                json.loads(hit["mesh_terms"])
                if isinstance(hit["mesh_terms"], str)
                else hit["mesh_terms"]
            )
            rerank_relevances.append(compute_relevance_score(mesh, eq))

        bi_only_results.append({
            "query": eq["query"],
            "ndcg_5": ndcg(bi_relevances, 5),
            "ndcg_10": ndcg(bi_relevances, 10),
        })
        reranked_results.append({
            "query": eq["query"],
            "ndcg_5": ndcg(rerank_relevances, 5),
            "ndcg_10": ndcg(rerank_relevances, 10),
            "rerank_ms": rerank_time * 1000,
        })

    # Print comparison
    print(f"\n{'='*75}")
    print(f"  {'Query':<40} {'Bi-enc':>8} {'Re-rank':>8} {'Δ':>8} {'ms':>6}")
    print(f"{'='*75}")
    for b, r in zip(bi_only_results, reranked_results):
        query = b["query"][:38]
        delta = r["ndcg_5"] - b["ndcg_5"]
        marker = "+" if delta > 0 else ""
        print(f"  {query:<40} {b['ndcg_5']:>7.3f} {r['ndcg_5']:>7.3f} {marker}{delta:>7.3f} {r['rerank_ms']:>5.0f}")

    bi_mean = np.mean([r["ndcg_5"] for r in bi_only_results])
    rr_mean = np.mean([r["ndcg_5"] for r in reranked_results])
    bi_mean_10 = np.mean([r["ndcg_10"] for r in bi_only_results])
    rr_mean_10 = np.mean([r["ndcg_10"] for r in reranked_results])
    avg_ms = np.mean([r["rerank_ms"] for r in reranked_results])

    d5 = rr_mean - bi_mean
    d10 = rr_mean_10 - bi_mean_10
    print(f"{'─'*75}")
    print(f"  {'Mean NDCG@5':<40} {bi_mean:>7.3f} {rr_mean:>7.3f} {'+'if d5>0 else ''}{d5:>7.3f} {avg_ms:>5.0f}")
    print(f"  {'Mean NDCG@10':<40} {bi_mean_10:>7.3f} {rr_mean_10:>7.3f} {'+'if d10>0 else ''}{d10:>7.3f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Cross-encoder re-ranker for PubMed search")
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--train", action="store_true", help="Train the cross-encoder")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate re-ranking")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--max-examples", type=int, default=50_000)
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    conn = psycopg2.connect(args.db_url)

    if args.train:
        queries = generate_training_queries(conn, n_queries=args.n_queries)
        examples = build_training_examples(queries, max_examples=args.max_examples)
        train_cross_encoder(
            examples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.model_dir,
        )

    if args.evaluate:
        logger.info("Loading models for evaluation...")
        bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
        model_path = str(Path(args.model_dir).resolve())
        cross_encoder = CrossEncoder(model_path)
        evaluate_reranking(conn, bi_encoder, cross_encoder)

    if not args.train and not args.evaluate:
        parser.print_help()

    conn.close()
