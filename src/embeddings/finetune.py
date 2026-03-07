"""
Fine-tune MiniLM on PubMed abstracts using contrastive learning.

Papers sharing topic-relevant MeSH terms form positive pairs.
Uses MultipleNegativesRankingLoss (in-batch negatives): for each
(anchor, positive) pair, all other positives in the batch serve
as negatives. This scales well without explicit negative mining.

Usage:
    python -m src.embeddings.finetune --db-url postgresql://...
    python -m src.embeddings.finetune --db-url postgresql://... --epochs 3 --batch-size 64
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime

import psycopg2
import psycopg2.extras
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# MeSH terms that don't indicate topical similarity
SKIP_MESH = {
    # Demographics
    "Humans", "Male", "Female", "Adult", "Middle Aged", "Young Adult",
    "Aged", "Aged, 80 and over", "Adolescent", "Child", "Child, Preschool",
    "Infant", "Infant, Newborn", "Pregnancy",
    # Animals (too broad)
    "Animals", "Mice", "Rats", "Mice, Inbred C57BL",
    # Study design (methodology, not topic)
    "Cross-Sectional Studies", "Retrospective Studies", "Prospective Studies",
    "Surveys and Questionnaires", "Qualitative Research",
    "Randomized Controlled Trials as Topic", "Treatment Outcome",
    "Risk Factors", "Cohort Studies", "Follow-Up Studies",
    # Geographic
    "United States", "China", "Japan", "Brazil", "Europe",
    # Too generic
    "Time Factors", "Reproducibility of Results", "Pilot Projects",
    "Patient Satisfaction", "Patient Acceptance of Health Care",
    "Health Knowledge, Attitudes, Practice",
}

# Minimum shared topic MeSH terms to consider papers a positive pair
MIN_SHARED_MESH = 2


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


def build_mesh_index(papers: list[dict]) -> dict[str, list[int]]:
    """Build inverted index: MeSH term -> list of paper indices."""
    index = defaultdict(list)
    for i, p in enumerate(papers):
        for term in p["topic_mesh"]:
            index[term].append(i)
    return index


def generate_pairs(
    papers: list[dict],
    mesh_index: dict[str, list[int]],
    max_pairs: int = 100_000,
    seed: int = 42,
) -> list[InputExample]:
    """Generate positive pairs from papers sharing topic MeSH terms."""
    rng = random.Random(seed)
    seen = set()
    pairs = []

    # Iterate through MeSH terms, sampling pairs from each cluster
    terms = list(mesh_index.keys())
    rng.shuffle(terms)

    for term in terms:
        paper_indices = mesh_index[term]
        if len(paper_indices) < 2:
            continue

        # Sample pairs from this cluster
        sampled = rng.sample(paper_indices, min(len(paper_indices), 50))
        for i in range(len(sampled)):
            for j in range(i + 1, min(i + 5, len(sampled))):
                a, b = sampled[i], sampled[j]
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue

                # Verify they share enough topic MeSH terms
                shared = papers[a]["topic_mesh"] & papers[b]["topic_mesh"]
                if len(shared) >= MIN_SHARED_MESH:
                    seen.add(key)
                    pairs.append(InputExample(
                        texts=[papers[a]["text"], papers[b]["text"]],
                    ))

                    if len(pairs) >= max_pairs:
                        logger.info(f"Generated {len(pairs)} pairs (hit max)")
                        return pairs

    logger.info(f"Generated {len(pairs)} pairs from {len(seen)} unique paper combinations")
    return pairs


def finetune(
    pairs: list[InputExample],
    model_name: str = "all-MiniLM-L6-v2",
    epochs: int = 1,
    batch_size: int = 64,
    warmup_ratio: float = 0.1,
    output_dir: str | None = None,
) -> SentenceTransformer:
    """Fine-tune using MultipleNegativesRankingLoss."""
    model = SentenceTransformer(model_name)

    dataloader = DataLoader(pairs, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(dataloader) * epochs * warmup_ratio)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"models/minilm-pubmed-{timestamp}"

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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fine-tune MiniLM on PubMed abstracts")
    parser.add_argument(
        "--db-url", default="postgresql://pubmed:pubmed@localhost:5432/pubmed"
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=100_000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conn = psycopg2.connect(args.db_url)
    papers = load_papers(conn)
    mesh_index = build_mesh_index(papers)
    pairs = generate_pairs(papers, mesh_index, max_pairs=args.max_pairs, seed=args.seed)
    conn.close()

    logger.info(f"Pair examples:")
    for p in pairs[:3]:
        logger.info(f"  A: {p.texts[0][:80]}...")
        logger.info(f"  B: {p.texts[1][:80]}...")
        logger.info("")

    model = finetune(
        pairs,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
