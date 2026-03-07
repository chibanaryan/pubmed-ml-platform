"""
Distill PubMedBERT (768-dim) into MiniLM (384-dim).

The teacher (PubMedBERT) has domain-specific biomedical knowledge from
pretraining on PubMed/PMC text. The student (MiniLM) is smaller and faster.
Distillation transfers the teacher's similarity structure into the student
via KL divergence on pairwise similarity distributions.

For each batch of texts:
  1. Teacher computes pairwise cosine similarities → softmax → distribution
  2. Student computes pairwise cosine similarities → log_softmax → log probs
  3. KL(teacher || student) pushes the student to rank documents the same way

Usage:
    python -m src.embeddings.distill --db-url postgresql://...
    python -m src.embeddings.distill --db-url postgresql://... --epochs 3 --batch-size 32
    python -m src.embeddings.distill --db-url postgresql://... --eval-only
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
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

from src.embeddings.evaluate import EVAL_QUERIES, compute_relevance_score, ndcg

logger = logging.getLogger(__name__)

TEACHER_MODEL = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
STUDENT_MODEL = "all-MiniLM-L6-v2"
TEACHER_DIM = 768
STUDENT_DIM = 384
OUTPUT_DIR = Path("models/minilm-distilled")
DB_MODEL_NAME = "minilm-distilled"
BASE_DB_MODEL = "all-MiniLM-L6-v2"

# Temperature for softmax over similarity scores.
# Higher temperature → softer distribution → more knowledge transfer from
# less-similar pairs. Lower → sharper → focuses on top similarities.
TEMPERATURE = 2.0


class TextDataset(Dataset):
    """Simple dataset of text strings."""

    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def load_texts(conn, max_texts: int = 50_000, seed: int = 42) -> list[str]:
    """Load paper texts for distillation training."""
    rng = random.Random(seed)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT title, abstract
            FROM papers
            WHERE abstract IS NOT NULL
              AND LENGTH(abstract) > 100
        """)
        rows = cur.fetchall()

    texts = [f"{r['title']}. {r['abstract'][:512]}" for r in rows]
    rng.shuffle(texts)
    texts = texts[:max_texts]
    logger.info(f"Loaded {len(texts)} texts for distillation")
    return texts


def distill(
    texts: list[str],
    teacher_name: str = TEACHER_MODEL,
    student_name: str = STUDENT_MODEL,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    temperature: float = TEMPERATURE,
    output_dir: str | None = None,
) -> SentenceTransformer:
    """Distill teacher's similarity structure into student."""
    if output_dir is None:
        output_dir = str(OUTPUT_DIR)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading teacher: {teacher_name}")
    teacher = SentenceTransformer(teacher_name, device=str(device))
    teacher.eval()

    logger.info(f"Loading student: {student_name}")
    student = SentenceTransformer(student_name, device=str(device))

    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Get the tokenizer and transformer module for forward pass
    student_tokenizer = student.tokenizer
    student_module = student[0].auto_model

    optimizer = torch.optim.AdamW(student_module.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=int(total_steps * 0.1),
    )

    logger.info(f"Training: {len(texts)} texts, {epochs} epochs, batch_size={batch_size}")
    logger.info(f"Total steps: {total_steps}, temperature: {temperature}")

    student_module.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        start = time.time()

        for batch_texts in dataloader:
            batch_texts = list(batch_texts)

            # Teacher similarities (no grad, using encode is fine)
            with torch.no_grad():
                teacher_embs = teacher.encode(
                    batch_texts, convert_to_tensor=True,
                    normalize_embeddings=True, show_progress_bar=False,
                )
                teacher_sims = torch.mm(teacher_embs, teacher_embs.t()) / temperature
                teacher_probs = F.softmax(teacher_sims, dim=-1)

            # Student: manual forward pass to keep gradients
            encoded = student_tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            ).to(device)

            outputs = student_module(**encoded)
            token_embs = outputs.last_hidden_state  # (batch, seq, dim)

            # Mean pooling with attention mask
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            sum_embs = (token_embs * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            student_embs = sum_embs / sum_mask

            # L2 normalize
            student_embs = F.normalize(student_embs, p=2, dim=-1)

            student_sims = torch.mm(student_embs, student_embs.t()) / temperature
            student_log_probs = F.log_softmax(student_sims, dim=-1)

            # KL divergence: sum over distribution, mean over batch
            loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                avg = epoch_loss / n_batches
                logger.info(f"  Epoch {epoch+1}, batch {n_batches}/{len(dataloader)}, loss: {avg:.4f}")

        elapsed = time.time() - start
        avg_loss = epoch_loss / n_batches
        logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, time={elapsed:.0f}s")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    student.save(str(output_path.resolve()))
    logger.info(f"Distilled model saved to {output_dir}")

    return student


def embed_and_store(conn, model: SentenceTransformer, batch_size: int = 256):
    """Re-embed all papers with the distilled model."""
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

    logger.info(f"Embedding {len(texts)} papers with distilled model...")
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
        for pmid, emb in zip(pmids, embeddings):
            cur.execute(sql, (pmid, DB_MODEL_NAME, emb.tolist()))
    conn.commit()
    logger.info(f"Stored {len(pmids)} embeddings as '{DB_MODEL_NAME}'")


def evaluate_distilled(conn, student, teacher=None):
    """Evaluate distilled model against base MiniLM and optionally teacher."""
    base_model = SentenceTransformer(STUDENT_MODEL)

    models = [
        ("base-minilm", base_model, BASE_DB_MODEL),
        ("distilled", student, DB_MODEL_NAME),
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
                           1 - (e.embedding::vector({STUDENT_DIM}) <=> %s::vector({STUDENT_DIM})) as similarity
                    FROM embeddings e
                    JOIN papers p ON e.pmid = p.pmid
                    WHERE e.model_name = %s
                    ORDER BY e.embedding::vector({STUDENT_DIM}) <=> %s::vector({STUDENT_DIM})
                    LIMIT 10
                    """,
                    (query_vec, db_model_name, query_vec),
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
    print(f"  {'Query':<45} {'Base':>8} {'Dist':>8} {'D':>8}")
    print(f"{'='*70}")
    for b, d in zip(all_results["base-minilm"], all_results["distilled"]):
        query = b["query"][:43]
        delta = d["ndcg_5"] - b["ndcg_5"]
        marker = "+" if delta > 0 else ""
        print(f"  {query:<45} {b['ndcg_5']:>7.3f} {d['ndcg_5']:>7.3f} {marker}{delta:>7.3f}")

    base_5 = np.mean([r["ndcg_5"] for r in all_results["base-minilm"]])
    dist_5 = np.mean([r["ndcg_5"] for r in all_results["distilled"]])
    base_10 = np.mean([r["ndcg_10"] for r in all_results["base-minilm"]])
    dist_10 = np.mean([r["ndcg_10"] for r in all_results["distilled"]])

    print(f"{'─'*70}")
    d5 = dist_5 - base_5
    d10 = dist_10 - base_10
    print(f"  {'Mean NDCG@5':<45} {base_5:>7.3f} {dist_5:>7.3f} {'+'if d5>0 else ''}{d5:>7.3f}")
    print(f"  {'Mean NDCG@10':<45} {base_10:>7.3f} {dist_10:>7.3f} {'+'if d10>0 else ''}{d10:>7.3f}")

    return {"base_ndcg5": base_5, "distilled_ndcg5": dist_5, "delta": d5}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Distill PubMedBERT into MiniLM")
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-texts", type=int, default=50_000)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate")
    parser.add_argument("--skip-embed", action="store_true", help="Skip re-embedding")
    args = parser.parse_args()

    conn = psycopg2.connect(args.db_url)

    if args.eval_only:
        logger.info("Loading distilled model for evaluation...")
        student = SentenceTransformer(str(Path(args.output_dir).resolve()))
        evaluate_distilled(conn, student)
    else:
        texts = load_texts(conn, max_texts=args.max_texts)
        conn.close()

        student = distill(
            texts,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            output_dir=args.output_dir,
        )

        # Reconnect for embedding + evaluation
        conn = psycopg2.connect(args.db_url)

        if not args.skip_embed:
            embed_and_store(conn, student)

        evaluate_distilled(conn, student)

    conn.close()
