"""
Export embedding model to ONNX and quantize to INT8.

Benchmarks inference latency and measures NDCG degradation
compared to the PyTorch model.

Usage:
    # Export and quantize
    python -m src.embeddings.onnx_export --export --quantize

    # Benchmark latency
    python -m src.embeddings.onnx_export --benchmark

    # Full pipeline: export, quantize, benchmark, evaluate
    python -m src.embeddings.onnx_export --all --db-url postgresql://...
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import psycopg2
import psycopg2.extras
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.embeddings.evaluate import EVAL_QUERIES, compute_relevance_score, ndcg

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_MODEL_NAME = "all-MiniLM-L6-v2"  # model_name stored in DB
DIM = 384
ONNX_DIR = Path("models/minilm-onnx")
ONNX_PATH = ONNX_DIR / "model.onnx"
QUANTIZED_PATH = ONNX_DIR / "model_int8.onnx"


def export_to_onnx(model_name: str = MODEL_NAME, output_path: Path = ONNX_PATH):
    """Export sentence transformer to ONNX."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get the transformer module and move to CPU for export
    transformer = model[0]
    auto_model = transformer.auto_model.cpu()

    # Create dummy input
    dummy_text = "This is a test sentence for ONNX export."
    encoded = tokenizer(
        dummy_text,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )

    auto_model.eval()
    with torch.no_grad():
        torch.onnx.export(
            auto_model,
            (encoded["input_ids"], encoded["attention_mask"]),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "last_hidden_state": {0: "batch_size", 1: "sequence"},
            },
            opset_version=14,
        )

    # Verify
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Exported ONNX model: {output_path} ({size_mb:.1f} MB)")
    return output_path


def quantize_model(input_path: Path = ONNX_PATH, output_path: Path = QUANTIZED_PATH):
    """Quantize ONNX model to INT8."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )

    orig_size = input_path.stat().st_size / 1024 / 1024
    quant_size = output_path.stat().st_size / 1024 / 1024
    reduction = (1 - quant_size / orig_size) * 100

    logger.info(f"Quantized: {orig_size:.1f} MB → {quant_size:.1f} MB ({reduction:.0f}% smaller)")
    return output_path


class OnnxEmbedder:
    """ONNX-based embedding model with mean pooling."""

    def __init__(self, onnx_path: str, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )

    def encode(self, texts: str | list[str], normalize_embeddings: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np",
        )

        outputs = self.session.run(
            None,
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            },
        )

        # Mean pooling
        token_embeddings = outputs[0]
        attention_mask = encoded["attention_mask"]
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

        return embeddings


def benchmark(n_iterations: int = 100):
    """Benchmark latency: PyTorch vs ONNX vs INT8."""
    test_queries = [
        "effects of creatine supplementation on muscle recovery",
        "gut microbiome and mental health connection",
        "resistance training for older adults",
    ]

    results = {}

    # PyTorch
    logger.info("Benchmarking PyTorch...")
    pt_model = SentenceTransformer(MODEL_NAME)
    # Warmup
    for q in test_queries:
        pt_model.encode(q, normalize_embeddings=True)

    times = []
    for _ in range(n_iterations):
        for q in test_queries:
            start = time.perf_counter()
            pt_model.encode(q, normalize_embeddings=True)
            times.append(time.perf_counter() - start)
    results["pytorch"] = {
        "mean_ms": np.mean(times) * 1000,
        "p50_ms": np.percentile(times, 50) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
    }

    # ONNX FP32
    if ONNX_PATH.exists():
        logger.info("Benchmarking ONNX FP32...")
        onnx_model = OnnxEmbedder(str(ONNX_PATH))
        for q in test_queries:
            onnx_model.encode(q)

        times = []
        for _ in range(n_iterations):
            for q in test_queries:
                start = time.perf_counter()
                onnx_model.encode(q)
                times.append(time.perf_counter() - start)
        results["onnx_fp32"] = {
            "mean_ms": np.mean(times) * 1000,
            "p50_ms": np.percentile(times, 50) * 1000,
            "p95_ms": np.percentile(times, 95) * 1000,
        }

    # ONNX INT8
    if QUANTIZED_PATH.exists():
        logger.info("Benchmarking ONNX INT8...")
        int8_model = OnnxEmbedder(str(QUANTIZED_PATH))
        for q in test_queries:
            int8_model.encode(q)

        times = []
        for _ in range(n_iterations):
            for q in test_queries:
                start = time.perf_counter()
                int8_model.encode(q)
                times.append(time.perf_counter() - start)
        results["onnx_int8"] = {
            "mean_ms": np.mean(times) * 1000,
            "p50_ms": np.percentile(times, 50) * 1000,
            "p95_ms": np.percentile(times, 95) * 1000,
        }

    # Print results
    print(f"\n{'='*55}")
    print(f"  {'Model':<15} {'Mean':>8} {'P50':>8} {'P95':>8}")
    print(f"{'='*55}")
    for name, m in results.items():
        print(f"  {name:<15} {m['mean_ms']:>7.2f}ms {m['p50_ms']:>7.2f}ms {m['p95_ms']:>7.2f}ms")

    if "pytorch" in results and "onnx_int8" in results:
        speedup = results["pytorch"]["mean_ms"] / results["onnx_int8"]["mean_ms"]
        print(f"\n  INT8 speedup vs PyTorch: {speedup:.1f}x")

    return results


def evaluate_onnx(conn, onnx_path: str, label: str):
    """Evaluate ONNX model against stored embeddings."""
    model = OnnxEmbedder(onnx_path)
    results = []

    for eq in EVAL_QUERIES:
        query_vec = model.encode(eq["query"], normalize_embeddings=True)[0].tolist()

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
                (query_vec, DB_MODEL_NAME, query_vec),
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

    mean_5 = np.mean([r["ndcg_5"] for r in results])
    mean_10 = np.mean([r["ndcg_10"] for r in results])
    return {"label": label, "ndcg_5": mean_5, "ndcg_10": mean_10, "details": results}


def evaluate_all(conn):
    """Compare PyTorch, ONNX FP32, and ONNX INT8 on NDCG."""
    # PyTorch baseline
    pt_model = SentenceTransformer(MODEL_NAME)
    pt_results = []
    for eq in EVAL_QUERIES:
        query_vec = pt_model.encode(eq["query"], normalize_embeddings=True).tolist()
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
                (query_vec, DB_MODEL_NAME, query_vec),
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
        pt_results.append({"ndcg_5": ndcg(relevances, 5), "ndcg_10": ndcg(relevances, 10)})

    all_evals = [{"label": "pytorch", "ndcg_5": np.mean([r["ndcg_5"] for r in pt_results]),
                  "ndcg_10": np.mean([r["ndcg_10"] for r in pt_results])}]

    if ONNX_PATH.exists():
        all_evals.append(evaluate_onnx(conn, str(ONNX_PATH), "onnx_fp32"))
    if QUANTIZED_PATH.exists():
        all_evals.append(evaluate_onnx(conn, str(QUANTIZED_PATH), "onnx_int8"))

    print(f"\n{'='*50}")
    print(f"  {'Model':<15} {'NDCG@5':>10} {'NDCG@10':>10}")
    print(f"{'='*50}")
    for e in all_evals:
        print(f"  {e['label']:<15} {e['ndcg_5']:>10.4f} {e['ndcg_10']:>10.4f}")

    if len(all_evals) > 1:
        baseline = all_evals[0]["ndcg_5"]
        for e in all_evals[1:]:
            delta = e["ndcg_5"] - baseline
            print(f"  {e['label']} vs pytorch: {'+' if delta >= 0 else ''}{delta:.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="ONNX export and quantization")
    parser.add_argument("--export", action="store_true", help="Export to ONNX")
    parser.add_argument("--quantize", action="store_true", help="Quantize to INT8")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark latency")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate NDCG")
    parser.add_argument("--all", action="store_true", help="Export, quantize, benchmark, evaluate")
    parser.add_argument("--db-url", default=None, help="DB URL for evaluation")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    if args.all:
        args.export = args.quantize = args.benchmark = args.evaluate = True

    if args.export:
        export_to_onnx()

    if args.quantize:
        quantize_model()

    if args.benchmark:
        benchmark(n_iterations=args.iterations)

    if args.evaluate:
        if not args.db_url:
            logger.error("--db-url required for evaluation")
        else:
            conn = psycopg2.connect(args.db_url)
            evaluate_all(conn)
            conn.close()

    if not any([args.export, args.quantize, args.benchmark, args.evaluate]):
        parser.print_help()
