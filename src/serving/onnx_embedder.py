"""
Torch-free query embedder for memory-constrained serving.

Runs the INT8-quantized ONNX export of all-MiniLM-L6-v2 (see
src/embeddings/onnx_export.py) with onnxruntime + the bare `tokenizers`
library. Peak RSS stays around ~300MB vs ~1.5GB for the PyTorch stack,
which is what lets the API fit on a 512MB free-tier instance.

Model files are pulled from the HF Hub on first use (cached thereafter).
"""

import logging
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

ONNX_MODEL_REPO = os.environ.get("ONNX_MODEL_REPO", "chibanaryan/minilm-pubmed-onnx")
MAX_LENGTH = 256


class OnnxEmbedder:
    """Duck-types SentenceTransformer.encode() for the query path."""

    def __init__(self, model_path: str, tokenizer_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(MAX_LENGTH)

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        enc = self.tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)

        hidden = self.session.run(
            ["last_hidden_state"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]

        # Mean pooling over non-padding tokens, then L2 normalize — mirrors
        # sentence-transformers' pooling head, which isn't in the ONNX graph.
        mask = attention_mask[..., np.newaxis].astype(np.float32)
        emb = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        if normalize_embeddings:
            emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
        return emb[0]


def load_onnx_embedder() -> OnnxEmbedder:
    """Load from local paths if set, otherwise download from the HF Hub."""
    model_path = os.environ.get("ONNX_MODEL_PATH")
    tokenizer_path = os.environ.get("ONNX_TOKENIZER_PATH")
    if not (model_path and tokenizer_path):
        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading ONNX model from {ONNX_MODEL_REPO}...")
        model_path = hf_hub_download(ONNX_MODEL_REPO, "model_int8.onnx")
        tokenizer_path = hf_hub_download(ONNX_MODEL_REPO, "tokenizer.json")
    return OnnxEmbedder(model_path, tokenizer_path)
