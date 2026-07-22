FROM apache/airflow:2.10.0-python3.11

# The ingest DAG's embedding task runs the INT8 ONNX model directly. Installing
# torch + sentence-transformers here instead would add ~1.5GB to the scheduler
# image to compute vectors that are, at this precision, the same ones the API
# already produces for queries.
RUN pip install --no-cache-dir \
    "onnxruntime>=1.18" \
    "tokenizers>=0.19" \
    "huggingface_hub>=0.23"
