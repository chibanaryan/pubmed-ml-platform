# Build stage: install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install ".[all]"

# Runtime stage: slim image with only what's needed
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
