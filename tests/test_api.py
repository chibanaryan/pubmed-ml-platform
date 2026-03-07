"""Tests for the FastAPI serving layer."""

import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked DB pool and model."""
    with patch("src.serving.api.SentenceTransformer") as mock_st:
        import numpy as np

        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        mock_st.return_value = mock_model

        # Mock asyncpg pool and connection
        mock_conn = AsyncMock()
        mock_pool = MagicMock()

        # Make pool.acquire() work as async context manager
        acm = AsyncMock()
        acm.__aenter__ = AsyncMock(return_value=mock_conn)
        acm.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire.return_value = acm
        mock_pool.close = AsyncMock()

        from src.serving.api import app
        import src.serving.api as api_module
        api_module._models["all-MiniLM-L6-v2"] = mock_model
        api_module._pool = mock_pool

        yield TestClient(app, raise_server_exceptions=False), mock_conn, mock_model


class TestHealthEndpoint:
    def test_health_returns_paper_count(self, client):
        test_client, mock_conn, _ = client
        mock_conn.fetchval = AsyncMock(return_value=42)

        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["papers_count"] == 42


class TestSearchEndpoint:
    def test_search_requires_query(self, client):
        test_client, _, _ = client
        resp = test_client.post("/search", json={"query": ""})
        assert resp.status_code == 422

    def test_search_enforces_top_k_bounds(self, client):
        test_client, _, _ = client
        resp = test_client.post("/search", json={"query": "test", "top_k": 0})
        assert resp.status_code == 422

        resp = test_client.post("/search", json={"query": "test", "top_k": 101})
        assert resp.status_code == 422

    def test_search_returns_results(self, client):
        test_client, mock_conn, mock_model = client
        import numpy as np

        mock_model.encode.return_value = np.zeros(384)
        mock_conn.fetch = AsyncMock(return_value=[
            {
                "pmid": 12345,
                "title": "Test Paper",
                "abstract": "Test abstract",
                "authors": json.dumps(["Smith, John"]),
                "journal": "Test Journal",
                "pub_date": date(2024, 1, 1),
                "mesh_terms": json.dumps(["Creatine"]),
                "similarity": 0.95,
            }
        ])

        resp = test_client.post("/search", json={"query": "creatine muscle"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["pmid"] == 12345
        assert data["results"][0]["similarity"] == 0.95
        assert "latency_ms" in data

    def test_search_rejects_unknown_model(self, client):
        test_client, _, _ = client
        resp = test_client.post("/search", json={"query": "test", "model_name": "nonexistent-model"})
        assert resp.status_code == 400


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, client):
        test_client, _, _ = client
        resp = test_client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        body = resp.text
        assert "pubmed_requests_total" in body
        assert "pubmed_models_loaded" in body
        assert "pubmed_search_latency_seconds" in body

    def test_metrics_tracks_requests(self, client):
        test_client, mock_conn, mock_model = client
        import numpy as np
        import src.serving.api as api_module

        # Reset metrics
        api_module._metrics["requests_total"] = 0
        api_module._metrics["requests_by_endpoint"] = {}

        mock_model.encode.return_value = np.zeros(384)
        mock_conn.fetch = AsyncMock(return_value=[])

        test_client.post("/search", json={"query": "test"})
        resp = test_client.get("/metrics")
        assert 'pubmed_endpoint_requests_total{endpoint="search"} 1' in resp.text


class TestPaperEndpoint:
    def test_get_paper_not_found(self, client):
        test_client, mock_conn, _ = client
        mock_conn.fetchrow = AsyncMock(return_value=None)

        resp = test_client.get("/paper/99999")
        assert resp.status_code == 404

    def test_get_paper_returns_data(self, client):
        test_client, mock_conn, _ = client
        mock_conn.fetchrow = AsyncMock(return_value={
            "pmid": 12345,
            "title": "Test",
            "abstract": "Abstract",
            "authors": ["Smith"],
            "journal": "Journal",
            "pub_date": date(2024, 1, 1),
            "mesh_terms": ["Term"],
        })

        resp = test_client.get("/paper/12345")
        assert resp.status_code == 200
        assert resp.json()["pmid"] == 12345
