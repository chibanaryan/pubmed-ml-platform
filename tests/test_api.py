"""Tests for the FastAPI serving layer."""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked DB and model."""
    with patch("src.serving.api.SentenceTransformer") as mock_st, \
         patch("src.serving.api.psycopg2") as mock_pg:

        import numpy as np

        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        mock_st.return_value = mock_model

        # Mock the DB connection
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn

        from src.serving.api import app
        import src.serving.api as api_module
        api_module._models["all-MiniLM-L6-v2"] = mock_model
        api_module._conn = mock_conn

        yield TestClient(app, raise_server_exceptions=False), mock_conn, mock_model


class TestHealthEndpoint:
    def test_health_returns_paper_count(self, client):
        test_client, mock_conn, _ = client
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

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

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
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
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        resp = test_client.post("/search", json={"query": "creatine muscle"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["pmid"] == 12345
        assert data["results"][0]["similarity"] == 0.95
        assert "latency_ms" in data


class TestPaperEndpoint:
    def test_get_paper_not_found(self, client):
        test_client, mock_conn, _ = client
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        resp = test_client.get("/paper/99999")
        assert resp.status_code == 404

    def test_get_paper_returns_data(self, client):
        test_client, mock_conn, _ = client
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "pmid": 12345,
            "title": "Test",
            "abstract": "Abstract",
            "authors": ["Smith"],
            "journal": "Journal",
            "pub_date": date(2024, 1, 1),
            "mesh_terms": ["Term"],
        }
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        resp = test_client.get("/paper/12345")
        assert resp.status_code == 200
        assert resp.json()["pmid"] == 12345
