"""Tests for the embedding pipeline (no DB or model downloads required)."""

from unittest.mock import MagicMock

import numpy as np

from src.embeddings.embed_pipeline import (
    generate_embeddings,
    get_unembedded_abstracts,
    store_embeddings,
)


def _mock_cursor(conn):
    """Wire up conn.cursor() as a context manager returning a mock cursor."""
    cur = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cur)
    cm.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cm
    return cur


class TestGenerateEmbeddings:
    def test_batches_and_stacks(self):
        model = MagicMock()
        model.encode.side_effect = lambda batch, **kw: np.zeros((len(batch), 384))

        texts = [f"text {i}" for i in range(10)]
        result = generate_embeddings(model, texts, batch_size=4, show_progress=False)

        assert result.shape == (10, 384)
        assert model.encode.call_count == 3  # 4 + 4 + 2

    def test_normalizes_embeddings(self):
        model = MagicMock()
        model.encode.return_value = np.zeros((1, 384))

        generate_embeddings(model, ["text"], batch_size=8, show_progress=False)

        _, kwargs = model.encode.call_args
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["convert_to_numpy"] is True


class TestStoreEmbeddings:
    def test_upserts_each_row_and_commits(self):
        conn = MagicMock()
        cur = _mock_cursor(conn)
        embeddings = np.ones((3, 384))

        store_embeddings(conn, [1, 2, 3], embeddings, "all-MiniLM-L6-v2")

        assert cur.execute.call_count == 3
        sql, params = cur.execute.call_args[0]
        assert "ON CONFLICT (pmid, model_name)" in sql
        assert params[0] == 3
        assert params[1] == "all-MiniLM-L6-v2"
        assert params[2] == [1.0] * 384
        conn.commit.assert_called_once()


class TestGetUnembeddedAbstracts:
    def test_queries_by_model_name(self):
        conn = MagicMock()
        cur = _mock_cursor(conn)
        cur.fetchall.return_value = [{"pmid": 1, "title": "t", "abstract": "a", "mesh_terms": []}]

        rows = get_unembedded_abstracts(conn, "all-MiniLM-L6-v2")

        assert rows[0]["pmid"] == 1
        sql, params = cur.execute.call_args[0]
        assert "LEFT JOIN embeddings" in sql
        assert "LIMIT" not in sql
        assert params == ("all-MiniLM-L6-v2",)

    def test_applies_limit(self):
        conn = MagicMock()
        cur = _mock_cursor(conn)
        cur.fetchall.return_value = []

        get_unembedded_abstracts(conn, "all-MiniLM-L6-v2", limit=100)

        sql, _ = cur.execute.call_args[0]
        assert "LIMIT 100" in sql
