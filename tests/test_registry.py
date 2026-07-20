"""Tests for the MLflow model registry management script."""

from unittest.mock import MagicMock, patch

from src.embeddings.registry import list_models, load_production_model, promote_model


def _version(version="1", aliases=(), run_id="abc12345def", stage="None"):
    mv = MagicMock()
    mv.version = version
    mv.aliases = list(aliases)
    mv.run_id = run_id
    mv.current_stage = stage
    mv.source = "s3://bucket/model"
    return mv


class TestListModels:
    def test_empty_registry(self, capsys):
        client = MagicMock()
        client.search_registered_models.return_value = []

        list_models(client)

        assert "No registered models found" in capsys.readouterr().out

    def test_lists_versions_with_aliases(self, capsys):
        client = MagicMock()
        rm = MagicMock()
        rm.name = "pubmed-minilm"
        rm.description = None
        client.search_registered_models.return_value = [rm]
        client.search_model_versions.return_value = [
            _version("2", aliases=["production"]),
            _version("1", run_id=None),  # run_id can be None; must not crash
        ]

        list_models(client)

        out = capsys.readouterr().out
        assert "pubmed-minilm" in out
        assert "[production]" in out
        assert "run=abc12345" in out
        assert "run=unknown" in out


class TestPromoteModel:
    def test_sets_production_alias(self, capsys):
        client = MagicMock()
        client.get_model_version.return_value = _version("3")
        client.search_model_versions.return_value = []

        promote_model(client, "pubmed-minilm", 3)

        client.set_registered_model_alias.assert_called_once_with(
            "pubmed-minilm", "production", "3"
        )
        assert "Set alias 'production'" in capsys.readouterr().out

    def test_missing_version_does_not_promote(self, capsys):
        client = MagicMock()
        client.get_model_version.side_effect = Exception("version not found")

        promote_model(client, "pubmed-minilm", 99)

        client.set_registered_model_alias.assert_not_called()
        assert "Error" in capsys.readouterr().out


class TestLoadProductionModel:
    def test_loads_from_registry(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.0] * 384

        with patch("src.embeddings.registry.mlflow") as mock_mlflow:
            mock_mlflow.sentence_transformers.load_model.return_value = mock_model
            model = load_production_model("pubmed-minilm")

        assert model is mock_model
        mock_mlflow.sentence_transformers.load_model.assert_called_once_with(
            "models:/pubmed-minilm@production"
        )

    def test_returns_none_when_no_production_alias(self, capsys):
        with patch("src.embeddings.registry.mlflow") as mock_mlflow:
            mock_mlflow.sentence_transformers.load_model.side_effect = Exception("no alias")
            model = load_production_model("pubmed-minilm")

        assert model is None
        assert "Error loading model" in capsys.readouterr().out
