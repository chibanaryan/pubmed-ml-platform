"""
MLflow model registry management.

Handles model promotion (staging → production) and loading models
from the registry for serving.

Usage:
    # List registered models and their versions
    python -m src.embeddings.registry list

    # Promote a model version to production
    python -m src.embeddings.registry promote pubmed-minilm --version 3

    # Load the production model for a given name
    python -m src.embeddings.registry load pubmed-minilm
"""

import argparse
import logging

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def list_models(client: MlflowClient):
    """List all registered models and their versions."""
    models = client.search_registered_models()
    if not models:
        print("No registered models found.")
        return

    for rm in models:
        print(f"\n{rm.name}")
        print(f"  Description: {rm.description or '(none)'}")
        for mv in client.search_model_versions(f"name='{rm.name}'"):
            alias_str = ""
            if mv.aliases:
                alias_str = f" [{', '.join(mv.aliases)}]"
            run_id = mv.run_id[:8] if mv.run_id else "unknown"
            print(f"  Version {mv.version}: {mv.current_stage}{alias_str} (run={run_id})")


def promote_model(client: MlflowClient, model_name: str, version: int):
    """Promote a model version by setting the 'production' alias."""
    # Check model version exists
    try:
        mv = client.get_model_version(model_name, str(version))
    except Exception as e:
        print(f"Error: {e}")
        return

    # Set the production alias (replaces any previous production version)
    client.set_registered_model_alias(model_name, "production", str(version))
    print(f"Set alias 'production' on {model_name} version {version}")
    print(f"  Run ID: {mv.run_id}")
    print(f"  Source: {mv.source}")

    # Also set staging alias on the previous production version if it exists
    all_versions = client.search_model_versions(f"name='{model_name}'")
    for v in all_versions:
        if v.aliases and "production" in v.aliases and str(v.version) != str(version):
            # The alias was already moved, but log it for clarity
            print(f"  (Previous production was version {v.version})")


def load_production_model(model_name: str):
    """Load the production version of a model from the registry."""
    model_uri = f"models:/{model_name}@production"
    try:
        model = mlflow.sentence_transformers.load_model(model_uri)
        print(f"Loaded {model_name}@production from registry")
        # Quick test
        test_embedding = model.encode("test query", normalize_embeddings=True)
        print(f"  Embedding dimension: {len(test_embedding)}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure a version has the 'production' alias set.")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="MLflow model registry management")
    parser.add_argument("--mlflow-uri", default="http://localhost:5001")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List registered models")

    promote_parser = subparsers.add_parser("promote", help="Promote a model version to production")
    promote_parser.add_argument("model_name", help="Registered model name (e.g., pubmed-minilm)")
    promote_parser.add_argument("--version", type=int, required=True, help="Version number to promote")

    load_parser = subparsers.add_parser("load", help="Load the production model")
    load_parser.add_argument("model_name", help="Registered model name")

    args = parser.parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient(args.mlflow_uri)

    if args.command == "list":
        list_models(client)
    elif args.command == "promote":
        promote_model(client, args.model_name, args.version)
    elif args.command == "load":
        load_production_model(args.model_name)
    else:
        parser.print_help()
