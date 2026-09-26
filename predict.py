#!/usr/bin/env python
"""Predict movie genres from a plot string using the saved pipeline artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

# The saved pipeline references text_utils.preprocess_text, so this module must
# be importable before the artifact is unpickled.
from text_utils import ensure_nltk_resources

DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "movie_genre_classifier.joblib"


def load_artifact(artifact_path: Path):
    artifact = joblib.load(artifact_path)
    return artifact["model"], artifact["label_encoder"]


def predict_top_k(model, label_encoder, texts, k: int = 3):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(texts)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
    else:
        raise ValueError("Model does not expose probability or decision scores.")

    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    top_indices = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    return label_encoder.inverse_transform(top_indices.ravel()).reshape(top_indices.shape)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a movie genre from a plot string.")
    parser.add_argument("text", help="Movie plot or description to classify.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="Path to the saved joblib artifact.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of genre guesses to print.")
    args = parser.parse_args()

    if not args.artifact.exists():
        parser.error(f"Artifact not found: {args.artifact}. Run `python evaluate_models.py` first.")
    if not args.text.strip():
        parser.error("Text must not be empty.")

    ensure_nltk_resources()
    model, label_encoder = load_artifact(args.artifact)
    top_k = max(1, min(args.top_k, len(label_encoder.classes_)))
    top_labels = predict_top_k(model, label_encoder, [args.text], k=top_k)[0]

    print(f"Top-1 prediction: {top_labels[0]}")
    print("Top-k predictions:")
    for rank, label in enumerate(top_labels, 1):
        print(f"{rank}. {label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
