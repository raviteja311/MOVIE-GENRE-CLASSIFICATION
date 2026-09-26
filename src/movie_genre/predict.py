"""Predict movie genres from a plot string using the saved pipeline artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from movie_genre.config import MODEL_PATH
from movie_genre.model import load_artifact, predict_top_k
from movie_genre.preprocessing import ensure_nltk_resources

DEFAULT_ARTIFACT = MODEL_PATH


def main() -> int:
    parser = argparse.ArgumentParser(prog="movie-genre-predict", description="Predict a movie genre from a plot string.")
    parser.add_argument("text", help="Movie plot or description to classify.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="Path to the saved joblib artifact.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of genre guesses to print.")
    args = parser.parse_args()

    if not args.artifact.exists():
        parser.error(f"Artifact not found: {args.artifact}. Run `movie-genre-train` first.")
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
