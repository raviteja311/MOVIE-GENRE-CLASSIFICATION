"""Predict movie genres from a plot string, or a file of plots, using the saved pipeline artifact."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from movie_genre.config import MODEL_PATH
from movie_genre.model import load_artifact, predict_top_k, to_frame

DEFAULT_ARTIFACT = MODEL_PATH


def read_batch(path: Path):
    """Return (descriptions, titles) from a CSV with a description column, or a text file with one plot per line."""
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        columns = {column.strip().lower(): column for column in frame.columns}
        if "description" not in columns:
            raise ValueError(f"{path} needs a 'description' column (and optionally 'title').")
        descriptions = frame[columns["description"]].str.strip()
        titles = frame[columns["title"]].str.strip() if "title" in columns else pd.Series([""] * len(frame))
        keep = descriptions != ""
        return descriptions[keep].tolist(), titles[keep].tolist()

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    descriptions = [line for line in lines if line]
    return descriptions, [""] * len(descriptions)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="movie-genre-predict", description="Predict a movie genre from a plot string or a file of plots.")
    parser.add_argument("text", nargs="?", help="Movie plot or description to classify.")
    parser.add_argument("--title", default="", help='Optional title with year, e.g. "Heat (1995)". Improves accuracy.')
    parser.add_argument(
        "--file", type=Path,
        help="Classify many plots in one run: a .csv with a 'description' (and optional 'title') column, "
             "or a text file with one description per line. Prints CSV to stdout.",
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="Path to the saved joblib artifact.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of genre guesses to print.")
    args = parser.parse_args(argv)

    if args.file and args.text:
        parser.error("Pass either a plot text or --file, not both.")
    if args.file and not args.file.exists():
        parser.error(f"File not found: {args.file}")
    if not args.file and not (args.text or "").strip():
        parser.error("Text must not be empty.")
    if not args.artifact.exists():
        parser.error(f"Artifact not found: {args.artifact}. Run `movie-genre-train` first.")

    if args.file:
        try:
            descriptions, titles = read_batch(args.file)
        except ValueError as error:
            parser.error(str(error))
        if not descriptions:
            parser.error(f"No descriptions found in {args.file}")

    model, label_encoder = load_artifact(args.artifact)
    top_k = max(1, min(args.top_k, len(label_encoder.classes_)))

    if args.file:
        top_labels = predict_top_k(model, label_encoder, to_frame(descriptions, titles), k=top_k)
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(["row", "title", *[f"genre_{rank}" for rank in range(1, top_k + 1)]])
        for row, (title, labels) in enumerate(zip(titles, top_labels, strict=True), 1):
            writer.writerow([row, title, *labels])
        return 0

    top_labels = predict_top_k(model, label_encoder, to_frame([args.text], [args.title]), k=top_k)[0]

    print(f"Top-1 prediction: {top_labels[0]}")
    print("Top-k predictions:")
    for rank, label in enumerate(top_labels, 1):
        print(f"{rank}. {label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
