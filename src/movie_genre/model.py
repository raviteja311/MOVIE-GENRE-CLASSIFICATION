"""Pipeline definition and helpers shared by training and inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from movie_genre.preprocessing import preprocess_text

FEATURE_COLUMNS = ["DESCRIPTION", "TITLE"]


def build_pipeline(classifier):
    features = ColumnTransformer([
        ("description", TfidfVectorizer(
            preprocessor=preprocess_text,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100_000,
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
        ), "DESCRIPTION"),
        # Character n-grams pick up the release year and TV-episode quoting in titles.
        ("title", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=50_000,
            min_df=3,
            sublinear_tf=True,
        ), "TITLE"),
    ])
    return Pipeline([("features", features), ("clf", classifier)])


def to_frame(descriptions, titles=None):
    """Build model input from plot texts and optional titles (missing titles become "")."""
    descriptions = list(descriptions)
    titles = [""] * len(descriptions) if titles is None else list(titles)
    return pd.DataFrame({"DESCRIPTION": descriptions, "TITLE": titles})


def predict_top_k(model, label_encoder, frame, k: int = 3):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(frame)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(frame)
    else:
        raise ValueError("Model does not expose probability or decision scores.")

    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    top_indices = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    return label_encoder.inverse_transform(top_indices.ravel()).reshape(top_indices.shape)


def load_artifact(artifact_path: Path):
    artifact = joblib.load(artifact_path)
    return artifact["model"], artifact["label_encoder"]
