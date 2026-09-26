"""Pipeline definition and helpers shared by training and inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def build_pipeline(classifier):
    # Inputs are already cleaned by preprocess_many, so no preprocessor here.
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100_000,
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("clf", classifier),
    ])


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


def load_artifact(artifact_path: Path):
    artifact = joblib.load(artifact_path)
    return artifact["model"], artifact["label_encoder"]
