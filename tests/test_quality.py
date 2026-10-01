"""Minimum-quality regression checks for the shipped model."""

import json

import pandas as pd
import pytest

from movie_genre.config import DATA_DIR, METRICS_PATH, MODEL_PATH
from movie_genre.model import load_artifact, to_frame

# The shipped model scores about 59.4% test accuracy; fail if a change drops it noticeably.
MIN_TEST_ACCURACY = 0.58
MIN_TEST_WEIGHTED_F1 = 0.58
# A fixed 2,000-row test sample; its score sits within a couple of points of the full test set.
SAMPLE_ROWS = 2000
MIN_SAMPLE_ACCURACY = 0.56

SOLUTION_PATH = DATA_DIR / "test_data_solution.txt"


@pytest.mark.skipif(not METRICS_PATH.exists(), reason="metrics.json not available")
def test_recorded_shipped_model_metrics_meet_minimum():
    test = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["shipped_model"]["test"]

    assert test["accuracy"] >= MIN_TEST_ACCURACY
    assert test["weighted_f1"] >= MIN_TEST_WEIGHTED_F1


@pytest.mark.skipif(not (MODEL_PATH.exists() and SOLUTION_PATH.exists()), reason="trained model or test data not available")
def test_saved_model_accuracy_on_test_sample():
    sample = pd.read_csv(
        SOLUTION_PATH, sep=":::", names=["ID", "TITLE", "GENRE", "DESCRIPTION"], engine="python", nrows=SAMPLE_ROWS,
    )
    model, label_encoder = load_artifact(MODEL_PATH)

    predicted = label_encoder.inverse_transform(model.predict(to_frame(sample["DESCRIPTION"].str.strip(), sample["TITLE"].str.strip())))

    assert (predicted == sample["GENRE"].str.strip().str.lower()).mean() >= MIN_SAMPLE_ACCURACY
