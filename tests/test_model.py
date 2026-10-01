import pytest

from movie_genre.config import MODEL_PATH
from movie_genre.model import load_artifact, predict_top_k, to_frame


def test_predict_top_k_returns_ranked_distinct_labels(tiny_model):
    model, label_encoder = tiny_model

    top = predict_top_k(model, label_encoder, to_frame(["alien space battle"], ["Galaxy (2001)"]), k=3)[0]

    assert top[0] == "sci-fi"
    assert sorted(top) == ["romance", "sci-fi", "thriller"]


def test_predict_works_without_title(tiny_model):
    model, label_encoder = tiny_model

    top = predict_top_k(model, label_encoder, to_frame(["a couple falls in love at a wedding"]), k=1)[0]

    assert top[0] == "romance"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not available")
def test_saved_model_predicts_from_raw_text():
    model, label_encoder = load_artifact(MODEL_PATH)
    frame = to_frame(["A detective hunts a serial killer in a rain-soaked city."], ["Rain (1998)"])

    top = predict_top_k(model, label_encoder, frame, k=3)[0]

    assert len(top) == 3
    assert set(top) <= set(label_encoder.classes_)
