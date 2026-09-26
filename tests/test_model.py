import pytest
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from movie_genre.config import MODEL_PATH
from movie_genre.model import build_pipeline, load_artifact, predict_top_k, to_frame

TEXTS = [
    "space ship alien planet laser",
    "alien invasion space fleet",
    "love wedding romance couple",
    "romantic couple fall in love",
    "murder detective killer police",
    "detective hunts serial killer",
]
TITLES = ["Star Run (1999)", "Invaders (2004)", "June Vows (2010)", "Paris Hearts (2012)", "Night Cop (1987)", "The Hunt (1991)"]
GENRES = ["sci-fi", "sci-fi", "romance", "romance", "thriller", "thriller"]


def fit_tiny_model():
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(GENRES)
    model = build_pipeline(LinearSVC(random_state=0))
    model.set_params(features__description__min_df=1, features__description__max_df=1.0, features__title__min_df=1)
    model.fit(to_frame(TEXTS, TITLES), y)
    return model, label_encoder


def test_predict_top_k_returns_ranked_distinct_labels():
    model, label_encoder = fit_tiny_model()

    top = predict_top_k(model, label_encoder, to_frame(["alien space battle"], ["Galaxy (2001)"]), k=3)[0]

    assert top[0] == "sci-fi"
    assert sorted(top) == ["romance", "sci-fi", "thriller"]


def test_predict_works_without_title():
    model, label_encoder = fit_tiny_model()

    top = predict_top_k(model, label_encoder, to_frame(["a couple falls in love at a wedding"]), k=1)[0]

    assert top[0] == "romance"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not available")
def test_saved_model_predicts_from_raw_text():
    model, label_encoder = load_artifact(MODEL_PATH)
    frame = to_frame(["A detective hunts a serial killer in a rain-soaked city."], ["Rain (1998)"])

    top = predict_top_k(model, label_encoder, frame, k=3)[0]

    assert len(top) == 3
    assert set(top) <= set(label_encoder.classes_)
