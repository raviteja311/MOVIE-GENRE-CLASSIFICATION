import pytest
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from movie_genre.config import MODEL_PATH
from movie_genre.model import build_pipeline, load_artifact, predict_top_k
from movie_genre.preprocessing import ensure_nltk_resources

TEXTS = [
    "space ship alien planet laser",
    "alien invasion space fleet",
    "love wedding romance couple",
    "romantic couple fall in love",
    "murder detective killer police",
    "detective hunts serial killer",
]
GENRES = ["sci-fi", "sci-fi", "romance", "romance", "thriller", "thriller"]


def test_predict_top_k_returns_ranked_distinct_labels():
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(GENRES)
    model = build_pipeline(LinearSVC(random_state=0))
    model.set_params(tfidf__min_df=1, tfidf__max_df=1.0)
    model.fit(TEXTS, y)

    top = predict_top_k(model, label_encoder, ["alien space battle"], k=3)[0]

    assert top[0] == "sci-fi"
    assert sorted(top) == ["romance", "sci-fi", "thriller"]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not available")
def test_saved_model_predicts_from_raw_text():
    ensure_nltk_resources()
    model, label_encoder = load_artifact(MODEL_PATH)

    top = predict_top_k(model, label_encoder, ["A detective hunts a serial killer in a rain-soaked city."], k=3)[0]

    assert len(top) == 3
    assert set(top) <= set(label_encoder.classes_)
