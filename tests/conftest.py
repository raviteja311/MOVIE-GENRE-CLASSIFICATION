import pytest
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from movie_genre.model import build_pipeline, to_frame

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


@pytest.fixture
def tiny_model():
    """A six-document pipeline and its label encoder, fast enough for unit tests."""
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(GENRES)
    model = build_pipeline(LinearSVC(random_state=0))
    model.set_params(features__description__min_df=1, features__description__max_df=1.0, features__title__min_df=1)
    model.fit(to_frame(TEXTS, TITLES), y)
    return model, label_encoder
