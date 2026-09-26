from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from movie_genre.config import MODEL_PATH

APP_PATH = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not available")


def run_app():
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


def test_app_renders_form_and_metrics():
    at = run_app()

    assert not at.exception
    assert at.title[0].value == "Movie genre classifier"
    assert [metric.label for metric in at.metric] == ["Test accuracy", "Weighted F1", "Top-3 accuracy"]
    assert at.text_area(key="plot").value == ""


def test_example_fills_form_and_predicts():
    at = run_app()

    at.pills(key="example").set_value("Game show").run()
    assert "trivia" in at.text_area(key="plot").value
    assert at.text_input(key="title").value == '"Brain Rush" (2014)'

    at.button[0].click().run()

    assert not at.exception
    assert at.subheader[0].value == "Predicted genre: game-show"


def test_empty_plot_shows_warning():
    at = run_app()

    at.button[0].click().run()

    assert not at.exception
    assert at.warning[0].value == "Enter a plot description first."
    assert not at.subheader
