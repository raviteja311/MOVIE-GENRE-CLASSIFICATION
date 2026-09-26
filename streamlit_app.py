"""Streamlit demo: predict a movie's genre from its plot and optional title."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from movie_genre.config import METRICS_PATH, MODEL_PATH
from movie_genre.model import genre_scores, load_artifact, to_frame

REPO_URL = "https://github.com/raviteja311/MOVIE-GENRE-CLASSIFICATION"

EXAMPLES = {
    "Space mystery": (
        "A mining crew on a distant moon picks up a strange signal and discovers an alien ship buried beneath the ice.",
        "Deep Signal (2019)",
    ),
    "Game show": (
        "Three contestants race against the clock, answering trivia questions to win a cash prize.",
        '"Brain Rush" (2014)',
    ),
    "Detective thriller": (
        "A weary detective hunts a serial killer who leaves riddles at every crime scene in a rain-soaked city.",
        "Riddle Man (1997)",
    ),
    "Nature documentary": (
        "Filmmakers follow beekeepers across three continents to understand why honeybee colonies are collapsing.",
        "The Last Hive (2016)",
    ),
    "Romantic comedy": (
        "Two rival wedding planners are forced to work together for one summer and slowly fall for each other.",
        "Plus One (2008)",
    ),
}


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return load_artifact(MODEL_PATH)


@st.cache_data
def load_test_metrics():
    if not METRICS_PATH.exists():
        return None
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    for row in payload["metrics"]:
        if row["model"] == payload["best_model_name"] and row["split"] == "held-out test":
            return {**row, "top3_accuracy": payload["top3_accuracy"]}
    return None


def apply_example():
    example = st.session_state.example
    if example:
        st.session_state.plot, st.session_state.title = EXAMPLES[example]


st.set_page_config(page_title="Movie genre classifier", page_icon=":material/movie:")
st.session_state.setdefault("plot", "")
st.session_state.setdefault("title", "")

st.title("Movie genre classifier", icon=":material/movie:")
st.caption(
    f"Predicts one of 27 genres from a plot description and an optional title, using TF-IDF features "
    f"and a linear SVM trained on 54k movies. [Source on GitHub]({REPO_URL})"
)

metrics = load_test_metrics()
if metrics:
    with st.container(horizontal=True):
        st.metric("Test accuracy", f"{metrics['accuracy']:.1%}", border=True)
        st.metric("Weighted F1", f"{metrics['weighted_f1']:.1%}", border=True)
        st.metric("Top-3 accuracy", f"{metrics['top3_accuracy']:.1%}", border=True)

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `movie-genre-train` first.", icon=":material/error:")
    st.stop()

st.pills("Try an example", list(EXAMPLES), key="example", on_change=apply_example)

with st.form("predict"):
    st.text_area(
        "Plot description",
        key="plot",
        height=140,
        placeholder="A small-town detective investigates a string of unsettling disappearances.",
    )
    st.text_input(
        "Title (optional)",
        key="title",
        placeholder="Hollow Creek (2015)",
        help="Including the release year in brackets improves accuracy.",
    )
    st.segmented_control("Number of guesses", [1, 3, 5], default=3, required=True, key="top_k")
    submitted = st.form_submit_button("Predict genre", type="primary", icon=":material/auto_awesome:")

model, label_encoder = load_model()

if submitted:
    plot = st.session_state.plot.strip()
    if not plot:
        st.warning("Enter a plot description first.", icon=":material/edit_note:")
    else:
        scores = genre_scores(model, to_frame([plot], [st.session_state.title.strip()]))[0]
        ranking = (
            pd.DataFrame({"genre": label_encoder.classes_, "score": scores})
            .sort_values("score", ascending=False)
            .head(st.session_state.top_k)
        )
        with st.container(border=True):
            st.subheader(f"Predicted genre: {ranking.iloc[0]['genre']}", icon=":material/theaters:")
            if len(ranking) > 1:
                # Dots rather than bars: SVM scores are often all negative, so bar length from zero would mislead.
                chart = (
                    alt.Chart(ranking)
                    .mark_circle(size=160)
                    .encode(
                        x=alt.X("score:Q", title="Score", scale=alt.Scale(zero=False, nice=False, padding=20)),
                        y=alt.Y("genre:N", title="Genre", sort="-x"),
                        tooltip=["genre", alt.Tooltip("score:Q", format=".3f")],
                    )
                )
                st.altair_chart(chart)
            if hasattr(model, "predict_proba"):
                st.caption("Scores are predicted probabilities.")
            else:
                st.caption("Scores are the SVM's decision values: further right means more likely. They are not probabilities.")
