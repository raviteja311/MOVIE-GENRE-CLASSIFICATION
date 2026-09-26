"""Text cleaning shared by training and inference."""

from __future__ import annotations

import re

import nltk
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = None
LEMMATIZER = None

NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}


def ensure_nltk_resources():
    # Only hit the network when a resource is actually missing.
    for name, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.data.find(f"{path}.zip")
            except LookupError:
                nltk.download(name, quiet=True)


def _ensure_resources():
    global STOP_WORDS, LEMMATIZER
    if STOP_WORDS is None:
        STOP_WORDS = set(stopwords.words("english"))
    if LEMMATIZER is None:
        LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text):
    _ensure_resources()
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def _preprocess_chunk(texts):
    return [preprocess_text(text) for text in texts]


def preprocess_many(texts, n_jobs=-1, chunk_size=2000):
    """Clean a list of texts once, in parallel, preserving order."""
    texts = list(texts)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    results = Parallel(n_jobs=n_jobs)(delayed(_preprocess_chunk)(chunk) for chunk in chunks)
    return [text for chunk in results for text in chunk]
