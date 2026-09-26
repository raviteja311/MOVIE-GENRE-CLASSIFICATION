"""Text cleaning shared by training and inference."""

from __future__ import annotations

import re

# Applied in order. Stopwords are removed later by the TF-IDF vectorizer.
_CLEANING_PATTERNS = [
    re.compile(r"\S+@\S+"),        # emails
    re.compile(r"[@#]\w+"),        # mentions and hashtags
    re.compile(r"<.*?>"),          # html tags
    re.compile(r"\d+"),            # numbers
    re.compile(r"[^\w\s]"),        # punctuation
    re.compile(r"\b[a-zA-Z]\b"),   # single letters
]
_WHITESPACE = re.compile(r"\s+")


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for pattern in _CLEANING_PATTERNS:
        text = pattern.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()
