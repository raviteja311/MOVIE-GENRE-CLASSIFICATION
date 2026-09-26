from movie_genre.preprocessing import ensure_nltk_resources, preprocess_many, preprocess_text

ensure_nltk_resources()


def test_preprocess_text_removes_noise_and_stopwords():
    text = "The 3 DETECTIVES <b>chase</b> a thief! Contact: someone@example.com #crime"
    assert preprocess_text(text) == "detective chase thief contact"


def test_preprocess_text_handles_non_strings():
    assert preprocess_text(None) == ""
    assert preprocess_text(float("nan")) == ""


def test_preprocess_many_preserves_order_across_chunks():
    texts = [f"movie number {word}" for word in ["alpha", "bravo", "charlie", "delta", "echo"]]
    expected = [preprocess_text(text) for text in texts]
    assert preprocess_many(texts, n_jobs=2, chunk_size=2) == expected
