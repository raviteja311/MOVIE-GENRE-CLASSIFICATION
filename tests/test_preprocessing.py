from movie_genre.preprocessing import preprocess_text


def test_preprocess_text_removes_noise():
    text = "The 3 DETECTIVES <b>chase</b> a thief! Contact: someone@example.com #crime"
    assert preprocess_text(text) == "the detectives chase thief contact"


def test_preprocess_text_handles_non_strings():
    assert preprocess_text(None) == ""
    assert preprocess_text(float("nan")) == ""
