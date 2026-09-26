import pandas as pd

from movie_genre.data import clean_training_data, load_data


def write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_data_strips_fields_and_merges_labels(tmp_path):
    write(tmp_path / "train_data.txt", [
        "1 ::: Movie A (2000) ::: Drama ::: A sad story.",
        "2 ::: Movie B (2001) ::: comedy ::: A funny story.",
    ])
    write(tmp_path / "test_data.txt", [
        "1 ::: Movie C (2002) ::: A scary story.",
    ])
    write(tmp_path / "test_data_solution.txt", [
        "1 ::: Movie C (2002) ::: Horror ::: A scary story.",
    ])

    train_data, test_eval_data = load_data(tmp_path)

    assert train_data["GENRE"].tolist() == ["drama", "comedy"]
    assert train_data["TITLE"].iloc[0] == "Movie A (2000)"
    assert train_data["DESCRIPTION"].iloc[0] == "A sad story."
    assert test_eval_data[["ID", "TITLE", "GENRE"]].to_dict("records") == [
        {"ID": 1, "TITLE": "Movie C (2002)", "GENRE": "horror"},
    ]


def test_clean_training_data_drops_leaks_conflicts_and_duplicates():
    train = pd.DataFrame({
        "DESCRIPTION": ["seen in test", "kept once", "kept once", "label clash", "label clash", "unique"],
        "GENRE": ["drama", "comedy", "comedy", "drama", "horror", "action"],
    })
    test = pd.DataFrame({"DESCRIPTION": ["seen in test", "other"]})

    cleaned, removed = clean_training_data(train, test)

    assert cleaned["DESCRIPTION"].tolist() == ["kept once", "unique"]
    assert removed == {"shared_with_test": 1, "conflicting_labels": 2, "duplicates": 1}
