"""Load the ::: delimited train and test files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from movie_genre.config import DATA_DIR


def load_data(data_dir: Path = DATA_DIR):
    """Return (train_data, test_eval_data), where test_eval_data carries the true GENRE."""
    columns = ["ID", "TITLE", "GENRE", "DESCRIPTION"]
    train_data = pd.read_csv(data_dir / "train_data.txt", sep=":::", names=columns, engine="python")
    test_data = pd.read_csv(data_dir / "test_data.txt", sep=":::", names=["ID", "TITLE", "DESCRIPTION"], engine="python")
    test_solution = pd.read_csv(data_dir / "test_data_solution.txt", sep=":::", names=columns, engine="python")

    # The " ::: " separator leaves padding around every field.
    for frame in (train_data, test_data, test_solution):
        for column in frame.columns.drop("ID"):
            frame[column] = frame[column].str.strip()
    train_data["GENRE"] = train_data["GENRE"].str.lower()
    test_solution["GENRE"] = test_solution["GENRE"].str.lower()

    test_eval_data = test_data.merge(test_solution[["ID", "GENRE"]], on="ID", how="inner", validate="one_to_one")
    return train_data, test_eval_data
