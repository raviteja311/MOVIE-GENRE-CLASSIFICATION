"""Load the ::: delimited train and test files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from movie_genre.config import DATA_DIR, RANDOM_STATE, VALIDATION_SIZE


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


def clean_training_data(train_data, test_data):
    """Drop training rows that leak into the test set or carry conflicting labels.

    Returns the cleaned frame and a dict with the number of rows removed per reason.
    """
    in_test = train_data["DESCRIPTION"].isin(set(test_data["DESCRIPTION"]))
    conflicting = train_data.groupby("DESCRIPTION")["GENRE"].transform("nunique") > 1
    kept = train_data[~in_test & ~conflicting]
    deduplicated = kept.drop_duplicates("DESCRIPTION").reset_index(drop=True)

    removed = {
        "shared_with_test": int(in_test.sum()),
        "conflicting_labels": int((conflicting & ~in_test).sum()),
        "duplicates": len(kept) - len(deduplicated),
    }
    return deduplicated, removed


def train_validation_split(frame, y):
    """Stratified split shared by training and tuning so both see the same rows."""
    return train_test_split(frame, y, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y)
