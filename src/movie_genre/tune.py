"""Choose C for the Linear SVC with stratified k-fold cross-validation on the training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from movie_genre.config import RANDOM_STATE, SVC_C_GRID, TUNING_PATH
from movie_genre.data import clean_training_data, load_data, train_validation_split
from movie_genre.model import FEATURE_COLUMNS, build_pipeline


def save_tuning(path: Path, best_c: float, best_score: float, folds: int, grid) -> None:
    """Write the chosen C where config.py picks it up for training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "svc_c": float(best_c),
        "cv_mean_weighted_f1": round(float(best_score), 6),
        "cv_folds": folds,
        "grid": [float(c) for c in grid],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(prog="movie-genre-tune", description=__doc__)
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--grid", type=float, nargs="+", default=SVC_C_GRID, help="C values to try.")
    parser.add_argument("--output", type=Path, default=TUNING_PATH, help="Where to write the best C (default: config/tuning.json).")
    parser.add_argument("--no-save", action="store_true", help="Print the result without writing it.")
    args = parser.parse_args()

    train_data, test_eval_data = load_data()
    train_data, _ = clean_training_data(train_data, test_eval_data)
    y = LabelEncoder().fit_transform(train_data["GENRE"])
    X_train, _, y_train, _ = train_validation_split(train_data[FEATURE_COLUMNS], y)

    search = GridSearchCV(
        build_pipeline(LinearSVC(random_state=RANDOM_STATE, max_iter=5000, class_weight="balanced")),
        {"clf__C": args.grid},
        cv=StratifiedKFold(args.folds, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_weighted",
        n_jobs=-1,
    )
    print(f"Running {args.folds}-fold cross-validation over C={args.grid}...")
    search.fit(X_train, y_train)

    results = pd.DataFrame({
        "C": search.cv_results_["param_clf__C"].astype(float),
        "mean_weighted_f1": search.cv_results_["mean_test_score"],
        "std": search.cv_results_["std_test_score"],
    })
    print(results.round(4).to_string(index=False))
    best_c = search.best_params_["clf__C"]
    print(f"\nBEST_C {best_c}")
    if not args.no_save:
        save_tuning(args.output, best_c, search.best_score_, args.folds, args.grid)
        print(f"Saved {args.output} (used by movie-genre-train)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
