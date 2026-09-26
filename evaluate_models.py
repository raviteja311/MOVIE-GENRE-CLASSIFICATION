#!/usr/bin/env python
"""Train text classifiers, select on validation, and report held-out test metrics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from predict import predict_top_k
from text_utils import ensure_nltk_resources, preprocess_many, preprocess_text

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def load_data(root: Path = ROOT):
    columns = ["ID", "TITLE", "GENRE", "DESCRIPTION"]
    train_data = pd.read_csv(root / "train_data.txt", sep=":::", names=columns, engine="python")
    test_data = pd.read_csv(root / "test_data.txt", sep=":::", names=["ID", "TITLE", "DESCRIPTION"], engine="python")
    test_solution = pd.read_csv(root / "test_data_solution.txt", sep=":::", names=columns, engine="python")

    # The " ::: " separator leaves padding around every field.
    for frame in (train_data, test_data, test_solution):
        for column in frame.columns.drop("ID"):
            frame[column] = frame[column].str.strip()
    train_data["GENRE"] = train_data["GENRE"].str.lower()
    test_solution["GENRE"] = test_solution["GENRE"].str.lower()

    test_eval_data = test_data.merge(test_solution[["ID", "GENRE"]], on="ID", how="inner", validate="one_to_one")
    return train_data, test_eval_data


def build_pipeline(classifier):
    # Inputs are already cleaned by preprocess_many, so no preprocessor here.
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100_000,
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("clf", classifier),
    ])


def metric_block(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
    }


def save_confusion_matrix(y_true, y_pred, class_names, title, path):
    cm_normalized = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted genre")
    plt.ylabel("Actual genre")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> int:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    ensure_nltk_resources()

    train_data, test_eval_data = load_data()

    print("Cleaning text (once, in parallel)...")
    train_clean = pd.Series(preprocess_many(train_data["DESCRIPTION"].fillna("")), index=train_data.index)
    X_test_text = pd.Series(preprocess_many(test_eval_data["DESCRIPTION"].fillna("")), index=test_eval_data.index)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_data["GENRE"])
    y_test = label_encoder.transform(test_eval_data["GENRE"])

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        train_clean,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        "Complement NB": build_pipeline(ComplementNB()),
        "Linear SVC": build_pipeline(LinearSVC(C=0.3, random_state=42, max_iter=5000, class_weight="balanced")),
    }

    metrics = []
    test_predictions = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_text, y_train)
        test_predictions[name] = model.predict(X_test_text)
        metrics.append({"model": name, "split": "validation", **metric_block(y_val, model.predict(X_val_text))})
        metrics.append({"model": name, "split": "held-out test", **metric_block(y_test, test_predictions[name])})

    # Select on validation only; the test set is scored but never used to choose.
    metrics_df = pd.DataFrame(metrics)
    best_row = metrics_df[metrics_df["split"] == "validation"].sort_values("weighted_f1", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_model = models[best_model_name]
    best_test_pred = test_predictions[best_model_name]

    class_names = label_encoder.classes_
    print(f"\nHeld-out test classification report ({best_model_name}):\n")
    print(classification_report(y_test, best_test_pred, target_names=class_names, zero_division=0))
    save_confusion_matrix(
        y_test, best_test_pred, class_names,
        f"Row-normalized confusion matrix: {best_model_name}",
        PLOTS_DIR / "confusion_matrix.png",
    )

    top_3 = predict_top_k(best_model, label_encoder, X_test_text, k=min(3, len(class_names)))
    top3_accuracy = float(np.mean([true in row for true, row in zip(label_encoder.inverse_transform(y_test), top_3)]))
    baseline = float(pd.Series(y_test).value_counts(normalize=True).max())

    final_model = best_model
    final_model.fit(train_clean, y)
    # The model was fit on cleaned text; attaching the cleaner lets it accept raw plots.
    # preprocess_text lowercases, so this matches the default preprocessor used in fit.
    final_model.set_params(tfidf__preprocessor=preprocess_text)
    sample = slice(0, 200)
    raw_pred = final_model.predict(test_eval_data["DESCRIPTION"].iloc[sample])
    final_model.set_params(tfidf__preprocessor=None)
    clean_pred = final_model.predict(X_test_text.iloc[sample])
    final_model.set_params(tfidf__preprocessor=preprocess_text)
    assert (raw_pred == clean_pred).all(), "Raw-text and cleaned-text predictions disagree."

    artifact_path = ARTIFACTS_DIR / "movie_genre_classifier.joblib"
    joblib.dump({"model": final_model, "label_encoder": label_encoder, "best_model_name": best_model_name}, artifact_path, compress=3)

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metrics_payload = {
        "best_model_name": best_model_name,
        "selection": "highest validation weighted F1",
        "majority_class_baseline": baseline,
        "top3_accuracy": top3_accuracy,
        "metrics": metrics_df.round(6).to_dict(orient="records"),
        "best_model_test_report": classification_report(
            y_test, best_test_pred, target_names=class_names, output_dict=True, zero_division=0,
        ),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    sample_rows = test_eval_data.head(2)
    sample_top_3 = predict_top_k(final_model, label_encoder, sample_rows["DESCRIPTION"], k=3)
    print("Sample predictions (final model):")
    for (_, row), labels in zip(sample_rows.iterrows(), sample_top_3):
        print(f"- {row['TITLE']}: actual={row['GENRE']}, top-3={list(labels)}")

    print()
    print(metrics_df.sort_values(["split", "weighted_f1"], ascending=[True, False]).to_string(index=False))
    print(f"\nBEST_MODEL {best_model_name} (selected on validation weighted F1)")
    print(f"TOP3_ACCURACY {top3_accuracy:.6f}")
    print(f"BASELINE_TEST {baseline:.6f}")
    print(f"ARTIFACT {artifact_path}")
    print(f"METRICS {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
