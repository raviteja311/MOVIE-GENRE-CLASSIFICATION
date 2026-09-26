"""Train text classifiers, select on validation, and report held-out test metrics."""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from movie_genre.config import FIGURES_DIR, METRICS_PATH, MODEL_PATH, RANDOM_STATE, SVC_C
from movie_genre.data import clean_training_data, load_data, train_validation_split
from movie_genre.model import FEATURE_COLUMNS, build_pipeline, predict_top_k


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
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_data, test_eval_data = load_data()
    train_data, rows_removed = clean_training_data(train_data, test_eval_data)
    print(f"Removed training rows: {rows_removed}; {len(train_data)} rows left.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_data["GENRE"])
    y_test = label_encoder.transform(test_eval_data["GENRE"])

    X = train_data[FEATURE_COLUMNS]
    X_test = test_eval_data[FEATURE_COLUMNS]
    X_train, X_val, y_train, y_val = train_validation_split(X, y)

    models = {
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")),
        "Complement NB": build_pipeline(ComplementNB()),
        "Linear SVC": build_pipeline(LinearSVC(C=SVC_C, random_state=RANDOM_STATE, max_iter=5000, class_weight="balanced")),
    }

    metrics = []
    test_predictions = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        test_predictions[name] = model.predict(X_test)
        metrics.append({"model": name, "split": "validation", **metric_block(y_val, model.predict(X_val))})
        metrics.append({"model": name, "split": "held-out test", **metric_block(y_test, test_predictions[name])})

    # Select on validation only; the test set is scored but never used to choose.
    metrics_df = pd.DataFrame(metrics)
    best_row = metrics_df[metrics_df["split"] == "validation"].sort_values("weighted_f1", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_model = models[best_model_name]
    best_test_pred = test_predictions[best_model_name]

    # Predictions without a title, as happens when predict is called with only a plot.
    no_title_pred = best_model.predict(X_test.assign(TITLE=""))
    metrics_df = pd.concat([metrics_df, pd.DataFrame([
        {"model": best_model_name, "split": "held-out test (no title)", **metric_block(y_test, no_title_pred)},
    ])], ignore_index=True)

    class_names = label_encoder.classes_
    print(f"\nHeld-out test classification report ({best_model_name}):\n")
    print(classification_report(y_test, best_test_pred, target_names=class_names, zero_division=0))
    save_confusion_matrix(
        y_test, best_test_pred, class_names,
        f"Row-normalized confusion matrix: {best_model_name}",
        FIGURES_DIR / "confusion_matrix.png",
    )

    top_3 = predict_top_k(best_model, label_encoder, X_test, k=min(3, len(class_names)))
    top3_accuracy = float(np.mean([true in row for true, row in zip(label_encoder.inverse_transform(y_test), top_3)]))
    baseline = float(pd.Series(y_test).value_counts(normalize=True).max())

    final_model = best_model
    final_model.fit(X, y)
    artifact_path = MODEL_PATH
    joblib.dump({"model": final_model, "label_encoder": label_encoder, "best_model_name": best_model_name}, artifact_path, compress=3)

    metrics_path = METRICS_PATH
    metrics_payload = {
        "best_model_name": best_model_name,
        "selection": "highest validation weighted F1",
        "training_rows_removed": rows_removed,
        "majority_class_baseline": baseline,
        "top3_accuracy": top3_accuracy,
        "metrics": metrics_df.round(6).to_dict(orient="records"),
        "best_model_test_report": classification_report(
            y_test, best_test_pred, target_names=class_names, output_dict=True, zero_division=0,
        ),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    sample_rows = test_eval_data.head(2)
    sample_top_3 = predict_top_k(final_model, label_encoder, sample_rows[FEATURE_COLUMNS], k=3)
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
