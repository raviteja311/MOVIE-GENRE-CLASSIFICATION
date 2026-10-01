"""Project paths and shared constants."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "movie_genre_classifier.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
TUNING_PATH = PROJECT_ROOT / "config" / "tuning.json"

DATA_FILES = ["train_data.txt", "test_data.txt", "test_data_solution.txt"]
# SHA-256 of each file with LF line endings (a Windows checkout with autocrlf is normalized before hashing).
DATA_SHA256 = {
    "train_data.txt": "b4159255d29287a55b23b74a6328a63a6bb27974eb3602b767d517a384aaef54",
    "test_data.txt": "ff1e5fe712dc6ef7dd00e651db7f234efc0816142ceb0f1cfbb1c43ce8925843",
    "test_data_solution.txt": "b6781457355dab45ac0962da327ee4140d579ec0dd9f0b59492cd1531132e334",
}
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2

DEFAULT_SVC_C = 0.5
SVC_C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]


def load_svc_c(path: Path = TUNING_PATH) -> float:
    """C written by `movie-genre-tune`, or DEFAULT_SVC_C if the file is missing."""
    if not path.exists():
        return DEFAULT_SVC_C
    return float(json.loads(path.read_text(encoding="utf-8"))["svc_c"])


SVC_C = load_svc_c()
