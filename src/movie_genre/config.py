"""Project paths and shared constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "movie_genre_classifier.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

DATA_FILES = ["train_data.txt", "test_data.txt", "test_data_solution.txt"]
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2

# Chosen with `movie-genre-tune` (5-fold cross-validation on the training split).
SVC_C = 0.5
SVC_C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
