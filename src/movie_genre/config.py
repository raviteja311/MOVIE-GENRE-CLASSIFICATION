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
