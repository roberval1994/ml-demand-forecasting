"""Central configuration for the demand-forecasting pipeline.

Keeping paths and hyperparameters in one place makes the project easy to
reconfigure and keeps the other modules free of magic constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Project paths --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --- Dataset --------------------------------------------------------------
# UCI Bike Sharing dataset (hourly). Public, no licensing restrictions.
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
)
RAW_CSV_NAME = "hour.csv"          # file inside the downloaded archive
TARGET = "cnt"                     # total rentals per hour

# Columns that leak the target (partial counts) must be dropped.
LEAKAGE_COLUMNS = ("casual", "registered")


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and split settings for training."""

    random_state: int = 42
    test_size: float = 0.2          # fraction held out (chronologically)
    cv_splits: int = 5              # TimeSeriesSplit folds
    n_jobs: int = -1
    # Lag/rolling windows (in hours) used by feature engineering.
    lags: tuple[int, ...] = (1, 2, 3, 24, 168)
    rolling_windows: tuple[int, ...] = (3, 24)
    # Model search space is defined in model.py; kept small for fast runs.
    scoring: str = "neg_root_mean_squared_error"


DEFAULT_CONFIG = TrainConfig()


def ensure_dirs() -> None:
    """Create the data/output directories if they do not exist."""
    for d in (RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
