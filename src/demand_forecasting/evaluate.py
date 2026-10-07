"""Evaluation metrics and diagnostic plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for CI / scripts
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE, RMSE and R² as a plain dictionary."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


def leaderboard(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn a {model: metrics} mapping into a sorted leaderboard (best RMSE first)."""
    frame = pd.DataFrame(results).T
    return frame.sort_values("RMSE")


def plot_predictions(
    y_true, y_pred, out_path: Path, title: str = "Actual vs. Predicted"
) -> Path:
    """Scatter of predicted against actual values with the identity line."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.3)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_residuals(y_true, y_pred, out_path: Path) -> Path:
    """Histogram of residuals to inspect bias and spread."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=50)
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
