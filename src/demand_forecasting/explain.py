"""Model explainability.

Provides permutation importance (always available) and an optional SHAP summary
when the ``shap`` package is installed. Interpreting *why* a model predicts what
it does is as important as the score itself.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402


def permutation_importances(
    model, X, y, random_state: int = 42, n_repeats: int = 10
) -> pd.DataFrame:
    """Return permutation importances as a sorted DataFrame."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    frame = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_importances(importances: pd.DataFrame, out_path: Path, top_n: int = 15) -> Path:
    """Horizontal bar chart of the top-N most important features."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("Permutation importance (increase in error)")
    ax.set_title(f"Top {top_n} features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def try_shap_summary(model, X, out_path: Path) -> Path | None:
    """Produce a SHAP summary plot if ``shap`` is available, else return None."""
    try:
        import shap  # noqa: WPS433 - optional dependency
    except ImportError:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    estimator = model.named_steps.get("model", model)
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path
