"""End-to-end pipeline runner.

Usage:
    python scripts/run_pipeline.py

Downloads the data, engineers features, trains and tunes every candidate model,
evaluates them on a chronological hold-out set, and writes a metrics leaderboard
and diagnostic plots to ``outputs/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``src`` importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecasting import config, data, evaluate, explain, features, model  # noqa: E402


def main() -> None:
    config.ensure_dirs()

    print("[1/5] Loading data ...")
    clean = data.load_clean()

    print("[2/5] Building features ...")
    feat = features.build_features(clean)

    print("[3/5] Splitting (chronological) ...")
    split = model.chronological_split(feat)

    print("[4/5] Training & tuning models ...")
    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, object] = {}
    for name, pipeline in model.build_models().items():
        best = model.tune_model(name, pipeline, split)
        preds = best.predict(split.X_test)
        results[name] = evaluate.regression_metrics(split.y_test, preds)
        fitted[name] = best
        print(f"    - {name}: {results[name]}")

    board = evaluate.leaderboard(results)
    board.to_csv(config.OUTPUTS_DIR / "leaderboard.csv")
    (config.OUTPUTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    best_name = board.index[0]
    best_model = fitted[best_name]
    print(f"[5/5] Best model: {best_name} -> diagnostics & explainability ...")

    preds = best_model.predict(split.X_test)
    evaluate.plot_predictions(
        split.y_test, preds, config.OUTPUTS_DIR / "actual_vs_predicted.png"
    )
    evaluate.plot_residuals(
        split.y_test, preds, config.OUTPUTS_DIR / "residuals.png"
    )

    importances = explain.permutation_importances(best_model, split.X_test, split.y_test)
    importances.to_csv(config.OUTPUTS_DIR / "feature_importances.csv", index=False)
    explain.plot_importances(importances, config.OUTPUTS_DIR / "feature_importances.png")
    explain.try_shap_summary(best_model, split.X_test, config.OUTPUTS_DIR / "shap_summary.png")

    print(f"\nDone. Artifacts written to: {config.OUTPUTS_DIR}")
    print(board.to_string())


if __name__ == "__main__":
    main()
