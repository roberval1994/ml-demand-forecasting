"""Model definition, chronological splitting, and hyperparameter tuning.

Everything is wrapped in scikit-learn ``Pipeline`` objects so preprocessing is
fitted only on training folds, and validation uses ``TimeSeriesSplit`` to keep
the temporal order intact (no peeking into the future).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .config import DEFAULT_CONFIG, TARGET, TrainConfig

# XGBoost is an OPTIONAL dependency. The whole pipeline runs without it; when it
# is installed, XGBoost automatically joins the model comparison. This keeps the
# project lightweight for casual users while rewarding those who install extras.
try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_XGBOOST = False


def xgboost_available() -> bool:
    """Return True if the optional ``xgboost`` package is importable."""
    return HAS_XGBOOST


@dataclass
class SplitData:
    """Container for a chronological train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def chronological_split(
    df: pd.DataFrame,
    target: str = TARGET,
    config: TrainConfig = DEFAULT_CONFIG,
) -> SplitData:
    """Split by time: the last ``test_size`` fraction becomes the test set.

    Random shuffling would leak future information into training, so we cut the
    series at a fixed point in time instead.
    """
    feature_cols = [c for c in df.columns if c not in (target, "timestamp")]
    n_test = int(len(df) * config.test_size)
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    return SplitData(
        X_train=train[feature_cols],
        X_test=test[feature_cols],
        y_train=train[target],
        y_test=test[target],
    )


def build_models(
    config: TrainConfig = DEFAULT_CONFIG,
    include_xgboost: bool | None = None,
) -> dict[str, Pipeline]:
    """Return the candidate models, each as a scikit-learn Pipeline.

    When ``include_xgboost`` is ``None`` (default), XGBoost is added only if the
    package is installed. Pass ``True`` to require it explicitly (raising if it
    is missing) or ``False`` to force it off.
    """
    models: dict[str, Pipeline] = {
        "baseline_mean": Pipeline(
            [("model", DummyRegressor(strategy="mean"))]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        random_state=config.random_state,
                        n_jobs=config.n_jobs,
                    ),
                )
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                (
                    "model",
                    GradientBoostingRegressor(random_state=config.random_state),
                )
            ]
        ),
    }

    want_xgb = HAS_XGBOOST if include_xgboost is None else include_xgboost
    if want_xgb:
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost was requested but is not installed. "
                "Install it with: pip install xgboost"
            )
        models["xgboost"] = Pipeline(
            [
                (
                    "model",
                    XGBRegressor(
                        random_state=config.random_state,
                        n_jobs=config.n_jobs,
                        objective="reg:squarederror",
                        tree_method="hist",
                    ),
                )
            ]
        )

    return models


# Small search spaces keep the demo fast while still showing the mechanics.
SEARCH_SPACES: dict[str, dict] = {
    "random_forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 16],
        "model__min_samples_leaf": [1, 2],
    },
    "gradient_boosting": {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    },
    # Used only when XGBoost is installed.
    "xgboost": {
        "model__n_estimators": [300, 600],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [4, 6],
        "model__subsample": [0.8, 1.0],
    },
}


def tune_model(
    name: str,
    pipeline: Pipeline,
    data: SplitData,
    config: TrainConfig = DEFAULT_CONFIG,
) -> Pipeline:
    """Grid-search a model with time-series cross-validation.

    Models without a defined search space (e.g. the baseline) are simply fitted.
    Returns the best fitted estimator.
    """
    space = SEARCH_SPACES.get(name)
    if not space:
        pipeline.fit(data.X_train, data.y_train)
        return pipeline

    cv = TimeSeriesSplit(n_splits=config.cv_splits)
    search = GridSearchCV(
        pipeline,
        param_grid=space,
        scoring=config.scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        refit=True,
    )
    search.fit(data.X_train, data.y_train)
    return search.best_estimator_
