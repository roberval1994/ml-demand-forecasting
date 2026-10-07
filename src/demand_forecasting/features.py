"""Feature engineering for hourly demand forecasting.

The functions here are deliberately pure (DataFrame in, DataFrame out) so they
are trivial to unit-test and compose. Three families of features are built:

1. Calendar features extracted from the timestamp.
2. Cyclical (sin/cos) encodings so the model understands that hour 23 is close
   to hour 0 and December is close to January.
3. Lag and rolling-window features that give the model recent history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, TARGET, TrainConfig


def add_calendar_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Extract hour, day-of-week, month and weekend flag from the timestamp."""
    out = df.copy()
    ts = out[ts_col]
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return out


def _encode_cyclical(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """Replace a periodic integer column with its sin/cos projection."""
    out = df.copy()
    radians = 2.0 * np.pi * out[col] / period
    out[f"{col}_sin"] = np.sin(radians)
    out[f"{col}_cos"] = np.cos(radians)
    return out


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encodings for hour, day-of-week and month."""
    out = df
    for col, period in (("hour", 24), ("dayofweek", 7), ("month", 12)):
        if col in out.columns:
            out = _encode_cyclical(out, col, period)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target: str = TARGET,
    config: TrainConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Add lagged and rolling-mean features of the target.

    Rolling windows are shifted by one step so that the feature at time ``t``
    never includes the target at ``t`` (which would leak the answer).
    """
    out = df.copy()
    for lag in config.lags:
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    for window in config.rolling_windows:
        out[f"{target}_rollmean_{window}"] = (
            out[target].shift(1).rolling(window=window).mean()
        )
    return out


def build_features(
    df: pd.DataFrame,
    target: str = TARGET,
    config: TrainConfig = DEFAULT_CONFIG,
    dropna: bool = True,
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline.

    Rows with NaNs introduced by the initial lags are dropped by default so the
    training matrix is dense.
    """
    out = add_calendar_features(df)
    out = add_cyclical_features(out)
    out = add_lag_features(out, target=target, config=config)
    if dropna:
        out = out.dropna().reset_index(drop=True)
    return out
