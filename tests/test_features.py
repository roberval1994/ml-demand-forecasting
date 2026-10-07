"""Unit tests for the feature-engineering module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecasting import features  # noqa: E402
from demand_forecasting.config import TrainConfig  # noqa: E402


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """A small, 10-day hourly series with a trivial target."""
    ts = pd.date_range("2021-01-01", periods=24 * 10, freq="h")
    return pd.DataFrame({"timestamp": ts, "cnt": np.arange(len(ts), dtype=float)})


def test_calendar_features_ranges(toy_frame):
    out = features.add_calendar_features(toy_frame)
    assert out["hour"].between(0, 23).all()
    assert out["dayofweek"].between(0, 6).all()
    assert set(out["is_weekend"].unique()).issubset({0, 1})


def test_cyclical_encoding_is_bounded(toy_frame):
    out = features.add_cyclical_features(features.add_calendar_features(toy_frame))
    for col in ("hour_sin", "hour_cos", "month_sin", "month_cos"):
        assert out[col].between(-1.0, 1.0).all()


def test_cyclical_hour_wraps_around(toy_frame):
    """Hour 23 and hour 0 should be adjacent in the sin/cos space."""
    out = features.add_cyclical_features(features.add_calendar_features(toy_frame))
    h0 = out.loc[out["hour"] == 0].iloc[0]
    h23 = out.loc[out["hour"] == 23].iloc[0]
    dist = np.hypot(h0["hour_sin"] - h23["hour_sin"], h0["hour_cos"] - h23["hour_cos"])
    # One hour apart on the unit circle: distance ~= 2*sin(pi/24) ~= 0.26.
    assert dist < 0.3


def test_lag_features_shift_correctly(toy_frame):
    cfg = TrainConfig(lags=(1,), rolling_windows=(3,))
    out = features.add_lag_features(toy_frame, config=cfg)
    # cnt is 0,1,2,...; lag_1 at row i must equal cnt at row i-1.
    assert out["cnt_lag_1"].iloc[1] == toy_frame["cnt"].iloc[0]
    assert pd.isna(out["cnt_lag_1"].iloc[0])


def test_build_features_drops_initial_nans(toy_frame):
    cfg = TrainConfig(lags=(1, 24), rolling_windows=(3,))
    out = features.build_features(toy_frame, config=cfg)
    assert not out.isna().any().any()
    # The largest lag is 24, so we lose at most 24 rows.
    assert len(out) >= len(toy_frame) - 24
