"""Unit tests for the data module (offline, no network required)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecasting import config, data  # noqa: E402


def _write_fake_hour_csv(path: Path) -> None:
    """Create a minimal CSV mimicking the UCI ``hour.csv`` schema."""
    frame = pd.DataFrame(
        {
            "instant": [1, 2, 3],
            "dteday": ["2011-01-01", "2011-01-01", "2011-01-01"],
            "hr": [0, 1, 2],
            "temp": [0.24, 0.22, 0.22],
            "casual": [3, 8, 5],
            "registered": [13, 32, 27],
            "cnt": [16, 40, 32],
        }
    )
    frame.to_csv(path, index=False)


def test_load_raw_builds_timestamp(tmp_path, monkeypatch):
    fake = tmp_path / config.RAW_CSV_NAME
    _write_fake_hour_csv(fake)

    df = data.load_raw(fake)
    assert "timestamp" in df.columns
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].iloc[1].hour == 1


def test_load_clean_drops_leakage_columns(tmp_path):
    fake = tmp_path / config.RAW_CSV_NAME
    _write_fake_hour_csv(fake)

    df = data.load_clean(fake)
    for leaked in ("casual", "registered", "instant", "dteday"):
        assert leaked not in df.columns
    assert "cnt" in df.columns  # target preserved
