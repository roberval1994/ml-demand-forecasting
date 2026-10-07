"""Data acquisition and loading.

Downloads the UCI Bike Sharing dataset (if not already present) and loads the
hourly file into a tidy, well-typed :class:`pandas.DataFrame`.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from . import config


def download_raw(url: str = config.DATASET_URL, raw_dir: Path | None = None) -> Path:
    """Download and extract the dataset archive into ``raw_dir``.

    Returns the path to the extracted hourly CSV. If the file already exists,
    the download is skipped so the pipeline stays idempotent and offline-friendly.
    """
    raw_dir = raw_dir or config.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / config.RAW_CSV_NAME

    if target.exists():
        return target

    with urlopen(url) as response:  # noqa: S310 - trusted UCI URL
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        with archive.open(config.RAW_CSV_NAME) as src, open(target, "wb") as dst:
            dst.write(src.read())

    return target


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the hourly CSV into a DataFrame with a proper datetime index.

    The dataset ships calendar fields split across several columns; here we
    reconstruct a single timezone-naive timestamp and sort chronologically,
    which every downstream time-series step relies on.
    """
    path = path or (config.RAW_DIR / config.RAW_CSV_NAME)
    if not path.exists():
        path = download_raw()

    df = pd.read_csv(path)

    # ``dteday`` is the calendar day; ``hr`` is the hour of day (0-23).
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_clean(path: Path | None = None) -> pd.DataFrame:
    """Return a cleaned frame ready for feature engineering.

    Drops identifier and leakage columns (``casual``/``registered`` sum to the
    target) and keeps the reconstructed timestamp as the time key.
    """
    df = load_raw(path)
    drop = ["instant", "dteday", *config.LEAKAGE_COLUMNS]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    return df
