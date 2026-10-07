"""Data acquisition and loading.

Downloads the UCI Bike Sharing dataset (if not already present) and loads the
hourly file into a tidy, well-typed :class:`pandas.DataFrame`.
"""
from __future__ import annotations

import io
import ssl
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from . import config


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context backed by a trust store that works everywhere.

    Certificate verification can fail for two unrelated reasons here:

    * the UCI host sometimes serves an expired/incomplete chain, and
    * on machines where antivirus/proxy software inspects HTTPS traffic
      (e.g. Avast), the served certificate is re-signed by a local CA that
      lives only in the operating-system trust store.

    We therefore prefer ``truststore`` (which validates against the OS store,
    covering the antivirus case), then fall back to ``certifi``'s up-to-date
    bundle, and finally to Python's default context. Verification is never
    disabled.
    """
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch(url: str) -> bytes:
    """Download the raw bytes at ``url`` using a verified SSL context."""
    with urlopen(url, context=_ssl_context()) as response:  # noqa: S310
        return response.read()


def download_raw(url: str | None = None, raw_dir: Path | None = None) -> Path:
    """Download and extract the dataset archive into ``raw_dir``.

    Returns the path to the extracted hourly CSV. If the file already exists,
    the download is skipped so the pipeline stays idempotent and offline-friendly.

    The primary (UCI) and any configured mirror URLs are tried in order. If all
    fail — e.g. the UCI certificate has expired and no mirror is reachable — a
    clear error explains how to place the file manually.
    """
    raw_dir = raw_dir or config.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / config.RAW_CSV_NAME

    if target.exists():
        return target

    candidates = [url] if url else list(config.DATASET_URLS)
    errors: list[str] = []
    payload: bytes | None = None
    for candidate in candidates:
        try:
            payload = _fetch(candidate)
            break
        except (URLError, ssl.SSLError) as exc:  # pragma: no cover - network
            errors.append(f"  - {candidate}\n      {exc}")

    if payload is None:
        joined = "\n".join(errors)
        raise RuntimeError(
            "Could not download the dataset from any source:\n"
            f"{joined}\n\n"
            "This is usually a transient problem on the UCI server (an expired "
            "TLS certificate). To proceed manually:\n"
            f"  1. Download the archive from: {config.DATASET_URLS[0]}\n"
            f"  2. Extract '{config.RAW_CSV_NAME}' into: {raw_dir}\n"
            "Then re-run the pipeline."
        )

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
