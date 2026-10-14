from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from predictive_maintenance.config import (
    DATASET_FILENAME,
    DATASET_URL,
    RAW_DATA_DIR,
    SAMPLE_DATA_DIR,
)


def download_dataset(destination: Path | None = None) -> Path:
    """Download the official UCI archive and extract its CSV without altering it."""
    target = destination or RAW_DATA_DIR / DATASET_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(DATASET_URL, timeout=30) as response:
        archive = ZipFile(BytesIO(response.read()))
    member = next(
        (name for name in archive.namelist() if name.lower().endswith("ai4i2020.csv")),
        None,
    )
    if member is None:
        raise ValueError("The UCI archive did not contain ai4i2020.csv.")
    target.write_bytes(archive.read(member))
    return target


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the official dataset or the clearly labelled local demo sample."""
    if path is not None:
        source = Path(path)
    else:
        official = RAW_DATA_DIR / DATASET_FILENAME
        sample = SAMPLE_DATA_DIR / "synthetic_ai4i_demo.csv"
        source = official if official.exists() else sample
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found at {source}. Run `py -3.14 main.py train` "
            "to download the UCI dataset or create an offline synthetic sample."
        )
    return pd.read_csv(source)


def make_synthetic_demo(rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Create explicitly synthetic AI4I-shaped data for offline demonstration only."""
    rng = np.random.default_rng(random_state)
    types = rng.choice(["L", "M", "H"], rows, p=[0.5, 0.3, 0.2])
    air = rng.normal(300.0, 2.0, rows)
    process = air + rng.normal(10.0, 1.0, rows)
    speed = np.clip(rng.normal(1538.0, 180.0, rows), 900, 2500)
    torque = np.clip(rng.normal(40.0, 10.0, rows), 3, 75)
    wear = rng.integers(0, 251, rows)
    delta = process - air
    power = torque * speed * (2 * np.pi / 60)
    twf = (wear >= rng.integers(200, 241, rows)).astype(int)
    hdf = ((delta < 8.6) & (speed < 1380)).astype(int)
    pwf = ((power < 3500) | (power > 9000)).astype(int)
    limits = np.select([types == "L", types == "M"], [11000, 12000], 13000)
    osf = ((wear * torque) > limits).astype(int)
    rnf = (rng.random(rows) < 0.001).astype(int)
    failure = np.maximum.reduce([twf, hdf, pwf, osf, rnf])
    return pd.DataFrame(
        {
            "UDI": np.arange(1, rows + 1),
            "Product ID": [f"{kind}{i:05d}" for i, kind in enumerate(types, 1)],
            "Type": types,
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": speed.round().astype(int),
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
            "Machine failure": failure,
            "TWF": twf,
            "HDF": hdf,
            "PWF": pwf,
            "OSF": osf,
            "RNF": rnf,
        }
    )


def ensure_dataset(allow_download: bool = True) -> tuple[pd.DataFrame, str]:
    """Prefer the official UCI file; fall back to a reproducible synthetic sample."""
    official = RAW_DATA_DIR / DATASET_FILENAME
    if official.exists():
        return pd.read_csv(official), "UCI AI4I 2020 (official downloaded file)"
    try:
        if allow_download:
            return pd.read_csv(download_dataset()), "UCI AI4I 2020 (official download)"
    except Exception as exc:
        download_error = f"{type(exc).__name__}: {exc}"
    else:
        download_error = "download disabled"

    sample_path = SAMPLE_DATA_DIR / "synthetic_ai4i_demo.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if not sample_path.exists():
        make_synthetic_demo().to_csv(sample_path, index=False)
    frame = pd.read_csv(sample_path)
    return frame, f"Synthetic AI4I-shaped offline demo ({download_error})"
