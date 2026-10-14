from __future__ import annotations

import numpy as np
import pandas as pd

from predictive_maintenance.config import REQUIRED_DATASET_COLUMNS, TARGET_COLUMN


def validate_dataset(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_DATASET_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required AI4I columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset contains no rows.")
    identifiers = pd.to_numeric(frame["UDI"], errors="coerce")
    if identifiers.isna().any() or identifiers.duplicated().any():
        raise ValueError("UDI must contain unique numeric row identifiers.")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing values.")
    target_values = set(frame[TARGET_COLUMN].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN!r} must contain only 0 and 1; got {target_values}.")
    for name in (
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ):
        numeric = pd.to_numeric(frame[name], errors="coerce")
        malformed = frame[name].notna() & numeric.isna()
        if malformed.any():
            raise ValueError(f"Column {name!r} contains non-numeric values.")
        values = numeric.dropna().to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Column {name!r} contains infinite values.")


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate labels and remove rows with unusable sensor inputs; imputers fit later."""
    validate_dataset(frame)
    cleaned = frame.copy()
    cleaned = cleaned.dropna(subset=["Type", TARGET_COLUMN])
    return cleaned.sort_values("UDI", kind="stable").reset_index(drop=True)
