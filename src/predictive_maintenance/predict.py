from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from predictive_maintenance.config import FEATURE_COLUMNS, MODEL_DIR
from predictive_maintenance.feature_engineering import engineer_features


def validate_reading(reading: dict) -> pd.DataFrame:
    missing = sorted(set(FEATURE_COLUMNS) - set(reading))
    if missing:
        raise ValueError(f"Reading is missing required fields: {missing}")
    row = pd.DataFrame([{name: reading[name] for name in FEATURE_COLUMNS}])
    if row["Type"].iloc[0] not in {"L", "M", "H"}:
        raise ValueError("Type must be one of 'L', 'M', or 'H'.")
    for column in FEATURE_COLUMNS[1:]:
        row[column] = pd.to_numeric(row[column], errors="coerce")
        value = row[column].iloc[0]
        if pd.isna(value) or not float("-inf") < float(value) < float("inf"):
            raise ValueError(f"{column} must be a finite number.")
    if (row["Rotational speed [rpm]"] <= 0).any() or (row["Torque [Nm]"] < 0).any():
        raise ValueError("Rotational speed must be positive and torque cannot be negative.")
    if (row["Tool wear [min]"] < 0).any():
        raise ValueError("Tool wear cannot be negative.")
    return row


def predict_reading(
    reading: dict, model_path: str | Path | None = None
) -> dict:
    row = validate_reading(reading)
    path = Path(model_path) if model_path else MODEL_DIR / "failure_classifier.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}. Run `py -3.14 main.py train` first.")
    model = joblib.load(path)
    features = engineer_features(row)
    label = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, 1]) if hasattr(model, "predict_proba") else None
    return {
        "machine_id": str(reading.get("machine_id", "UNSPECIFIED")),
        "timestamp": str(reading.get("timestamp", "")),
        "prediction": "FAILURE" if label else "NORMAL",
        "model_failure_probability": probability,
        "model_probability_is_calibrated": False,
    }
