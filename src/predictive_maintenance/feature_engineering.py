from __future__ import annotations

import numpy as np
import pandas as pd

from predictive_maintenance.config import FEATURE_COLUMNS


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep known-at-reading inputs and add physically interpretable derived features."""
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing sensor features: {missing}")
    result = frame.loc[:, FEATURE_COLUMNS].copy()
    result["Temperature difference [K]"] = (
        result["Process temperature [K]"] - result["Air temperature [K]"]
    )
    result["Estimated mechanical power [W]"] = (
        result["Torque [Nm]"]
        * result["Rotational speed [rpm]"]
        * (2 * np.pi / 60)
    )
    return result
