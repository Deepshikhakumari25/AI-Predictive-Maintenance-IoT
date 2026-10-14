from __future__ import annotations

import pandas as pd
import plotly.express as px


def sensor_trend_figure(history: pd.DataFrame):
    available = [
        name for name in [
            "Air temperature [K]", "Process temperature [K]",
            "Vibration [mm/s]", "Current [A]",
        ] if name in history.columns
    ]
    if history.empty or not available:
        return None
    long_frame = history.melt(
        id_vars=["timestamp"], value_vars=available,
        var_name="simulated sensor", value_name="reading",
    )
    return px.line(
        long_frame, x="timestamp", y="reading", color="simulated sensor",
        markers=True, title="Virtual sensor readings (simulated)",
    )
