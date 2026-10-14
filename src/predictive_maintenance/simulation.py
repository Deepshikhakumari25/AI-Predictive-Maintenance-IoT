from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

SCENARIOS = [
    "normal",
    "gradual_temperature_rise",
    "abnormal_vibration",
    "current_fluctuation",
    "multiple_abnormal_readings",
    "invalid_reading",
    "simulated_shutdown",
]


class SensorSimulator:
    """Seedable virtual sensor generator; every value is simulated, not measured."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.sequence = 0

    def generate(self, scenario: str = "normal", machine_id: str = "SIM-001") -> dict:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}.")
        self.sequence += 1
        air = float(self.rng.normal(300.0, 0.8))
        process = air + float(self.rng.normal(10.0, 0.5))
        speed = float(self.rng.normal(1538.0, 55.0))
        torque = float(self.rng.normal(40.0, 4.0))
        wear = float(self.rng.uniform(15.0, 110.0))
        vibration = float(max(0.3, self.rng.normal(2.2, 0.4)))
        current = float(max(0.1, self.rng.normal(8.0, 0.5)))
        state = "RUNNING"

        if scenario == "gradual_temperature_rise":
            process += 5.0 + min(self.sequence, 20) * 0.35
        elif scenario == "abnormal_vibration":
            vibration = float(self.rng.normal(9.0, 0.5))
        elif scenario == "current_fluctuation":
            current = float(self.rng.choice([2.0, 15.0]) + self.rng.normal(0, 0.3))
            torque = float(self.rng.normal(62.0, 4.0))
        elif scenario == "multiple_abnormal_readings":
            air = float(self.rng.normal(301.0, 0.3))
            process = air + float(self.rng.uniform(6.5, 8.2))
            speed = float(self.rng.normal(1280.0, 30.0))
            torque = float(self.rng.normal(66.0, 2.0))
            wear = float(self.rng.uniform(215.0, 245.0))
            vibration = float(self.rng.normal(10.0, 0.4))
            current = float(self.rng.normal(13.5, 0.4))
        elif scenario == "invalid_reading":
            process = float("nan")
        elif scenario == "simulated_shutdown":
            state = "SHUTDOWN"
            speed = 0.0
            current = 0.0

        return {
            "machine_id": machine_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "simulation": True,
            "scenario": scenario,
            "operating_state": state,
            "Type": str(self.rng.choice(["L", "M", "H"], p=[0.5, 0.3, 0.2])),
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
            "Vibration [mm/s]": vibration,
            "Current [A]": current,
        }
