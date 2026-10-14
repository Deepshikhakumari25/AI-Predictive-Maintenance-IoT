from __future__ import annotations

import math


def assess_sensor_risk(reading: dict) -> dict:
    """Apply transparent sensor heuristics; the score is not a failure probability."""
    reasons: list[str] = []
    score = 0
    if reading.get("operating_state") == "SHUTDOWN":
        score = 100
        reasons.append("Simulated machine shutdown")
    else:
        sensor_values = [
            reading.get(name) for name in [
                "Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
            ]
        ]
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in sensor_values
        ):
            score += 30
            reasons.append("Missing or invalid required sensor reading")
        air = reading.get("Air temperature [K]")
        process = reading.get("Process temperature [K]")
        speed = reading.get("Rotational speed [rpm]")
        torque = reading.get("Torque [Nm]")
        wear = reading.get("Tool wear [min]")
        vibration = reading.get("Vibration [mm/s]")
        current = reading.get("Current [A]")

        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in (air, process)):
            if process - air < 8.6:
                score += 25
                reasons.append("Temperature difference below 8.6 K")
        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in (speed, torque)):
            power = torque * speed * (2 * 3.141592653589793 / 60)
            if power < 3500 or power > 9000:
                score += 30
                reasons.append(f"Estimated mechanical power outside 3,500-9,000 W ({power:.0f} W)")
        if isinstance(wear, (int, float)) and math.isfinite(wear):
            if wear >= 200:
                score += 25
                reasons.append("Tool wear at or above 200 min")
            elif wear >= 150:
                score += 15
                reasons.append("Tool wear at or above 150 min")
        if isinstance(vibration, (int, float)) and math.isfinite(vibration):
            if vibration >= 8:
                score += 30
                reasons.append("Simulated vibration at or above 8 mm/s")
            elif vibration >= 6:
                score += 15
                reasons.append("Simulated vibration at or above 6 mm/s")
        if isinstance(current, (int, float)) and math.isfinite(current) and (current < 3 or current > 13):
            score += 15
            reasons.append("Simulated current outside 3-13 A")

    score = min(score, 100)
    category = (
        "CRITICAL" if score >= 80 else
        "HIGH" if score >= 55 else
        "MODERATE" if score >= 30 else
        "LOW"
    )
    return {
        "heuristic_risk_score": score,
        "risk_category": category,
        "risk_score_is_failure_probability": False,
        "triggering_conditions": reasons or ["No configured sensor threshold was triggered"],
    }
