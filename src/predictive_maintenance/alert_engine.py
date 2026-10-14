from __future__ import annotations

from predictive_maintenance.risk_scoring import assess_sensor_risk


def assess_reading(reading: dict, prediction: dict | None = None) -> dict:
    risk = assess_sensor_risk(reading)
    predicted_failure = bool(prediction and prediction.get("prediction") == "FAILURE")
    alert = risk["risk_category"] in {"MODERATE", "HIGH", "CRITICAL"} or predicted_failure
    if reading.get("operating_state") == "SHUTDOWN":
        recommendation = "Inspect the simulated shutdown cause and follow the site's restart procedure."
    elif alert:
        recommendation = "Verify the indicated sensor readings and schedule an appropriate maintenance inspection."
    else:
        recommendation = "Continue routine monitoring."
    return {
        "machine_id": str(reading.get("machine_id", "UNSPECIFIED")),
        "timestamp": str(reading.get("timestamp", "")),
        "alert": alert,
        "risk_category": risk["risk_category"],
        "heuristic_risk_score": risk["heuristic_risk_score"],
        "risk_score_is_failure_probability": False,
        "triggering_conditions": risk["triggering_conditions"],
        "model_prediction": prediction,
        "recommended_investigation": recommendation,
        "scenario": reading.get("scenario", "unspecified"),
        "simulated": bool(reading.get("simulation", False)),
    }
