import pytest

from predictive_maintenance.alert_engine import assess_reading
from predictive_maintenance.risk_scoring import assess_sensor_risk
from predictive_maintenance.simulation import SCENARIOS, SensorSimulator


def test_scenarios_cover_expected_states_and_are_seedable():
    first = SensorSimulator(seed=17)
    second = SensorSimulator(seed=17)
    first_reading = first.generate("normal")
    second_reading = second.generate("normal")
    first_reading.pop("timestamp")
    second_reading.pop("timestamp")
    assert first_reading == second_reading
    for scenario in SCENARIOS:
        assert first.generate(scenario)["simulation"] is True


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="Unknown scenario"):
        SensorSimulator().generate("not-a-scenario")


def test_risk_category_mapping_and_shutdown_alert():
    assert assess_sensor_risk({})["risk_category"] == "MODERATE"
    stopped = SensorSimulator().generate("simulated_shutdown")
    alert = assess_reading(stopped)
    assert alert["alert"] is True
    assert alert["risk_category"] == "CRITICAL"
    assert alert["simulated"] is True


def test_invalid_sensor_value_triggers_data_quality_warning():
    reading = SensorSimulator().generate("invalid_reading")
    result = assess_sensor_risk(reading)
    assert "Missing or invalid required sensor reading" in result["triggering_conditions"]
    with pytest.raises(ValueError, match="finite number"):
        from predictive_maintenance.predict import validate_reading
        validate_reading(reading)


def test_model_prediction_can_raise_alert_independently_of_heuristic():
    reading = SensorSimulator().generate("normal")
    result = assess_reading(reading, {"prediction": "FAILURE"})
    assert result["alert"] is True
    assert result["model_prediction"]["prediction"] == "FAILURE"
