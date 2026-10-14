import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from predictive_maintenance.data_loader import make_synthetic_demo
from predictive_maintenance.feature_engineering import engineer_features
from predictive_maintenance.predict import predict_reading, validate_reading
from predictive_maintenance.preprocessing import build_preprocessor


def sample_reading():
    return {
        "machine_id": "TEST-001",
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 90,
    }


def test_reading_requires_all_expected_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        validate_reading({"Type": "M"})


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), "not-a-number"])
def test_non_finite_or_non_numeric_input_is_rejected(bad_value):
    reading = sample_reading()
    reading["Torque [Nm]"] = bad_value
    with pytest.raises(ValueError, match="finite number"):
        validate_reading(reading)


def test_prediction_returns_structured_result(tmp_path):
    training = make_synthetic_demo(rows=800)
    features = engineer_features(training)
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    model.fit(features, training["Machine failure"])
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    result = predict_reading(sample_reading(), model_path)
    assert result["machine_id"] == "TEST-001"
    assert result["prediction"] in {"NORMAL", "FAILURE"}
    assert 0 <= result["model_failure_probability"] <= 1
    assert result["model_probability_is_calibrated"] is False


def test_missing_model_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        predict_reading(sample_reading(), tmp_path / "missing.joblib")
