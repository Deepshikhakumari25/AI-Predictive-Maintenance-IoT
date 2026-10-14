import pytest

from predictive_maintenance.data_loader import make_synthetic_demo
from predictive_maintenance.validation import clean_dataset, validate_dataset


def test_valid_ai4i_schema_is_accepted():
    validate_dataset(make_synthetic_demo(rows=30))


def test_invalid_schema_is_rejected():
    with pytest.raises(ValueError, match="missing required AI4I columns"):
        validate_dataset(make_synthetic_demo(rows=30).drop(columns=["Torque [Nm]"]))


def test_invalid_target_is_rejected():
    frame = make_synthetic_demo(rows=30)
    frame.loc[0, "Machine failure"] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_dataset(frame)


def test_duplicate_row_identifier_is_rejected():
    frame = make_synthetic_demo(rows=30)
    frame.loc[1, "UDI"] = frame.loc[0, "UDI"]
    with pytest.raises(ValueError, match="unique numeric row identifiers"):
        validate_dataset(frame)


def test_malformed_numeric_sensor_value_is_rejected():
    frame = make_synthetic_demo(rows=30)
    frame["Torque [Nm]"] = frame["Torque [Nm]"].astype(object)
    frame.loc[0, "Torque [Nm]"] = "invalid"
    with pytest.raises(ValueError, match="non-numeric values"):
        validate_dataset(frame)


def test_sensor_nulls_are_left_for_train_fitted_imputer():
    frame = make_synthetic_demo(rows=30)
    frame.loc[0, "Torque [Nm]"] = float("nan")
    cleaned = clean_dataset(frame)
    assert len(cleaned) == 30
    assert cleaned["Torque [Nm]"].isna().sum() == 1
