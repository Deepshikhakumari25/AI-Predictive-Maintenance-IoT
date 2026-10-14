from predictive_maintenance.data_loader import make_synthetic_demo
from predictive_maintenance.feature_engineering import engineer_features
from predictive_maintenance.preprocessing import build_preprocessor
from predictive_maintenance.validation import clean_dataset


def test_engineered_features_are_known_at_reading_time():
    frame = clean_dataset(make_synthetic_demo(rows=100))
    features = engineer_features(frame)
    assert "Temperature difference [K]" in features
    assert "Estimated mechanical power [W]" in features
    assert "Machine failure" not in features
    assert "TWF" not in features


def test_preprocessor_fits_and_transforms_sensor_features():
    frame = clean_dataset(make_synthetic_demo(rows=120))
    features = engineer_features(frame)
    transformer = build_preprocessor()
    transformed = transformer.fit_transform(features.iloc[:90])
    assert transformed.shape[0] == 90
    assert transformer.transform(features.iloc[90:]).shape[0] == 30
