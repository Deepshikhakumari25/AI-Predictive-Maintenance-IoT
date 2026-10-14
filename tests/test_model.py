import json

from predictive_maintenance.data_loader import make_synthetic_demo
from predictive_maintenance.train import train_model


def test_training_compares_and_saves_models_offline(tmp_path, monkeypatch):
    import predictive_maintenance.train as training

    monkeypatch.setattr(training, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(training, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(training, "FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(training, "PROCESSED_DATA_DIR", tmp_path / "processed")
    data_path = tmp_path / "synthetic_fixture.csv"
    make_synthetic_demo(rows=900, random_state=5).to_csv(data_path, index=False)

    report = train_model(str(data_path))

    assert report["selected_model"] in {"LogisticRegression", "RandomForest"}
    assert set(report["models"]) == {"LogisticRegression", "RandomForest"}
    assert report["test_rows"] > 0
    assert report["split"]["type"] == "chronological UDI holdout"
    assert report["split"]["train_udi_max"] < report["split"]["test_udi_min"]
    assert report["target"].startswith("Machine failure")
    assert (tmp_path / "models" / "failure_classifier.joblib").exists()
    assert (tmp_path / "processed" / "ai4i_engineered_features.csv").exists()
    assert (tmp_path / "figures" / "confusion_matrix.png").exists()
    saved = json.loads((tmp_path / "reports" / "model_metrics.json").read_text())
    assert saved["dataset_source"] == f"Local file: {data_path}"
