from __future__ import annotations

import argparse
import json

from predictive_maintenance.alert_engine import assess_reading
from predictive_maintenance.config import MODEL_DIR
from predictive_maintenance.predict import predict_reading
from predictive_maintenance.simulation import SCENARIOS, SensorSimulator
from predictive_maintenance.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Powered Predictive Maintenance prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="download data if needed and train/evaluate models")
    train_parser.add_argument("--dataset", help="optional path to an existing AI4I CSV")
    simulation_parser = subparsers.add_parser("simulate", help="generate and assess one virtual sensor reading")
    simulation_parser.add_argument("--scenario", choices=SCENARIOS, default="multiple_abnormal_readings")
    simulation_parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.command == "train":
        report = train_model(args.dataset)
        print(f"Dataset: {report['dataset_source']}")
        print(f"Selected model: {report['selected_model']}")
        for name, metrics in report["models"].items():
            print(
                f"{name}: test F1={metrics['f1_failure']:.3f}, "
                f"recall={metrics['recall_failure']:.3f}, "
                f"precision={metrics['precision_failure']:.3f}, "
                f"accuracy={metrics['accuracy']:.3f}"
            )
    else:
        reading = SensorSimulator(args.seed).generate(args.scenario)
        prediction = None
        model_path = MODEL_DIR / "failure_classifier.joblib"
        if model_path.exists() and reading["operating_state"] != "SHUTDOWN":
            try:
                prediction = predict_reading(reading)
            except ValueError as exc:
                prediction = {"error": str(exc)}
        print(json.dumps({"reading": reading, "assessment": assess_reading(reading, prediction)}, indent=2))


if __name__ == "__main__":
    main()
