from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from predictive_maintenance.alert_engine import assess_reading
from predictive_maintenance.config import (
    MODEL_DIR,
    PROCESSED_FEATURES_PATH,
    PROJECT_ROOT,
    REPORT_DIR,
    TARGET_COLUMN,
)
from predictive_maintenance.data_loader import load_dataset
from predictive_maintenance.predict import predict_reading
from predictive_maintenance.simulation import SCENARIOS, SensorSimulator
from predictive_maintenance.utils import append_jsonl, load_recent_events
from predictive_maintenance.visualization import sensor_trend_figure

st.set_page_config(page_title="Predictive Maintenance | IoT Lab", page_icon="⚙", layout="wide")
st.title("AI-Powered Predictive Maintenance")
st.caption("LOCAL PROTOTYPE  /  VIRTUAL SENSOR DATA ONLY  /  NOT FOR SAFETY-CRITICAL USE")

model_path = MODEL_DIR / "failure_classifier.joblib"
metrics_path = REPORT_DIR / "model_metrics.json"
history_path = PROJECT_ROOT / "outputs" / "predictions" / "event_history.jsonl"
model_ready = model_path.exists()
report = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None

if "sensor_simulator" not in st.session_state:
    st.session_state.sensor_simulator = SensorSimulator(seed=42)
if "events" not in st.session_state:
    try:
        st.session_state.events = load_recent_events(history_path)
    except (OSError, json.JSONDecodeError) as exc:
        st.session_state.events = []
        st.sidebar.warning(f"Saved event history could not be read: {exc}")

with st.sidebar:
    st.header("Virtual machine")
    machine_id = st.text_input("Machine identifier", "SIM-001")
    scenario = st.selectbox("Scenario", SCENARIOS, index=4)
    batch_size = st.slider("Readings to generate", min_value=1, max_value=30, value=8)
    simulation_enabled = st.toggle("Simulation enabled", value=True)
    generate = st.button("Generate readings", type="primary", disabled=not simulation_enabled, use_container_width=True)
    if st.button("Clear session history", use_container_width=True):
        st.session_state.events = []
        st.session_state.sensor_simulator = SensorSimulator(seed=42)
        st.rerun()
    st.divider()
    st.write("Model status")
    st.success("Trained model loaded" if model_ready else "Model not trained")

if generate:
    for _ in range(batch_size):
        reading = st.session_state.sensor_simulator.generate(scenario, machine_id)
        prediction = None
        inference_error = None
        if model_ready and reading["operating_state"] != "SHUTDOWN":
            try:
                prediction = predict_reading(reading, model_path)
            except (ValueError, TypeError) as exc:
                inference_error = str(exc)
        assessment = assess_reading(reading, prediction)
        event = {"reading": reading, "prediction": prediction, "assessment": assessment}
        if inference_error:
            event["inference_error"] = inference_error
        st.session_state.events.append(event)
        append_jsonl(history_path, event)

events = st.session_state.events
readings = pd.DataFrame([event["reading"] for event in events])
assessments = pd.DataFrame([event["assessment"] for event in events])
alerts = assessments[assessments["alert"]] if not assessments.empty else assessments

overview_tab, data_tab, monitoring_tab, analytics_tab, alerts_tab, model_tab, test_tab = st.tabs(
    ["Overview", "Data preparation", "Sensor monitoring", "Predictive analytics", "Alerts", "Dataset & model", "Tests"]
)

with overview_tab:
    columns = st.columns(3)
    columns[0].metric("Readings this session", len(events))
    columns[1].metric("Alerts this session", len(alerts))
    columns[2].metric("Model", report["selected_model"] if report else "Not trained")
    st.caption("All virtual sensor readings are simulated.")
    st.subheader("Risk distribution")
    if assessments.empty:
        st.info("Generate a virtual reading batch to populate the overview.")
    else:
        distribution = assessments["risk_category"].value_counts().reindex(
            ["LOW", "MODERATE", "HIGH", "CRITICAL"], fill_value=0
        ).rename_axis("Risk category").reset_index(name="Readings")
        st.plotly_chart(px.bar(distribution, x="Risk category", y="Readings", color="Risk category"), use_container_width=True)
    st.info("Risk scores are rule-based indicators, not calibrated probabilities or predictions of future failure.")

with data_tab:
    st.subheader("Engineered training table")
    st.caption("Derived features are deterministic; learned imputers and scaling are fitted on training rows only.")
    if PROCESSED_FEATURES_PATH.exists():
        processed_frame = pd.read_csv(PROCESSED_FEATURES_PATH)
        st.write(f"{len(processed_frame):,} rows | {len(processed_frame.columns)} columns")
        st.dataframe(processed_frame.head(12), use_container_width=True, hide_index=True)
    else:
        st.info("No processed table yet. Run `py -3.14 main.py train` to create it.")

with monitoring_tab:
    st.subheader("Virtual sensor stream")
    if readings.empty:
        st.info("No readings yet. Enable simulation and generate a batch.")
    else:
        figure = sensor_trend_figure(readings)
        if figure:
            st.plotly_chart(figure, use_container_width=True)
        display_columns = [
            column for column in ["timestamp", "machine_id", "scenario", "operating_state", "Type",
                                  "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
                                  "Torque [Nm]", "Tool wear [min]", "Vibration [mm/s]", "Current [A]"]
            if column in readings.columns
        ]
        st.dataframe(readings[display_columns].iloc[::-1], use_container_width=True, hide_index=True)

with analytics_tab:
    st.subheader("Model output")
    if not model_ready:
        st.warning("No trained artifact found. From the project root, run `py -3.14 main.py train`.")
    elif not events:
        st.info("Generate readings to see row-level predictions.")
    else:
        result_rows = []
        for event in reversed(events):
            prediction = event["prediction"]
            result_rows.append({
                "Machine": event["reading"]["machine_id"],
                "Scenario": event["reading"]["scenario"],
                "Model prediction": prediction.get("prediction", "Unavailable") if prediction else "Unavailable",
                "Failure score (uncalibrated)": prediction.get("model_failure_probability") if prediction else None,
                "Heuristic risk": event["assessment"]["risk_category"],
                "Inference issue": event.get("inference_error", ""),
            })
        st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)
    if report:
        st.subheader("Held-out evaluation")
        model_metrics = report["models"]
        metric_frame = pd.DataFrame(model_metrics).T.reset_index(names="Model")
        st.dataframe(metric_frame[["Model", "accuracy", "precision_failure", "recall_failure", "f1_failure", "roc_auc"]], use_container_width=True, hide_index=True)
        matrix = model_metrics[report["selected_model"]]["confusion_matrix_labels_0_1"]
        st.plotly_chart(px.imshow(matrix, x=["NORMAL", "FAILURE"], y=["NORMAL", "FAILURE"],
                                  text_auto=True, labels={"x": "Predicted", "y": "Actual", "color": "Records"},
                                  title="Selected model: held-out confusion matrix"), use_container_width=True)
        importance = pd.DataFrame(report.get("selected_feature_importance", [])).head(12)
        if not importance.empty:
            st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature",
                                   orientation="h", title="Model feature importance"), use_container_width=True)

with alerts_tab:
    st.subheader("Maintenance alert history")
    if alerts.empty:
        st.success("No alerts in this session.")
    else:
        for event in reversed(events):
            alert = event["assessment"]
            if alert["alert"]:
                with st.expander(f"{alert['risk_category']}  |  {alert['machine_id']}  |  {alert['timestamp']}"):
                    st.write("Trigger:", "; ".join(alert["triggering_conditions"]))
                    st.write("Heuristic score:", f"{alert['heuristic_risk_score']}/100 (not a probability)")
                    st.write("Suggested investigation:", alert["recommended_investigation"])
                    st.json({key: value for key, value in event["reading"].items() if key not in {"simulation"}})
    st.caption("Events also append to outputs/predictions/event_history.jsonl. All readings shown here are simulated.")

with model_tab:
    st.subheader("Dataset and task")
    st.write("Dataset: UCI AI4I 2020 Predictive Maintenance Dataset. The source dataset is synthetic, not real machine telemetry.")
    st.write("Target: `Machine failure` for the same recorded row. This prototype does not forecast a future time horizon.")
    try:
        frame = load_dataset()
        st.write(f"Available rows: {len(frame):,} | columns: {len(frame.columns)}")
        st.dataframe(frame.head(10), use_container_width=True, hide_index=True)
        st.write("Observed failure labels:", frame[TARGET_COLUMN].value_counts().sort_index().to_dict())
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.warning(f"Dataset is not available locally yet: {exc}")
    if report:
        st.write("Source used for model:", report["dataset_source"])
        st.write("Model selected by held-out failure-class F1, then recall:", report["selected_model"])
        st.write("Input columns:", ", ".join(report["input_features"]))
        st.write("Excluded leakage/identifier columns:", ", ".join(report["excluded_columns"]))
    else:
        st.info("Training report not found. Run `py -3.14 main.py train` to create the model and evaluation report.")

with test_tab:
    st.subheader("Automated test results")
    test_log = REPORT_DIR / "pytest_results.txt"
    if test_log.exists():
        log_bytes = test_log.read_bytes()
        encoding = "utf-16" if log_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        st.code(log_bytes.decode(encoding), language="text")
    else:
        st.info("No saved test run yet. Run `py -3.14 -m pytest -q` from the project root.")
