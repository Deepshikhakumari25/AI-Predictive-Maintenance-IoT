from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from predictive_maintenance.config import (
    FIGURE_DIR,
    FEATURE_COLUMNS,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from predictive_maintenance.data_loader import ensure_dataset
from predictive_maintenance.feature_engineering import engineer_features
from predictive_maintenance.preprocessing import build_preprocessor
from predictive_maintenance.validation import clean_dataset


def _metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_failure": float(precision_score(y_test, predicted, zero_division=0)),
        "recall_failure": float(recall_score(y_test, predicted, zero_division=0)),
        "f1_failure": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predicted, labels=[0, 1], target_names=["NORMAL", "FAILURE"],
            output_dict=True, zero_division=0,
        ),
    }


def train_model(dataset_path: str | None = None) -> dict:
    if dataset_path:
        frame = pd.read_csv(dataset_path)
        data_source = f"Local file: {dataset_path}"
    else:
        frame, data_source = ensure_dataset()
    frame = clean_dataset(frame)
    x = engineer_features(frame)
    y = frame[TARGET_COLUMN].astype(int)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed = x.copy()
    processed[TARGET_COLUMN] = y
    processed.to_csv(PROCESSED_DATA_DIR / "ai4i_engineered_features.csv", index=False)
    split_index = int(len(frame) * (1 - TEST_SIZE))
    if split_index <= 0 or split_index >= len(frame):
        raise ValueError("Dataset needs enough rows for a non-empty chronological train/test split.")
    x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    candidates = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1500, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    evaluated: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, classifier in candidates.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor()), ("classifier", classifier)]
        )
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        evaluated[name] = _metrics(pipeline, x_test, y_test)

    selected_name = max(
        evaluated,
        key=lambda name: (evaluated[name]["f1_failure"], evaluated[name]["recall_failure"]),
    )
    selected = fitted[selected_name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, MODEL_DIR / "failure_classifier.joblib")

    matrix = evaluated[selected_name]["confusion_matrix_labels_0_1"]
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="YlOrRd")
    axis.set(title=f"{selected_name}: held-out confusion matrix", xlabel="Predicted label", ylabel="Actual label")
    axis.set_xticks([0, 1], labels=["NORMAL", "FAILURE"])
    axis.set_yticks([0, 1], labels=["NORMAL", "FAILURE"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Records")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    classifier = selected.named_steps["classifier"]
    preprocessor = selected.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out().tolist()
    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    else:
        importance = abs(classifier.coef_[0])
    importance_rows = sorted(
        [
            {"feature": name, "importance": float(value)}
            for name, value in zip(feature_names, importance)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": data_source,
        "dataset_rows": int(len(frame)),
        "target": TARGET_COLUMN,
        "target_definition": "Failure is labelled on this recorded operating row; this is not a future-horizon forecast.",
        "input_features": FEATURE_COLUMNS,
        "excluded_columns": ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
        "split": {
            "type": "chronological UDI holdout",
            "test_size": TEST_SIZE,
            "train_udi_max": int(frame["UDI"].iloc[split_index - 1]),
            "test_udi_min": int(frame["UDI"].iloc[split_index]),
        },
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "class_counts": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
        "selection_metric": "failure-class F1, then failure-class recall",
        "training_configuration": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "split": "chronological; ordered by unique UDI",
            "logistic_regression": {"class_weight": "balanced", "max_iter": 1500},
            "random_forest": {
                "n_estimators": 250,
                "class_weight": "balanced_subsample",
                "min_samples_leaf": 2,
                "n_jobs": -1,
            },
            "numeric_preprocessing": "median imputation and StandardScaler, fit on training rows",
            "categorical_preprocessing": "most-frequent imputation and one-hot encoding, fit on training rows",
        },
        "selected_model": selected_name,
        "models": evaluated,
        "selected_feature_importance": importance_rows,
    }
    (REPORT_DIR / "model_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "model_metrics.txt").write_text(
        "Held-out test-set model comparison\n"
        + "\n".join(
            f"{name}: F1={values['f1_failure']:.3f}, recall={values['recall_failure']:.3f}, "
            f"precision={values['precision_failure']:.3f}, accuracy={values['accuracy']:.3f}"
            for name, values in evaluated.items()
        )
        + f"\nSelected: {selected_name}\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = train_model()
    print(f"Dataset: {result['dataset_source']}")
    print(f"Selected model: {result['selected_model']}")
    print((REPORT_DIR / "model_metrics.txt").read_text(encoding="utf-8"))
