# Project Architecture

## Data Flow

1. `data_loader.py` reads the original UCI CSV from `data/raw/`. If it is missing, training attempts the official download and falls back to a deterministic, clearly labelled synthetic sample if unavailable.
2. `validation.py` checks the expected AI4I schema, binary target, unique numeric UDI values, and finite numeric sensor values. Rows with missing required product type or target are removed; missing sensor values are retained for train-fitted imputers.
3. Rows are sorted by UDI. `feature_engineering.py` selects only six known-at-reading inputs and derives temperature difference and estimated mechanical power. Identifier and failure-mode columns never enter model features.
4. Training uses UDI rows 1-8000 as development data and 8001-10000 as chronological holdout for the official data. A scikit-learn `Pipeline` fits median imputers, a numeric scaler, categorical one-hot encoding, and each classifier on training rows only.
5. Class-weighted Logistic Regression and Random Forest are evaluated on the same held-out period. Selection is by failure-class F1, then failure-class recall. Model, JSON/text metrics, processed feature CSV, and confusion-matrix figure are saved locally.
6. `predict.py` validates one reading, engineers its features, loads the fitted pipeline, and returns a structured same-row classifier output. Model output probability is explicitly marked uncalibrated.
7. `simulation.py` produces virtual readings only. `risk_scoring.py` applies documented heuristic thresholds separately from the model; `alert_engine.py` combines the two while retaining their distinct origins. Events append to local JSON Lines history.
8. The Streamlit dashboard displays dataset/model status, training evaluation, virtual sensor trends, model predictions, heuristic categories, and alerts. It does not communicate with physical equipment or external services.

## Trust Boundaries

- UCI source data is synthetic research data; locally generated fallback and virtual sensor readings are also synthetic.
- No model score or heuristic score is a calibrated failure probability.
- No exact future horizon, remaining useful life, or breakdown time is predicted.
- No real device, actuator, cloud IoT service, or API is connected.
