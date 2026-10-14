# Project Status

## Current Phase

Part 1 is complete. Part 2 README and ignore-rule review are complete; the local `main` repository has initial commit `362d51f`. No GitHub remote has been configured and nothing has been uploaded. The local Streamlit server is running at `http://localhost:8501`.

## Completed

- Inspected an empty workspace and verified Python 3.14.6 and 3.13.9 are installed; selected an isolated Python 3.14 virtual environment.
- Installed the declared CPU-only stack and editable project package in `.venv`.
- Documented UCI AI4I schema, synthetic provenance, CC BY 4.0 license, target, and known limitations.
- Implemented validated loading/download with deterministic synthetic fallback, feature engineering, train-only preprocessing, chronological UDI split, model comparison, prediction, simulation, risk scoring, alerts, event history, and Streamlit dashboard.
- Generated the official-dataset model artifact, processed feature file, reports, and confusion-matrix figure.
- Added offline tests for loading, validation, feature preparation, training, prediction, simulation, alerts, and event serialization.
- Verified the dashboard in a desktop browser and at a 390 px viewport; no horizontal page overflow was present. Generated simulated readings and confirmed prediction, risk, and alert views render.
- Captured all 11 requested PNG screenshots from the actual Streamlit app and test report.
- Expanded the publication README with an architecture diagram, exact dataset columns, a real seeded CLI prediction, embedded screenshots, references, learning outcomes, future improvements, and disclaimer.
- Hardened `.gitignore` and confirmed source, docs, tests, metrics, report, confusion figure, and all screenshots are the staged publication set; local data/model/history/cache/egg-info files are ignored.
- Initialized a local `main` repository with GitHub Desktop's bundled Git 2.53.0.windows.4 and reviewed the staged file list. No remote is configured.
- Created the local root commit `362d51f` (`Add predictive maintenance portfolio project`); the working tree was clean immediately afterward.

## Actual Training Results

Source: UCI AI4I 2020 official download, 10,000 rows. Training: first 8,000 UDI rows. Test: UDI 8001-10000, 2,000 rows / 39 failures. Selected model: class-weighted Random Forest.

| Model | Accuracy | Failure precision | Failure recall | Failure F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9455 | 0.2177 | 0.6923 | 0.3313 | 0.9091 |
| Random Forest | 0.9925 | 1.0000 | 0.6154 | 0.7619 | 0.9476 |

Random Forest held-out confusion matrix: `[[1961, 0], [15, 24]]`. These are measured results, not expected guarantees.

## Verification

- Environment setup: `.venv` created and `requirements.txt` plus editable package installed successfully.
- Training command: `.\.venv\Scripts\python.exe main.py train` completed against the official UCI download.
- Test command: `.\.venv\Scripts\python.exe -m pytest -q` completed with **24 passed**; actual console summary saved in `outputs/reports/pytest_results.txt`.
- README relative links: 10 checked, all resolve. Secret/private-key pattern scan found no matches in project text/code files.
- Dashboard browser smoke check: passed at desktop and 390 px mobile viewport; server remains running.
- Screenshot files: complete under `images/screenshots/`.

## Important Outputs

- `outputs/reports/model_metrics.json` and `model_metrics.txt`
- `outputs/reports/pytest_results.txt`
- `outputs/figures/confusion_matrix.png`
- `data/processed/ai4i_engineered_features.csv` (ignored by Git)
- `models/failure_classifier.joblib` (ignored; retrain locally)
- `data/raw/ai4i2020.csv` (ignored; downloadable under documented CC BY 4.0 terms)

## Known Limitations

- The data is synthetic and the classifier predicts the label for the same row, not a future failure horizon.
- The holdout has 39 failures; Random Forest missed 15. This is a learning demonstration, not production-ready.
- Sensor simulation is virtual. Vibration/current are demo-only and are not model inputs.
- The project has no real machine IDs/timestamps, physical connections, API, or calibrated probability estimates.

## GitHub Handoff

- Project directory: workspace root `AI-Predictive-Maintenance-IoT`.
- Proposed repository name: `AI-Predictive-Maintenance-IoT`; public repository search found zero exact-name matches. A private repository under the user's account cannot be checked until sign-in.
- Suggested description and topics are in the README's GitHub Presentation section.
- Suggested exclude items: `.venv/`, `data/raw/ai4i2020.csv` (downloaded; cite UCI if redistributed), `data/processed/ai4i_engineered_features.csv`, `models/failure_classifier.joblib`, `outputs/predictions/event_history.jsonl`, and local cache files.
- Reports and the small confusion-matrix figure are retained in the project so documented metrics are inspectable.
- GitHub Desktop 3.6.6 is installed; its bundled Git works. Standalone Git is not on PATH. GitHub CLI 2.102.0 is installed but `gh auth status` reports no authenticated account. Global Git author identity is configured.
- The exact-name public GitHub repository search returned zero matches. The user's private repositories cannot be inspected without signing in.
- The project code has no separate license file. Choose a code license before publication if reuse rights are intended; the dataset's CC BY 4.0 license does not license this project's code.
- Remaining action: sign in through GitHub Desktop, verify the intended account, publish this existing local repository as Public or Private as desired, then verify README/images/files on the actual GitHub URL. No URL, visibility, or successful push is claimed until that is done.

## Screenshot Inventory

- `images/screenshots/01_project_overview.png`
- `images/screenshots/02_dataset_preview.png`
- `images/screenshots/03_preprocessing_output.png`
- `images/screenshots/04_model_training.png`
- `images/screenshots/05_model_evaluation.png`
- `images/screenshots/06_confusion_matrix.png`
- `images/screenshots/07_prediction_output.png`
- `images/screenshots/08_sensor_simulation.png`
- `images/screenshots/09_alert_dashboard.png`
- `images/screenshots/10_final_dashboard.png`
- `images/screenshots/11_pytest_results.png`

## Exact Commands

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -e .
.\.venv\Scripts\python.exe main.py train
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
.\.venv\Scripts\python.exe -m pytest -q
```
