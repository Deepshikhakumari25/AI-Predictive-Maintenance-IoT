# Model Card

## Model and Intended Use

The saved artifact is a scikit-learn pipeline containing fitted preprocessing and a class-weighted Random Forest classifier. It classifies the AI4I `Machine failure` label for an individual recorded row using the six available sensor/product fields. It is intended for local education and portfolio demonstration only.

The comparison baseline is class-weighted Logistic Regression. On this run, model selection by failure-class F1 (then recall) selected Random Forest. Both model results are recorded in `outputs/reports/model_metrics.json`.

## Evaluation

The official UCI dataset contains 10,000 rows and 339 positive labels. Rows are sorted by UDI, the first 8,000 train the models, and UDI 8001-10000 is the held-out test set (2,000 rows, 39 failures). Random state 42 is used by model estimators; the chronological split itself is deterministic.

| Held-out measure | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.9455 | 0.9925 |
| Failure precision | 0.2177 | 1.0000 |
| Failure recall | 0.6923 | 0.6154 |
| Failure F1 | 0.3313 | 0.7619 |
| ROC AUC | 0.9091 | 0.9476 |

Random Forest confusion matrix, labels ordered NORMAL then FAILURE: `[[1961, 0], [15, 24]]`. It missed 15 of 39 failures. Accuracy is dominated by normal rows and must not be read alone. These results are from synthetic source data and a single chronological holdout.

## Inputs and Exclusions

Inputs: product `Type`, air/process temperature, rotational speed, torque, tool wear. Derived features: temperature difference and estimated mechanical power. Excluded: `UDI`, `Product ID`, and `TWF`/`HDF`/`PWF`/`OSF`/`RNF` to avoid identifiers and direct label leakage.

## Scores and Thresholds

`predict_proba` output is exposed as `model_failure_probability` for inspection only and is marked uncalibrated. The separate 0-100 heuristic score is a sum of configured sensor-rule points, capped at 100; it is not a probability. The alert engine shows the classifier result, heuristic category, and threshold reasons as separate fields.

Current illustrative heuristic points are: simulated shutdown = 100; any missing/non-finite required model sensor = 30; temperature difference below 8.6 K = 25; estimated mechanical power outside 3,500-9,000 W = 30; tool wear at least 200 min = 25 (otherwise at least 150 min = 15); simulated vibration at least 8 mm/s = 30 (otherwise at least 6 mm/s = 15); simulated current below 3 A or above 13 A = 15. Points add together and cap at 100. Categories are LOW 0-29, MODERATE 30-54, HIGH 55-79, and CRITICAL 80-100. MODERATE or above produces a heuristic alert; a model FAILURE prediction also produces an alert independently. Vibration/current are demo-only thresholds and not AI4I model inputs. These values are examples, not validated operating limits.

## Limitations and Risks

This model does not predict future failures, time-to-failure, or remaining useful life. It is not validated on real equipment, cross-machine data, changing operating regimes, or safety-critical workflows. Missing/invalid readings and unseen operating conditions require domain review; do not use these outputs to operate machinery or defer qualified inspection.
