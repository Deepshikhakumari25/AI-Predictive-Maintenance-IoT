# Limitations

- UCI AI4I 2020 is explicitly synthetic, despite being modeled after industrial behavior. The offline fallback and dashboard sensor stream are also synthetic.
- The prediction target is the failure label for the same recorded row. There is no future prediction horizon, remaining-useful-life target, or exact breakdown-time estimate.
- The source has no explicit machine ID or timestamp field. UDI is used as the available ordering proxy for a chronological holdout; the test does not measure generalization to a new machine.
- Only 39 failures occur in the final 2,000-row held-out period. The selected Random Forest found 24 and missed 15. A perfect observed precision on 24 predicted failures is not a guarantee of future precision.
- Source failure labels are imbalanced. Class weights help training but do not make the outputs calibrated or remove the need to select thresholds using operational costs.
- Simulated vibration and current have no corresponding columns in the selected UCI schema; those values are only visual/demo signals and do not enter the model. Their dashboard heuristics are illustrative.
- Risk thresholds and category points are configurable examples, not validated industrial alarm limits.
- The random-failure mode is very rare, so this project cannot establish reliable detection of that mode.
- This local prototype has no authentication, cloud service, physical sensor connection, API, production observability, or safety interlock.
- Joblib/scikit-learn artifacts should be re-trained in the target environment; do not load model files from untrusted sources.
