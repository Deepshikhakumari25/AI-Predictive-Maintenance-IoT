# Dataset Documentation

## Source and Conditions

- Name: AI4I 2020 Predictive Maintenance Dataset.
- Maintainer: UCI Machine Learning Repository, dataset 601.
- Official page: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Official ZIP: https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip
- DOI: https://doi.org/10.24432/C5HS5C
- License: CC BY 4.0 according to the UCI dataset page; attribution is required when sharing/adapting.
- Format: CSV (`ai4i2020.csv`), downloaded under `data/raw/` and never modified in place.
- Size/schema: 10,000 rows, 14 columns, no missing values in the published dataset.
- Provenance: UCI explicitly says the measurements are synthetic and intended to reflect industrial predictive-maintenance data. They are not actual factory sensor readings.

## Published Columns

| Column | Meaning | Model use |
|---|---|---|
| `UDI` | Unique row identifier | Ordering/split only; excluded as feature. |
| `Product ID` | Product identifier, with L/M/H quality variant and serial | Excluded identifier. |
| `Type` | Product quality type L/M/H | Input feature. |
| `Air temperature [K]` | Air temperature in kelvin | Input feature. |
| `Process temperature [K]` | Process temperature in kelvin | Input feature. |
| `Rotational speed [rpm]` | Rotational speed | Input feature. |
| `Torque [Nm]` | Torque | Input feature. |
| `Tool wear [min]` | Tool wear time in minutes | Input feature. |
| `Machine failure` | Whether failure occurs at this recorded data point | Binary target. |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Tool-wear, heat-dissipation, power, overstrain, random failure modes | Excluded: direct failure-mode/target leakage. |

## Target and Preparation

The task is to classify the `Machine failure` label attached to the same row from the six listed inputs. It is **not** a future-window prediction. The preprocessing code checks required fields, unique numeric UDI values, binary target labels, and numeric sensor values. Required target/product-type nulls are dropped; optional missing numeric values are median-imputed inside each training pipeline. The published source has no missing values.

The input records are sorted by UDI. The first 80% are used for training and the final 20% are reserved as a chronological test period; no future UDI records enter model fitting. Derived fields are temperature difference and estimated mechanical power. No target labels, failure-mode flags, product serial identifiers, or learned whole-dataset transformations enter the feature matrix.

When download is unavailable, `data_loader.py` creates a deterministic 1,200-row AI4I-shaped synthetic fallback based on documented variable ranges and simplified published failure rules. Its source name is reported in model metrics; it must not be described as real or official UCI observations.

## Limitations

The published data is synthetic, has no explicit machine identifier or timestamp column, and represents a single dataset's simulated process. UDI ordering is used as the available sequence proxy. Random failures and rare failure modes have few examples. Results do not establish future or cross-machine performance. Consult [LIMITATIONS.md](LIMITATIONS.md).
