from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FEATURES_PATH = PROCESSED_DATA_DIR / "ai4i_engineered_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
SCREENSHOT_DIR = PROJECT_ROOT / "images" / "screenshots"

DATASET_FILENAME = "ai4i2020.csv"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
TARGET_COLUMN = "Machine failure"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_FEATURES = FEATURE_COLUMNS[1:]
CATEGORICAL_FEATURES = [FEATURE_COLUMNS[0]]
REQUIRED_DATASET_COLUMNS = [
    "UDI",
    "Product ID",
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]
