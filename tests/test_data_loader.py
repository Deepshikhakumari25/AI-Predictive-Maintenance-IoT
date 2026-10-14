import pandas as pd
import pytest

from predictive_maintenance.data_loader import load_dataset, make_synthetic_demo


def test_synthetic_demo_is_repeatable_and_ai4i_shaped():
    first = make_synthetic_demo(rows=300, random_state=9)
    second = make_synthetic_demo(rows=300, random_state=9)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 300
    assert {0, 1}.issuperset(set(first["Machine failure"].unique()))


def test_missing_dataset_path_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(tmp_path / "missing.csv")
