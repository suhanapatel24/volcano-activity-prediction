import pandas as pd
import pytest
from volcano.pipeline import prepare, evaluate_and_impute

def test_prior_gap_and_missing_vei():
    frame = pd.DataFrame({
        "Volcano Name": ["A", "A", "B"],
        "Volcano Number": [1, 1, 2],
        "Start Year": [1990, 2000, 2001],
        "VEI": [2, None, 4],
    })
    result = prepare(frame)
    assert result.loc[1, "dormancy_years"] == 10
    assert pd.isna(result.loc[1, "vei"])
    assert pd.isna(result.loc[0, "dormancy_years"])

def test_missing_required_field():
    with pytest.raises(ValueError):
        prepare(pd.DataFrame({"Start Year": [2000]}))

def test_insufficient_data_does_not_invent_metrics(tmp_path):
    frame = pd.DataFrame({
        "Volcano Name": ["A", "A"],
        "Start Year": [2000, 2010],
        "VEI": [2, None],
    })
    prepared = prepare(frame)
    output, metrics = evaluate_and_impute(prepared, tmp_path)
    assert metrics["missing_vei"] == 1
    assert "vei_mae_holdout" not in metrics
    assert pd.isna(output.loc[1, "vei_estimated"])
