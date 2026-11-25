import json

import numpy as np
import pandas as pd
import pytest

from volcano.pipeline import load_gvp, model, prepare, run


def test_prepare_dormancy_and_null_vei():
    frame = pd.DataFrame({
        "Volcano Name": ["A", "A", "B"],
        "Volcano Number": [1, 1, 2],
        "Start Year": [1990, 2000, 2001],
        "VEI": [2, "NULL", 4],
    })
    result = prepare(frame)
    a = result[result["volcano_number"] == "1"].reset_index(drop=True)
    assert pd.isna(a.loc[0, "dormancy_years"])
    assert a.loc[1, "dormancy_years"] == 10
    assert pd.isna(a.loc[1, "vei"])
    assert result["evidence_method"].isna().all()


def test_prepare_rejects_out_of_range_vei_and_missing_years():
    frame = pd.DataFrame({
        "Volcano Name": ["A", "A", "A"],
        "Volcano Number": [1, 1, 1],
        "Start Year": [1900, None, 1950],
        "VEI": [9, 3, -1],
    })
    result = prepare(frame)
    assert len(result) == 2  # row without a start year is dropped
    assert result["vei"].isna().all()  # 9 and -1 are outside the 0-8 VEI scale


def test_prepare_missing_required_columns():
    with pytest.raises(ValueError, match="Missing GVP columns"):
        prepare(pd.DataFrame({"Start Year": [2000]}))


def _synthetic_catalog(n_volcanoes=30, eruptions=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_volcanoes):
        year = int(rng.integers(1000, 1500))
        for e in range(eruptions):
            year += int(rng.integers(5, 120))
            vei = int(rng.integers(0, 6))
            rows.append({
                "Volcano Number": 100000 + v,
                "Volcano Name": f"Volcano {v}",
                "Start Year": year,
                "VEI": "NULL" if e == eruptions - 1 else vei,
                "Evidence Method (dating)": rng.choice(
                    ["Historical Observations", "Radiocarbon (corrected)", "NULL"]
                ),
            })
    return pd.DataFrame(rows)


def test_model_writes_artifacts_and_only_imputes_missing(tmp_path):
    df = prepare(_synthetic_catalog())
    metrics = model(df, tmp_path)

    assert metrics["rows"] == len(df)
    assert metrics["unique_volcanoes"] == 30
    assert metrics["missing_vei"] == 30
    assert metrics["known_vei"] + metrics["missing_vei"] == metrics["rows"]
    for key in ("vei_mae_group_holdout", "intensity_accuracy_group_holdout",
                "intensity_f1_group_holdout"):
        assert key in metrics and np.isfinite(metrics[key])

    for name in ("metrics.json", "cleaned_eruptions.csv", "vei_imputation_model.joblib"):
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics

    out = pd.read_csv(tmp_path / "cleaned_eruptions.csv")
    missing = out["vei"].isna()
    assert out.loc[missing, "vei_estimated"].between(0, 8).all()
    assert out.loc[~missing, "vei_estimated"].isna().all()


def test_run_from_csv(tmp_path):
    src = tmp_path / "eruptions.csv"
    _synthetic_catalog(seed=1).to_csv(src, index=False)
    assert len(load_gvp(src)) == 180
    metrics = run(src, tmp_path / "out")
    assert metrics["rows"] == 180
