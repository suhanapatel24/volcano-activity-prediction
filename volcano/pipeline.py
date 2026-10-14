import argparse
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["start_year", "dormancy_years", "evidence_method"]

def load_gvp(path):
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        # GVP export has a title/provenance row above the real header.
        frame = pd.read_excel(path, sheet_name="Eruption List", header=1)
    else:
        frame = pd.read_csv(path)
    return frame

def prepare(frame):
    required = {"Volcano Number", "Volcano Name", "Start Year", "VEI"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing GVP columns: {sorted(missing)}")
    df = frame.copy()
    df["volcano_number"] = df["Volcano Number"].astype(str)
    df["volcano_name"] = df["Volcano Name"].astype(str)
    df["start_year"] = pd.to_numeric(df["Start Year"], errors="coerce")
    df["vei"] = pd.to_numeric(df["VEI"].replace("NULL", np.nan), errors="coerce")
    df.loc[~df["vei"].between(0, 8), "vei"] = np.nan
    if "Evidence Method (dating)" in df:
        df["evidence_method"] = df["Evidence Method (dating)"].replace("NULL", np.nan)
    else:
        df["evidence_method"] = np.nan
    df = df.dropna(subset=["start_year"]).sort_values(
        ["volcano_number", "start_year"], kind="stable"
    )
    df["dormancy_years"] = df.groupby("volcano_number")["start_year"].diff()
    df.loc[df["dormancy_years"] < 0, "dormancy_years"] = np.nan
    return df.reset_index(drop=True)

def estimator(classifier=False):
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), ["start_year", "dormancy_years"]),
        ("evidence", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), ["evidence_method"]),
    ])
    model = (
        RandomForestClassifier(
            n_estimators=150, random_state=42, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1
        )
        if classifier else
        RandomForestRegressor(
            n_estimators=150, random_state=42, min_samples_leaf=2, n_jobs=-1
        )
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])

def model(df, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    observed = df.dropna(subset=["vei"]).copy()
    metrics = {
        "rows": int(len(df)),
        "unique_volcanoes": int(df["volcano_number"].nunique()),
        "known_vei": int(observed.shape[0]),
        "missing_vei": int(df["vei"].isna().sum()),
        "missing_vei_percent": float(100 * df["vei"].isna().mean()),
    }
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(
        split.split(observed, groups=observed["volcano_number"])
    )
    train, test = observed.iloc[train_idx], observed.iloc[test_idx]

    reg = estimator()
    reg.fit(train[FEATURES], train["vei"])
    reg_pred = reg.predict(test[FEATURES])
    metrics["vei_mae_group_holdout"] = float(
        mean_absolute_error(test["vei"], reg_pred)
    )

    clf = estimator(classifier=True)
    y_train = (train["vei"] >= 3).astype(int)
    y_test = (test["vei"] >= 3).astype(int)
    clf.fit(train[FEATURES], y_train)
    clf_pred = clf.predict(test[FEATURES])
    metrics["intensity_accuracy_group_holdout"] = float(
        accuracy_score(y_test, clf_pred)
    )
    metrics["intensity_f1_group_holdout"] = float(
        f1_score(y_test, clf_pred, zero_division=0)
    )

    # Final imputation model learns from all observed VEI rows.
    reg.fit(observed[FEATURES], observed["vei"])
    joblib.dump(reg, output / "vei_imputation_model.joblib")
    df["vei_estimated"] = np.nan
    missing = df["vei"].isna()
    df.loc[missing, "vei_estimated"] = np.clip(
        reg.predict(df.loc[missing, FEATURES]), 0, 8
    )
    df.to_csv(output / "cleaned_eruptions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

def run(input_path, output):
    return model(prepare(load_gvp(input_path)), output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), indent=2))

if __name__ == "__main__":
    main()
