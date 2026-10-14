import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ALIASES = {
    "volcano_number": ["volcano_number", "volcano no", "volcano number", "volcano id"],
    "volcano_name": ["volcano_name", "volcano name", "volcano"],
    "start_year": ["start_year", "start year", "eruption start year"],
    "vei": ["vei", "volcanic explosivity index"],
    "region": ["region", "volcanic region", "subregion"],
}
FEATURES = ["start_year", "dormancy_years", "region"]

def normalize_columns(frame):
    def clean(name):
        return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    by_name = {clean(col): col for col in frame.columns}
    renames = {}
    for target, options in ALIASES.items():
        for option in options:
            source = by_name.get(clean(option))
            if source is not None:
                renames[source] = target
                break
    frame = frame.rename(columns=renames).copy()
    required = {"volcano_name", "start_year", "vei"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Check GVP workbook schema.")
    if "volcano_number" not in frame:
        frame["volcano_number"] = frame["volcano_name"]
    if "region" not in frame:
        frame["region"] = "Unknown"
    return frame

def prepare(frame):
    df = normalize_columns(frame)
    df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
    df["vei"] = pd.to_numeric(df["vei"], errors="coerce")
    df.loc[~df["vei"].between(0, 8), "vei"] = np.nan
    df = df.dropna(subset=["start_year"]).copy()
    df["volcano_number"] = df["volcano_number"].astype(str)
    df["region"] = df["region"].fillna("Unknown").astype(str)
    df = df.sort_values(["volcano_number", "start_year"], kind="stable")
    # Prior eruption gap is calculated from past eruption start years only.
    df["dormancy_years"] = df.groupby("volcano_number")["start_year"].diff()
    df.loc[df["dormancy_years"] < 0, "dormancy_years"] = np.nan
    return df.reset_index(drop=True)

def make_estimator(classifier=False):
    pre = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), ["start_year", "dormancy_years"]),
        ("category", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), ["region"]),
    ])
    model = (RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=2)
             if classifier else RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2))
    return Pipeline([("preprocess", pre), ("model", model)])

def evaluate_and_impute(df, output):
    observed = df.dropna(subset=["vei"]).copy()
    summary = {
        "rows": int(len(df)),
        "known_vei": int(len(observed)),
        "missing_vei": int(df["vei"].isna().sum()),
        "unique_volcanoes": int(df["volcano_number"].nunique()),
    }
    df["vei_estimated"] = np.nan
    if len(observed) < 20 or observed["volcano_number"].nunique() < 4:
        summary["model_status"] = "insufficient labeled volcano groups"
        return df, summary
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(observed, groups=observed["volcano_number"]))
    train, test = observed.iloc[train_idx], observed.iloc[test_idx]
    regression = make_estimator()
    regression.fit(train[FEATURES], train["vei"])
    summary["vei_mae_holdout"] = float(mean_absolute_error(test["vei"], regression.predict(test[FEATURES])))
    # Refit on all known values before imputing missing values.
    regression.fit(observed[FEATURES], observed["vei"])
    joblib.dump(regression, output / "vei_imputation_model.joblib")
    missing = df["vei"].isna()
    if missing.any():
        df.loc[missing, "vei_estimated"] = np.clip(regression.predict(df.loc[missing, FEATURES]), 0, 8)
    # Intensity is a retrospective VEI >= 3 label, not a forecast of future eruptions.
    train_y, test_y = (train["vei"] >= 3).astype(int), (test["vei"] >= 3).astype(int)
    if train_y.nunique() > 1:
        classifier = make_estimator(classifier=True)
        classifier.fit(train[FEATURES], train_y)
        pred = classifier.predict(test[FEATURES])
        summary["intensity_accuracy_holdout"] = float(accuracy_score(test_y, pred))
        summary["intensity_f1_holdout"] = float(f1_score(test_y, pred, zero_division=0))
        classifier.fit(observed[FEATURES], (observed["vei"] >= 3).astype(int))
        joblib.dump(classifier, output / "intensity_classifier.joblib")
    else:
        summary["classifier_status"] = "single class in training partition"
    return df, summary

def run(input_path, output_dir):
    source = Path(input_path)
    if source.suffix.lower() in {".xlsx", ".xlsm"}:
        raw = pd.read_excel(source)
    elif source.suffix.lower() == ".csv":
        raw = pd.read_csv(source)
    else:
        raise ValueError("Provide .csv or .xlsx input")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = prepare(raw)
    df, summary = evaluate_and_impute(df, output)
    df.to_csv(output / "cleaned_eruptions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4))
    df["vei"].dropna().hist(ax=ax, bins=np.arange(-0.5, 9.5, 1))
    ax.set(title="Observed eruption VEI distribution", xlabel="VEI", ylabel="Count")
    fig.tight_layout()
    fig.savefig(output / "vei_distribution.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    run(args.input, args.output)

if __name__ == "__main__":
    main()
