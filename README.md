# Volcano Activity Prediction System

Machine-learning analysis of the supplied Smithsonian Global Volcanism Program (GVP) Holocene confirmed-eruption workbook.

## Dataset used

`GVP_Eruption_List_Holocene_20260424.xlsx` contains **9,918 confirmed eruption records**, **854 volcanoes**, and 22 eruption fields. The supplied workbook has **7,725 known VEI values** and **2,193 missing VEI values (22.11%)**.

The workbook's Info sheet provides this citation:
Global Volcanism Program, 2024. *Volcanoes of the World* (v. 5.2.8; 6 May 2025), Smithsonian Institution, compiled by Venzke, E. https://doi.org/10.5479/si.GVP.VOTW5-2024.5.2

The Eruption List sheet itself identifies the export as Volcanoes of the World 5.3.5, downloaded 24 April 2026. Both pieces of provenance are retained because they appear in the supplied file.

## Pipeline

- cleans GVP `NULL` values and numeric fields
- engineers previous-eruption dormancy by volcano
- uses eruption start year, dormancy, and dating-evidence method
- evaluates VEI imputation with Random Forest regression
- evaluates retrospective higher-intensity classification (`VEI >= 3`)
- uses a volcano-group holdout to reduce same-volcano train/test leakage
- refits the VEI model on all known labels and estimates only missing VEI rows

## Measured baseline on the supplied workbook

Using random_state=42 and a 20% volcano-group holdout:

- VEI regression holdout: **MAE 0.741 VEI units**
- Higher-intensity classification: **78.31% accuracy**
- Higher-intensity classification: **F1 0.588**

These are exploratory retrospective metrics, not future-eruption forecasting performance and not public-safety validation.

## Run

```bash
pip install -r requirements.txt
python -m volcano.pipeline --input data/raw/GVP_Eruption_List_Holocene_20260424.xlsx --output outputs
pytest -q
```

The source workbook is not committed automatically; place your supplied file under `data/raw/`.

## Resume-safe summary

Analyzed ~10K Smithsonian GVP confirmed-eruption records, engineered historical dormancy features, and built reproducible Random Forest pipelines for missing-VEI estimation and retrospective eruption-intensity analysis.

## Safety / interpretation

This project does **not** predict whether or when a volcano will erupt. Historical eruption catalogs have incompleteness and uncertainty, especially farther back in time. Outputs are for data-science education and portfolio analysis only.
