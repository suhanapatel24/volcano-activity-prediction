# Volcano Activity Prediction System

Reconstructed portfolio implementation based on Suhana Patel's resume description (the original source files were unavailable). This project analyzes Smithsonian Global Volcanism Program (GVP) eruption records, engineers historical dormancy features, and trains models for **VEI imputation** and **eruption-intensity classification**.

**Important:** This is retrospective analysis, not a real-time eruption forecast or hazard-warning system. No measured results or 10,000-row dataset are claimed until an actual dataset is processed.

## Data

Download the official **Confirmed Holocene Eruptions** spreadsheet from https://volcano.si.edu/search_eruption.cfm and place it at `data/raw/eruptions.xlsx`. The Smithsonian publishes a current workbook and its database citation at that page. Dataset contents and schema may change across releases.

GVP citation: Global Volcanism Program (2026), *Volcanoes of the World*, v5.4.0, Smithsonian Institution, https://doi.org/10.5479/si.GVP.VOTW5-2026.5.4.

## Run

```bash
pip install -r requirements.txt
python -m volcano.pipeline --input data/raw/eruptions.xlsx --output outputs
pytest -q
```

The pipeline writes `cleaned_eruptions.csv`, `metrics.json`, `vei_imputation_model.joblib`, `intensity_classifier.joblib`, and `figures/` where enough labeled data is available.

## Methods

- Normalizes common GVP workbook column-name variants
- Parses eruption start years and missing VEI
- Computes prior-eruption gap by volcano without using future eruption information
- Predicts known VEI values using cross-validated Random Forest regression
- Classifies observed eruptions into lower/higher intensity categories
- Produces data-quality summaries and exploratory figures
- Keeps missing-value estimates separate from observed VEI

**Evaluation caution:** historical records are incomplete, and random splits can overstate generalization across volcanoes and time. Metrics are exploratory and should not be used for public-safety decisions.

## Cloud extension

The original resume mentions GCP storage/notebooks. This reconstructed version runs locally first. `docs/gcp.md` explains how to upload artifacts to Cloud Storage; no cloud deployment is claimed.

## Project provenance

Original notebooks and the original ~10,000-row merged dataset were not supplied. This repository is a new reconstruction based on the resume scope, not a recovery of the original experiment or its metrics.
