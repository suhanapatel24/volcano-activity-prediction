# Optional GCP workflow

The reconstructed code is local-first. To store reproducible results in Cloud Storage:

1. Run the pipeline on a local workstation or a GCP notebook with the same dependencies.
2. Create a dedicated Cloud Storage bucket with appropriate access controls.
3. Upload sanitized outputs with `gcloud storage cp --recursive outputs gs://YOUR_BUCKET/volcano-analysis/`.
4. Record the dataset version, row count, pipeline commit SHA, metrics, and run date.

Do not commit cloud credentials or claim that these steps were executed until evidence exists.
