# Input fields

Required: `Volcano Name`, `Start Year`, `VEI`.

Optional: `Volcano Number` (used to group eruptions by volcano), `Region`.

Derived: `dormancy_years` (difference between this and the previous recorded eruption start year for the same volcano); `vei_estimated` (model prediction only where observed VEI is missing).

The GVP workbook may use different column names. Inspect its header and extend `ALIASES` in `volcano/pipeline.py` when necessary. Missing and uncertain dates should be reviewed before analysis. No synthetic eruption records are bundled.
