# GOTTA variability and classification analysis

Versioned analysis code and derived validation products supporting the RAA manuscript on variability extraction and source-level classification with the GOTTA Prototype data.

## Contents

- `scripts/`: analysis, validation, time-system sensitivity, and figure-generation scripts.
- `derived_data/`: source splits, predictions, corrected features, GP held-out products, statistical summaries, and figures used in the revised analysis.
- `docs/DATA_AVAILABILITY.md`: distinction between restricted raw GOTTA photometry and public code/derived products.
- `requirements.txt`: recorded Python dependencies.
- `CITATION.cff`: citation metadata; replace the GitHub username before publication.

## Quick reproduction from public derived products

Create an environment and install the recorded dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Reconstruct the GP validation summaries without raw GOTTA photometry:

```bash
python scripts/reproduce_gp_validation_summary.py \
  --output-dir rerun_summary
```

The script reads the point-level predictions and occupancy products in `derived_data/` and regenerates the aggregate table, class-resolved coverage, paired source bootstrap, and diagnostic figure.

## Full reruns with restricted inputs

Raw GOTTA photometry is not distributed. Authorized collaboration members can provide local paths without editing the scripts:

```bash
export GOTTA_LIGHT_CURVES=/path/to/gotta/light_curves
export GOTTA_MATCH_TABLE=/path/to/match.csv
export ZTF_LIGHT_CURVES=/path/to/ztf/light_curves
```

For example, rerun the fixed GP validation with:

```bash
python scripts/run_gp_validation_from_raw.py \
  --light-curves "$GOTTA_LIGHT_CURVES" \
  --match-table "$GOTTA_MATCH_TABLE"
```

Run the time-system sensitivity analysis with:

```bash
python scripts/time_system_sensitivity.py
```

## Reproducibility record

The recorded environment and random seeds are in `derived_data/software_versions_and_seeds.json`. The primary classification seed and GP validation seed are both 1; repeated source partitions use seeds 1 through 10.

Release tag: `v1.0.0`  
Release URL: `https://github.com/zhangajingyi/gotta-variability-classification/releases/tag/v1.0.0`  
Commit: replace with the output of `git rev-parse HEAD` after the release commit.

## Data policy

Raw GOTTA Prototype photometry cannot be publicly released or redistributed under the collaboration data policy. The restriction does not apply to the analysis code and the derived validation products included here. See `docs/DATA_AVAILABILITY.md`.

## License

A reuse license must be approved by the collaboration before public release. See `docs/LICENSE_NOTE.md`.
