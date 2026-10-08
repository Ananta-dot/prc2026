# PRC 2026 Starter Pack

A minimal pipeline to go from the raw parquet files you downloaded to a
submittable predictions file. It's deliberately schema-agnostic: the
2026 bucket layout (`training_*.parquet`, `ranking.parquet`,
`submitting.parquet`) differs from the 2024 challenge docs, so step 1
is always "inspect the actual columns" rather than assuming them.

## Layout

```
prc2026-starter/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # put your `mc cp` parquet files here
│   └── processed/      # optional: any cached/cleaned data you build later
├── outputs/            # model.txt and submission.csv land here
├── notebooks/          # optional: EDA notebooks
└── src/
    ├── config.py       # paths + column-name assumptions
    ├── inspect_data.py # run FIRST: prints schema/dtypes/sample rows
    ├── data.py         # loads + concatenates monthly training files
    ├── train.py        # trains a baseline LightGBM regressor, reports val RMSE
    └── predict.py       # scores submitting.parquet, writes submission csv
```

`config.py` already points `DATA_DIR` at `data/raw/` and outputs at
`outputs/` relative to the project root — no path editing needed if
you keep this structure. Just drop your parquet files into `data/raw/`.

## Workflow

1. `pip install -r requirements.txt`
2. Put the parquet files from `mc cp` into `data/raw/`.
3. `python src/inspect_data.py`
   This prints column names/dtypes/null counts/sample rows for
   `training_2025-01-01_2025-02-01.parquet`, `ranking.parquet`, and
   `submitting.parquet`. **Read this output before going further** —
   `train.py` and `predict.py` both have a `TARGET_COL` /
   `ID_COL` constant at the top you may need to correct once you see
   the real column names (I've guessed `tow` and `flight_id` based on
   the 2024 docs, which may or may not still hold).
4. `python src/train.py`
   Loads all monthly training files, does a quick train/validation
   split, trains a LightGBM baseline, prints validation RMSE, and
   saves the model to `model.txt`.
5. `python src/predict.py`
   Loads `submitting.parquet`, applies the same feature pipeline,
   predicts, and writes `submission.csv` in `flight_id,tow` format —
   matching the upload convention from the challenge docs
   (`mc cp submission.csv dc24/<submissions-bucket>/<team_name>_v<num>_<team_id>.csv`).
   **Check whether the 2026 submissions bucket name/convention has
   changed** — `mc ls dc24` again to confirm before uploading.

## Known unknowns (check these before trusting the baseline)

- Exact column names/target — confirmed only by running `inspect_data.py`.
- Whether `ranking.parquet` has labels (a local validation set) or is
  purely features (like 2024's `challenge_set.csv`) — the script treats
  it as unlabeled by default.
- Categorical columns are auto-detected and target-encoded; if there
  are high-cardinality ID-like columns (airport codes, aircraft
  registration) that shouldn't be dropped as "IDs," you may need to
  adjust `CATEGORICAL_OVERRIDE` in `config.py`.
