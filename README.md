# prc2026

Team **courageous-onion**'s entry for the EUROCONTROL
[PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/): predict the
taxi-out time (seconds) of departures at 10 major European airports for January and
July 2026, training on all 2025 movements. Scored by RMSE.

Licence: GPL-3.0-or-later (see `LICENSE`).

## Set up on a new machine

```
git clone https://github.com/Ananta-dot/prc2026.git && cd prc2026
pip3 install -r requirements.txt
# copy credentials.json (the OpenSky access key file) into this folder; it is git-ignored
cd pipeline
python3 upload.py pull          # downloads the challenge files into ../data/raw
```

## Run

From `pipeline/`:

```
python3 get_weather.py           # METAR observations into ../data/external (open data)
python3 prep.py ../data/raw ../data/processed/compact.parquet
python3 features.py ../data/processed/compact.parquet ../data/processed/feat.parquet ../data/external
python3 stage2.py --feat ../data/processed/feat.parquet --compact ../data/processed/compact.parquet \
    --template ../data/raw/submitting.parquet --out ../outputs \
    --team courageous-onion --version 3 --stages 3 --cap 5400
python3 upload.py put ../outputs/courageous-onion_v3.parquet
```

Both `train.py` (single model) and `stage2.py` (two-stage model) hold out January and
July 2025, print the validation RMSE, then refit on all of 2025 and write
`<team>_v<version>.parquet` in the template's row order.
`--cap` keeps training rows whose taxi time is at most that many seconds; `--winsor`
clips instead of dropping; `--threads 0` (default) uses every core.

## Layout

| Path | What it is |
|---|---|
| `pipeline/prep.py` | raw monthly parquet files to one compact frame |
| `pipeline/features.py` | one row per departure (training + ranking); uses no departure off-block time and no target |
| `pipeline/get_weather.py`, `pipeline/weather.py` | download METAR observations; turn them into snow, frost, thunderstorm, visibility and wind features |
| `pipeline/model.py` | out-of-fold target encodings, LightGBM helpers, validation report |
| `pipeline/train.py` | single model: validation, final fit, submission file |
| `pipeline/stage2.py` | two-stage model: out-of-fold first pass, surface-state features from the estimated off-block times, final model |
| `pipeline/upload.py` | list / download / upload against the OpenSky object store |
| `outputs/report_v*.txt` | validation report for each submitted version |
| `src/` | the earlier starter pipeline, kept for reference; not used by `pipeline/` |

## What the data showed

- Timestamps carry about ±6 s of added noise.
- For roughly a fifth of departures the recorded off-block time equals the Network
  Manager's `AOBT_3_flt` to the minute, so the taxi time is take-off minus that value.
- At Rome (LIRF) about 17% of off-block times equal the scheduled time, including most
  of the hours-long "taxi times".
- Rows above one hour are 0.2% of departures but more than half of the squared error
  against an airport-mean baseline, so which rows are scored matters a great deal.

## External data

Airport weather observations (METAR) from the Iowa Environmental Mesonet ASOS archive,
https://mesonet.agron.iastate.edu/request/download.phtml, fetched by `pipeline/get_weather.py`.
No other external data is used.

## Results

| Version | Trained on | Validation RMSE, Jan+Jul 2025, rows with true taxi time <= 1 h |
|---|---|---|
| v1 | `train.py`, rows <= 3600 s, 194 features, LightGBM | 205.87 |
| v2 | `stage2.py` (two-stage, before the extra push-back-order features were added), rows <= 3600 s | 199.29 |
| - | `stage2.py --stages 3` with weather, rows <= 3600 s (not submitted) | 195.85 |

Validation for other cut-offs is in `outputs/report_v*.txt`.
