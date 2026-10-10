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
    --team courageous-onion --version 3 --stages 3
python3 upload.py put ../outputs/courageous-onion_v3.parquet
python3 leaderboard.py           # the team's scores from the official leaderboard API
```

`stage2.py` holds out January and July 2025, prints the validation RMSE over every held-out
row, then refits on all of 2025 and writes `<team>_v<version>.parquet` in the template's
row order. `--threads 0` (default) uses every core. On a machine with less than about 8 GB
free, add `--mm ../data/processed/park.npy`: the feature matrix is parked in that file while
models train. `train.py` is the earlier single model (version 1).

## Final phase

The final ranking uses `final_ranking.parquet` (January, February, June and July 2026)
and one file named `<team>_final.parquet`:

```
python3 upload.py pull
python3 prep.py ../data/raw ../data/processed/compact_final.parquet final_ranking.parquet
python3 features.py ../data/processed/compact_final.parquet ../data/processed/feat_final.parquet ../data/external
python3 stage2.py --feat ../data/processed/feat_final.parquet --compact ../data/processed/compact_final.parquet \
    --template ../data/raw/final_submitting.parquet --out ../outputs --name courageous-onion_final \
    --stages 3 --seed 11 --lr 0.05 --rounds 2420,300,300
python3 upload.py put ../outputs/courageous-onion_final.parquet
```

`--rounds` (regressor, gate, push-delay model) skips the validation pass and uses the
round counts found on the held-out months.

## Layout

| Path | What it is |
|---|---|
| `pipeline/prep.py` | raw monthly parquet files to one compact frame |
| `pipeline/features.py` | one row per departure (training + ranking); uses no departure off-block time and no target |
| `pipeline/get_weather.py`, `pipeline/weather.py` | download METAR observations; turn them into snow, frost, thunderstorm, visibility and wind features |
| `pipeline/model.py` | out-of-fold target encodings, LightGBM helpers, validation report |
| `pipeline/train.py` | single model: validation, final fit, submission file |
| `pipeline/stage2.py` | the model: out-of-fold light passes, surface-state features from the estimated off-block times, a gate for late flights whose off-block time is near the schedule, rules for the rarest record types, final regressor |
| `pipeline/upload.py` | list / download / upload against the OpenSky object store |
| `pipeline/leaderboard.py` | read the team's scores from the official leaderboard API |
| `outputs/report_v*.txt` | validation report for each submitted version |
| `src/` | the earlier starter pipeline, kept for reference; not used by `pipeline/` |

## What the data showed

- Timestamps carry about ±6 s of added noise.
- For roughly a fifth of departures the recorded off-block time equals the Network
  Manager's `AOBT_3_flt` to the minute, so the taxi time is take-off minus that value.
- Every row is scored, and the score is dominated by a few hundred records. Taxi times
  above one hour are 0.2% of departures; in the held-out months about 1,000 such rows
  carry almost 90% of the squared error of a model that ignores them.
- Most of those are not taxi times but records of a known form:
  - **Off-block time equal to the schedule** (18% of Rome departures; elsewhere rare).
    For a flight that took off hours late the "taxi time" is then take-off minus schedule.
    Among Rome flights with no matched Network Manager record that took off on a later
    UTC day than scheduled, 44 of 46 are of this kind.
  - **Off-block time with the schedule's date** for flights delayed past about 13.5 h:
    the taxi time is 24 h plus a normal taxi time (13 records in 2025, 12 at Rome).
  - **Real long holds** after an on-time push-back (Heathrow, Paris CDG, Istanbul in
    snow): taxi time is take-off minus schedule minus a short push delay.
- How `stage2.py` uses this: for flights that took off more than an hour after schedule,
  a classifier (the gate) estimates the probability that the off-block time is within
  45 minutes of the schedule, a small regressor estimates that push delay, and the
  prediction is the probability-weighted mean of "take-off minus schedule minus delay"
  and an ordinary regressor. Class shares from the training year replace the gate for
  the record types with too few cases to learn (the date-shifted ones, the UTC-day rule).
- Long holds come in events (snow days), and a gate under-states an event it has not
  seen. Out-of-fold gate predictions measure that per airport, and the gate is shifted
  upwards by that amount (at most 0.7 logits). On six held-out Istanbul snow days this
  lowers the RMSE from 276.7 to 271.2; on ordinary months it changes little.
- A handful of records match nothing (for example two Paris departures in January 2025
  with "taxi times" of 16 h and 23 h and nothing unusual in any other field). They are
  not predicted.

## External data

Airport weather observations (METAR) from the Iowa Environmental Mesonet ASOS archive,
https://mesonet.agron.iastate.edu/request/download.phtml, fetched by `pipeline/get_weather.py`.
No other external data is used.

## Results

Validation: January and July 2025 held out (344,419 departures), the rest of 2025 for training.

| Version | Model | Validation RMSE, rows with taxi time <= 1 h | Validation RMSE, all rows | Leaderboard |
|---|---|---|---|---|
| v1 | `train.py`, trained on rows <= 3600 s | 205.87 | about 565 | 538.92 |
| v2 | two-stage, trained on rows <= 3600 s | 199.29 | about 563 | 534.77 |
| v3 | `stage2.py --stages 3`: weather, gate for near-schedule off-block times, record-type rules | 206.69 | 304.31 | 264.31 |
| v4 | mean of two `stage2.py --stages 3` runs (seed 7 / lr 0.08 and seed 11 / lr 0.05), refined rules for flights more than 8 h late | 203.0 | 299.8 | 262.69 |
| v5 | mean of two `stage2.py --stages 3 --lr 0.05` runs (seed 7 with the per-airport gate correction, seed 11 without) | 203.4 | 297.9 | 262.61 |

Two of the held-out rows (the Paris records above) account for 173 of the 304 in quadrature; without
them the all-rows figure is 250.83 for v3, 245.3 for v4 and 243.0 for v5. The first two versions were built on the wrong
assumption that extreme rows were not scored.

v3 was written by the run reported in `outputs/report_v3.txt`; the rules for unmatched
flights more than 8 h late were refined afterwards and applied to 22 rows of that file.
Those rules are now part of `stage2.py`, so a fresh run includes them.

Reports for each version are in `outputs/report_v*.txt`.
