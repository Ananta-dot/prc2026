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

From v7 on, the submitted file is a blend of several `stage2.py` runs with the pairing rules
(see "What the data showed") enforced on top:

```
python3 link.py ../data/processed/compact.parquet ../data/processed/link.parquet
python3 tzcand.py ../data/processed/compact.parquet ../data/processed/tz.parquet
python3 dfl_rate.py ../data/processed/compact.parquet ../data/processed/dfl.parquet
python3 build.py --compact ../data/processed/compact.parquet --link ../data/processed/link.parquet \
    --tz ../data/processed/tz.parquet --dflt ../data/processed/dfl.parquet \
    --template ../data/raw/submitting.parquet --out ../outputs/courageous-onion_v10.parquet \
    --parts ../outputs/parts_a.npz,../outputs/parts_b.npz,../outputs/parts_c.npz \
    --extra ../outputs/final_rules.parquet:3
```

`--extra file:weight` averages in another prediction file for the rows it shares (here the
final-phase runs, which cover the same January and July flights).

Each `stage2.py` run saves `parts_<tag>.npz` (regressor, class probabilities, prediction) next
to its submission file; `build.py` applies the rules to every run and averages them.

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
    --template ../data/raw/final_submitting.parquet --out ../outputs --name fin_seed11 \
    --stages 3 --seed 11 --lr 0.05 --rounds 2420,300,300
python3 stage2.py --feat ../data/processed/feat_final.parquet --compact ../data/processed/compact_final.parquet \
    --template ../data/raw/final_submitting.parquet --out ../outputs --name fin_seed7 \
    --stages 3 --seed 7 --lr 0.05 --rounds 2681,300,851
python3 upload.py put ../outputs/courageous-onion_final.parquet
```

`--rounds` (regressor, gate, push-delay model) skips the validation pass and uses the
round counts found on the held-out months. Further runs use `--seed 23`, `31` and `47` with
`--rounds 2550,300,575`. The final file is then built like the leaderboard files:

```
python3 link.py ../data/processed/compact_final.parquet ../data/processed/link_final.parquet
python3 tzcand.py ../data/processed/compact_final.parquet ../data/processed/tz_final.parquet
python3 dfl_rate.py ../data/processed/compact_final.parquet ../data/processed/dfl_final.parquet
python3 build.py --compact ../data/processed/compact_final.parquet --link ../data/processed/link_final.parquet \
    --tz ../data/processed/tz_final.parquet --dflt ../data/processed/dfl_final.parquet \
    --template ../data/raw/final_submitting.parquet --out ../outputs/courageous-onion_final.parquet \
    --parts ../outputs/parts_fin7.npz,../outputs/parts_fin11.npz,../outputs/parts_fin23.npz,../outputs/parts_fin31.npz,../outputs/parts_fin47.npz \
    --extra ../outputs/lb_rules.parquet:3
python3 upload.py put ../outputs/courageous-onion_final.parquet --replace
```

`lb_rules.parquet` is the rule-corrected mean of the three leaderboard-set runs (the same
`build.py` call as for v10 without `--extra`); it covers the January and July flights, which
are in both ranking sets. See `outputs/report_final.txt`.

From v11 on (`pipeline/run_prod.sh lb|final` does all of it): each `stage2.py` run takes
`--dump-all ../outputs/prod/mat_<set>` for the first seed, three `nn.py --final` runs (seeds 7,
23, 31) train on that matrix, and `build.py` takes `--nn <mean of the three>.npz:0.35`. On a
machine where SciPy no longer loads (macOS 27 with the Python 3.10 wheels), use a fresh
environment: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy
pandas==2.3.3 pyarrow lightgbm==4.7.0 scipy torch minio`. A validation run takes about 8 minutes
on an M4 Max (14 cores), an MLP about 90 s on its GPU.

The final file now holds seven runs on the final ranking set (seeds 7, 11, 23, 31, 47, 59, 71)
with the MLP blend, and the five leaderboard-set runs of v12 (weight 5) for January and July.

## Layout

| Path | What it is |
|---|---|
| `pipeline/prep.py` | raw monthly parquet files to one compact frame |
| `pipeline/features.py` | one row per departure (training + ranking); uses no departure off-block time and no target |
| `pipeline/get_weather.py`, `pipeline/weather.py` | download METAR observations; turn them into snow, frost, thunderstorm, visibility and wind features |
| `pipeline/model.py` | out-of-fold target encodings, LightGBM helpers, validation report |
| `pipeline/train.py` | single model: validation, final fit, submission file |
| `pipeline/stage2.py` | the model: out-of-fold light passes, surface-state features from the estimated off-block times, a gate for late flights whose off-block time is near the schedule, rules for the rarest record types, final regressor |
| `pipeline/link.py` | flight plans of departures without NM columns, recovered from the arrival record at the destination |
| `pipeline/posthoc.py` | the pairing rules: taxi time inside (or outside) one hour of take-off minus the plan's off-block time |
| `pipeline/tzcand.py` | Rome departures whose off-block time is likely shifted by the UTC offset |
| `pipeline/dfl_rate.py` | Rome flight numbers and airlines whose off-block time is (nearly) always the schedule |
| `pipeline/build.py` | blend runs of `stage2.py` (and, with `--nn`, the MLP) and enforce the rules; writes the submission file |
| `pipeline/nn.py` | MLP on the matrix saved by `stage2.py --dump-all`, trained on the GPU (Apple MPS or CUDA); blended into the regressor |
| `pipeline/exp.py`, `pipeline/postval.py` | experiments on the saved matrix; validation scores with the rules applied |
| `pipeline/run_prod.sh` | the production runs behind v11 and the final file |
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
- **How movements were paired with Network Manager flights.** A departure carries NM
  columns only if its off-block time is within one hour of the flight plan's last filed
  off-block time (`LOBT_flt`): over the 2.06 million matched departures of 2025 the
  difference runs from -3606 s to +3606 s (one hour plus the 6 s noise) and stops dead.
  Arrivals are paired the same way on landing time against `ARVT_3_flt`. Two things follow.
  - *Matched departure:* the taxi time lies within 3606 s of take-off minus LOBT. The
    model's blend of "ordinary taxi time" and "off-block at schedule" often falls outside
    that window, or one of the two is impossible; `posthoc.py` drops the impossible part
    and clips the rest. When the plan is so old that even the lower end of the window is
    above an ordinary taxi time, the off-block time is known to be near the plan's.
  - *Unmatched departure:* the taxi time lies outside that window, but LOBT is not given.
    When the destination is another of the ten airports, the same flight's arrival record
    is matched there (on landing time) and carries the plan; `link.py` finds it for about
    90% of such flights. If the schedule is within the hour of LOBT, an off-block time at
    schedule would have been paired, so the record is an ordinary one (1,764 of 1,764
    cases in 2025). If LOBT is within about an hour of take-off, an ordinary off-block time
    would have been paired, so the off-block time is the schedule (25 of 27 at Rome) or
    just over an hour before the plan (Istanbul).
  - *Rome, off-block time two hours early.* Some Rome off-block times were entered in UTC
    and stored as local time. In summer the shift is two hours, which always breaks the
    pairing. So a Rome departure whose flight number is normally paired, which was not
    late and yet has no NM columns, is such a record: of 25 in summer 2025, 18 have a
    "taxi time" of an ordinary taxi time plus 7200 s and 4 more are longer still. The
    regressor had learned an average of about 5,000 s for this group; `posthoc.py` raises
    the summer ones to 7,900 s. (In winter the shift is one hour and the regressor's value
    is already about right.) Other airports show nothing of the kind.
  - *Rome, always the schedule.* For some airlines at Rome the off-block time equals the
    schedule on every flight, so the taxi time is exactly take-off minus schedule. A tree
    model only approximates that identity (RMSE about 300 s on those flights, against 4 s
    for the identity). `posthoc.py` predicts take-off minus schedule when the flight
    number's 2025 rate is 99% or more, and leans towards it from 90%.
  On held-out months these rules alone, applied to unchanged models, lower the RMSE from
  244.59 to 240.34 (January+July 2025, without the two Paris records below) and from
  241.53 to 239.57 (February+August); on six Istanbul snow days from 265.16 to 262.88.
- A handful of records match nothing (for example two Paris departures in January 2025
  with "taxi times" of 16 h and 23 h and nothing unusual in any other field). They are
  not predicted.

## Things that did not help

Each was checked on two held-out month pairs (January+July and February+August 2025):

- features linking a departure to the same airline's previous arrival on its stand (an
  arrival whose in-block time is a default predicts a schedule-default off-block time,
  72% against 29% at Rome, but the gate already had that information);
- a separate gate for Rome, and a separate regressor for late flights;
- other thresholds for the near-schedule class (20 to 90 minutes);
- arrival taxi-in times as a signal of year-on-year drift in taxi-out times;
- pairing-window features and airline-level shares of unmatched late flights as inputs to
  the gate (the rules above already take what there is; the gate did not improve);
- a small separate model for departures with an old flight plan (worse than the rule);
- a second-stage gate that sees the first gate's probabilities for other late flights at
  the airport around the same time, event features counting matched neighbours whose
  window certifies a long taxi time (strongly related to the class, 10 to 40 times the
  base rate, but the gate did not improve), a schedule-default rate computed only on
  clearly delayed flights, and day-level shares of flight plans left at the schedule;
- a physical lower bound from the next aircraft parking on the same stand (holds for 98%
  of cases when that happens within 20 minutes of take-off, but not beyond: stand codes
  are shared or aircraft are towed);
- the first-seen time of the aircraft in OpenSky's public flights API as a proxy for the
  off-block time: at Heathrow, Schiphol and Rome it is the take-off moment (median gap
  -34 s, +15 s, -19 s); at Zurich it is on the ground but adding it made a held-out day
  worse (RMSE 191 against 157). `pipeline/opensky_probe.py` is the probe; no OpenSky data
  is used in any submission;
- a gate for schedule-default off-block times on flights less than an hour late (Rome:
  27% of flights 10 to 30 minutes late), with a regressor fit without them: under 1 s;
- a recalibration of the gate (shift and temperature in logits): about 1 s, within noise;
- the share of arrivals around the schedule whose in-block time equals their schedule, as a
  sign of a system-wide default (Rome: 17.8% against 23.0% across its deciles);
- the rate at which a flight number is paired with an NM flight when on time (almost every
  flight number is paired, so it does not separate the late unmatched classes);
- 255 or 511 leaves for the final regressor;
- a larger gate for flights 4 to 8 h late: on Jan+Jul 2025 the gate is far too low there
  (predicted 0.23, actual 0.77) and +2 logits gives 238.5 -> 234.7, but on Feb+Aug the same
  shift costs 2.7 (228.6 -> 231.2): the January/July shortfall came from a few event days;
- a global +0.25 logit shift of the gate: better on three 2025 validation runs (-0.4 to -0.9)
  but worse on the leaderboard (v13 against v14, about +1.1).

## Same-day rule (`pipeline/calib.py`)

An off-block time "with the schedule's date" is a day early only when the take-off is on a later
local day than the schedule: 12 of 12 Rome records of that kind in 2025. For a take-off on the
schedule's local day the day-shift class is impossible, and its probability goes to the
near-schedule class. Feb+Aug 2025: 228.57 -> 227.62; leaderboard v12 -> v14: 255.85 -> 254.86
(one row, predicted 66,751 s, now 53,619 s). `build.py` applies it (and `--gate-shift`, 0 by default).

## External data

Airport weather observations (METAR) from the Iowa Environmental Mesonet ASOS archive,
https://mesonet.agron.iastate.edu/request/download.phtml, fetched by `pipeline/get_weather.py`.

From v15 on: OpenSky Network ground positions (ADS-B state vectors on the ground, historical
database via Trino, https://opensky-network.org/data/trino), fetched by `pipeline/ground_fetch.py`
(needs an OpenSky account with access to the historical database; one login per run).

## OpenSky ground tracks (v15 on)

`ground_fetch.py` downloads, for each airport and UTC day, the ground positions inside a box around
the airport, one point per 10 s per aircraft (about 5 s per airport-day). Ground coverage near the
stands exists at Zurich, Amsterdam, Heathrow, Munich and Barcelona; at Madrid, Istanbul and Rome
there is almost none, and at Paris CDG and Frankfurt it covers few departures, so only the first
five are fetched. `ground.py` matches each departure to its track on the callsign and the take-off
time and searches back from the take-off for the stand (the last stop of 20 min or more within 50 m,
or a gap of 15 min or more: transponder off). Only at Zurich does the first movement give the
off-block time itself (56% within 60 s); elsewhere the aircraft are first seen partway along the
taxi, but the observed part still carries the queue the model cannot see. Seven features
(`g_*`, see `ground.py`) are appended to the feature file by `ground_feats.py`:

```
python3 ground_fetch.py 2026-01-01:2026-03-01,2026-06-01:2026-08-01,2025-01-01:2026-01-01 LSZH EHAM EGLL EDDM LEBL
python3 ground_feats.py ../data/processed/compact.parquet ../data/processed/feat.parquet ../data/processed/feat_g.parquet
python3 ground_feats.py ../data/processed/compact_final.parquet ../data/processed/feat_final.parquet ../data/processed/feat_final_g.parquet
OUT=../outputs/prod_g G=_g ./run_prod.sh lb && OUT=../outputs/prod_g G=_g ./run_prod.sh final
```

Validation (seed 7, rules applied, without the two Paris records): Jan+Jul 2025 239.34 -> 235.19,
Feb+Aug 2025 228.57 -> 223.88. Rows up to 1 h at Zurich 161 -> 115 (Jan+Jul) and 150 -> 102
(Feb+Aug), Amsterdam 159 -> 137 and 147 -> 133, Barcelona 209 -> 199 and 197 -> 184.

## Results

Validation: January and July 2025 held out (344,419 departures), the rest of 2025 for training.

| Version | Model | Validation RMSE, rows with taxi time <= 1 h | Validation RMSE, all rows | Leaderboard |
|---|---|---|---|---|
| v1 | `train.py`, trained on rows <= 3600 s | 205.87 | about 565 | 538.92 |
| v2 | two-stage, trained on rows <= 3600 s | 199.29 | about 563 | 534.77 |
| v3 | `stage2.py --stages 3`: weather, gate for near-schedule off-block times, record-type rules | 206.69 | 304.31 | 264.31 |
| v4 | mean of two `stage2.py --stages 3` runs (seed 7 / lr 0.08 and seed 11 / lr 0.05), refined rules for flights more than 8 h late | 203.0 | 299.8 | 262.69 |
| v5 | mean of two `stage2.py --stages 3 --lr 0.05` runs (seed 7 with the per-airport gate correction, seed 11 without) | 203.4 | 297.9 | 262.61 |
| v6 | as v5 with a third run (seed 23) | - | - | 261.73 |
| v7 | v6 with the pairing rules of `posthoc.py` (window for matched departures, linked flight plans for unmatched ones) | 206.5 | 296.1 | (see below) |
| v8 | v7 averaged with the three final-phase runs on the same flights | - | - | |
| v9 | v8 with the Rome summer-time rule | 206.5 | 295.9 | |
| v10 | v9 with the Rome always-schedule rule | 206.3 | 295.7 | 257.51 |
| v11 | three fresh `stage2.py` runs (seeds 7, 11, 23) with the regressor blended 65/35 with three `nn.py` MLPs, rules, final-phase runs blended in | 202.7 | 294.1 | 256.20 |
| v12 | v11 with two more runs (seeds 31, 47) | - | - | 255.85 |
| v13 | v12 with the gate shifted up by 0.25 logits and the same-day rule (`calib.py`) | - | - | 255.94 |
| v14 | v12 with the same-day rule only (one row changes) | - | - | 254.86 |

The validation figures for v7, v9 and v10 are the seed-7 run of v5 (299.2 on all rows) with the
rules applied, since the rules need no retraining. The v11 figures are the mean of the seed-7 and
seed-11 validation runs with the MLP blend and the rules (`postval.py`); without the two Paris
records: 239.34 (seed 7, rules), 238.43 (seeds 7 + 11), 238.32 (with the MLP).

Two of the held-out rows (the Paris records above) account for 173 of the 304 in quadrature; without
them the all-rows figure is 250.83 for v3, 245.3 for v4, 243.0 for v5 and, for the seed-7 run,
244.6 before the rules and 240.4 after all of them. The first two versions were built on the wrong
assumption that extreme rows were not scored.

v3 was written by the run reported in `outputs/report_v3.txt`; the rules for unmatched
flights more than 8 h late were refined afterwards and applied to 22 rows of that file.
Those rules are now part of `stage2.py`, so a fresh run includes them.

Reports for each version are in `outputs/report_v*.txt`.
