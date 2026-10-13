# prc2026

Team **courageous-onion**, EUROCONTROL [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/):
predict the taxi-out time (seconds) of departures at ten European airports in 2026, trained on 2025.
Scored by RMSE over every departure. Leaderboard (January and July 2026): **250.48**.

Licence: GPL-3.0-or-later (see `LICENSE`).

## Approach

- **Features** (`features.py`): one row per departure from the schedule, flight plan (NM), runway, stand,
  airline and aircraft; out-of-fold target encodings; airport traffic around the take-off; METAR weather.
- **OpenSky ground tracks** (`ground_fetch.py`, `ground.py`): ADS-B positions on the ground at Zurich,
  Amsterdam, Heathrow, Munich and Barcelona (the airports with stand-level coverage). Per departure: the
  off-block time seen in the track, path length, stops, time since the last stop; per airport: aircraft
  taxiing and holding before the take-off.
- **Model** (`stage2.py`, LightGBM):
  1. two out-of-fold passes predict every taxi time; surface-state features (aircraft taxiing, queue at the
     runway) are rebuilt from the estimated off-block times;
  2. for flights more than an hour late, a classifier (the gate) estimates whether the recorded off-block
     time is the schedule (a recording default, or a real long hold) and a regressor the push delay;
  3. a regressor for ordinary taxi times, blended 70/30 with three MLPs (`nn.py`, GPU);
  4. the prediction is the probability-weighted mixture of the two.
- **Rules** (`posthoc.py`, `calib.py`), from how movements were paired with NM flight plans: a departure
  carries NM columns only if its off-block time is within an hour of the plan, which bounds the taxi time
  (matched flights) or excludes a range (unmatched ones, plan recovered from the arrival record, `link.py`);
  Rome records whose off-block time is the schedule, shifted by the UTC offset, or carries the schedule's
  date (impossible for a take-off on the schedule's local day).
- **Ensemble** (`build.py`): rules applied to each of 10 (leaderboard) or 12 (final) seeds, then averaged;
  January and July flights averaged across both ranking sets.

Validation: January+July 2025 held out (February+August as a second check), the rest of 2025 for training.

## Replicate

Needs Python 3.12, an OpenSky Network account with access keys for the challenge bucket (`credentials.json`
in the repository root) and access to the OpenSky historical database (Trino; one browser login per
download run).

```
git clone https://github.com/Ananta-dot/prc2026.git && cd prc2026
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cd pipeline
PY=../.venv/bin/python; P=../data/processed

# data
$PY upload.py pull                                          # challenge files -> ../data/raw
$PY get_weather.py                                          # METAR -> ../data/external
$PY ground_fetch.py 2026-01-01:2026-03-01,2026-06-01:2026-08-01,2025-01-01:2026-01-01 LSZH EHAM EGLL EDDM LEBL

# features and rule inputs, for the leaderboard set ("") and the final set ("_final")
$PY prep.py ../data/raw $P/compact.parquet
$PY prep.py ../data/raw $P/compact_final.parquet final_ranking.parquet
for s in "" _final; do
  $PY features.py $P/compact$s.parquet $P/feat$s.parquet ../data/external
  $PY ground_feats.py $P/compact$s.parquet $P/feat$s.parquet $P/feat${s}_g2.parquet
  $PY link.py $P/compact$s.parquet $P/link$s.parquet
  $PY tzcand.py $P/compact$s.parquet $P/tz$s.parquet
  $PY dfl_rate.py $P/compact$s.parquet $P/dfl$s.parquet
done

# models: 10 + 12 LightGBM runs and the MLPs (about 6 h on a 14-core Mac)
export OUT=../outputs/prod_g2 G=_g2 LR=0.03 RR=5500,300,267
./run_prod.sh lb && ./run_prod.sh final

# submission files
O=$OUT; LB=$(ls $O/parts_lb*.npz | paste -sd, -); FIN=$(ls $O/parts_fin*.npz | paste -sd, -)
$PY build.py --compact $P/compact.parquet --link $P/link.parquet --tz $P/tz.parquet --dflt $P/dfl.parquet \
    --template ../data/raw/submitting.parquet --parts $LB --nn $O/nn_lb_avg.npz:0.3 --out $O/lb_rules.parquet
$PY build.py --compact $P/compact_final.parquet --link $P/link_final.parquet --tz $P/tz_final.parquet --dflt $P/dfl_final.parquet \
    --template ../data/raw/final_submitting.parquet --parts $FIN --nn $O/nn_fin_avg.npz:0.3 \
    --extra $O/lb_rules.parquet:10 --out ../outputs/courageous-onion_final.parquet
$PY upload.py put ../outputs/courageous-onion_final.parquet --replace
```

The leaderboard file: the second `build.py` command without `--extra` and with `--out $O/fin_rules.parquet`,
then the first one with `--extra $O/fin_rules.parquet:12` and `--out ../outputs/courageous-onion_v20.parquet`.
To validate a change: `stage2.py ... --skip-final --val-months 1,7` (or `2,8`), then
`postval.py --val ../outputs/exp/val_<tag>.npz`.

## External data

- METAR observations, Iowa Environmental Mesonet ASOS archive (https://mesonet.agron.iastate.edu/request/download.phtml).
- OpenSky Network ground positions (historical database, https://opensky-network.org/data/trino).
