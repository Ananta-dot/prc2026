# SPDX-License-Identifier: GPL-3.0-or-later
"""Blend runs of stage2.py and enforce the pairing rules (posthoc.py).

usage: python3 build.py --compact C --link L --template T --parts a.npz,b.npz,... --out file.parquet [--extra other.parquet:weight]

Each parts file is what a stage2.py run saves (id, r, P, p).  The rules are applied to each run,
the runs are averaged, and rows whose plain average is above 4 h come from the first run.
`--extra` blends in another prediction file for the rows it shares (weight = number of runs it stands for).
"""
import argparse, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import posthoc

ap = argparse.ArgumentParser()
ap.add_argument("--compact", required=True); ap.add_argument("--link", required=True); ap.add_argument("--template", required=True)
ap.add_argument("--parts", required=True); ap.add_argument("--out", required=True); ap.add_argument("--extra", default="")
ap.add_argument("--no-rules", action="store_true")
ap.add_argument("--tz", default="", help="candidates from tzcand.py (Rome records shifted by the UTC offset)")
ap.add_argument("--dflt", default="", help="rates from dfl_rate.py (Rome flight numbers that are always schedule-default)")
a = ap.parse_args()
t = pd.read_parquet(a.template)
c = pd.read_parquet(a.compact, columns=["MVT_ID_mvt", "APT", "t_MVT_TIME_UTC_mvt", "t_SCHED_TIME_UTC_mvt", "t_LOBT_flt", "t_AOBT_3_flt", "is_rank", "is_dep"])
c = c[c.is_rank & c.is_dep].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").reindex(t.MVT_ID_mvt.to_numpy())
lk = pd.read_parquet(a.link).set_index("MVT_ID_mvt").lk_LOBT.reindex(c.index).to_numpy()
T, S, LOBT, AOBT, lirf = c.t_MVT_TIME_UTC_mvt.to_numpy(), c.t_SCHED_TIME_UTC_mvt.to_numpy(), c.t_LOBT_flt.to_numpy(), c.t_AOBT_3_flt.to_numpy(), (c.APT == "LIRF").to_numpy()
tzs = None
if a.tz:
    tz = pd.read_parquet(a.tz); tzs = pd.Series(True, index=tz.MVT_ID_mvt[tz.off == 7200]).reindex(c.index).fillna(False).to_numpy(bool)
dq = pd.read_parquet(a.dflt).set_index("MVT_ID_mvt").q.reindex(c.index).to_numpy() if a.dflt else None
raw, new = [], []
for f in a.parts.split(","):
    z = np.load(f); o = pd.Series(np.arange(len(z["id"])), index=z["id"]).reindex(c.index).to_numpy()
    p, r, P = z["p"][o], z["r"][o], z["P"][o]
    st = {}
    raw.append(p); new.append(p if a.no_rules else posthoc.apply(p, r, P[:, 1], P[:, 2], T, S, LOBT, AOBT, lk, lirf, st, tzs, dq))
    print(f"{os.path.basename(f)}: {st}")
m0 = np.mean(raw, axis=0); far = m0 > 14400
out = np.where(far, new[0], np.mean(new, axis=0)); base = np.where(far, raw[0], m0)
n = len(new)
if a.extra:
    path, w = a.extra.rsplit(":", 1); w = float(w)
    e = pd.read_parquet(path).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(c.index).to_numpy()
    k = np.isfinite(e) & ~far & (e <= 14400)
    out = np.where(k, (n * out + w * np.nan_to_num(e)) / (n + w), out)
    print(f"blended {int(k.sum())} rows with {os.path.basename(path)} (weight {w:g} to {n})")
out = np.clip(out, 0, None)
assert np.isfinite(out).all() and len(out) == len(t)
t["TAXITIME_SEC_mvt"] = out
t.to_parquet(a.out, index=False)
d = out - base
print(f"wrote {a.out}: {len(t)} rows, mean {out.mean():.1f}, max {out.max():.1f}; rows moved by more than 60 s: {(np.abs(d) > 60).sum()}, "
      f"more than 600 s: {(np.abs(d) > 600).sum()}, RMS change {np.sqrt((d ** 2).mean()):.2f} s")
