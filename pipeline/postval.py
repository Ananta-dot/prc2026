# SPDX-License-Identifier: GPL-3.0-or-later
"""Score a validation file of stage2.py (val_<tag>.npz) with the pairing rules of posthoc.py applied,
as build.py does for the submission.  An optional NN prediction is blended into the regressor first.

  python3 postval.py --val ../outputs/exp/val_val7.npz [--nn ../outputs/exp/nn7.npz:0.3]
"""
import argparse, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import posthoc

P_ = "../data/processed/"
ap = argparse.ArgumentParser()
ap.add_argument("--val", required=True, help="comma-separated val npz files: averaged after the rules")
ap.add_argument("--nn", default="", help="file:weight, NN regressor blended into r (comma-separated, one per val file)")
ap.add_argument("--compact", default=P_ + "compact.parquet"); ap.add_argument("--link", default=P_ + "link.parquet")
ap.add_argument("--tz", default=P_ + "tz.parquet"); ap.add_argument("--dflt", default=P_ + "dfl_val17.parquet")
a = ap.parse_args()


def load_rows(ids):
    c = pd.read_parquet(a.compact, columns=["MVT_ID_mvt", "APT", "t_MVT_TIME_UTC_mvt", "t_SCHED_TIME_UTC_mvt", "t_LOBT_flt", "t_AOBT_3_flt", "is_dep"])
    c = c[c.is_dep].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").reindex(ids)
    lk = pd.read_parquet(a.link).set_index("MVT_ID_mvt").lk_LOBT.reindex(ids).to_numpy()
    tz = pd.read_parquet(a.tz); tzs = pd.Series(True, index=tz.MVT_ID_mvt[tz.off == 7200]).reindex(ids).fillna(False).to_numpy(bool)
    dq = pd.read_parquet(a.dflt).set_index("MVT_ID_mvt").q.reindex(ids).to_numpy()
    return (c.t_MVT_TIME_UTC_mvt.to_numpy(), c.t_SCHED_TIME_UTC_mvt.to_numpy(), c.t_LOBT_flt.to_numpy(), c.t_AOBT_3_flt.to_numpy(),
            lk, (c.APT == "LIRF").to_numpy(), tzs, dq)


files = a.val.split(","); nns = a.nn.split(",") if a.nn else [""] * len(files)
z0 = np.load(files[0]); ids, y, cls = z0["id"], z0["y"], z0["cls"]
T, S, LOBT, AOBT, lk, lirf, tzs, dq = load_rows(ids)
wild = (y > 14400) & (cls == 0)
rm = lambda q, k: float(np.sqrt(np.mean((y[k] - q[k]) ** 2)))
raw, new = [], []
for f, nnspec in zip(files, nns):
    z = np.load(f); assert (z["id"] == ids).all()
    p, r, P = z["p"].copy(), z["r"].copy(), z["P"]
    if nnspec:
        path, w = nnspec.rsplit(":", 1); w = float(w)
        n = np.load(path); assert (n["id"] == ids).all()
        rb = (1 - w) * r + w * n["nn"]; p = p + P[:, 0] * (rb - r); r = rb
    q = posthoc.apply(p, r, P[:, 1], P[:, 2], T, S, LOBT, AOBT, lk, lirf, None, tzs, dq)
    raw.append(p); new.append(q)
    print(f"{os.path.basename(f)}{' + ' + nnspec if nnspec else ''}: no rules {rm(p, ~wild):.2f} | rules {rm(q, ~wild):.2f} (all rows {rm(q, np.ones_like(wild)):.2f}, <=1h {rm(q, y <= 3600):.2f})")
if len(files) > 1:
    m0 = np.mean(raw, axis=0); far = m0 > 14400
    out = np.where(far, new[0], np.mean(new, axis=0))
    print(f"blend of {len(files)}: rules {rm(out, ~wild):.2f} (all rows {rm(out, np.ones_like(wild)):.2f}, <=1h {rm(out, y <= 3600):.2f})")
