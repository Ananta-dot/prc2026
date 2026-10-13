# SPDX-License-Identifier: GPL-3.0-or-later
"""CatBoost on the matrix of `stage2.py --dump-all`, a blend member for the stage2 regressor.

Besides the matrix (whose c_* columns become categoricals), the raw keys of the feature file (k_stand_rwy,
k_flight, k_callsign, ...) go in as categoricals, so CatBoost's ordered target statistics see the stand,
flight number and callsign directly rather than only through the out-of-fold means of model.py.

  --target raw    the taxi time
  --target resid  taxi time minus the anchor of members.py (anchor and its source are inputs too)

Uses the GPU when CatBoost sees a CUDA device (not Apple MPS); on CPU, --rsm 0.3 to 0.5 keeps it to tens of minutes.

  python3 cat.py --mat ../outputs/exp/mat7 --feat ../data/processed/feat_g2.parquet --val ../outputs/exp/val_val7.npz --out ../outputs/exp/cat7.npz
  python3 cat.py --mat $O/mat_lb_fin --feat ../data/processed/feat_g2.parquet --final --rounds <from validation> --out $O/cat_lb.npz
"""
import argparse, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
import members as K

ap = argparse.ArgumentParser()
ap.add_argument("--mat", required=True); ap.add_argument("--feat", default="", help="feature file the matrix was built from (raw keys)")
ap.add_argument("--keys", default="stand,stand_rwy,rwy_type,opr_stand,flight,callsign,ades", help="k_* keys used as categoricals")
ap.add_argument("--val", default=""); ap.add_argument("--out", required=True); ap.add_argument("--final", action="store_true")
ap.add_argument("--target", default="raw", choices=["raw", "resid"]); ap.add_argument("--anchor", default="g,A,te")
ap.add_argument("--lr", type=float, default=0.08); ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--iters", type=int, default=6000); ap.add_argument("--rounds", type=int, default=0, help="final fit: iterations (the validation run prints them)")
ap.add_argument("--rsm", type=float, default=0.5, help="CPU only: share of features per split")
ap.add_argument("--cpu", action="store_true"); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--threads", type=int, default=-1)
a = ap.parse_args()
t0 = time.time()
if a.final and not a.rounds:
    raise SystemExit("--final needs --rounds from the validation run")
D = K.load(a.mat, a.final)
X, names, y, tr, va, pr = D["X"], D["names"], D["y"], D["tr"], D["va"], D["pred"]
gpu = not a.cpu and get_gpu_device_count() > 0
num = [j for j, n in enumerate(names) if not n.startswith("c_")]
cat_cols = K.columns(X, names, [n for n in names if n.startswith("c_")])
if a.feat:
    keys = ["k_" + k for k in a.keys.split(",") if k]
    fid = pq.read_table(a.feat, columns=["m_id"]).column(0).to_numpy()
    assert len(fid) == len(y) and (fid == D["ids"]).all(), "feature file rows differ from the matrix rows"
    for k in keys:
        cat_cols[k] = pq.read_table(a.feat, columns=[k]).column(0).to_numpy().astype("float64")
anc, src = (None, None)
extra = {}
if a.target == "resid":
    anc, src = K.anchor(X, names, a.anchor)
    extra = {"anc": anc, "anc_src": src.astype("float64")}
    K.anchor_report(y[tr], anc[tr], src[tr])
label = y - anc if a.target == "resid" else y
print(f"{'GPU' if gpu else 'CPU'}; numeric {len(num) + len(extra)}, categorical {len(cat_cols)}: {list(cat_cols)} ({time.time()-t0:.0f}s)", flush=True)


def pool(idx, with_label=True):
    """Numeric block (float32) + integer categorical columns; NaN categories become -1."""
    N = np.empty((len(idx), len(num) + len(extra)), dtype="float32")
    for i in range(0, len(idx), 200_000):
        b = idx[i:i + 200_000]
        N[i:i + 200_000, :len(num)] = X[b][:, num]
        for k, v in enumerate(extra.values()):
            N[i:i + 200_000, len(num) + k] = v[b]
    df = pd.DataFrame(N, columns=[names[j] for j in num] + list(extra), copy=False)
    for n, v in cat_cols.items():
        df[n] = np.nan_to_num(v[idx], nan=-1.0).astype("int64")
    return Pool(df, label=label[idx] if with_label else None, cat_features=list(cat_cols))


params = dict(loss_function="RMSE", eval_metric="RMSE", learning_rate=a.lr, depth=a.depth, l2_leaf_reg=10.0,
              random_seed=a.seed, boosting_type="Plain", max_ctr_complexity=2, one_hot_max_size=8,
              thread_count=a.threads, verbose=250, allow_writing_files=False)
if gpu:
    params.update(task_type="GPU", border_count=254)
else:
    params.update(rsm=a.rsm, border_count=128)
ti = np.flatnonzero(tr)
dtr = pool(ti)
print(f"train pool {len(ti)} rows ({time.time()-t0:.0f}s)", flush=True)
if a.final:
    mdl = CatBoostRegressor(iterations=a.rounds, **params)
    mdl.fit(dtr)
    n_it = a.rounds
else:
    vi = np.flatnonzero(D["rows"] & va)
    mdl = CatBoostRegressor(iterations=a.iters, od_type="Iter", od_wait=150, use_best_model=True, **params)
    mdl.fit(dtr, eval_set=pool(vi))
    n_it = mdl.get_best_iteration() + 1
    print(f"best iteration {n_it}; for --final pass --rounds {int(n_it * 1.15)} ({time.time()-t0:.0f}s)", flush=True)
del dtr
pi = np.flatnonzero(pr)
p = np.empty(len(pi))
for i in range(0, len(pi), 500_000):
    p[i:i + 500_000] = mdl.predict(pool(pi[i:i + 500_000], with_label=False))
if anc is not None:
    p = p + anc[pi]
p = np.clip(p, 0, None)
if not a.final and a.val:
    K.compare(f"catboost ({a.target})", a.val, D["ids"][pi], p, None if src is None else src[pi], t0)
imp = pd.Series(mdl.get_feature_importance(), mdl.feature_names_).sort_values(ascending=False)
print("top features: " + ", ".join(f"{k} {v:.1f}" for k, v in imp.head(20).items()))
np.savez(a.out, id=D["ids"][pi], nn=p)
print(f"wrote {a.out} ({time.time()-t0:.0f}s)")
