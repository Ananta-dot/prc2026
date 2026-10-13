# SPDX-License-Identifier: GPL-3.0-or-later
"""Taxi time as anchor + correction: LightGBM fit to (taxi time - anchor), a blend member for the stage2 regressor.

The regressor's strongest inputs (ground-track off-block, NM AOBT) are near-direct estimates of the answer, and
trees rebuild "taxi = xA" only out of many steps.  Here the anchor (members.anchor: g, A, te) is subtracted, the
trees learn the correction, and the anchor and its source are inputs too.

  --mode resid  one regressor on the correction
  --mode mix    for rows anchored on NM AOBT, a classifier for "the off-block time is AOBT" (within 67 s)
                and the correction regressor fit without those rows; prediction p * anchor + (1 - p) * (anchor + correction)

  python3 resid.py --mat ../outputs/exp/mat7 --val ../outputs/exp/val_val7.npz --out ../outputs/exp/res7.npz
  python3 resid.py --mat $O/mat_lb_fin --final --rounds <from the validation run> --out $O/res_lb.npz
"""
import argparse, time
import numpy as np
import lightgbm as lgb
import model as M
import members as K

ap = argparse.ArgumentParser()
ap.add_argument("--mat", required=True); ap.add_argument("--val", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--final", action="store_true"); ap.add_argument("--mode", default="resid", choices=["resid", "mix"])
ap.add_argument("--anchor", default="g,A,te", help="anchor sources in order of preference")
ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--rounds", default="", help="final fit: 'regressor[,classifier]' rounds (the validation run prints them)")
a = ap.parse_args()
t0 = time.time()
D = K.load(a.mat, a.final)
X, names, y, tr, va, pr = D["X"], D["names"], D["y"], D["tr"], D["va"], D["pred"]
anc, src = K.anchor(X, names, a.anchor)
print(f"anchors ({time.time()-t0:.0f}s)", flush=True)
K.anchor_report(y[tr], anc[tr], src[tr])
names2 = names + ["anc", "anc_src"]
cats = [n for n in names if n.startswith("c_")]
REG = dict(M.BASE, learning_rate=a.lr, seed=a.seed, num_threads=a.threads)
CLF = dict(objective="binary", metric="binary_logloss", learning_rate=a.lr, num_leaves=63, min_data_in_leaf=50,
           feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=127,
           cat_smooth=20.0, cat_l2=10.0, verbose=-1, seed=a.seed, num_threads=a.threads)
rounds = [int(x) for x in a.rounds.split(",")] if a.rounds else []
if a.final and not rounds:
    raise SystemExit("--final needs --rounds from the validation run")
ext = (anc, src.astype("float64"))


def train(params, rows, label, n_rounds=None):
    """Fit on rows; in validation mode early-stop on the validation rows of the same kind."""
    idx = np.flatnonzero(tr & rows)
    d = lgb.Dataset(K.take(X, idx, ext), label=label[idx], feature_name=names2, categorical_feature=cats, params=params)
    if a.final:
        return lgb.train(params, d, n_rounds), n_rounds
    vi = np.flatnonzero(D["rows"] & va & rows)
    v = lgb.Dataset(K.take(X, vi, ext), label=label[vi], reference=d, categorical_feature=cats)
    m = lgb.train(params, d, 5000, valid_sets=[v], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    return m, m.best_iteration


pi = np.flatnonzero(pr)
Xp = K.take(X, pi, ext)
everything = np.ones(len(y), dtype=bool)
with np.errstate(invalid="ignore"):
    res = y - anc
    isA = (src == K.SRC["A"]) & (np.abs(res) <= 67)
if a.mode == "resid":
    reg, nr = train(REG, everything, res, rounds[0] if rounds else None)
    p = anc[pi] + reg.predict(Xp, num_iteration=nr)
    used = [nr]
else:
    reg, nr = train(REG, ~isA, res, rounds[0] if rounds else None)
    zone = src == K.SRC["A"]
    clf, nc = train(CLF, zone, isA.astype("float64"), rounds[1] if len(rounds) > 1 else None)
    corr = anc[pi] + reg.predict(Xp, num_iteration=nr)
    pa = np.where(zone[pi], clf.predict(Xp, num_iteration=nc), 0.0)
    p = pa * anc[pi] + (1 - pa) * corr
    used = [nr, nc]
    if not a.final:
        k = zone[pi] & D["rows"][pi]
        print(f"  classifier: share AOBT-exact {isA[pi][k].mean():.3f}, mean p on them {pa[k & isA[pi]].mean():.3f}, on the rest {pa[k & ~isA[pi]].mean():.3f}")
p = np.clip(p, 0, None)
del Xp
if not a.final:
    print(f"rounds {','.join(map(str, used))}; for --final pass --rounds {','.join(str(int(r * 1.15)) for r in used)}", flush=True)
    if a.val:
        K.compare(f"resid ({a.mode}, anchor {a.anchor})", a.val, D["ids"][pi], p, src[pi], t0)
np.savez(a.out, id=D["ids"][pi], nn=p)
print(f"wrote {a.out} ({time.time()-t0:.0f}s)")
