# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast experiments on the last stage, using the matrix saved by `stage2.py --dump-all`.

  python3 exp.py --mat ../outputs/exp/mat7 --val ../outputs/exp/val_val7.npz --exp dgate
"""
import argparse, time
import numpy as np, pandas as pd
import lightgbm as lgb
import model as M

ap = argparse.ArgumentParser()
ap.add_argument("--mat", required=True); ap.add_argument("--val", required=True)
ap.add_argument("--exp", default="dgate"); ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--threads", type=int, default=0); ap.add_argument("--save", default="")
a = ap.parse_args()
t0 = time.time()
X = np.load(a.mat + "_X.npy", mmap_mode="r")
names = open(a.mat + "_names.txt").read().split("\n")
m = pd.read_parquet(a.mat + "_meta.parquet")
y, xS, fold, lab, apt, cls = (m[c].to_numpy() for c in ["y", "xS", "fold", "lab", "apt", "cls"])
cats = [n for n in names if n.startswith("c_")]
ok = lab & np.isfinite(y) & (y >= 0)
va = lab & (fold == 0) & np.isfinite(y)
tr = ok & (fold > 0)
z = np.load(a.val)
assert (z["id"] == m.id.to_numpy()[va]).all()
yv, pb = z["y"], z["p"]
LIRF = M.APTS.index("LIRF")
av = apt[va]


def report(label, p):
    k = yv <= 3600
    print(f"{label:28s} all {M.rmse(yv, p):7.2f} | <=1h {M.rmse(yv[k], p[k]):6.2f} | LIRF<=1h {M.rmse(yv[k & (av == LIRF)], p[k & (av == LIRF)]):6.2f} "
          f"| others<=1h {M.rmse(yv[k & (av != LIRF)], p[k & (av != LIRF)]):6.2f}  ({time.time()-t0:.0f}s)", flush=True)


def sub(rows):
    return np.ascontiguousarray(X[rows])


REG = dict(M.BASE, learning_rate=a.lr, num_threads=a.threads)
CLF = dict(objective="binary", metric="binary_logloss", learning_rate=a.lr, num_leaves=63, min_data_in_leaf=50,
           feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=127,
           cat_smooth=20.0, cat_l2=10.0, verbose=-1, seed=7, num_threads=a.threads)


def train(params, rows, label, vrows, vlabel, rounds=5000):
    d = lgb.Dataset(sub(rows), label=label, feature_name=names, categorical_feature=cats, params=params)
    v = lgb.Dataset(sub(vrows), label=vlabel, reference=d, categorical_feature=cats)
    return lgb.train(params, d, rounds, valid_sets=[v], callbacks=[lgb.early_stopping(100, verbose=False)])


report("baseline (saved run)", pb)
Xv = sub(va)

if a.exp == "hp":
    # final-regressor settings, scored on the rows it is fit to (ordinary records up to 4 h)
    rows = ok & (y <= 14400.0) & (cls == 0)
    vr = rows[va]
    grid = [dict(), dict(num_leaves=255), dict(num_leaves=511, min_data_in_leaf=100), dict(feature_fraction=0.8),
            dict(min_data_in_leaf=50), dict(lambda_l2=1.0), dict(max_bin=255), dict(objective="huber", alpha=900.0)]
    for g in grid:
        mdl = train({**REG, **g}, tr & rows, y[tr & rows], va & rows, y[va & rows])
        r = np.clip(mdl.predict(Xv), 0, None)
        k = yv <= 3600
        print(f"{str(g):55s} rounds {mdl.best_iteration:5d} | fit rows {M.rmse(yv[vr], r[vr]):7.2f} | <=1h {M.rmse(yv[k], r[k]):6.2f}  ({time.time()-t0:.0f}s)", flush=True)

if a.exp == "agate":
    # identity gate: off-block time equal to NM's AOBT_3 (taxi = take-off - AOBT), for matched departures
    tail = 14400.0
    xA = m["xA"].to_numpy()                       # take-off minus AOBT_3 (0 where missing)
    hasA = np.isfinite(xA) & (xA > 0) & (xA <= 3600)
    with np.errstate(invalid="ignore"):
        isA = hasA & (np.abs(y - xA) <= 67)
    base_rows = ok & (y <= tail) & (cls == 0)
    r_all = train(REG, tr & base_rows, y[tr & base_rows], va & base_rows, y[va & base_rows])
    ra = np.clip(r_all.predict(Xv), 0, None)
    r0_rows = base_rows & ~isA
    r0 = train(REG, tr & r0_rows, y[tr & r0_rows], va & r0_rows, y[va & r0_rows])
    r0v = np.clip(r0.predict(Xv), 0, None)
    Z = hasA & base_rows
    g = train(CLF, tr & Z, isA[tr & Z].astype(int), va & Z, isA[va & Z].astype(int))
    gp = g.predict(Xv)
    xv = xA[va]; zv = hasA[va]
    vr = base_rows[va]; k1 = yv <= 3600
    sc = lambda p: f"fit rows {M.rmse(yv[vr], p[vr]):7.2f} | <=1h {M.rmse(yv[k1], p[k1]):6.2f}"
    print("like-for-like regressor        ", sc(ra), flush=True)
    pm = np.where(zv, gp * xv + (1 - gp) * r0v, ra)
    print("AOBT identity mixture          ", sc(pm), flush=True)
    for w in (0.5, 0.75):
        print(f"  blend {w} mixture + {1-w} reg    ", sc(w * pm + (1 - w) * ra), flush=True)
    t_ = zv & vr
    print("gate: mean p on isA %.3f, on not %.3f; share isA %.3f" % (gp[t_ & isA[va]].mean(), gp[t_ & ~isA[va]].mean(), isA[va][t_].mean()))
    for lo_, hi_ in [(0, .2), (.2, .5), (.5, .8), (.8, 1.01)]:
        kk = t_ & (gp >= lo_) & (gp < hi_)
        print(f"  p in [{lo_},{hi_}): n {kk.sum():6d}, actual {isA[va][kk].mean():.3f}")
    if a.save:
        np.savez(a.save, id=z["id"], ra=ra, r0=r0v, gp=gp, zv=zv, xA=xv)

if a.exp == "dgate":
    tail = 14400.0
    with np.errstate(invalid="ignore"):
        isS = np.abs(y - xS) <= 7
    Z = (np.nan_to_num(xS) > 0) & (np.nan_to_num(xS) <= 3600)
    base_rows = ok & (y <= tail) & (cls == 0)
    # like-for-like regressor at the experiment's learning rate
    r_all = train(REG, tr & base_rows, y[tr & base_rows], va & base_rows, y[va & base_rows])
    ra = np.clip(r_all.predict(Xv), 0, None)
    pa = np.where(Z[va], ra, pb)
    report(f"regressor lr{a.lr} in zone", pa)
    # regressor that never sees a schedule-default record in the zone, and a gate for them
    r0_rows = base_rows & ~(Z & isS)
    r0 = train(REG, tr & r0_rows, y[tr & r0_rows], va & r0_rows, y[va & r0_rows])
    r0v = np.clip(r0.predict(Xv), 0, None)
    g = train(CLF, tr & Z & base_rows, isS[tr & Z & base_rows].astype(int), va & Z & base_rows, isS[va & Z & base_rows].astype(int))
    gp = g.predict(Xv)
    xv = np.nan_to_num(xS[va])
    pm = np.where(Z[va], gp * xv + (1 - gp) * r0v, pb)
    report("dgate mixture in zone", pm)
    for w in (0.25, 0.5, 0.75):
        report(f"  blend {w} mix + {1-w} reg", np.where(Z[va], w * (gp * xv + (1 - gp) * r0v) + (1 - w) * ra, pb))
    print("gate AUC-ish: mean p on isS %.3f, on not %.3f" % (gp[Z[va] & isS[va]].mean(), gp[Z[va] & ~isS[va]].mean()))
    if a.save:
        np.savez(a.save, id=z["id"], ra=ra, r0=r0v, gp=gp, Z=Z[va])
