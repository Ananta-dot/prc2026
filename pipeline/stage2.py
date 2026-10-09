# SPDX-License-Identifier: GPL-3.0-or-later
"""Two-stage model.

Stage 1 predicts every departure's taxi time out of fold, which gives an estimated
off-block time for every flight (training and ranking alike). Stage 2 adds what those
estimates say about the surface: how many departures were taxiing when this one pushed
back, how the neighbours on the runway are doing against their usual time, and so on.

  python3 stage2.py --feat ../data/processed/feat.parquet --compact ../data/processed/compact.parquet \\
      --template ../data/raw/submitting.parquet --out ../outputs --version 2
"""
import argparse, gc, os, time
import numpy as np, pandas as pd
import model as M
from features import Events

ap = argparse.ArgumentParser()
ap.add_argument("--feat", default="cache/feat.parquet"); ap.add_argument("--compact", default="cache/compact.parquet")
ap.add_argument("--template", required=True); ap.add_argument("--out", default="out")
ap.add_argument("--team", default="courageous-onion"); ap.add_argument("--version", type=int, default=2)
ap.add_argument("--tag", default=""); ap.add_argument("--cap", type=float, default=3600.0)
ap.add_argument("--val-months", default="1,7"); ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--lr", type=float, default=0.08); ap.add_argument("--leaves", type=int, default=127)
ap.add_argument("--s1-rounds", type=int, default=300); ap.add_argument("--s1-lr", type=float, default=0.15)
ap.add_argument("--skip-final", action="store_true"); ap.add_argument("--rounds", type=int, default=0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--drop", default="doy", help="comma-separated feature-name prefixes to leave out")
ap.add_argument("--init", action="store_true", help="final model learns the correction to the last light pass")
ap.add_argument("--stages", type=int, default=2, help="2 = one out-of-fold pass before the final model, 3 = two passes")
a = ap.parse_args()
t0 = time.time(); os.makedirs(a.out, exist_ok=True); tag = a.tag or f"v{a.version}"
M.BASE.update(num_threads=a.threads, learning_rate=a.lr, num_leaves=a.leaves, seed=a.seed)
DROP = tuple(x for x in a.drop.split(",") if x)
LIGHT = dict(learning_rate=a.s1_lr, num_leaves=63, bagging_fraction=0.5)
S2 = ["s2_yhat", "s2_exc", "s2_ntaxi_apt", "s2_ntaxi_rwy", "s2_ntaxi_apt_T", "s2_ntaxi_rwy_T",
      "s2_npush_m300", "s2_npush_p300", "s2_npush_rwy_m600", "s2_npush_rwy_p600", "s2_narr_taxi_b",
      "s2_nto_rwy_during", "s2_nld_apt_during", "s2_exc_rwy_bef", "s2_exc_rwy_aft", "s2_exc_apt_bef",
      "s2_exc_apt_aft", "s2_exc_rel", "s2_prev1_exc", "s2_prev2_exc", "s2_next1_exc", "s2_next2_exc",
      "s2_shift_rwy", "s2_shift_apt", "s2_exc_push_rwy_bef", "s2_exc_push_rwy_aft", "s2_exc_push_apt_bef",
      "s2_exc_push_apt_aft", "s2_gap_push_prev_rwy", "s2_gap_push_next_rwy", "s2_exc_rwy_wide", "s2_exc_apt_wide"]

feats, meta = M.load(a.feat)
y = meta.m_y.to_numpy("float64"); rank = meta.m_rank.to_numpy(); lab = ~rank
month = meta.m_month.to_numpy(); apt = meta.m_apt.to_numpy().astype("int64"); ids = meta.m_id.to_numpy()
T = meta.m_T.to_numpy("float64")
keep = lab & np.isfinite(y) & (y >= 0) & (y <= a.cap)
yt = np.nan_to_num(np.clip(y, 0, a.cap))
vm = [int(x) for x in a.val_months.split(",")]
arr = pd.read_parquet(a.compact, columns=["APT", "is_dep", "t_MVT_TIME_UTC_mvt", "t_BLOCK_TIME_UTC_mvt"])
arr = arr[~arr.is_dep]
arr_k = arr.APT.cat.codes.to_numpy().astype("int64")
arr_land = Events(arr_k, arr.t_MVT_TIME_UTC_mvt.to_numpy())
arr_inbl = Events(arr_k, arr.t_BLOCK_TIME_UTC_mvt.to_numpy())
del arr


def surface_features(pred, rwy, usual):
    b = T - pred                       # estimated off-block time
    exc = pred - usual                 # estimated time above the usual for this stand and runway
    tA, tR = Events(apt, T, exc=exc), Events(rwy, T, exc=exc)
    bA, bR = Events(apt, b, exc=exc), Events(rwy, b, exc=exc)
    W, o = -172800.0, {}
    o["s2_yhat"], o["s2_exc"] = pred, exc
    o["s2_ntaxi_apt"] = bA.count(apt, b, W, 0.5) - tA.count(apt, b, W, 0.5) - 1
    o["s2_ntaxi_rwy"] = bR.count(rwy, b, W, 0.5) - tR.count(rwy, b, W, 0.5) - 1
    o["s2_ntaxi_apt_T"] = bA.count(apt, T, W, 0.5) - tA.count(apt, T, W, 0.5)
    o["s2_ntaxi_rwy_T"] = bR.count(rwy, T, W, 0.5) - tR.count(rwy, T, W, 0.5)
    o["s2_npush_m300"] = bA.count(apt, b, -300.0, -0.5)
    o["s2_npush_p300"] = bA.count(apt, b, 0.5, 300.0)
    o["s2_npush_rwy_m600"] = bR.count(rwy, b, -600.0, -0.5)
    o["s2_npush_rwy_p600"] = bR.count(rwy, b, 0.5, 600.0)
    o["s2_narr_taxi_b"] = arr_land.count(apt, b, W, 0.5) - arr_inbl.count(apt, b, W, 0.5)
    w = np.clip(pred, 0, 7200)
    o["s2_nto_rwy_during"] = tR.count(rwy, T, -w, -0.5)
    o["s2_nld_apt_during"] = arr_land.count(apt, T, -w, -0.5)
    o["s2_exc_rwy_bef"] = tR.mean("exc", rwy, T, -600.0, -0.5)
    o["s2_exc_rwy_aft"] = tR.mean("exc", rwy, T, 0.5, 600.0)
    o["s2_exc_apt_bef"] = tA.mean("exc", apt, T, -900.0, -0.5)
    o["s2_exc_apt_aft"] = tA.mean("exc", apt, T, 0.5, 900.0)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        o["s2_exc_rel"] = exc - np.nanmean(np.vstack([o["s2_exc_rwy_bef"], o["s2_exc_rwy_aft"]]), axis=0)
    for j in (1, 2):
        o[f"s2_prev{j}_exc"] = tR.take("exc", tR.prev(rwy, T, j, max_gap=3600.0)[1])
        o[f"s2_next{j}_exc"] = tR.take("exc", tR.next(rwy, T, j, max_gap=3600.0)[1])
    # take-off order against push-back order (0 in a first-come-first-served queue)
    o["s2_shift_rwy"] = tR.count(rwy, T, W, 0.5) - bR.count(rwy, b, W, 0.5)
    o["s2_shift_apt"] = tA.count(apt, T, W, 0.5) - bA.count(apt, b, W, 0.5)
    # flights that pushed back around the same time
    o["s2_exc_push_rwy_bef"] = bR.mean("exc", rwy, b, -600.0, -0.5)
    o["s2_exc_push_rwy_aft"] = bR.mean("exc", rwy, b, 0.5, 600.0)
    o["s2_exc_push_apt_bef"] = bA.mean("exc", apt, b, -600.0, -0.5)
    o["s2_exc_push_apt_aft"] = bA.mean("exc", apt, b, 0.5, 600.0)
    o["s2_gap_push_prev_rwy"] = np.clip(bR.prev(rwy, b, 1)[0], 0, 3600)
    o["s2_gap_push_next_rwy"] = np.clip(bR.next(rwy, b, 1)[0], 0, 3600)
    o["s2_exc_rwy_wide"] = tR.mean("exc", rwy, T, -2700.0, 2700.0)
    o["s2_exc_apt_wide"] = tA.mean("exc", apt, T, -5400.0, 5400.0)
    return o


def oof(ds, X, fit_rows, universe, fold, **params):
    """Out-of-fold predictions for `universe` rows. Every other row (validation or ranking) gets
    the average of the fold models, so its estimate comes from models of the same strength."""
    pred = np.zeros(len(y))
    folds = sorted(set(fold[universe]))
    for f in folds:
        m = M.fit(ds, fit_rows & (fold != f), rounds=a.s1_rounds, **params)
        rows = universe & (fold == f)
        pred[rows] = M.predict(m, X, rows)
        pred[~universe] += M.predict(m, X, ~universe) / len(folds)
    return np.clip(pred, 0, None)


def run(fold, final, pred_rows, rounds=None, va=None):
    X, names = M.matrix(a.feat, [f for f in feats if not (DROP and f.startswith(DROP))], meta, extra=S2)
    M.fill_encodings(X, names, meta, y, lab, fold, final=final)
    col = {n: j for j, n in enumerate(names)}
    universe = lab & ((fold > 0) | final)          # rows that get an out-of-fold prediction
    fit_rows = keep & universe
    rwy, usual = X[:, col["c_rwy"]].astype("int64"), X[:, col["te_stand_rwy"]].astype("float64")
    for stage in range(1, a.stages):               # light out-of-fold passes
        ds = M.dataset(X, names, yt, free=True).construct()
        pred = oof(ds, X, fit_rows, universe, fold, **LIGHT)
        del ds; gc.collect()
        for n, v in surface_features(pred, rwy, usual).items():
            X[:, col[n]] = v
        print(f"  pass {stage} done ({time.time()-t0:.0f}s)", flush=True)
    Xp = X[pred_rows].copy()
    ds = M.dataset(X, names, yt, free=True)
    if a.init:
        ds.set_init_score(pred)
    ds.construct()
    del X; gc.collect()
    m = M.fit(ds, fit_rows, va, rounds=rounds or 5000)
    return m, Xp, names, pred


log = [f"{tag}: {a.stages}-stage, init {a.init}, cap {a.cap}, lr {a.lr}, leaves {a.leaves}, light rounds {a.s1_rounds}"]
print(log[0], flush=True)
best = a.rounds
if not a.rounds:
    fold = M.month_folds(month, vm)
    va = lab & (fold == 0) & np.isfinite(y)
    m, Xv, names, pred1 = run(fold, False, va, va=va)
    best = m.best_iteration
    p = np.clip(m.predict(Xv, num_iteration=best) + (pred1[va] if a.init else 0), 0, None)
    k = y[va] <= 3600
    log.append(f"features {len(names)}, val rows {int(va.sum())}, rounds {best}; last light pass alone on truth<=3600: {M.rmse(y[va][k], pred1[va][k]):.2f}")
    log += M.val_lines(y[va], p, apt[va], month[va])
    imp = pd.Series(m.feature_importance("gain"), names).sort_values(ascending=False)
    log.append("  top features: " + ", ".join(f"{k_} {v:.3f}" for k_, v in (imp / imp.sum()).head(30).items()))
    print("\n".join(log[1:]), f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(os.path.join(a.out, f"val_{tag}.npz"), y=y[va], p=p, id=ids[va])
    (imp / imp.sum()).to_csv(os.path.join(a.out, f"imp_{tag}.csv"))
    del m, Xv; gc.collect()

if not a.skip_final:
    m, Xr, names, predf = run(M.month_folds(month, []), True, rank, rounds=int(best * 1.15))
    pr = np.clip(m.predict(Xr) + (predf[rank] if a.init else 0), 0, None)
    tpl = pd.read_parquet(a.template)
    res = tpl[["MVT_ID_mvt"]].merge(pd.DataFrame({"MVT_ID_mvt": ids[rank], "p": pr}), on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert len(res) == len(tpl) and res.p.notna().all() and (res.MVT_ID_mvt.to_numpy() == tpl.MVT_ID_mvt.to_numpy()).all()
    out = tpl.copy(); out["TAXITIME_SEC_mvt"] = res.p.to_numpy("float64")
    path = os.path.join(a.out, f"{a.team}_v{a.version}.parquet"); out.to_parquet(path, index=False)
    m.save_model(os.path.join(a.out, f"model_{tag}.txt"))
    log.append(f"wrote {path}: {len(out)} rows, mean {out.TAXITIME_SEC_mvt.mean():.1f}, min {out.TAXITIME_SEC_mvt.min():.1f}, max {out.TAXITIME_SEC_mvt.max():.1f}")
    print(log[-1], f"({time.time()-t0:.0f}s)", flush=True)
with open(os.path.join(a.out, f"report_{tag}.txt"), "w") as f:
    f.write("\n".join(log) + "\n")
