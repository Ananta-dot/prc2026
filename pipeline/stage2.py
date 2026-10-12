# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-pass model with an explicit tail.

Passes: light out-of-fold models predict every departure's taxi time, which gives an
estimated off-block time for every flight (training and ranking alike); surface-state
features are rebuilt from those estimates after each pass.

Tail: the score is RMSE over every row, and a small number of records carry "taxi times" of
hours. Most of those are not taxi times at all but recording artefacts with an exact form:
  * the off-block time is the schedule, or within 45 min after it, although the take-off
    was much later (a recording default, or a real push-back followed by a long hold)
                                                        -> taxi = take-off - schedule - push delay
  * the off-block time has the scheduled date          -> taxi = one day + a normal taxi time
For flights that took off more than an hour after schedule, a classifier (the gate) estimates
how likely the first case is and a small regressor estimates the push delay; a regressor
handles everything else; the prediction is the probability-weighted mean.

  python3 stage2.py --feat ../data/processed/feat.parquet --compact ../data/processed/compact.parquet \\
      --template ../data/raw/submitting.parquet --out ../outputs --version 4 --stages 3
"""
import argparse, ctypes, gc, os, time, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
import model as M
from features import Events

ap = argparse.ArgumentParser()
ap.add_argument("--feat", default="cache/feat.parquet"); ap.add_argument("--compact", default="cache/compact.parquet")
ap.add_argument("--template", required=True); ap.add_argument("--out", default="out")
ap.add_argument("--team", default="courageous-onion"); ap.add_argument("--version", type=int, default=2)
ap.add_argument("--tag", default="")
ap.add_argument("--name", default="", help="output file name without .parquet (default <team>_v<version>)")
ap.add_argument("--cap", type=float, default=3600.0, help="light passes train on rows with taxi time <= cap")
ap.add_argument("--tail-cap", type=float, default=14400.0, help="final regressor trains on rows with taxi time <= this")
ap.add_argument("--no-tail", action="store_true", help="single regressor, no gate (the earlier behaviour)")
ap.add_argument("--val-months", default="1,7"); ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--lr", type=float, default=0.08); ap.add_argument("--leaves", type=int, default=127)
ap.add_argument("--s1-rounds", type=int, default=300); ap.add_argument("--s1-lr", type=float, default=0.15)
ap.add_argument("--skip-final", action="store_true")
ap.add_argument("--rounds", default="", help="skip validation: 'regressor_rounds,gate_rounds,push_delay_rounds'")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--drop", default="doy", help="comma-separated feature-name prefixes to leave out")
ap.add_argument("--stages", type=int, default=3, help="number of models in the chain (light passes = stages - 1)")
ap.add_argument("--mm", default="", help="park the feature matrix in this .npy file while the light passes train (for machines short of memory)")
ap.add_argument("--val-range", default="", help="hold out these UTC days instead of whole months: 'YYYY-MM-DD:YYYY-MM-DD[,...]', end exclusive")
ap.add_argument("--cal-cap", type=float, default=0.7, help="largest per-airport upward correction of the gate, in logits (0 = off)")
ap.add_argument("--gate-rounds", type=int, default=300)
ap.add_argument("--dump-zone", default="", help="save the gate-zone rows of the matrix to this prefix and stop")
ap.add_argument("--dump-all", default="", help="save the whole matrix after the light passes to this prefix (final fit: <prefix>_fin), then continue")
ap.add_argument("--on-min", type=float, default=3600.0, help="near-schedule class needs a taxi time above this")
ap.add_argument("--on-thr", type=float, default=2700.0, help="near-schedule class: off-block at most this long after schedule")
a = ap.parse_args()
t0 = time.time(); os.makedirs(a.out, exist_ok=True); tag = a.tag or f"v{a.version}"
M.BASE.update(num_threads=a.threads, learning_rate=a.lr, num_leaves=a.leaves, seed=a.seed)
DROP = tuple(x for x in a.drop.split(",") if x)
LIGHT = dict(learning_rate=a.s1_lr, num_leaves=63, bagging_fraction=0.5)
GATE = dict(objective="binary", metric="binary_logloss", learning_rate=0.03, num_leaves=31,
            min_data_in_leaf=10, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
            max_bin=127, cat_smooth=20.0, cat_l2=10.0, verbose=-1, seed=a.seed, num_threads=a.threads)
DELTA = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
             feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=127,
             cat_smooth=20.0, cat_l2=10.0, verbose=-1, seed=a.seed, num_threads=a.threads)
GATE_XS = 3600.0     # the gate only looks at flights that took off more than an hour after schedule
ON_THR = a.on_thr    # "pushed back near schedule": off-block no more than 45 min after schedule
FAR_XS = 43200.0     # unmatched flights beyond 12 h: too few cases to learn from, use observed class shares
FAR_BANDS = [(49000.0, 66000.0), (66000.0, 100000.0), (100000.0, np.inf)]
S2 = ["s2_yhat", "s2_exc", "s2_ntaxi_apt", "s2_ntaxi_rwy", "s2_ntaxi_apt_T", "s2_ntaxi_rwy_T",
      "s2_npush_m300", "s2_npush_p300", "s2_npush_rwy_m600", "s2_npush_rwy_p600", "s2_narr_taxi_b",
      "s2_nto_rwy_during", "s2_nld_apt_during", "s2_exc_rwy_bef", "s2_exc_rwy_aft", "s2_exc_apt_bef",
      "s2_exc_apt_aft", "s2_exc_rel", "s2_prev1_exc", "s2_prev2_exc", "s2_next1_exc", "s2_next2_exc",
      "s2_shift_rwy", "s2_shift_apt", "s2_exc_push_rwy_bef", "s2_exc_push_rwy_aft", "s2_exc_push_apt_bef",
      "s2_exc_push_apt_aft", "s2_gap_push_prev_rwy", "s2_gap_push_next_rwy", "s2_exc_rwy_wide", "s2_exc_apt_wide"]
MID = ["mid_utc", "mid_s_utc"]

feats, meta = M.load(a.feat)
feats = [f for f in feats if not (DROP and f.startswith(DROP))]
y = meta.m_y.to_numpy("float64"); rank = meta.m_rank.to_numpy(); lab = ~rank
month = meta.m_month.to_numpy(); apt = meta.m_apt.to_numpy().astype("int64"); ids = meta.m_id.to_numpy()
T = meta.m_T.to_numpy("float64")
_x = pd.read_parquet(a.feat, columns=["xS", "xA", "flt_missing"])
xS, fmiss = _x["xS"].to_numpy("float64"), (_x["flt_missing"].to_numpy() > 0).astype("int64")
xA = np.nan_to_num(_x["xA"].to_numpy("float64"))
ok = lab & np.isfinite(y) & (y >= 0)
keep = ok & (y <= a.cap)                                    # rows the light passes learn from
yt = np.nan_to_num(np.clip(y, 0, a.cap))
gate_zone = np.nan_to_num(xS) > GATE_XS
# take-off time counted from the first UTC midnight after the schedule (negative: same UTC day).
# Records whose off-block falls on a later UTC day than the schedule are often defaulted to it.
sched_tod = np.mod(T - xS, 86400.0)
mid_utc = xS - (86400.0 - sched_tod)
with np.errstate(invalid="ignore"):
    push = xS - y                                            # off-block time minus schedule
    k = np.round((y - 900) / 86400)
    is_shift = lab & (k >= 1) & (np.abs(y - k * 86400 - 1200) <= 1500) & (np.abs(push) > 7)   # off-block a day early
    is_on = lab & gate_zone & (push <= ON_THR) & (y > a.on_min) & ~is_shift   # late take-off although pushed back near schedule
cls = np.where(is_on, 1, np.where(is_shift, 2, 0)).astype("int32")
reg_rows = ok & (y <= a.tail_cap) & (cls == 0)              # rows the final regressor learns from
y_reg = np.nan_to_num(np.clip(y, 0, a.tail_cap))
push_t = np.nan_to_num(np.clip(push, -1800.0, ON_THR))
vm = [int(x) for x in a.val_months.split(",")]
arr = pd.read_parquet(a.compact, columns=["APT", "is_dep", "t_MVT_TIME_UTC_mvt", "t_BLOCK_TIME_UTC_mvt"])
arr = arr[~arr.is_dep]
arr_k = arr.APT.cat.codes.to_numpy().astype("int64")
arr_land = Events(arr_k, arr.t_MVT_TIME_UTC_mvt.to_numpy())
arr_inbl = Events(arr_k, arr.t_BLOCK_TIME_UTC_mvt.to_numpy())
del arr


def trim():
    """Hand freed memory back to the system (matters on machines short of memory)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


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
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def apply_rules(P, train, rows, rate=None, cnt=None):
    """Overrides for the few flights no model can learn from, using class shares seen in the
    training rows. P is [other, near-schedule, day-shift] for `rows`; returns P and the anchor
    (the time the near-schedule expert counts back from: take-off minus schedule by default).

    Unmatched flights (no NM record), per airport:
      * take-off on a later UTC day than the schedule, or 8 h to 13.6 h (49000 s) late: where the
        training rows show the off-block time is then defaulted to the schedule (share >= 0.8),
        use that share;
      * more than 49000 s late: at an airport where such records are not ordinary ones, the
        off-block is either the schedule or a real time with the schedule's date (one day
        early); split by the shares seen in the band (shrunk to the airport's overall split),
        pulled half-way to the airline's own schedule-default rate (`rate`, from `cnt` late
        unmatched flights) except in the band where the two answers nearly coincide.
    Matched flights whose NM off-block time is itself more than 4 h before take-off: the record
    was off-block near the schedule / NM time (take-off logged a day late).
    """
    idx = np.flatnonzero(rows); x = np.nan_to_num(xS[idx]); anchor = x.copy(); fixed = np.zeros(len(idx), dtype=bool)
    tr = train & np.isfinite(y) & (fmiss > 0)
    for ap in np.unique(apt[idx]):
        A = (apt == ap) & (fmiss > 0); Ar = A[idx]
        k = A & (mid_utc > 600) & (np.nan_to_num(xS) > GATE_XS) & (np.nan_to_num(xS) <= 49000.0)
        t = k & tr; kr = k[idx]
        if kr.any() and t.sum() >= 10 and (cls[t] == 1).mean() >= 0.8:
            P[kr, 1] = (cls[t] == 1).sum() / (t.sum() + 1.0); P[kr, 2] = 0.0
        k = A & (np.nan_to_num(xS) > 28800.0) & (np.nan_to_num(xS) <= 49000.0)     # 8 h to 13.6 h late
        t = k & tr; kr = k[idx]
        if kr.any() and t.sum() >= 10 and (cls[t] == 1).mean() >= 0.8:
            P[kr, 1] = (cls[t] == 1).sum() / (t.sum() + 0.5); P[kr, 2] = 0.0
        far_t = A & tr & (np.nan_to_num(xS) > 49000.0)
        n_far, n_oth = far_t.sum(), (cls[far_t] == 0).sum()
        n1, n2 = (cls[far_t] == 1).sum(), (cls[far_t] == 2).sum()
        p_oth = (n_oth + 0.5) / (n_far + 1.0)              # share of ordinary records this late at this airport
        s_far = (n1 + 0.5) / (n1 + n2 + 1.0)               # schedule-default share among the rest
        for lo, hi in FAR_BANDS:
            k = A & (np.nan_to_num(xS) > lo) & (np.nan_to_num(xS) <= hi); kr = k[idx]
            if not kr.any():
                continue
            if n_oth == n_far:                             # never anything but ordinary records here
                P[kr, 1] = P[kr, 2] = 0.0
                continue
            t = k & tr
            b1, b2 = (cls[t] == 1).sum(), (cls[t] == 2).sum()
            sb = np.full(int(kr.sum()), (b1 + s_far) / (b1 + b2 + 1.0))
            if rate is not None and (lo, hi) != FAR_BANDS[1]:
                c = cnt[kr] >= 10
                sb[c] = 0.5 * sb[c] + 0.5 * rate[kr][c]
            P[kr, 1] = (1.0 - p_oth) * sb; P[kr, 2] = (1.0 - p_oth) * (1.0 - sb)
    k = (fmiss[idx] == 0) & (xA[idx] > 14400.0) & (x > GATE_XS)
    P[k, 1] = 0.85; P[k, 2] = 0.0; anchor[k] = 0.6 * x[k] + 0.4 * xA[idx][k]; fixed[k] = True
    P[:, 0] = 1.0 - P[:, 1] - P[:, 2]
    return P, anchor, fixed


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
    """-> prediction for pred_rows, parts dict, (regressor rounds, gate rounds), feature names, importances"""
    X, names = M.matrix(a.feat, feats, meta, extra=S2 + MID)
    X[:, names.index("mid_utc")] = mid_utc
    X[:, names.index("mid_s_utc")] = sched_tod
    M.fill_encodings(X, names, meta, y, lab, fold, final=final)
    trim()
    col = {n: j for j, n in enumerate(names)}
    cats = [n for n in names if n.startswith("c_")]
    universe = lab & ((fold > 0) | final)          # rows that get an out-of-fold prediction
    rwy, usual = X[:, col["c_rwy"]].astype("int64"), X[:, col["te_stand_rwy"]].astype("float64")
    def park(X):       # move the matrix to disk and read it back on demand: only the binned copy is needed to train
        np.save(a.mm, X); del X; gc.collect()
        return np.load(a.mm, mmap_mode="r")
    for stage in range(1, a.stages):               # light out-of-fold passes on ordinary rows
        if a.mm:
            X = park(X)
        ds = M.dataset(X, names, yt, free=True).construct()
        pred = oof(ds, X, keep & universe, universe, fold, **LIGHT)
        del ds; gc.collect()
        if a.mm:
            X = np.array(X)
        for n, v in surface_features(pred, rwy, usual).items():
            X[:, col[n]] = v
        trim()
        print(f"  pass {stage} done ({time.time()-t0:.0f}s)", flush=True)
    if a.dump_all:    # the whole matrix after the passes, for fast experiments on the last stage and for nn.py
        dp = a.dump_all + ("_fin" if final else "")
        np.save(dp + "_X.npy", X)
        pd.DataFrame({"id": ids, "y": y, "cls": cls, "xS": xS, "fmiss": fmiss, "apt": apt, "month": month, "fold": fold,
                      "T": T, "push": push, "lab": lab, "rank": rank, "xA": xA}).to_parquet(dp + "_meta.parquet")
        open(dp + "_names.txt", "w").write("\n".join(names))
        print(f"  dumped matrix {X.shape} ({time.time()-t0:.0f}s)", flush=True)
    if a.mm:
        X = park(X)
    Xp = np.array(X[pred_rows])
    xs_p = xS[pred_rows]
    if a.dump_zone:
        zr = gate_zone & (universe | pred_rows)
        np.save(a.dump_zone + "_X.npy", X[zr])
        pd.DataFrame({"id": ids[zr], "y": y[zr], "cls": cls[zr], "xS": xS[zr], "fmiss": fmiss[zr], "apt": apt[zr], "month": month[zr],
                      "fold": fold[zr], "val": pred_rows[zr], "T": T[zr], "push": push[zr], "lab": lab[zr]}).to_parquet(a.dump_zone + "_meta.parquet")
        open(a.dump_zone + "_names.txt", "w").write("\n".join(names))
        raise SystemExit(f"dumped {int(zr.sum())} zone rows")

    # gate: only flights that took off more than an hour after schedule
    gate, g_rounds, d_rounds, P = None, 0, 0, np.zeros((int(pred_rows.sum()), 3)); P[:, 0] = 1.0
    anchor = np.nan_to_num(xs_p)
    delta = np.zeros(int(pred_rows.sum()))
    if not a.no_tail:
        g_tr = universe & gate_zone & np.isfinite(y) & (cls != 2)
        Xg = X[g_tr]
        # errors cost in proportion to the square of the lateness, so weight the gate that way
        gw = (np.clip(xS[g_tr], GATE_XS, FAR_XS) / GATE_XS) ** 2
        dg = lgb.Dataset(Xg, label=(cls[g_tr] == 1).astype("int32"), weight=gw, feature_name=names, categorical_feature=cats, params=GATE)
        g_rounds = a.gate_rounds if va is not None else rounds[1]
        gate = lgb.train(GATE, dg, g_rounds)
        zone = np.nan_to_num(xs_p) > GATE_XS
        P[zone, 1] = gate.predict(Xp[zone], num_iteration=g_rounds)
        if a.cal_cap > 0:
            # Long holds come in events (snow days), and a gate that has not seen an event
            # under-states it. Out-of-fold predictions on the training rows measure by how much
            # at each airport; shift the gate by that many logits (shrunk, never downwards).
            lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
            oofp = np.full(len(y), np.nan); gl = (cls == 1).astype("float64"); gwa = (np.clip(np.nan_to_num(xS), GATE_XS, FAR_XS) / GATE_XS) ** 2
            for f in sorted(set(fold[g_tr])):
                kf, ko = g_tr & (fold != f), g_tr & (fold == f)
                mg = lgb.train(GATE, lgb.Dataset(X[kf], label=gl[kf].astype("int32"), weight=gwa[kf], feature_name=names, categorical_feature=cats, params=GATE), g_rounds)
                oofp[ko] = mg.predict(X[ko])
            ap_p, shifts = apt[pred_rows], {}
            for ap_ in np.unique(apt[g_tr]):
                k = g_tr & (apt == ap_); npos = gl[k].sum(); b = 0.0
                if npos >= 5:
                    for _ in range(50):
                        q = 1 / (1 + np.exp(-(lg(oofp[k]) + b))); b -= (gwa[k] * (q - gl[k])).sum() / ((gwa[k] * q * (1 - q)).sum() + 1e-9)
                b = float(np.clip(b * npos / (npos + 100.0), 0.0, a.cal_cap)); shifts[M.APTS[ap_]] = round(b, 2)
                kz = zone & (ap_p == ap_)
                P[kz, 1] = 1 / (1 + np.exp(-(lg(P[kz, 1]) + b)))
            print("  gate correction by airport (logits):", shifts, flush=True)
        # how long after schedule the push-back was, for flights in the "near schedule" class
        on_tr = g_tr & (cls == 1)
        dd = lgb.Dataset(X[on_tr], label=push_t[on_tr], feature_name=names, categorical_feature=cats, params=DELTA)
        if va is not None:
            on_va = va & gate_zone & (cls == 1)
            dm = lgb.train(DELTA, dd, 3000, valid_sets=[lgb.Dataset(X[on_va], label=push_t[on_va], reference=dd, categorical_feature=cats)],
                           callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
            d_rounds = dm.best_iteration
        else:
            d_rounds = rounds[2]
            dm = lgb.train(DELTA, dd, d_rounds)
        delta[zone] = np.clip(dm.predict(Xp[zone], num_iteration=d_rounds), -1800.0, ON_THR)
        del dd
        P, anchor, fixed = apply_rules(P, universe, pred_rows, Xp[:, col["rU_fpre"]].astype("float64"), Xp[:, col["cntU_fpre"]].astype("float64"))
        delta[fixed] = 0.0
        del Xg, dg; gc.collect()
        print(f"  gate done, rounds {g_rounds} ({time.time()-t0:.0f}s)", flush=True)

    # final regressor on everything that is not one of the two artefacts
    fit_rows = (keep if a.no_tail else reg_rows) & universe
    ds = M.dataset(X, names, yt if a.no_tail else y_reg, free=True).construct()
    del X; gc.collect()
    va_reg = None if va is None else (va & (ok if a.no_tail else reg_rows))
    reg = M.fit(ds, fit_rows, va_reg, rounds=(rounds[0] if rounds else 5000))
    r_rounds = reg.best_iteration or reg.current_iteration()
    r = np.clip(reg.predict(Xp, num_iteration=r_rounds), 0, None)
    final_pred = P[:, 0] * r + P[:, 1] * (anchor - delta) + P[:, 2] * (86400.0 + np.clip(r, 0, 3600))
    imp = pd.Series(reg.feature_importance("gain"), names).sort_values(ascending=False)
    gimp = None if gate is None else pd.Series(gate.feature_importance("gain"), names).sort_values(ascending=False)
    return final_pred, dict(r=r, P=P, delta=delta), (r_rounds, g_rounds, d_rounds), names, (imp / imp.sum(), gimp)


log = [f"{tag}: {a.stages}-stage, tail {not a.no_tail}, pass cap {a.cap}, tail cap {a.tail_cap}, lr {a.lr}, leaves {a.leaves}"]
print(log[0], flush=True)
rounds = tuple(int(x) for x in a.rounds.split(",")) if a.rounds else None
if not rounds:
    if a.val_range:
        fold = M.month_folds(month, [])
        for part in a.val_range.split(","):
            lo, hi = (pd.Timestamp(x, tz="UTC").timestamp() for x in part.split(":"))
            fold[(T >= lo) & (T < hi) & lab] = 0
    else:
        fold = M.month_folds(month, vm)
    va = lab & (fold == 0) & np.isfinite(y)
    p, parts, rounds, names, (imp, gimp) = run(fold, False, va, va=va)
    yv, cv, xv = y[va], cls[va], xS[va]
    wild = (yv > a.tail_cap) & (cv == 0)          # huge values that match neither artefact
    log.append(f"features {len(names)}, val rows {int(va.sum())}, regressor rounds {rounds[0]}, gate rounds {rounds[1]}, push-delay rounds {rounds[2]}")
    log.append(f"  val RMSE, all rows: {M.rmse(yv, p):.2f}")
    log.append(f"  val RMSE, all rows except the {int(wild.sum())} above {a.tail_cap:.0f}s that match neither artefact: {M.rmse(yv[~wild], p[~wild]):.2f}")
    log.append(f"  regressor alone, all rows: {M.rmse(yv, parts['r']):.2f}")
    for name, k_ in [("near-schedule push, taxi above 1 h", (cv == 1) & (yv > 3600)), ("near-schedule push, taxi up to 1 h", (cv == 1) & (yv <= 3600)),
                     ("day-shift rows", cv == 2), ("other rows above 1 h", (cv == 0) & (yv > 3600) & ~wild),
                     ("rows up to 1 h", yv <= 3600)]:
        if k_.any():
            log.append(f"    {name}: n={int(k_.sum())}, RMSE {M.rmse(yv[k_], p[k_]):.1f}, mean gate P(near-schedule) {parts['P'][k_, 1].mean():.3f}, P(shift) {parts['P'][k_, 2].mean():.3f}")
    log += M.val_lines(yv, p, apt[va], month[va])
    log.append("  regressor top features: " + ", ".join(f"{k_} {v:.3f}" for k_, v in imp.head(20).items()))
    if gimp is not None:
        log.append("  gate top features: " + ", ".join(f"{k_} {v:.3f}" for k_, v in (gimp / gimp.sum()).head(20).items()))
    print("\n".join(log[1:]), f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(os.path.join(a.out, f"val_{tag}.npz"), y=yv, p=p, id=ids[va], r=parts["r"], P=parts["P"], cls=cv, delta=parts["delta"])
    gc.collect()

if not a.skip_final:
    fr = tuple(max(1, int(x * 1.15)) for x in rounds)
    pr, parts, _, names, _ = run(M.month_folds(month, []), True, rank, rounds=fr)
    tpl = pd.read_parquet(a.template)
    res = tpl[["MVT_ID_mvt"]].merge(pd.DataFrame({"MVT_ID_mvt": ids[rank], "p": pr}), on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert len(res) == len(tpl) and res.p.notna().all() and (res.MVT_ID_mvt.to_numpy() == tpl.MVT_ID_mvt.to_numpy()).all()
    out = tpl.copy(); out["TAXITIME_SEC_mvt"] = res.p.to_numpy("float64")
    path = os.path.join(a.out, (a.name or f"{a.team}_v{a.version}") + ".parquet"); out.to_parquet(path, index=False)
    np.savez(os.path.join(a.out, f"parts_{tag}.npz"), id=ids[rank], r=parts["r"], P=parts["P"], p=pr, delta=parts["delta"])
    log.append(f"wrote {path}: {len(out)} rows, mean {out.TAXITIME_SEC_mvt.mean():.1f}, max {out.TAXITIME_SEC_mvt.max():.1f}, "
               f"rows predicted above 1 h: {int((pr > 3600).sum())}, expected near-schedule-push rows {parts['P'][:, 1].sum():.0f}, day-shift {parts['P'][:, 2].sum():.1f}")
    print(log[-1], f"({time.time()-t0:.0f}s)", flush=True)
with open(os.path.join(a.out, f"report_{tag}.txt"), "w") as f:
    f.write("\n".join(log) + "\n")
