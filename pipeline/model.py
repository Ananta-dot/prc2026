# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared modelling code for PRC 2026 taxi-out prediction (memory-lean: one float32 matrix)."""
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
TE_QUANT = ("stand", "stand_rwy", "rwy_type")
INTER = ("xA", "xS", "xE", "xmin_SELI")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, "float64") - np.asarray(p, "float64")) ** 2)))


def load(path):
    """-> feature names, meta frame (m_* bookkeeping and k_* encoding keys)."""
    cols = pq.read_schema(path).names
    feats = [c for c in cols if not c.startswith(("m_", "k_"))]
    meta = pd.read_parquet(path, columns=[c for c in cols if c.startswith(("m_", "k_"))])
    return feats, meta


CLS_KEYS = ("opr", "opr_rwy", "opr_stand", "stand", "callsign", "flight")
EXTRA = ("S_exp_opr", "S_exp_opr_stand", "S_exp_flight", "A_exp_opr_rwy", "A_exp_opr_stand")


def te_names(meta):
    names = []
    for kc in [c for c in meta.columns if c.startswith("k_")]:
        n = kc[2:]
        names += ["te_" + n, "cnt_" + n] + (["q10_" + n] if n in TE_QUANT else [])
        names += (["rS_" + n, "rA_" + n] if n in CLS_KEYS else [])
    return names + [f"{a}_m_{b}" for a in INTER for b in ("te", "q10")] + list(EXTRA)


def matrix(path, feats, meta, extra=()):
    """One float32 matrix: stored features, then room for the encodings (and `extra` columns)."""
    names = feats + te_names(meta) + list(extra)
    X = np.full((len(meta), len(names)), np.nan, dtype=np.float32)
    for j, c in enumerate(feats):
        X[:, j] = pq.read_table(path, columns=[c]).column(0).to_numpy()
    return X, names


def month_folds(month, val_months):
    """fold 0 = validation months; the other months are spread over folds 1..4."""
    rest = [m for m in range(1, 13) if m not in val_months]
    f = {m: 0 for m in val_months}
    f.update({m: 1 + i % 4 for i, m in enumerate(rest)})
    return np.array([f[int(m)] for m in month])


def _oof(key, val, src, fold, smooth):
    """Smoothed mean of `val` per key from the `src` rows, each src row excluding its own fold."""
    t = pd.DataFrame({"k": key[src], "f": fold[src], "v": val[src]})
    tot = t.groupby("k")["v"].agg(["sum", "count"])
    byf = t.groupby(["k", "f"])["v"].agg(["sum", "count"])
    s = pd.Series(key).map(tot["sum"]).fillna(0).to_numpy(dtype="float64").copy()
    c = pd.Series(key).map(tot["count"]).fillna(0).to_numpy(dtype="float64").copy()
    own = byf.reindex(pd.MultiIndex.from_arrays([key[src], fold[src]]))
    s[src] -= own["sum"].to_numpy()
    c[src] -= own["count"].to_numpy()
    return (s + float(np.mean(val[src])) * smooth) / (c + smooth), c, t


def fill_encodings(X, names, meta, y, labelled, fold, final=False, smooth=20.0, cap=3600.0):
    """Out-of-fold encodings per key, written into X.

    Per key: mean / 10th percentile / count of the (capped) taxi time, and for some keys the
    rate at which the off-block time equals the schedule (rS) or the NM off-block time (rA).

    Validation mode: fold-0 rows are encoded from folds 1..4 and contribute to nobody.
    Final mode: every labelled row contributes; labelled rows are encoded out of their own
    fold, ranking rows from all of them.
    """
    col = {n: j for j, n in enumerate(names)}
    yc = np.clip(y, 0, cap)
    src = labelled & np.isfinite(y) & ((fold > 0) | final)
    xS, xA = X[:, col["xS"]].astype("float64"), X[:, col["xA"]].astype("float64")
    with np.errstate(invalid="ignore"):
        isS = np.nan_to_num(np.abs(y - xS) <= 7).astype("float64")
        isA = np.nan_to_num(np.abs(y - xA) <= 67).astype("float64")
    folds = sorted(set(fold[src]))
    for kc in [c for c in meta.columns if c.startswith("k_")]:
        name, key = kc[2:], meta[kc].to_numpy()
        m, c, t = _oof(key, yc, src, fold, smooth)
        X[:, col["te_" + name]] = m
        X[:, col["cnt_" + name]] = c
        if name in CLS_KEYS:
            X[:, col["rS_" + name]] = _oof(key, isS, src, fold, smooth)[0]
            X[:, col["rA_" + name]] = _oof(key, isA, src, fold, smooth)[0]
        if name in TE_QUANT:
            q = np.full(len(key), np.nan, dtype="float32")
            for f in folds + [-1]:
                rows = (src & (fold == f)) if f >= 0 else ~src
                q10 = (t[t.f != f] if f >= 0 else t).groupby("k")["v"].quantile(0.1)
                q[rows] = pd.Series(key[rows]).map(q10).to_numpy(dtype="float32")
            X[:, col["q10_" + name]] = q
    base = X[:, col["te_stand_rwy"]]
    for a in INTER:
        X[:, col[a + "_m_te"]] = X[:, col[a]] - base
        X[:, col[a + "_m_q10"]] = X[:, col[a]] - X[:, col["q10_stand_rwy"]]
    for n in EXTRA:   # expected shift if the off-block time is the reference time
        kind, key = n[0], n[6:]
        X[:, col[n]] = X[:, col[f"r{kind}_{key}"]] * (X[:, col["x" + kind]] - base)


BASE = dict(objective="regression", metric="rmse", learning_rate=0.08, num_leaves=127,
            min_data_in_leaf=200, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1,
            lambda_l2=10.0, max_bin=127, cat_smooth=20.0, cat_l2=10.0, max_cat_to_onehot=8,
            verbose=-1, seed=7, num_threads=0)


def dataset(X, names, label, free=False, **params):
    cats = [n for n in names if n.startswith("c_")]
    return lgb.Dataset(X, label=label, feature_name=names, categorical_feature=cats,
                       free_raw_data=free, params={**BASE, **params})


def fit(ds, tr, va=None, rounds=5000, log=200, **params):
    """Train on rows `tr` of the parent dataset; early-stop on rows `va` if given."""
    p = {**BASE, **params}
    dtr = ds.subset(np.flatnonzero(tr))
    if va is None:
        m = lgb.train(p, dtr, rounds)
    else:
        m = lgb.train(p, dtr, rounds, valid_sets=[ds.subset(np.flatnonzero(va))],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(log)])
    gc.collect()
    return m


def predict(m, X, rows, chunk=200_000, **kw):
    idx = np.flatnonzero(rows)
    out = np.empty(len(idx))
    for i in range(0, len(idx), chunk):
        out[i:i + chunk] = m.predict(X[idx[i:i + chunk]], **kw)
    return out


def val_lines(y, p, apt, month, caps=(3000, 3600, 5400, 7200)):
    L = [f"  val RMSE on rows with truth <= {c}: {rmse(y[y <= c], p[y <= c]):.2f}" for c in caps]
    k = y <= 3600
    L.append("  by airport (truth<=3600): " + ", ".join(
        f"{APTS[i]} {rmse(y[k & (apt == i)], p[k & (apt == i)]):.0f}" for i in range(10)))
    L.append("  by month (truth<=3600):   " + ", ".join(
        f"{mo} {rmse(y[k & (month == mo)], p[k & (month == mo)]):.0f}" for mo in sorted(set(month))))
    return L
