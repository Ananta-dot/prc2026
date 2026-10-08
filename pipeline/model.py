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


def te_names(meta):
    names = []
    for kc in [c for c in meta.columns if c.startswith("k_")]:
        n = kc[2:]
        names += ["te_" + n, "cnt_" + n] + (["q10_" + n] if n in TE_QUANT else [])
    return names + [f"{a}_m_{b}" for a in INTER for b in ("te", "q10")]


def matrix(path, feats, meta):
    """One float32 matrix: stored features first, then room for the encodings."""
    names = feats + te_names(meta)
    X = np.empty((len(meta), len(names)), dtype=np.float32)
    for j, c in enumerate(feats):
        X[:, j] = pq.read_table(path, columns=[c]).column(0).to_numpy()
    return X, names


def month_folds(month, val_months):
    """fold 0 = validation months; the other months are spread over folds 1..4."""
    rest = [m for m in range(1, 13) if m not in val_months]
    f = {m: 0 for m in val_months}
    f.update({m: 1 + i % 4 for i, m in enumerate(rest)})
    return np.array([f[int(m)] for m in month])


def fill_encodings(X, names, meta, y, labelled, fold, final=False, smooth=20.0, cap=3600.0):
    """Out-of-fold mean / 10th percentile / count of the (capped) taxi time per key, written into X.

    Validation mode: fold-0 rows are encoded from folds 1..4 and contribute to nobody.
    Final mode: every labelled row contributes; labelled rows are encoded out of their own
    fold, ranking rows from all of them.
    """
    yc = np.clip(y, 0, cap)
    src = labelled & np.isfinite(y) & ((fold > 0) | final)
    prior = float(np.mean(yc[src]))
    folds = sorted(set(fold[src]))
    col = {n: j for j, n in enumerate(names)}
    for kc in [c for c in meta.columns if c.startswith("k_")]:
        name, key = kc[2:], meta[kc].to_numpy()
        t = pd.DataFrame({"k": key[src], "f": fold[src], "y": yc[src]})
        tot = t.groupby("k")["y"].agg(["sum", "count"])
        byf = t.groupby(["k", "f"])["y"].agg(["sum", "count"])
        s = pd.Series(key).map(tot["sum"]).fillna(0).to_numpy(dtype="float64").copy()
        c = pd.Series(key).map(tot["count"]).fillna(0).to_numpy(dtype="float64").copy()
        own = byf.reindex(pd.MultiIndex.from_arrays([key[src], fold[src]]))
        s[src] -= own["sum"].to_numpy()
        c[src] -= own["count"].to_numpy()
        X[:, col["te_" + name]] = (s + prior * smooth) / (c + smooth)
        X[:, col["cnt_" + name]] = c
        if name in TE_QUANT:
            q = np.full(len(key), np.nan, dtype="float32")
            for f in folds + [-1]:
                rows = (src & (fold == f)) if f >= 0 else ~src
                q10 = (t[t.f != f] if f >= 0 else t).groupby("k")["y"].quantile(0.1)
                q[rows] = pd.Series(key[rows]).map(q10).to_numpy(dtype="float32")
            X[:, col["q10_" + name]] = q
    for a in INTER:
        X[:, col[a + "_m_te"]] = X[:, col[a]] - X[:, col["te_stand_rwy"]]
        X[:, col[a + "_m_q10"]] = X[:, col[a]] - X[:, col["q10_stand_rwy"]]


BASE = dict(objective="regression", metric="rmse", learning_rate=0.08, num_leaves=127,
            min_data_in_leaf=200, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1,
            lambda_l2=10.0, max_bin=127, cat_smooth=20.0, cat_l2=10.0, max_cat_to_onehot=8,
            verbose=-1, seed=7, num_threads=0)


def dataset(X, names, label, **params):
    cats = [n for n in names if n.startswith("c_")]
    return lgb.Dataset(X, label=label, feature_name=names, categorical_feature=cats,
                       free_raw_data=False, params={**BASE, **params})


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
