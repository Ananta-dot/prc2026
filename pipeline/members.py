# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared code for the blend members fit on the matrix saved by `stage2.py --dump-all` (resid.py, cat.py).

Every member is fit to the rows the stage2 regressor learns from (ordinary records up to 4 h), early-stops on
the validation months (fold 0) in validation mode, and saves `id` + `nn` like nn.py, so postval.py, build.py
and blend.py take it as they take the MLP.

Anchors: the best available direct estimate of the taxi time, in order of preference
  g   take-off minus the off-block time seen in the OpenSky ground track (stop or gap found, 30 s to 1 h)
  A   take-off minus NM's actual off-block time AOBT_3 (1 min to 1 h)
  te  out-of-fold mean taxi time of the stand and runway (always there)
"""
import time
import numpy as np, pandas as pd

TAIL = 14400.0
SRC = {"g": 1, "A": 2, "te": 3}


def load(mat, final):
    """-> dict: X (memory-mapped), names, meta columns, row masks tr (fit), va (validation), pred (rows to predict)."""
    X = np.load(mat + "_X.npy", mmap_mode="r")
    names = open(mat + "_names.txt").read().split("\n")
    m = pd.read_parquet(mat + "_meta.parquet")
    y, fold, lab, cls, rank = (m[c].to_numpy() for c in ["y", "fold", "lab", "cls", "rank"])
    ok = lab & np.isfinite(y) & (y >= 0)
    rows = ok & (y <= TAIL) & (cls == 0)
    va = lab & (fold == 0) & np.isfinite(y)
    return dict(X=X, names=names, meta=m, y=y, ids=m.id.to_numpy(), rows=rows, va=va,
                tr=rows if final else rows & (fold > 0), pred=rank if final else va)


def columns(X, names, wanted, chunk=200_000):
    """Read a few columns of the (row-major, memory-mapped) matrix in one pass."""
    j = [names.index(n) for n in wanted]
    out = np.empty((X.shape[0], len(j)), dtype="float64")
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = X[i:i + chunk][:, j]
    return {n: out[:, k] for k, n in enumerate(wanted)}


def anchor(X, names, order="g,A,te"):
    """-> anchor (seconds), source code (SRC; 0 never left after the fallback)."""
    order = [o for o in order.split(",") if o]
    want = ["xA", "te_stand_rwy", "te_opr_rwy"] + (["g_est", "g_how"] if "g_est" in names else [])
    c = columns(X, names, want)
    cand = {"A": np.where((c["xA"] > 60) & (c["xA"] <= 3600), c["xA"], np.nan),
            "te": np.where(np.isfinite(c["te_stand_rwy"]), c["te_stand_rwy"], c["te_opr_rwy"])}
    if "g_est" in c:
        cand["g"] = np.where((c["g_est"] > 30) & (c["g_est"] <= 3600) & (c["g_how"] > 0), c["g_est"], np.nan)
    elif "g" in order:
        print("  no ground-track columns in this matrix: anchor g skipped", flush=True)
    a = np.full(X.shape[0], np.nan); src = np.zeros(X.shape[0], dtype="int8")
    for o in order:
        if o not in cand:
            continue
        k = np.isnan(a) & np.isfinite(cand[o])
        a[k], src[k] = cand[o][k], SRC[o]
    k = np.isnan(a)
    a[k] = np.nanmedian(cand["te"]); src[k] = SRC["te"]
    return a, src


def take(X, idx, extra=(), chunk=200_000):
    """Rows `idx` of X as one float32 array, with `extra` full-length columns appended."""
    out = np.empty((len(idx), X.shape[1] + len(extra)), dtype="float32")
    for i in range(0, len(idx), chunk):
        b = idx[i:i + chunk]
        out[i:i + chunk, :X.shape[1]] = X[b]
        for k, e in enumerate(extra):
            out[i:i + chunk, X.shape[1] + k] = e[b]
    return out


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, "float64") - np.asarray(p, "float64")) ** 2)))


def anchor_report(y, a, src):
    """How close each anchor is to the truth on the rows given."""
    inv = {v: k for k, v in SRC.items()}
    for s in sorted(set(src.tolist())):
        k = src == s; d = y[k] - a[k]
        print(f"  anchor {inv.get(s, s):3s}: n {k.sum():8d} | |truth - anchor| <= 60 s {np.mean(np.abs(d) <= 60):.3f}, <= 120 s "
              f"{np.mean(np.abs(d) <= 120):.3f} | median {np.median(d):7.1f} | RMSE {rmse(y[k], a[k]):7.1f}", flush=True)


def compare(label, val, ids, p, src=None, t0=None):
    """RMSE of a member against the stage2 regressor of the same validation run (fit rows and <= 1 h), and blends."""
    z = np.load(val); assert (z["id"] == ids).all(), "validation file and matrix rows differ"
    y, r, cls = z["y"], z["r"], z["cls"]
    fit = (cls == 0) & (y >= 0) & (y <= TAIL); k1 = y <= 3600
    msg = f"{label}: fit rows {rmse(y[fit], p[fit]):.2f} (lgb {rmse(y[fit], r[fit]):.2f}), <=1h {rmse(y[k1], p[k1]):.2f} (lgb {rmse(y[k1], r[k1]):.2f})"
    for w in (0.2, 0.3, 0.4, 0.5):
        q = w * p + (1 - w) * r
        msg += f" | blend {w}: {rmse(y[fit], q[fit]):.2f}"
    print(msg + (f" ({time.time()-t0:.0f}s)" if t0 else ""), flush=True)
    if src is not None:
        inv = {v: k for k, v in SRC.items()}
        for s in sorted(set(src.tolist())):
            k = fit & (src == s)
            print(f"    anchor {inv.get(s, s):3s} rows: n {k.sum():7d}, member {rmse(y[k], p[k]):7.2f}, lgb {rmse(y[k], r[k]):7.2f}, "
                  f"50/50 {rmse(y[k], 0.5 * (p[k] + r[k])):7.2f}", flush=True)
