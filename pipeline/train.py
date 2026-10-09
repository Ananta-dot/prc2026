# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate (hold out Jan+Jul 2025), then fit on all of 2025 and write a submission.

  python3 train.py --feat ../data/processed/feat.parquet --template ../data/raw/submitting.parquet \\
                   --out ../outputs --team courageous-onion --version 2 --cap 3600
"""
import argparse, gc, os, time
import numpy as np, pandas as pd
import model as M

ap = argparse.ArgumentParser()
ap.add_argument("--feat", default="cache/feat.parquet"); ap.add_argument("--template", required=True)
ap.add_argument("--out", default="out"); ap.add_argument("--team", default="courageous-onion")
ap.add_argument("--version", type=int, default=1); ap.add_argument("--tag", default="")
ap.add_argument("--cap", type=float, default=3600.0, help="train on rows with 0 <= taxi time <= cap")
ap.add_argument("--winsor", action="store_true", help="clip training targets to the cap instead of dropping rows")
ap.add_argument("--val-months", default="1,7"); ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--lr", type=float, default=0.08); ap.add_argument("--leaves", type=int, default=127)
ap.add_argument("--skip-final", action="store_true")
ap.add_argument("--rounds", type=int, default=0, help="skip validation and fit the final model with this many rounds")
ap.add_argument("--drop", default="", help="comma-separated feature-name prefixes to leave out")
a = ap.parse_args()
t0 = time.time(); os.makedirs(a.out, exist_ok=True); tag = a.tag or f"v{a.version}"
M.BASE.update(num_threads=a.threads, learning_rate=a.lr, num_leaves=a.leaves)

feats, meta = M.load(a.feat)
y = meta.m_y.to_numpy("float64"); rank = meta.m_rank.to_numpy(); lab = ~rank
month = meta.m_month.to_numpy(); apt = meta.m_apt.to_numpy(); ids = meta.m_id.to_numpy()
keep = lab & np.isfinite(y) & (y >= 0) & ((y <= a.cap) | a.winsor)
yt = np.nan_to_num(np.clip(y, 0, a.cap))
vm = [int(x) for x in a.val_months.split(",")]
drop = tuple(x for x in a.drop.split(",") if x)


def prepare(fold, final, pred_rows):
    """Matrix -> encodings -> binned dataset; keeps raw rows only where predictions are needed."""
    X, names = M.matrix(a.feat, feats, meta)
    M.fill_encodings(X, names, meta, y, lab, fold, final=final)
    use = [j for j, n in enumerate(names) if not n.startswith(drop)] if drop else None
    if use is not None:
        X, names = np.ascontiguousarray(X[:, use]), [names[j] for j in use]
    ds = M.dataset(X, names, yt, free=True).construct()
    Xp = X[pred_rows].copy()
    del X; gc.collect()
    return ds, Xp, names


log = [f"{tag}: cap {a.cap}, winsor {a.winsor}, lr {a.lr}, leaves {a.leaves}, drop {drop}"]
best = a.rounds
if not a.rounds:   # ---- 1) validation
    fold = M.month_folds(month, vm)
    tr, va = keep & (fold > 0), lab & (fold == 0) & np.isfinite(y)
    ds, Xv, names = prepare(fold, False, va)
    print(f"{log[0]}, features {len(names)} ({time.time()-t0:.0f}s)", flush=True)
    m = M.fit(ds, tr, va); best = m.best_iteration
    p = np.clip(m.predict(Xv, num_iteration=best), 0, None)
    log.append(f"features {len(names)}, train rows {int(tr.sum())}, val rows {int(va.sum())}, rounds {best}")
    log += M.val_lines(y[va], p, apt[va], month[va])
    imp = pd.Series(m.feature_importance("gain"), names).sort_values(ascending=False)
    log.append("  top features: " + ", ".join(f"{k} {v:.3f}" for k, v in (imp / imp.sum()).head(30).items()))
    print("\n".join(log[1:]), f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(os.path.join(a.out, f"val_{tag}.npz"), y=y[va], p=p, id=ids[va])
    (imp / imp.sum()).to_csv(os.path.join(a.out, f"imp_{tag}.csv"))
    del ds, m, Xv; gc.collect()

if not a.skip_final:   # ---- 2) final fit on all labelled rows, predict ranking departures
    ds, Xr, names = prepare(M.month_folds(month, []), True, rank)
    m = M.fit(ds, keep, rounds=int(best * 1.15))
    pr = np.clip(m.predict(Xr), 0, None)
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
