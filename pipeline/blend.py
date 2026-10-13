# SPDX-License-Identifier: GPL-3.0-or-later
"""Weights for the stage2 regressor and its blend members (nn.py, resid.py, cat.py), and the combined member.

Weights are least squares on the validation rows the regressor is fit to (ordinary records up to 4 h),
non-negative, with the regressor taking the rest; a two-fold check (rows split by id parity) shows how much
of the gain survives on rows the weights were not fit on.  The combined member is the weighted mean of the
members, so it goes into postval.py / build.py as one --nn file with the total weight W:

  python3 blend.py --val ../outputs/exp/val_val7.npz --members ../outputs/exp/nn7.npz,../outputs/exp/res7.npz,../outputs/exp/cat7.npz \\
      --out ../outputs/exp/mix7.npz
  python3 postval.py --val ../outputs/exp/val_val7.npz --nn ../outputs/exp/mix7.npz:W
  # production: same weights on the --final files of each member
  python3 blend.py --weights w1,w2,w3 --members $O/nn_lb_avg.npz,$O/res_lb.npz,$O/cat_lb.npz --out $O/mix_lb.npz
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--val", default="", help="val_<tag>.npz of stage2.py (fit the weights)")
ap.add_argument("--members", required=True, help="comma-separated npz files with id + nn")
ap.add_argument("--weights", default="", help="use these weights instead of fitting them")
ap.add_argument("--out", required=True)
a = ap.parse_args()
files = a.members.split(",")
Z = [np.load(f) for f in files]
ids = Z[0]["id"]
for f, z in zip(files, Z):
    assert (z["id"] == ids).all(), f"{f}: rows differ from {files[0]}"
Mx = np.stack([z["nn"] for z in Z], axis=1)


def nnls(D, t):
    """Non-negative least squares by dropping negative weights until none is left (few columns)."""
    act = list(range(D.shape[1]))
    while act:
        w = np.linalg.lstsq(D[:, act], t, rcond=None)[0]
        if (w >= 0).all():
            out = np.zeros(D.shape[1]); out[act] = w
            return out
        act = [c for c, x in zip(act, w) if x > 0]
    return np.zeros(D.shape[1])


if a.weights:
    w = np.array([float(x) for x in a.weights.split(",")])
    assert len(w) == len(files)
else:
    z = np.load(a.val); assert (z["id"] == ids).all(), "validation file rows differ from the members"
    y, r, cls = z["y"], z["r"], z["cls"]
    fit = (cls == 0) & (y >= 0) & (y <= 14400)
    rm = lambda p, k: float(np.sqrt(np.mean((y[k] - p[k]) ** 2)))
    D = Mx - r[:, None]; t = y - r          # prediction = r + sum w_i (m_i - r)
    w = nnls(D[fit], t[fit])
    print(f"regressor alone: fit rows {rm(r, fit):.2f}")
    for f, j in zip(files, range(len(files))):
        print(f"  {f}: alone {rm(Mx[:, j], fit):.2f}, weight {w[j]:.3f}")
    print(f"blend (regressor {1 - w.sum():.3f}): fit rows {rm(r + D @ w, fit):.2f}, <=1h {rm(r + D @ w, y <= 3600):.2f}")
    half = (ids.astype("int64") % 2 == 0) if np.issubdtype(ids.dtype, np.integer) else (np.arange(len(ids)) % 2 == 0)
    for h in (half, ~half):
        wh = nnls(D[fit & h], t[fit & h]); o = fit & ~h
        print(f"  two-fold: weights {np.round(wh, 3).tolist()} -> other half {rm(r + D @ wh, o):.2f} (regressor {rm(r, o):.2f}, all-row weights {rm(r + D @ w, o):.2f})")
W = float(w.sum())
if W <= 0:
    raise SystemExit("no member gets a positive weight: nothing to blend")
np.savez(a.out, id=ids, nn=Mx @ (w / W))
print(f"wrote {a.out}; weights {','.join(f'{x:.4f}' for x in w)}; use --nn {a.out}:{W:.4f}")
