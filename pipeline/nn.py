# SPDX-License-Identifier: GPL-3.0-or-later
"""MLP on the matrix saved by `stage2.py --dump-all`, trained on the GPU (Apple MPS / CUDA) when there is one.
A blend member for the ordinary-taxi-time regressor of stage2.py.

  python3 nn.py --mat ../outputs/exp/mat7 --val ../outputs/exp/val_val7.npz --out ../outputs/exp/nn7.npz
With --final the model is fit on every labelled row (no hold-out) for --epochs and predicts the ranking rows.
"""
import argparse, time
import numpy as np, pandas as pd
import torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--mat", required=True); ap.add_argument("--val", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--bs", type=int, default=4096)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--width", type=int, default=512)
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--final", action="store_true")
a = ap.parse_args()
t0 = time.time()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
X = np.load(a.mat + "_X.npy", mmap_mode="r")
names = open(a.mat + "_names.txt").read().split("\n")
m = pd.read_parquet(a.mat + "_meta.parquet")
y, fold, lab, cls, rank = (m[c].to_numpy() for c in ["y", "fold", "lab", "cls", "rank"])
ok = lab & np.isfinite(y) & (y >= 0)
rows = ok & (y <= 14400.0) & (cls == 0)                       # the rows the stage2 regressor is fit to
tr = rows if a.final else rows & (fold > 0)
va = lab & (fold == 0) & np.isfinite(y)
pred_rows = rank if a.final else va

ci = [j for j, n in enumerate(names) if n.startswith("c_")]
ni = [j for j, n in enumerate(names) if not n.startswith("c_")]
# numeric: clip to the 0.5/99.5 percentiles of the training rows, standardise, NaN -> 0 plus a flag
samp = np.flatnonzero(tr)[:: max(1, int(tr.sum() // 300_000))]
S = np.asarray(X[samp][:, ni], dtype="float64")
lo, hi = np.nanpercentile(S, 0.5, axis=0), np.nanpercentile(S, 99.5, axis=0)
Sc = np.clip(S, lo, hi); mu, sd = np.nanmean(Sc, axis=0), np.nanstd(Sc, axis=0) + 1e-6
mu, sd, lo, hi = (np.nan_to_num(v) for v in (mu, sd, lo, hi))
hasnan = np.isnan(S).any(axis=0)
nan_cols = np.flatnonzero(hasnan)
card = [int(np.nanmax(np.asarray(X[:, j], dtype="float64"))) + 2 for j in ci]
print(f"device {dev}; numeric {len(ni)} (+{len(nan_cols)} NaN flags), categorical {len(ci)} {card}; train rows {int(tr.sum())} ({time.time()-t0:.0f}s)", flush=True)


def encode(idx):
    B = np.asarray(X[idx], dtype="float32")
    N = B[:, ni]
    flags = np.isnan(N[:, nan_cols]).astype("float32")
    N = (np.clip(np.nan_to_num(N, nan=0.0), lo, hi) - mu) / sd
    N[np.isnan(B[:, ni])] = 0.0
    C = np.nan_to_num(B[:, ci], nan=-1.0).astype("int64") + 1
    C = np.clip(C, 0, np.array(card) - 1)
    return np.hstack([N, flags]).astype("float32"), C


class Net(nn.Module):
    def __init__(self, d_num, cards, w):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, min(16, (c + 1) // 2 + 1)) for c in cards])
        d = d_num + sum(e.embedding_dim for e in self.emb)
        self.f = nn.Sequential(nn.Linear(d, w), nn.SiLU(), nn.Dropout(0.1), nn.Linear(w, w), nn.SiLU(), nn.Dropout(0.1),
                               nn.Linear(w, w // 2), nn.SiLU(), nn.Linear(w // 2, 1))

    def forward(self, xn, xc):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.emb)]
        return self.f(torch.cat([xn] + e, dim=1)).squeeze(1)


# the matrix fits in memory once encoded; keep it on the host and stream batches to the GPU
tri = np.flatnonzero(tr)
Ntr, Ctr = [], []
for i in range(0, len(tri), 500_000):
    n_, c_ = encode(tri[i:i + 500_000]); Ntr.append(n_); Ctr.append(c_)
Ntr, Ctr = torch.from_numpy(np.vstack(Ntr)), torch.from_numpy(np.vstack(Ctr))
SCALE = 1000.0
ytr = torch.from_numpy((y[tri] / SCALE).astype("float32"))
net = Net(Ntr.shape[1], card, a.width).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
steps = a.epochs * (len(tri) // a.bs + 1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
print(f"encoded ({time.time()-t0:.0f}s)", flush=True)


def predict(idx):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 200_000):
            n_, c_ = encode(idx[i:i + 200_000])
            out.append(net(torch.from_numpy(n_).to(dev), torch.from_numpy(c_).to(dev)).float().cpu().numpy())
    net.train()
    return np.clip(np.concatenate(out) * SCALE, 0, None)


vi = np.flatnonzero(pred_rows)
if not a.final:
    zv = np.load(a.val); assert (zv["id"] == m.id.to_numpy()[va]).all()
    yv, rv = zv["y"], zv["r"]; fit_v = rows[va]; k1 = yv <= 3600
for ep in range(a.epochs):
    perm = torch.randperm(len(tri))
    tot = 0.0
    for i in range(0, len(tri), a.bs):
        b = perm[i:i + a.bs]
        xn, xc, yb = Ntr[b].to(dev, non_blocking=True), Ctr[b].to(dev, non_blocking=True), ytr[b].to(dev, non_blocking=True)
        loss = nn.functional.mse_loss(net(xn, xc), yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        tot += loss.item() * len(b)
    msg = f"epoch {ep+1}: train rmse {np.sqrt(tot/len(tri))*SCALE:.1f}"
    if not a.final and (ep % 3 == 2 or ep == a.epochs - 1):
        pv = predict(vi)
        msg += f" | val fit rows {np.sqrt(np.mean((yv[fit_v]-pv[fit_v])**2)):.2f} (lgb {np.sqrt(np.mean((yv[fit_v]-rv[fit_v])**2)):.2f})"
        for w in (0.2, 0.3, 0.4):
            q = w * pv + (1 - w) * rv
            msg += f" | blend {w}: {np.sqrt(np.mean((yv[fit_v]-q[fit_v])**2)):.2f} (<=1h {np.sqrt(np.mean((yv[k1]-q[k1])**2)):.2f})"
    print(msg + f" ({time.time()-t0:.0f}s)", flush=True)
pv = predict(vi)
np.savez(a.out, id=m.id.to_numpy()[vi], nn=pv)
print(f"wrote {a.out} ({time.time()-t0:.0f}s)")
