# SPDX-License-Identifier: GPL-3.0-or-later
"""Rules that follow from how movements were paired with NM flights.

A departure carries NM columns only if its off-block time is within one hour of the flight
plan's last off-block time (LOBT) -- 3600 s, plus the 6 s jitter.  So

  * for a departure with NM columns, the taxi time lies in [xL - 3606, xL + 3606], xL = take-off - LOBT;
  * for a departure without them whose flight plan is known from the arrival record at the
    destination (link.py), the taxi time lies outside that window.

`apply` takes the model's parts (regressor r, class probabilities P1/P2, blended prediction p)
and returns the prediction with these facts enforced.
"""
import numpy as np

PAD = 3606.0


def apply(p, r, P1, P2, T, S, LOBT, AOBT, lk_LOBT, is_lirf, stats=None, tz_summer=None):
    p, r, P1, P2 = (np.asarray(v, "float64").copy() for v in (p, r, P1, P2))
    xS = T - S
    xL = T - LOBT
    xA = T - AOBT
    m = np.isfinite(xL)
    out = p.copy()
    P0 = 1 - P1 - P2
    shift = 86400 + np.clip(r, 0, 3600)
    with np.errstate(invalid="ignore", divide="ignore"):
        anc = np.where(P1 > 1e-9, (p - P0 * r - P2 * shift) / np.where(P1 > 1e-9, P1, 1), xS)   # near-schedule value the model used
    # --- departures with NM columns: everything must lie in the window
    lo = np.maximum(xL - PAD, 0)
    hi = xL + PAD
    feas = (anc >= lo - 60) & (anc <= hi + 60)
    p1 = np.where(feas, P1, 0.0)
    stale = m & (lo > r)                       # plan is so old that even the lower end is above an ordinary taxi time
    over = np.where(lo <= 2400, 600.0, np.where(lo <= 3600, 1800.0, 2800.0))
    rc = np.clip(r, lo, hi)
    rc = np.where(stale, np.where((xA >= lo + 60) & (xA <= hi), xA, np.minimum(lo + over, hi)), rc)
    out = np.where(m, (1 - p1) * rc + p1 * np.clip(anc, lo, hi), out)
    # --- departures without NM columns whose flight plan is known from the arrival record
    xLl = T - lk_LOBT
    L = ~m & np.isfinite(xLl) & (xS > 3600)
    A = L & (np.abs(S - lk_LOBT) <= PAD)       # schedule within the hour of the plan: an off-block time at schedule would have paired
    B = L & ~A & (xLl > -900) & (xLl < 3900)   # plan is current: an ordinary off-block time would have paired
    C = L & ~A & ~B
    w1 = np.where(P1 + P2 > 1e-6, P1 / np.where(P1 + P2 > 1e-6, P1 + P2, 1), 1.0)
    on = np.where(is_lirf, xS, np.minimum(xLl + 4100, xS))
    out = np.where(A | C, r, out)
    out = np.where(C & is_lirf, 0.85 * r + 0.15 * xS, out)   # Rome: a schedule-default off-block time stays possible here (1 of 7 cases in 2025)
    out = np.where(B, w1 * on + (1 - w1) * shift, out)
    # not late, plan current and no room below the window: the off-block time is more than an hour before the plan's
    E = ~m & np.isfinite(xLl) & (xS <= 3600) & (xLl > -900) & (xLl - PAD < 240)
    out = np.where(E, np.maximum(out, xLl + 4000), out)
    # Rome, summer time: off-block times entered in UTC and stored as local time are two hours early, which always
    # breaks the pairing.  A flight number that is normally paired, not late and yet unpaired is such a record in
    # 18 of 25 cases in 2025 (taxi time = ordinary taxi time + 7200 s) and longer still in 3 more.
    G = np.zeros(len(out), bool) if tz_summer is None else (np.asarray(tz_summer, bool) & ~m & (xS <= 3600) & (xS >= -1800))
    out = np.where(G, np.maximum(out, 7900.0), out)
    if stats is not None:
        stats.update(G=int(G.sum()))
        stats.update(matched_changed=int((m & (np.abs(out - p) > 1)).sum()), stale=int(stale.sum()),
                     on_infeasible=int((m & ~feas & (P1 > 0.01)).sum()), linked=int(L.sum()), A=int(A.sum()), B=int(B.sum()), C=int(C.sum()), E=int(E.sum()))
    return out
