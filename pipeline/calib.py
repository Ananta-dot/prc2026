# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-lateness-band upward shift (in logits) of the gate's near-schedule probability, applied before the rules.
Shared by postval.py experiments and build.py."""
import numpy as np

BANDS = [(7200.0, 14400.0), (14400.0, 28800.0)]


def shift_gate(p, r, P, xS, shifts):
    """-> new (p, P) with P[:,1] shifted by shifts[i] logits in BANDS[i]; the near-schedule value the model used is kept."""
    P = P.copy(); P0, P1, P2 = P[:, 0], P[:, 1], P[:, 2]
    shift = 86400 + np.clip(r, 0, 3600)
    with np.errstate(all="ignore"):
        anc = np.where(P1 > 1e-9, (p - P0 * r - P2 * shift) / np.where(P1 > 1e-9, P1, 1), 0.0)
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    x = np.nan_to_num(xS)
    for (lo, hi), b in zip(BANDS, shifts):
        k = (x > lo) & (x <= hi) & (P1 > 1e-4) & (b != 0)
        n1 = np.minimum(1 / (1 + np.exp(-(lg(P1[k]) + b))), 1 - P2[k])
        P[k, 1] = n1; P[k, 0] = 1 - n1 - P2[k]
    p = P[:, 0] * r + P[:, 1] * anc + P[:, 2] * shift
    return p, P


TZ = {"EGLL": "Europe/London", "LTFM": "Europe/Istanbul"}      # the other eight airports keep Central European time


def same_local_day(apt, S, T):
    """True where the take-off is on the same local calendar day as the schedule.  An off-block time
    "with the schedule's date" then equals the real one, so the day-shift record type is impossible
    (12 of 12 Rome cases in 2025 took off on a later local day)."""
    import pandas as pd
    apt = np.asarray(apt).astype(str); out = np.zeros(len(apt), bool)
    for a in np.unique(apt):
        k = apt == a; tz = TZ.get(a, "Europe/Rome")
        s = pd.to_datetime(S[k], unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
        t = pd.to_datetime(T[k], unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
        out[k] = np.asarray(t == s)
    return out & np.isfinite(S) & np.isfinite(T)


def adjust(p, r, P, xS, S, T, apt, gate_shift=0.0):
    """Gate shift for every flight more than an hour late, then no day-shift mass on same-day take-offs."""
    p, P = shift_gate(p, r, P, xS, []) if gate_shift == 0 else _global(p, r, P, xS, gate_shift)
    same = same_local_day(apt, S, T) & (P[:, 2] > 0)
    if same.any():
        shift = 86400 + np.clip(r, 0, 3600)
        with np.errstate(all="ignore"):
            anc = np.where(P[:, 1] > 1e-9, (p - P[:, 0] * r - P[:, 2] * shift) / np.where(P[:, 1] > 1e-9, P[:, 1], 1), np.nan_to_num(xS))
        anc = np.where(P[:, 1] > 1e-9, anc, np.nan_to_num(xS))
        P = P.copy(); P[same, 1] += P[same, 2]; P[same, 2] = 0.0
        p = np.where(same, P[:, 0] * r + P[:, 1] * anc, p)
    return p, P


def _global(p, r, P, xS, b):
    P = P.copy(); P0, P1, P2 = P[:, 0], P[:, 1], P[:, 2]
    shift = 86400 + np.clip(r, 0, 3600)
    with np.errstate(all="ignore"):
        anc = np.where(P1 > 1e-9, (p - P0 * r - P2 * shift) / np.where(P1 > 1e-9, P1, 1), 0.0)
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    k = P1 > 1e-4
    n1 = np.minimum(1 / (1 + np.exp(-(lg(P1[k]) + b))), 1 - P2[k])
    P[k, 1] = n1; P[k, 0] = 1 - n1 - P2[k]
    return P[:, 0] * r + P[:, 1] * anc + P[:, 2] * shift, P
