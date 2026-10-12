# SPDX-License-Identifier: GPL-3.0-or-later
"""Off-block evidence from OpenSky ground positions (files of ground_fetch.py, one point per 10 s).

For each departure (matched on callsign, take-off time from the movement record) the aircraft's
ground track in the 4 h before take-off is searched backwards for the stand: the last stop of at
least 20 min within 50 m, or a gap of at least 15 min (transponder off at the stand).  The off-block
estimate is the first point after it; if neither is found, the first sighting.

Features (NaN when there is no track): g_est (take-off minus the off-block estimate), g_how (0 first
sighting, 1 stop, 2 gap), g_still (seconds the aircraft stayed within 25 m of its first point after
the estimate), g_n (points in the taxi), g_first (take-off minus first sighting), g_last (take-off
minus last ground point), g_v0 (mean speed of the first three points after the estimate).
"""
import os
import numpy as np, pandas as pd

APT = {"EDDF": (50.0379, 8.5622), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543), "EHAM": (52.3086, 4.7639),
       "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
       "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519)}
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "external", "opensky", "ground10")
COLS = ["g_est", "g_how", "g_still", "g_n", "g_first", "g_last", "g_v0"]


def _offblock(t, x, y, R=50.0, MIN_STOP=1200, GAP=900):
    i = len(t) - 1
    while i > 0:
        if t[i] - t[i - 1] >= GAP:
            return i, 2
        j = i - 1
        while j > 0 and np.hypot(x[j - 1] - x[i - 1], y[j - 1] - y[i - 1]) <= R:
            j -= 1
        if t[i - 1] - t[j] >= MIN_STOP:
            return i, 1
        i = j if j < i - 1 else i - 1
    return 0, 0


def features(apt, day, deps):
    """deps: DataFrame with columns id, cs (callsign), T (take-off, epoch s) for departures of `apt`
    taking off on UTC `day`.  -> DataFrame id + COLS (rows only for departures with a track)."""
    path = os.path.join(DIR, f"{apt}_{day}.parquet")
    if not os.path.exists(path) or not len(deps):
        return pd.DataFrame(columns=["id"] + COLS)
    sv = pd.read_parquet(path)
    if not len(sv):
        return pd.DataFrame(columns=["id"] + COLS)
    lat = APT[apt][0]
    tm = sv["time"]
    sv["t"] = (pd.to_datetime(tm, utc=True).astype("int64") / 1e9) if str(tm.dtype).startswith(("datetime", "timestamp")) else pd.to_numeric(tm).astype("float64")
    for k in ("lat", "lon", "velocity"):
        sv[k] = pd.to_numeric(sv[k], errors="coerce").astype("float64")
    sv["cs"] = sv["callsign"].astype(str).str.strip()
    sv = sv.sort_values(["cs", "t"])
    groups = {k: g for k, g in sv.groupby("cs")}
    out = []
    for r in deps.itertuples():
        g = groups.get(r.cs)
        if g is None:
            continue
        g = g[(g.t > r.T - 4 * 3600) & (g.t <= r.T + 60)]
        if len(g) < 3 or r.T - g.t.iloc[-1] > 600:        # no ground track ending near the take-off
            continue
        t = g.t.to_numpy(); x = (g.lon.to_numpy() - g.lon.iloc[-1]) * 111320 * np.cos(np.radians(lat)); y = (g.lat.to_numpy() - g.lat.iloc[-1]) * 110540
        i, how = _offblock(t, x, y)
        d = np.hypot(x[i:] - x[i], y[i:] - y[i]); far = np.flatnonzero(d > 25)
        still = t[i + far[0]] - t[i] if len(far) else t[-1] - t[i]
        v = g.velocity.to_numpy()[i:i + 3]
        out.append((r.id, r.T - t[i], how, still, len(t) - i, r.T - t[0], r.T - t[-1], np.nanmean(v) if len(v) else np.nan))
    return pd.DataFrame(out, columns=["id"] + COLS)
