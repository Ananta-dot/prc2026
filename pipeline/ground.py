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
COLS = ["g_est", "g_how", "g_still", "g_n", "g_first", "g_last", "g_v0",
        "g_path", "g_stop", "g_nstop", "g_lastrun",                 # this flight's observed taxi
        "q_move10", "q_hold10", "q_move30", "q_hold30"]             # every aircraft on the ground: taxiing / holding


def _surface(sv, lat):
    """Per minute: distinct aircraft taxiing (over 3 m/s) and holding (under 1 m/s, but moved within the
    last 15 min, so not parked).  Speeds from the 10 s positions of each aircraft."""
    sv = sv.sort_values(["icao24", "t"])
    same = sv.icao24.to_numpy()[1:] == sv.icao24.to_numpy()[:-1]
    dt = np.diff(sv.t.to_numpy()); dx = np.diff(sv.lon.to_numpy()) * 111320 * np.cos(np.radians(lat)); dy = np.diff(sv.lat.to_numpy()) * 110540
    sp = np.full(len(sv), np.nan)
    ok = same & (dt > 0) & (dt <= 60)
    sp[1:][ok] = np.hypot(dx, dy)[ok] / dt[ok]
    mov = sp > 3
    lastmov = pd.Series(np.where(mov, sv.t.to_numpy(), np.nan)).groupby(sv.icao24.to_numpy()).ffill().to_numpy()
    hold = (sp < 1) & (sv.t.to_numpy() - lastmov <= 900)
    minute = (sv.t.to_numpy() // 60).astype("int64")
    f = pd.DataFrame({"m": minute, "a": sv.icao24.to_numpy(), "mov": mov, "hold": hold})
    nm = f[f.mov].groupby("m").a.nunique(); nh = f[f.hold].groupby("m").a.nunique()
    m0, m1 = minute.min(), minute.max() + 1
    idx = np.arange(m0, m1)
    cm = np.concatenate([[0], np.cumsum(nm.reindex(idx, fill_value=0).to_numpy())])
    ch = np.concatenate([[0], np.cumsum(nh.reindex(idx, fill_value=0).to_numpy())])

    def window(T, lo, hi):          # mean per minute over [T+lo, T+hi) seconds
        a = np.clip((T + lo) // 60 - m0, 0, len(idx)).astype("int64"); b = np.clip((T + hi) // 60 - m0, 0, len(idx)).astype("int64")
        n = np.maximum(b - a, 1)
        return (cm[b] - cm[a]) / n, (ch[b] - ch[a]) / n
    return window


def _taxi(t, x, y):
    """Path length, stopped time, stops of 30 s or more, and time from the end of the last such stop to the last point."""
    if len(t) < 2:
        return 0.0, 0.0, 0, np.nan
    step = np.hypot(np.diff(x), np.diff(y)); dt = np.diff(t)
    still = step < 15                                   # under 1.5 m/s over a 10 s step
    runs, cur, end = 0, 0.0, np.nan
    for k in range(len(still)):
        if still[k]:
            cur += dt[k]
        else:
            if cur >= 30:
                runs += 1; end = t[k]
            cur = 0.0
    if cur >= 30:
        runs += 1; end = t[-1]
    return float(step.sum()), float(dt[still].sum()), runs, (t[-1] - end) if np.isfinite(end) else np.nan


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
    window = _surface(sv, lat)
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
        v0 = np.nanmean(v) if len(v) and np.isfinite(v).any() else np.nan
        path, stop, nstop, lastrun = _taxi(t[i:], x[i:], y[i:])
        out.append((r.id, r.T - t[i], how, still, len(t) - i, r.T - t[0], r.T - t[-1], v0, path, stop, nstop, lastrun))
    f = pd.DataFrame(out, columns=["id"] + COLS[:11])
    # the airport's surface state does not need this flight's own track: every departure gets it
    T = deps["T"].to_numpy("float64")
    m10, h10 = window(T, -600, 0); m30, h30 = window(T, -1800, -1200)
    q = pd.DataFrame({"id": deps["id"].to_numpy(), "q_move10": m10, "q_hold10": h10, "q_move30": m30, "q_hold30": h30})
    return q.merge(f, on="id", how="left")[["id"] + COLS]
