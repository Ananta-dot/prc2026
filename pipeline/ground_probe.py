#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Probe: do OpenSky ground positions (Trino historical database) give the off-block time?

For one airport and one UTC day of 2025 (labelled), downloads the state vectors on the ground
inside a box around the airport, finds for each departing aircraft the first moment it moves away
from where it was first seen, and compares take-off minus that moment with the recorded taxi time.

  python3 ground_probe.py LSZH 2025-01-14
Needs an OpenSky account with access to the historical database; pyopensky asks for the login
(username in its settings.conf; the first query opens the browser / asks for the password).
"""
import sys, os, time
import numpy as np, pandas as pd

APT = {"EDDF": (50.0379, 8.5622), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543), "EHAM": (52.3086, 4.7639),
       "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
       "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519)}
here = os.path.dirname(os.path.abspath(__file__))
apt, day = sys.argv[1], sys.argv[2]
lat, lon = APT[apt]
bounds = (lon - 0.07, lat - 0.045, lon + 0.07, lat + 0.045)          # west, south, east, north
start = pd.Timestamp(day, tz="UTC"); stop = start + pd.Timedelta(days=1)
out = os.path.join(here, "..", "data", "external", "opensky", f"ground_{apt}_{day}.parquet")

if os.path.exists(out):
    sv = pd.read_parquet(out)
else:
    from pyopensky.trino import Trino
    t0 = time.time()
    sv = Trino().history(start - pd.Timedelta(hours=1), stop, bounds=bounds,
                         selected_columns=("time", "icao24", "callsign", "lat", "lon", "velocity", "onground"))
    if sv is None or len(sv) == 0:
        sys.exit("no state vectors returned (no access, or no coverage)")
    sv.to_parquet(out)
    print(f"downloaded {len(sv):,} state vectors in {time.time()-t0:.0f}s -> {out}")
sv["t"] = pd.to_datetime(sv["time"], utc=True).astype("int64") / 1e9
sv["callsign"] = sv["callsign"].astype(str).str.strip()
print(f"{len(sv):,} vectors, {sv.icao24.nunique()} aircraft, on ground {sv.onground.mean():.2f}")

# per aircraft: take-off = first airborne point after a ground segment; first movement = first ground
# point more than 60 m from where the aircraft was first seen on the ground that day (within 3 h before take-off)
def first_moves(g):
    g = g.sort_values("t")
    gr = g[g.onground]
    air = g[~g.onground]
    res = []
    for ta in air.t[(air.t.diff().fillna(1e9) > 600)]:                 # starts of airborne segments
        seg = gr[(gr.t < ta) & (gr.t > ta - 10800)]
        if len(seg) < 3:
            continue
        x = (seg.lon - seg.lon.iloc[0]) * 111320 * np.cos(np.radians(lat)); y = (seg.lat - seg.lat.iloc[0]) * 110540
        moved = np.hypot(x, y) > 60
        res.append(dict(icao24=g.icao24.iloc[0], callsign=seg.callsign.iloc[-1], t_air=ta, t_first=seg.t.iloc[0],
                        t_move=seg.t[moved.to_numpy()].iloc[0] if moved.any() else np.nan, n=len(seg)))
    return res

rows = [r for _, g in sv.groupby("icao24") for r in first_moves(g)]
dep = pd.DataFrame(rows)
print(f"{len(dep)} take-offs with ground track")

c = pd.read_parquet(os.path.join(here, "..", "data", "processed", "compact.parquet"),
                    columns=["APT", "is_dep", "is_rank", "CALLSIGN_flt", "FLIGHT_mvt", "t_MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "t_SCHED_TIME_UTC_mvt"])
c = c[(c.APT == apt) & c.is_dep & ~c.is_rank & (c.t_MVT_TIME_UTC_mvt >= start.timestamp()) & (c.t_MVT_TIME_UTC_mvt < stop.timestamp())].copy()
c["cs"] = c.CALLSIGN_flt.astype(str).where(c.CALLSIGN_flt.notna(), c.FLIGHT_mvt.astype(str)).str.strip()
m = c.merge(dep, left_on="cs", right_on="callsign")
m = m[(m.t_air - m.t_MVT_TIME_UTC_mvt).abs() < 300]
m["est"] = m.t_MVT_TIME_UTC_mvt - m.t_move
y = m.TAXITIME_SEC_mvt
ok = m.est.notna() & (y <= 3600)
print(f"departures that day: {len(c)}, matched to a ground track: {len(m)} ({len(m)/max(len(c),1):.0%}), with a movement: {ok.sum()}")
if ok.any():
    e = (m.est - y)[ok]
    print(f"take-off minus first movement vs recorded taxi time: median error {e.median():.0f}s, MAE {e.abs().mean():.0f}s, "
          f"RMSE {np.sqrt((e**2).mean()):.0f}s, within 60s {np.mean(np.abs(e)<=60):.2f}, within 120s {np.mean(np.abs(e)<=120):.2f}, corr {np.corrcoef(m.est[ok], y[ok])[0,1]:.3f}")
    print(f"(for scale: taxi-time std that day {y[ok].std():.0f}s; our model's ordinary-row RMSE at this airport ~150-200s)")
    first = m.t_MVT_TIME_UTC_mvt - m.t_first
    e2 = (first - y)[ok]
    print(f"take-off minus first ground sighting: median error {e2.median():.0f}s, RMSE {np.sqrt((e2**2).mean()):.0f}s")
