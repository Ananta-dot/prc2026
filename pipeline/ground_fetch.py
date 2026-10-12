#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Download OpenSky ground positions (Trino historical database), one point per 10 s per aircraft,
inside a box around each airport, one file per airport and UTC day.  Resumable: existing files are skipped.

  python3 ground_fetch.py 2025-01-14 2025-01-15 LIRF EGLL      # dates UTC, end exclusive
Needs an OpenSky account with access to the historical database (pyopensky login).
"""
import os, sys, time
import pandas as pd

APT = {"EDDF": (50.0379, 8.5622), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543), "EHAM": (52.3086, 4.7639),
       "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
       "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519)}
here = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(here, "..", "data", "external", "opensky", "ground10")
os.makedirs(OUT, exist_ok=True)


def sql(apt, day):
    lat, lon = APT[apt]
    t0 = int(pd.Timestamp(day, tz="UTC").timestamp()); t1 = t0 + 86400
    h0, h1 = t0 - 4 * 3600, t1                     # 4 h before midnight: stands of early departures
    return f"""SELECT time, icao24, callsign, lat, lon, velocity FROM state_vectors_data4
      WHERE hour >= {h0} AND hour < {h1} AND time >= {h0} AND time < {t1} AND time % 10 = 0 AND onground
      AND lat BETWEEN {lat-0.045} AND {lat+0.045} AND lon BETWEEN {lon-0.07} AND {lon+0.07}"""


def fetch(engine, apt, day):
    """engine: one SQLAlchemy engine for the whole run (pyopensky's Trino.query builds a new one, and a
    new browser login, for every query)."""
    from sqlalchemy import text
    path = os.path.join(OUT, f"{apt}_{day}.parquet")
    if os.path.exists(path):
        return "skip", 0
    with engine.connect() as conn:
        df = pd.read_sql(text(sql(apt, day)), conn)
    df.to_parquet(path)
    return "ok", len(df)


if __name__ == "__main__":
    # usage: ground_fetch.py START:END[,START:END...] APT [APT ...]  -- one login, three queries at a time
    from concurrent.futures import ThreadPoolExecutor
    from pyopensky.trino import Trino
    ranges, apts = sys.argv[1], sys.argv[2:]
    days = [d for part in ranges.split(",") for d in pd.date_range(*part.split(":"), inclusive="left").strftime("%Y-%m-%d")]
    eng = Trino().engine()                         # one engine: the login token is kept for the whole run
    fetch(eng, apts[0], days[0])                   # first query alone, so the login happens once

    def job(ad):
        apt, day = ad; t = time.time()
        for attempt in range(3):
            try:
                st, n = fetch(eng, apt, day); break
            except Exception as e:
                st, n = f"error {type(e).__name__}: {str(e)[:120]}", 0
                time.sleep(10)
        print(f"{apt} {day}: {st} {n:,} rows ({time.time()-t:.0f}s)", flush=True)

    with ThreadPoolExecutor(3) as ex:
        list(ex.map(job, [(a, d) for d in days for a in apts]))
    print("all done", flush=True)
