#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Download airport weather observations (METAR) for the 10 challenge airports.

Source: Iowa Environmental Mesonet ASOS/METAR archive (open data),
https://mesonet.agron.iastate.edu/request/download.phtml
Writes ../data/external/metar_<ICAO>.csv, one file per airport, UTC timestamps.

  python3 get_weather.py
"""
import os, sys, time, urllib.parse, urllib.request

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
FIELDS = ["tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "skyc1", "skyl1", "wxcodes", "alti"]
BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
here = os.path.dirname(os.path.abspath(__file__))
dest = os.path.join(here, "..", "data", "external")
os.makedirs(dest, exist_ok=True)


def fetch(station, fields):
    q = [("station", station)] + [("data", f) for f in fields] + [
        ("year1", "2025"), ("month1", "1"), ("day1", "1"),
        ("year2", "2026"), ("month2", "8"), ("day2", "2"),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", "3"), ("report_type", "4")]
    req = urllib.request.Request(BASE + "?" + urllib.parse.urlencode(q), headers={"User-Agent": "prc2026-courageous-onion"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read().decode("utf-8", "replace")


for ap in AIRPORTS:
    path = os.path.join(dest, f"metar_{ap}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 500_000:
        print("have   ", path); continue
    for attempt in range(3):
        try:
            txt = fetch(ap, FIELDS)
            if "wxcodes" not in txt.split("\n", 1)[0]:
                txt = fetch(ap, ["all"])
            break
        except Exception as e:                      # the archive rate-limits; wait and retry
            print(f"  {ap}: {e}; retrying in 20 s"); time.sleep(20)
    else:
        sys.exit(f"could not download {ap}")
    with open(path, "w") as f:
        f.write(txt)
    print(f"fetched {ap}: {txt.count(chr(10)):,} lines, header: {txt.split(chr(10), 1)[0][:90]}")
    time.sleep(5)
