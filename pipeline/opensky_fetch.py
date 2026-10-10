#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Download departures from the public OpenSky flights API, two days per request, resumable.

  python3 opensky_fetch.py 2026-01-01 2026-02-01 EHAM EGLL LSZH     # dates are UTC, end exclusive

Each request costs 30 API credits (a standard account has 4,000 a day). Files that already
exist are skipped, so the same command can be run again the next day to carry on. Stops by
itself when the credits run out. Login: ../opensky_api.json (clientId, clientSecret) from the
"API client" section of the OpenSky account page; without it the script tries anonymously.
"""
import datetime as dt, json, os, sys, time, urllib.parse, urllib.request, urllib.error

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "data", "external", "opensky"); os.makedirs(out, exist_ok=True)
API = "https://opensky-network.org/api/flights/departure"
AUTH = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
if len(sys.argv) < 4:
    sys.exit(__doc__)
day = lambda s: int(dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())
t0, t1, apts = day(sys.argv[1]), day(sys.argv[2]), sys.argv[3:]
_tok = {"v": None, "t": 0}


def token():
    path = os.path.join(here, "..", "opensky_api.json")
    if not os.path.exists(path):
        return None
    if _tok["v"] and time.time() - _tok["t"] < 1200:
        return _tok["v"]
    c = json.load(open(path))
    data = urllib.parse.urlencode({"grant_type": "client_credentials", "client_id": c.get("clientId") or c.get("client_id"),
                                   "client_secret": c.get("clientSecret") or c.get("client_secret")}).encode()
    with urllib.request.urlopen(urllib.request.Request(AUTH, data=data), timeout=30) as r:
        _tok.update(v=json.loads(r.read())["access_token"], t=time.time())
    return _tok["v"]


done = skipped = 0
for b in range(t0, t1, 172800):
    e = min(b + 172800, t1)
    for apt in apts:
        path = os.path.join(out, f"dep_{apt}_{b}_{e}.json")
        if os.path.exists(path):
            skipped += 1
            continue
        tok = token()
        req = urllib.request.Request(f"{API}?airport={apt}&begin={b}&end={e}",
                                     headers={"User-Agent": "prc2026", **({"Authorization": "Bearer " + tok} if tok else {})})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                rows = json.loads(r.read()); left = r.headers.get("X-Rate-Limit-Remaining")
        except urllib.error.HTTPError as err:
            if err.code == 404:                      # the API answers 404 when it has no flights
                rows, left = [], err.headers.get("X-Rate-Limit-Remaining")
            elif err.code == 429:
                sys.exit(f"out of credits after {done} requests; retry in {err.headers.get('X-Rate-Limit-Retry-After-Seconds')} s. Run the same command again then.")
            else:
                sys.exit(f"{apt} {b}: HTTP {err.code} {err.reason} {err.read()[:200]!r}")
        json.dump(rows, open(path, "w")); done += 1
        print(f"{apt} {dt.datetime.fromtimestamp(b, dt.timezone.utc):%Y-%m-%d}: {len(rows)} flights; credits left {left}", flush=True)
        time.sleep(1)
print(f"finished: {done} requests made, {skipped} files were already there")
