#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small test of the public OpenSky flights API: does its "first seen" time track off-block times?

Fetches the departures of four airports for two days of January 2025 (a labelled training
month) into ../data/external/opensky/. Uses about 120 of the day's API credits.

  python3 opensky_probe.py

Works without an account if anonymous access is allowed. If the API asks for a login, create
an API client on your OpenSky account page, save the file it gives you as ../opensky_api.json
(keys clientId and clientSecret; it is git-ignored) and run this again.
"""
import json, os, sys, time, urllib.parse, urllib.request, urllib.error

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "data", "external", "opensky"); os.makedirs(out, exist_ok=True)
API = "https://opensky-network.org/api/flights/departure"
AUTH = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
BEGIN, END = 1736812800, 1736985600          # 2025-01-14 00:00 UTC to 2025-01-16 00:00 UTC


def token():
    path = os.path.join(here, "..", "opensky_api.json")
    if not os.path.exists(path):
        return None
    c = json.load(open(path))
    data = urllib.parse.urlencode({"grant_type": "client_credentials", "client_id": c.get("clientId") or c.get("client_id"),
                                   "client_secret": c.get("clientSecret") or c.get("client_secret")}).encode()
    with urllib.request.urlopen(urllib.request.Request(AUTH, data=data), timeout=30) as r:
        return json.loads(r.read())["access_token"]


tok = token()
print("using", "an API client login" if tok else "anonymous access")
for apt in ["LSZH", "EGLL", "EHAM", "LIRF"]:
    url = f"{API}?airport={apt}&begin={BEGIN}&end={END}"
    req = urllib.request.Request(url, headers={"User-Agent": "prc2026-probe", **({"Authorization": "Bearer " + tok} if tok else {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read(); left = r.headers.get("X-Rate-Limit-Remaining")
        rows = json.loads(body)
        json.dump(rows, open(os.path.join(out, f"probe_{apt}.json"), "w"))
        print(f"{apt}: {len(rows)} flights saved; credits left: {left}; fields: {sorted(rows[0].keys()) if rows else '-'}")
    except urllib.error.HTTPError as e:
        print(f"{apt}: HTTP {e.code} {e.reason}; retry-after: {e.headers.get('X-Rate-Limit-Retry-After-Seconds')}; body: {e.read()[:200]!r}")
        if e.code in (401, 403):
            sys.exit("The API wants a login: create an API client on your OpenSky account page, save its file as ../opensky_api.json, run again.")
    except Exception as e:
        print(f"{apt}: failed: {e}")
    time.sleep(2)
print("done; files are in", os.path.abspath(out))
