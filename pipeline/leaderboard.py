#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read the official PRC 2026 leaderboard API and show this team's entries.

The API and competition id are the ones published on the challenge's Ranking page.

  python3 leaderboard.py                 # team courageous-onion
  python3 leaderboard.py some-other-team
"""
import json, re, sys, urllib.request

COMP = "bb3693e1-26bc-4a9e-8619-4fe78b4eab0c"
BASE = "https://datacomp.opensky-network.org/api"
team = sys.argv[1] if len(sys.argv) > 1 else "courageous-onion"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "prc2026-" + team, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def items_of(page):
    if isinstance(page, list):
        return page
    for v in page.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []


rows, cursor, pages = [], None, 0
while pages < 400:
    url = f"{BASE}/competitions/{COMP}/leaderboard" + (f"?cursor={cursor}" if cursor else "")
    try:
        page = get(url)
    except Exception as e:
        print(f"request failed on page {pages + 1}: {e}")
        break
    pages += 1
    if pages == 1:
        print("top-level keys:", list(page.keys()) if isinstance(page, dict) else "list")
    got = items_of(page)
    rows += got
    print(f"  page {pages}: {len(got)} entries (total {len(rows)})", flush=True)
    cursor = page.get("nextCursor") if isinstance(page, dict) else None
    if not cursor or not got:
        break

print(f"pages read: {pages}, submissions listed: {len(rows)}")
if rows:
    print("fields per entry:", list(rows[0].keys()))
    stamp = lambda r: max([str(v) for v in r.values() if isinstance(v, str) and re.match(r"\d{4}-\d{2}-\d{2}T", v)] or [""])
    newest = sorted(rows, key=stamp)[-5:]
    print("\n5 most recent entries on the whole leaderboard:")
    for r in newest:
        print("  ", json.dumps(r)[:300])
    mine = [r for r in rows if team in json.dumps(r)]
    print(f"\nentries mentioning {team}: {len(mine)}")
    for r in mine:
        print("  ", json.dumps(r)[:400])

# list the API's other endpoints, in case one reports per-team submission status or errors
for doc in ("/v3/api-docs", "/openapi.json", "/swagger.json", "/api-docs"):
    try:
        spec = get(BASE + doc)
        print(f"\nendpoints listed in {doc}:")
        for p, ops in spec.get("paths", {}).items():
            print("  ", ", ".join(k.upper() for k in ops), p)
        break
    except Exception:
        continue
