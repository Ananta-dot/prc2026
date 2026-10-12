#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Append the OpenSky ground-track features of ground.py to a feature file of features.py.

  python3 ground_feats.py ../data/processed/compact.parquet ../data/processed/feat.parquet ../data/processed/feat_g.parquet

Departures without a downloaded track (other airports, missing days, no callsign match) get NaN.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
import ground

comp, feat, out = sys.argv[1:4]


def one(args):
    apt, day, deps = args
    return ground.features(apt, day, deps)


if __name__ == "__main__":
    c = pd.read_parquet(comp, columns=["MVT_ID_mvt", "APT", "is_dep", "CALLSIGN_flt", "FLIGHT_mvt", "t_MVT_TIME_UTC_mvt"])
    c = c[c.is_dep & c.APT.astype(str).isin(["LSZH", "EHAM", "EGLL", "EDDM", "LEBL"])].drop_duplicates("MVT_ID_mvt")
    c["cs"] = c.CALLSIGN_flt.astype(str).where(c.CALLSIGN_flt.notna(), c.FLIGHT_mvt.astype(str)).str.strip()
    c["day"] = pd.to_datetime(c.t_MVT_TIME_UTC_mvt, unit="s").dt.strftime("%Y-%m-%d")
    jobs = [(str(a), d, pd.DataFrame({"id": g.MVT_ID_mvt.to_numpy(), "cs": g.cs.to_numpy(), "T": g.t_MVT_TIME_UTC_mvt.to_numpy()}))
            for (a, d), g in c.groupby([c.APT.astype(str), "day"])]
    with ProcessPoolExecutor() as ex:
        parts = [p for p in ex.map(one, jobs, chunksize=8) if len(p)]
    g = pd.concat(parts).drop_duplicates("id").set_index("id")
    print(f"ground features for {len(g):,} of {len(c):,} departures at the five airports")
    t = pq.read_table(feat)
    ids = t.column("m_id").to_numpy()
    for k in ground.COLS:
        t = t.append_column(k, pa.array(g[k].reindex(ids).to_numpy("float32")))
    pq.write_table(t, out)
    print(f"wrote {out}: {t.num_rows:,} rows, {t.num_columns} columns")
