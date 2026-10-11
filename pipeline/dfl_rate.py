# SPDX-License-Identifier: GPL-3.0-or-later
"""Rome: how often a flight number's (or, for a new flight number, its airline's) 2025 departures have an
off-block time equal to the schedule.  Some airlines' off-block times at Rome are always the schedule.

usage: python3 dfl_rate.py <compact.parquet> <out.parquet> [months to leave out and score, e.g. 1,7]"""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq

c = pq.read_table(sys.argv[1], columns=['MVT_ID_mvt', 't_MVT_TIME_UTC_mvt', 't_SCHED_TIME_UTC_mvt', 't_BLOCK_TIME_UTC_mvt', 'FLIGHT_mvt', 'is_rank'],
                  filters=[('APT', '=', 'LIRF'), ('is_dep', '=', True)]).to_pandas()
out_m = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else []
c['dflt'] = (c.t_SCHED_TIME_UTC_mvt - c.t_BLOCK_TIME_UTC_mvt).abs() <= 7
mon = pd.to_datetime(c.t_MVT_TIME_UTC_mvt, unit='s').dt.month
fl = c.FLIGHT_mvt.astype(str)
al = fl.str[:3]
src = (~c.is_rank & ~mon.isin(out_m) & c.FLIGHT_mvt.notna()).to_numpy()
rf = c[src].groupby(fl[src]).dflt.agg(['mean', 'count'])
ra = c[src].groupby(al[src]).dflt.agg(['mean', 'count'])
qf, nf = fl.map(rf['mean']).to_numpy(), fl.map(rf['count']).fillna(0).to_numpy()
qa, na = al.map(ra['mean']).to_numpy(), al.map(ra['count']).fillna(0).to_numpy()
q = np.where(nf >= 10, qf, np.where((na >= 50) & c.FLIGHT_mvt.notna().to_numpy(), qa, np.nan))
tgt = (c.is_rank | mon.isin(out_m)).to_numpy() if out_m else c.is_rank.to_numpy()
o = pd.DataFrame({'MVT_ID_mvt': c.MVT_ID_mvt.to_numpy(), 'q': q})[tgt & np.isfinite(q) & (q >= 0.9)]
o.to_parquet(sys.argv[2])
print(f"{len(o)} Rome departures whose flight number or airline is schedule-default at least 90% of the time; {int((o.q >= 0.99).sum())} at 99% or more")
