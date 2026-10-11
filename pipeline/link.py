# SPDX-License-Identifier: GPL-3.0-or-later
"""Flight plans of departures that have no matched NM flight, recovered from the arrival record
at the destination when the destination is one of the ten airports.

A movement and an NM flight are paired only when the off-block time is within one hour of the
flight plan's last off-block time (LOBT); arrivals are paired on the landing time.  A departure
whose off-block time is more than an hour from the LOBT therefore has no NM columns, but the
same flight's arrival record at the destination does.

usage: python3 link.py <compact.parquet> <out.parquet>
"""
import sys
import numpy as np
import pandas as pd

TEN = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
COLS = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'ADEP_mvt', 'ADES_mvt', 't_MVT_TIME_UTC_mvt', 'AIRCRAFT_TYPE_mvt',
        't_LOBT_flt', 't_EOBT_1_flt', 't_AOBT_3_flt', 't_IOBT_flt', 'is_dep', 'APT']


def link(compact):
    c = pd.read_parquet(compact, columns=COLS).rename(columns={'t_MVT_TIME_UTC_mvt': 'T'})
    c['m'] = c.t_LOBT_flt.notna()
    d = c[c.is_dep & c.ADES_mvt.isin(TEN)].copy()
    a = c[~c.is_dep & c.ADEP_mvt.isin(TEN) & c.m].copy()
    del c
    for k in ('ADEP_mvt', 'ADES_mvt', 'APT', 'AIRCRAFT_TYPE_mvt'):
        d[k] = d[k].astype(str)
        a[k] = a[k].astype(str)
    dm = d[d.m]
    pr = dm[['FLIGHT_ID_mvt', 'APT', 'ADES_mvt', 'T']].merge(
        a[['FLIGHT_ID_mvt', 'T']].rename(columns={'T': 'Ta'}), on='FLIGHT_ID_mvt')
    pr['ft'] = pr.Ta - pr['T']
    ftm = pr.groupby(['APT', 'ADES_mvt']).ft.agg(med='median', lo=lambda x: x.quantile(0.005), hi=lambda x: x.quantile(0.995))
    orph = a[~a.FLIGHT_ID_mvt.isin(set(dm.FLIGHT_ID_mvt))]          # arrivals whose NM flight is on no departure record
    u = d[~d.m][['MVT_ID_mvt', 'APT', 'ADES_mvt', 'T', 'AIRCRAFT_TYPE_mvt']]
    o = orph[['ADEP_mvt', 'APT', 'T', 'AIRCRAFT_TYPE_mvt', 't_LOBT_flt', 't_EOBT_1_flt', 't_AOBT_3_flt', 't_IOBT_flt', 'MVT_ID_mvt']].rename(
        columns={'ADEP_mvt': 'o', 'APT': 'ADES_mvt', 'T': 'Ta', 'AIRCRAFT_TYPE_mvt': 'typa', 'MVT_ID_mvt': 'aid',
                 't_LOBT_flt': 'lk_LOBT', 't_EOBT_1_flt': 'lk_EOBT', 't_AOBT_3_flt': 'lk_AOBT', 't_IOBT_flt': 'lk_IOBT'})
    cand = u.merge(o, left_on=['APT', 'ADES_mvt'], right_on=['o', 'ADES_mvt']).merge(ftm, left_on=['APT', 'ADES_mvt'], right_index=True)
    cand['ft'] = cand.Ta - cand['T']
    cand = cand[(cand.ft > cand.lo - 600) & (cand.ft < cand.hi + 600)]
    cand['dft'] = (cand.ft - cand.med).abs()
    ta, tb = cand.AIRCRAFT_TYPE_mvt.fillna('nan').astype(str).str.lower(), cand.typa.fillna('nan').astype(str).str.lower()
    ok = (ta == tb) | (ta == 'nan') | (tb == 'nan') | ((ta.str[:1] == tb.str[:1]) & (cand.dft < 900))   # same type, or type not given
    cand = cand[ok & (cand.APT != cand.ADES_mvt)]
    cand = cand.sort_values('dft')
    cand = cand.drop_duplicates('MVT_ID_mvt').drop_duplicates('aid')    # closest arrival per departure, each arrival used once
    return cand[['MVT_ID_mvt', 'lk_LOBT', 'lk_EOBT', 'lk_AOBT', 'lk_IOBT', 'dft']].reset_index(drop=True)


if __name__ == '__main__':
    L = link(sys.argv[1])
    L.to_parquet(sys.argv[2])
    print(f"{len(L)} unmatched departures linked to an arrival record")
