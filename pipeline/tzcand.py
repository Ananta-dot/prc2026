"""Rome departures whose off-block time is likely shifted by the UTC offset: no NM columns, not late, a flight number that is
normally paired, destination not Rome.  usage: python3 new/tzcand.py <compact> <out.parquet>"""
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
cols=['MVT_ID_mvt','APT','FLIGHT_mvt','ADES_mvt','t_MVT_TIME_UTC_mvt','t_SCHED_TIME_UTC_mvt','is_rank','is_dep','t_LOBT_flt']
c=pq.read_table(sys.argv[1],columns=cols,filters=[('is_dep','=',True),('APT','=','LIRF')]).to_pandas()
T=c.t_MVT_TIME_UTC_mvt.to_numpy(); xS=T-c.t_SCHED_TIME_UTC_mvt.to_numpy(); unm=c.t_LOBT_flt.isna().to_numpy()
fl=c.FLIGHT_mvt.astype(str).to_numpy(); ont=xS<=3000
d=pd.DataFrame({'k':fl,'s':ont.astype(float),'i':(ont&unm).astype(float)})
n=d.groupby('k').s.transform('sum').to_numpy()-d.s.to_numpy(); i=d.groupby('k').i.transform('sum').to_numpy()-d.i.to_numpy()   # other on-time departures of this flight number
usual=(n>=5)&(i/np.maximum(n,1)<0.05)&(fl!='nan')
dt=pd.to_datetime(T,unit='s'); md=dt.strftime('%m%d')
off=np.where((md>='0330')&(md<'1026'),7200.0,3600.0)       # CEST from the last Sunday of March to the last Sunday of October (29/30 March, 25/26 October)
cand=unm&(xS<=3600)&usual&(c.ADES_mvt.astype(str).to_numpy()!='LIRF')
out=pd.DataFrame({'MVT_ID_mvt':c.MVT_ID_mvt.to_numpy()[cand],'off':off[cand],'xS':xS[cand],'n_other':n[cand],'is_rank':c.is_rank.to_numpy()[cand],'FLIGHT':fl[cand]})
out.to_parquet(sys.argv[2]); print(len(out),'candidates;',int(out.is_rank.sum()),'in the ranking rows')
