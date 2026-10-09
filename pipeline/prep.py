# SPDX-License-Identifier: GPL-3.0-or-later
"""Raw monthly parquet files -> one compact frame (categoricals + float epoch seconds).

  python3 prep.py ../data/raw ../data/processed/compact.parquet [final_ranking.parquet]
"""
import glob, os, sys
import numpy as np, pandas as pd
from pandas.api.types import union_categoricals

TARGET, ID = "TAXITIME_SEC_mvt", "MVT_ID_mvt"
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt",
             "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def secs(s):
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return (s - EPOCH).dt.total_seconds().to_numpy(dtype="float64")


def compact(d, is_rank):
    out = {}
    for c in d.columns:
        if c in TIME_COLS:
            out["t_" + c] = secs(d[c])
        elif pd.api.types.is_numeric_dtype(d[c]):
            out[c] = d[c].to_numpy(dtype="float64")
        else:
            out[c] = pd.Categorical(d[c])
    o = pd.DataFrame(out)
    o["is_rank"] = is_rank
    o["is_dep"] = (o["PHASE_mvt"].astype(str).str.upper() == "DEP").to_numpy()
    o["APT"] = pd.Categorical(np.where(o["is_dep"], o["ADEP_mvt"].astype(str), o["ADES_mvt"].astype(str)))
    return o


def concat(frames):
    out = {}
    for c in frames[0].columns:
        if isinstance(frames[0][c].dtype, pd.CategoricalDtype):
            out[c] = union_categoricals([f[c] for f in frames])
        else:
            out[c] = np.concatenate([f[c].to_numpy() for f in frames])
    return pd.DataFrame(out)


def load_raw(data_dir, ranking="ranking.parquet"):
    files = sorted(glob.glob(os.path.join(data_dir, "training_*.parquet")))
    if not files:
        sys.exit(f"no training_*.parquet in {data_dir}")
    frames = [compact(pd.read_parquet(f), False) for f in files]
    frames.append(compact(pd.read_parquet(os.path.join(data_dir, ranking)), True))
    return concat(frames)


if __name__ == "__main__":
    data_dir, out = sys.argv[1], sys.argv[2]
    df = load_raw(data_dir, *sys.argv[3:4])     # optional third argument: the ranking file, e.g. final_ranking.parquet
    df.to_parquet(out, index=False)
    print(df.shape, round(df.memory_usage(deep=True).sum() / 1e6), "MB")
