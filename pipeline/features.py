# SPDX-License-Identifier: GPL-3.0-or-later
"""Feature matrix for every departure (training + ranking), built from the compact frame.

Nothing here uses the target or any departure's off-block time, so every feature is
available for the ranking rows exactly as it is for the training rows.

  python3 features.py compact.parquet feat.parquet [folder with metar_<ICAO>.csv files]
"""
import gc
import os
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd

TARGET, ID = "TAXITIME_SEC_mvt", "MVT_ID_mvt"
B = 1e11  # (group code, epoch seconds) -> one sortable float


class Events:
    """Time-stamped events grouped by an integer key, with fast window queries."""

    def __init__(self, key, t, **vals):
        ok = ~np.isnan(t) & (key >= 0)
        k = key[ok].astype("float64") * B + t[ok]
        o = np.argsort(k, kind="stable")
        self.k = k[o]
        self.vals = {n: v[ok][o] for n, v in vals.items()}
        self.cs = {}
        for n, v in self.vals.items():
            v = v.astype("float64")
            good = ~np.isnan(v)
            self.cs[n] = (np.concatenate([[0.0], np.cumsum(np.where(good, v, 0.0))]),
                          np.concatenate([[0.0], np.cumsum(good)]))

    def _idx(self, qk, qt, lo, hi):
        q = qk.astype("float64") * B + qt
        return np.searchsorted(self.k, q + lo, side="left"), np.searchsorted(self.k, q + hi, side="left")

    def count(self, qk, qt, lo, hi):
        """events with time in [qt+lo, qt+hi); lo/hi may be arrays."""
        a, b = self._idx(qk, qt, lo, hi)
        n = (b - a).astype("float32")
        n[np.isnan(qt) | np.isnan(lo + hi)] = np.nan
        return n

    def mean(self, name, qk, qt, lo, hi):
        a, b = self._idx(qk, qt, lo, hi)
        s, c = self.cs[name]
        n = c[b] - c[a]
        return np.where(n > 0, (s[b] - s[a]) / np.maximum(n, 1), np.nan).astype("float32")

    def prev(self, qk, qt, j=1, strict=True, max_gap=7 * 86400.0):
        """j-th most recent event before qt in the same group -> (gap seconds, position or -1)."""
        q = qk.astype("float64") * B + qt
        i = np.searchsorted(self.k, q, side="left" if strict else "right") - j
        ok = i >= 0
        gap = np.where(ok, q - self.k[np.maximum(i, 0)], np.nan)
        ok &= (gap >= 0) & (gap <= max_gap)
        return np.where(ok, gap, np.nan).astype("float32"), np.where(ok, i, -1)

    def next(self, qk, qt, j=1, max_gap=7 * 86400.0):
        q = qk.astype("float64") * B + qt
        i = np.searchsorted(self.k, q, side="right") + j - 1
        ok = i < len(self.k)
        gap = np.where(ok, self.k[np.minimum(i, len(self.k) - 1)] - q, np.nan)
        ok &= (gap >= 0) & (gap <= max_gap)
        return np.where(ok, gap, np.nan).astype("float32"), np.where(ok, i, -1)

    def take(self, name, pos):
        v = self.vals[name].astype("float64")
        return np.where(pos >= 0, v[np.maximum(pos, 0)], np.nan)


def combo(*cats):
    """Integer code for a tuple of categorical columns (-1 never returned; NaN is its own level)."""
    code = np.zeros(len(cats[0]), dtype="int64")
    for c in cats:
        cc = c.cat.codes.to_numpy().astype("int64") + 1
        code = code * (len(c.cat.categories) + 1) + cc
    return pd.factorize(code)[0].astype("int64")


def utc_offset(apt, t):
    """Local UTC offset in seconds (EU DST for 2025/2026; Istanbul is UTC+3 all year)."""
    dst = ((t >= 1743296400) & (t < 1761440400)) | ((t >= 1774746000) & (t < 1792890000))
    base = np.where(apt == "EGLL", 0, 3600)
    off = base + np.where(dst, 3600, 0)
    return np.where(apt == "LTFM", 3 * 3600, off)


WINS = [(-3600, -1800), (-1800, -900), (-900, -300), (-300, -0.5), (0.5, 300), (300, 900), (900, 1800)]
ANCHORS = {"S": "t_SCHED_TIME_UTC_mvt", "E": "t_EOBT_1_flt", "L": "t_LOBT_flt",
           "I": "t_IOBT_flt", "A": "t_AOBT_3_flt"}


def build(df, metar_dir=None, spill=None):
    dep = df["is_dep"].to_numpy()
    T_all = df["t_MVT_TIME_UTC_mvt"].to_numpy()
    blk = df["t_BLOCK_TIME_UTC_mvt"].to_numpy()
    k_apt = df["APT"].cat.codes.to_numpy().astype("int64")
    k_rwy = combo(df["APT"], df["RUNWAY_mvt"])
    k_std = combo(df["APT"], df["STAND_mvt"])
    typ = df["AIRCRAFT_TYPE_mvt"].cat.codes.to_numpy().astype("float64")
    opr = df["AIRCRAFT_OPERATOR_flt"].cat.codes.to_numpy().astype("float64")
    rwc = df["RUNWAY_mvt"].cat.codes.to_numpy().astype("float64")
    txi = df[TARGET].to_numpy()
    sched = df["t_SCHED_TIME_UTC_mvt"].to_numpy()
    arr = ~dep
    k_opr = combo(df["APT"], df["AIRCRAFT_OPERATOR_flt"])
    a_key = combo(df["APT"], df["STAND_mvt"], df["RUNWAY_mvt"])[arr]
    arvt3_arr = df["t_ARVT_3_flt"].to_numpy()[arr]

    d = df[dep].reset_index(drop=True)
    del df
    gc.collect()
    T = d["t_MVT_TIME_UTC_mvt"].to_numpy()
    qa, qr, qs = k_apt[dep], k_rwy[dep], k_std[dep]
    apt = d["APT"].astype(str).to_numpy()
    class F32(dict):
        """float32 columns; kept on disk (memory-mapped) when a spill folder is given."""
        def __setitem__(self, k, v):
            v = np.asarray(v, dtype="float32")
            if spill:
                path = os.path.join(spill, f"{len(self):04d}.npy")
                np.save(path, v)
                v = np.load(path, mmap_mode="r")
            super().__setitem__(k, v)
    F = F32()

    # --- anchors: take-off time minus each reference off-block time -----------------------
    X = {k: T - d[c].to_numpy() for k, c in ANCHORS.items()}
    for k, v in X.items():
        F["x" + k] = v
    F["xA_min_floor"] = (np.floor(T / 60) * 60 - d["t_AOBT_3_flt"].to_numpy()) / 60
    F["T_sec"] = T % 60
    F["xmin_SELI"] = np.fmin(np.fmin(X["S"], X["E"]), np.fmin(X["L"], X["I"]))
    F["xmax_SELI"] = np.fmax(np.fmax(X["S"], X["E"]), np.fmax(X["L"], X["I"]))
    for a, b in [("S", "E"), ("S", "A"), ("E", "A"), ("L", "E"), ("I", "L"), ("I", "E")]:
        F[f"d_{a}{b}"] = X[a] - X[b]
    F["plan_dur"] = (d["t_ARVT_1_flt"] - d["t_EOBT_1_flt"]).to_numpy()
    F["flown_dur"] = d["t_ARVT_3_flt"].to_numpy() - T
    F["nm_dur"] = (d["t_ARVT_3_flt"] - d["t_AOBT_3_flt"]).to_numpy()
    F["dur_diff"] = F["nm_dur"] - F["plan_dur"]
    F["flt_missing"] = d["t_EOBT_1_flt"].isna().to_numpy().astype("int8")
    F["type_mismatch"] = (d["AIRCRAFT_TYPE_mvt"].astype(str) != d["AIRCRAFT_TYPE_flt"].astype(str)).to_numpy().astype("int8")
    F["diverted"] = (d["ADES_flt"].astype(str) != d["ADES_FILED_flt"].astype(str)).to_numpy().astype("int8")

    # how common is this flight's NM taxi value among same-day departures (NM default detector)
    day = np.floor(T / 86400)
    xa_min = np.round(X["A"] / 60)
    g = pd.DataFrame({"a": qa, "r": qr, "day": day, "m": xa_min})
    for name, keys in [("apt_day", ["a", "day"]), ("rwy_day", ["r", "day"]), ("rwy", ["r"])]:
        tot = g.groupby(keys)["m"].transform("count").to_numpy()
        same = g.groupby(keys + ["m"])["m"].transform("count").to_numpy()
        F["xA_share_" + name] = np.where(np.isnan(xa_min), np.nan, same / np.maximum(tot, 1))
    F["xA_mode_rwy_day"] = g.groupby(["r", "day"])["m"].transform(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan).to_numpy()
    F["xA_minus_mode"] = xa_min - F["xA_mode_rwy_day"]

    # --- traffic around the take-off -------------------------------------------------------
    xa_c = np.clip(X["A"], -600, 5400)
    xs_c = np.clip(X["S"], -1800, 10800)
    ev = {
        "to_rwy": Events(k_rwy[dep], T, xA=xa_c, xS=xs_c),
        "to_apt": Events(k_apt[dep], T, xA=xa_c, xS=xs_c),
        "ld_rwy": Events(k_rwy[arr], T_all[arr]),
        "ld_apt": Events(k_apt[arr], T_all[arr], txi=np.clip(txi[arr], 0, 3600),
                         late=np.clip(T_all[arr] - sched[arr], -3600, 10800)),
        "ib_apt": Events(k_apt[arr], blk[arr]),
        "sd_apt": Events(k_apt[dep], sched[dep]),      # scheduled departures (demand)
    }
    for name, qk in [("to_rwy", qr), ("to_apt", qa), ("ld_rwy", qr), ("ld_apt", qa), ("ib_apt", qa), ("sd_apt", qa)]:
        for lo, hi in WINS:
            F[f"n_{name}_{int(lo)}_{int(hi)}"] = ev[name].count(qk, T, lo, hi)
    for lo, hi in [(-1800, -600), (-600, 600), (600, 1800), (-3600, 3600)]:
        tag = f"{int(lo)}_{int(hi)}"
        F["txi_in_" + tag] = ev["ld_apt"].mean("txi", qa, T, lo, hi)
        F["arr_late_" + tag] = ev["ld_apt"].mean("late", qa, T, lo, hi)
        for nm in ("to_rwy", "to_apt"):
            F[f"xA_{nm}_{tag}"] = ev[nm].mean("xA", qr if nm == "to_rwy" else qa, T, lo, hi)
            F[f"xS_{nm}_{tag}"] = ev[nm].mean("xS", qr if nm == "to_rwy" else qa, T, lo, hi)
    F["xA_vs_rwy"] = xa_c - F["xA_to_rwy_-600_600"]
    F["xS_vs_rwy"] = xs_c - F["xS_to_rwy_-600_600"]
    for j in (1, 2, 3, 5, 10):
        F[f"gap_prev{j}_to_rwy"] = np.clip(ev["to_rwy"].prev(qr, T, j)[0], 0, 7200)
        F[f"gap_next{j}_to_rwy"] = np.clip(ev["to_rwy"].next(qr, T, j)[0], 0, 7200)
    F["gap_prev1_to_apt"] = np.clip(ev["to_apt"].prev(qa, T, 1)[0], 0, 7200)
    F["gap_prev1_ld_rwy"] = np.clip(ev["ld_rwy"].prev(qr, T, 1)[0], 0, 7200)
    F["gap_next1_ld_rwy"] = np.clip(ev["ld_rwy"].next(qr, T, 1)[0], 0, 7200)
    # take-offs from the same runway since each candidate off-block time
    for k in ("S", "E", "A"):
        w = np.clip(X[k], 0, 7200)
        F[f"n_to_rwy_since_{k}"] = ev["to_rwy"].count(qr, T, -w, -0.5)
        F[f"n_to_apt_since_{k}"] = ev["to_apt"].count(qa, T, -w, -0.5)
        F[f"n_ld_apt_since_{k}"] = ev["ld_apt"].count(qa, T, -w, -0.5)

    # --- the stand: who was on it before, who needed it next ---------------------------------
    # arrivals whose recorded in-block time is just the schedule / whose landing matches NM
    a_dflt = (np.abs(blk[arr] - sched[arr]) <= 7).astype("float64")
    a_nm = (np.abs(T_all[arr] - arvt3_arr) <= 67).astype("float64")
    a_late = np.clip(T_all[arr] - sched[arr], -3600, 10800)
    qo = k_opr[dep]
    ao = Events(k_opr[arr], blk[arr], dflt=a_dflt, nm=a_nm)
    aa = Events(k_apt[arr], blk[arr], dflt=a_dflt, nm=a_nm)
    for lo, hi in [(-10800, 10800), (-43200, 43200)]:
        tag = f"{int(hi / 3600)}h"
        F["arr_dflt_opr_" + tag] = ao.mean("dflt", qo, T, lo, hi)
        F["arr_nm_opr_" + tag] = ao.mean("nm", qo, T, lo, hi)
        F["arr_dflt_apt_" + tag] = aa.mean("dflt", qa, T, lo, hi)
    ib_std = Events(k_std[arr], blk[arr], typ=typ[arr], opr=opr[arr], txi=txi[arr], land=T_all[arr],
                    rwy=rwc[arr], dflt=a_dflt, late=a_late)
    to_std = Events(k_std[dep], T)
    my_typ = d["AIRCRAFT_TYPE_mvt"].cat.codes.to_numpy().astype("float64")
    my_opr = d["AIRCRAFT_OPERATOR_flt"].cat.codes.to_numpy().astype("float64")
    for j in (1, 2, 3):
        gap, pos = ib_std.prev(qs, T, j)
        F[f"std_ib_prev{j}"] = gap
        F[f"std_ib_prev{j}_sametype"] = np.where(pos >= 0, ib_std.take("typ", pos) == my_typ, np.nan)
        F[f"std_ib_prev{j}_sameopr"] = np.where(pos >= 0, ib_std.take("opr", pos) == my_opr, np.nan)
        if j <= 2:
            F[f"std_ib_prev{j}_dflt"] = ib_std.take("dflt", pos)
        if j == 1:
            F["std_ib_prev1_txi"] = ib_std.take("txi", pos)
            F["std_ib_prev1_late"] = ib_std.take("late", pos)
    F["std_ib_next1"] = ib_std.next(qs, T, 1)[0]
    for j in (1, 2):
        F[f"std_to_prev{j}"] = to_std.prev(qs, T, j)[0]
    F["std_to_next1"] = to_std.next(qs, T, 1)[0]
    # arrivals that reached this stand after each candidate off-block time (must be 0 if it is right)
    for k in ("S", "E", "A"):
        w = np.clip(X[k], 0, 86400)
        F[f"std_ib_since_{k}"] = ib_std.count(qs, T, -w, -0.5)
    F["std_ib_prev1_minus_xS"] = F["std_ib_prev1"] - X["S"]
    F["std_ib_prev1_minus_xA"] = F["std_ib_prev1"] - X["A"]

    # --- surface state from arrivals (their in-block times are known for every row) ----------
    for k in (0, 300, 600, 900, 1200, 1800, 2700):
        F[f"n_arr_taxiing_{k}"] = (ev["ld_apt"].count(qa, T - k, -172800.0, 0.5)
                                   - ev["ib_apt"].count(qa, T - k, -172800.0, 0.5))
    a_txi = pd.Series(np.clip(txi[arr], 0, 3600))
    exc = (a_txi - a_txi.groupby(a_key).transform("mean")).to_numpy()
    ax = Events(k_apt[arr], T_all[arr], exc=exc)
    for lo, hi in [(-900, 900), (-3600, 3600), (-21600, 21600)]:
        F[f"txi_in_exc_{int(lo)}_{int(hi)}"] = ax.mean("exc", qa, T, lo, hi)

    # --- how the whole day looks at this airport ---------------------------------------------
    off = utc_offset(apt, T)
    gd = pd.DataFrame({"a": qa, "d": np.floor((T + off) / 86400), "xs": xs_c, "xa": xa_c,
                       "fm": np.asarray(F["flt_missing"], dtype="float64")}).groupby(["a", "d"])
    F["day_n_dep"] = gd["xs"].transform("size").to_numpy()
    F["day_xS_mean"] = gd["xs"].transform("mean").to_numpy()
    F["day_xS_med"] = gd["xs"].transform("median").to_numpy()
    F["day_xA_mean"] = gd["xa"].transform("mean").to_numpy()
    F["day_fm_share"] = gd["fm"].transform("mean").to_numpy()

    # --- length of the uninterrupted take-off stream this flight sits in ---------------------
    kk = ev["to_rwy"].k
    ii = np.arange(len(kk))
    pos = np.minimum(np.searchsorted(kk, qr.astype("float64") * B + T, side="left"), len(kk) - 1)
    for thr in (120, 180):
        brk = np.diff(kk, prepend=-np.inf) >= thr
        run_b = ii - np.maximum.accumulate(np.where(brk, ii, 0))
        brk2 = np.diff(kk, append=np.inf) >= thr
        run_f = np.minimum.accumulate(np.where(brk2, ii, len(kk))[::-1])[::-1] - ii
        F[f"run_back_{thr}"] = run_b[pos]
        F[f"run_fwd_{thr}"] = run_f[pos]

    # --- weather at the airport (optional: needs the METAR files from get_weather.py) ---------
    if metar_dir:
        import weather
        for k, v in weather.features(metar_dir, apt, T, d["RUNWAY_mvt"].astype(str).to_numpy()).items():
            F[k] = v

    # --- calendar / categoricals ------------------------------------------------------------
    loc = T + off
    F["loc_minute"] = (loc % 86400) / 60
    F["utc_minute"] = (T % 86400) / 60
    F["dow"] = ((np.floor(loc / 86400) + 4) % 7)
    ts = pd.to_datetime(T, unit="s")
    F["doy"] = ts.dayofyear.to_numpy()
    del ev, ib_std, to_std, ao, aa, ax
    cats = {"c_apt": d["APT"], "c_type": d["AIRCRAFT_TYPE_mvt"], "c_wake": d["WK_TBL_CAT_flt"],
            "c_seg": d["MARKET_SEGMENT_flt"], "c_opr": d["AIRCRAFT_OPERATOR_flt"],
            "c_ftype": d["FLIGHT_TYPE_flt"], "c_rule": d["FLIGHT_RULE_flt"]}
    extra = {k: v.cat.codes.to_numpy().astype("int32") for k, v in cats.items()}
    extra["c_rwy"] = qr.astype("int32")

    # keys for target encoding (done at training time, out of fold)
    te = {"stand": qs, "stand_rwy": combo(d["APT"], d["STAND_mvt"], d["RUNWAY_mvt"]),
          "rwy_type": combo(d["APT"], d["RUNWAY_mvt"], d["AIRCRAFT_TYPE_mvt"]),
          "opr_rwy": combo(d["APT"], d["AIRCRAFT_OPERATOR_flt"], d["RUNWAY_mvt"]),
          "opr_stand": combo(d["APT"], d["AIRCRAFT_OPERATOR_flt"], d["STAND_mvt"]),
          "opr": qo, "ades": combo(d["APT"], d["ADES_mvt"]), "flight": combo(d["APT"], d["FLIGHT_mvt"]),
          "callsign": combo(d["APT"], d["CALLSIGN_flt"])}
    for k, v in te.items():
        extra["k_" + k] = v.astype("int32")

    # --- bookkeeping columns (never used as features) ---------------------------------------
    extra["m_id"] = d[ID].to_numpy()
    extra["m_y"] = d[TARGET].to_numpy()
    extra["m_rank"] = d["is_rank"].to_numpy()
    extra["m_apt"] = d["APT"].cat.codes.to_numpy().astype("int8")
    extra["m_month"] = ts.month.to_numpy().astype("int8")
    extra["m_T"] = T
    out = dict(F)
    out.update(extra)
    return out, list(d["APT"].cat.categories)


if __name__ == "__main__":
    import pyarrow as pa
    import pyarrow.parquet as pq
    src, out = sys.argv[1], sys.argv[2]
    metar_dir = sys.argv[3] if len(sys.argv) > 3 else None
    spill = tempfile.mkdtemp(prefix="feat_", dir=os.path.dirname(os.path.abspath(out)))
    try:
        cols, apts = build(pd.read_parquet(src), metar_dir, spill)
        n = len(next(iter(cols.values())))
        writer = None
        for i in range(0, n, 200_000):          # row chunks keep the peak memory small
            t = pa.table({k: np.asarray(v[i:i + 200_000]) for k, v in cols.items()})
            writer = writer or pq.ParquetWriter(out, t.schema)
            writer.write_table(t)
        writer.close()
    finally:
        shutil.rmtree(spill, ignore_errors=True)
    print(n, "rows,", len(cols), "columns; airports:", apts)
