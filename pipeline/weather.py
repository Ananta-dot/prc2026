# SPDX-License-Identifier: GPL-3.0-or-later
"""Weather features from METAR observations (files written by get_weather.py)."""
import os
import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01")
COLS = ["w_temp", "w_spread", "w_wind", "w_gust", "w_vis", "w_ceil", "w_qnh", "w_snow", "w_frz", "w_rain",
        "w_fog", "w_ts", "w_snow_3h", "w_snow_12h", "w_snow_48h", "w_frz_6h", "w_precip_3h", "w_ts_3h",
        "w_ts_next1h", "w_tmin_12h", "w_frost", "w_vis_min_3h", "w_headwind", "w_crosswind", "w_age"]


def station_table(path):
    d = pd.read_csv(path, na_values=["M"])
    d["t"] = (pd.to_datetime(d["valid"]) - EPOCH).dt.total_seconds()
    d = d.dropna(subset=["t"]).sort_values("t").drop_duplicates("t").reset_index(drop=True)
    wx = d["wxcodes"].fillna("").astype(str)
    o = pd.DataFrame({"t": d["t"].to_numpy()})
    o["w_temp"] = (d["tmpf"] - 32) / 1.8
    o["w_spread"] = (d["tmpf"] - d["dwpf"]) / 1.8
    o["w_wind"] = d["sknt"]
    o["w_gust"] = d["gust"].fillna(d["sknt"])
    o["w_vis"] = d["vsby"]
    o["w_ceil"] = d["skyl1"].fillna(30000.0)
    o["w_qnh"] = d["alti"]
    o["w_snow"] = wx.str.contains("SN|SG|PL").astype(float)
    o["w_frz"] = wx.str.contains("FZ").astype(float)
    o["w_rain"] = wx.str.contains("RA|DZ").astype(float)
    o["w_fog"] = wx.str.contains("FG").astype(float)
    o["w_ts"] = wx.str.contains("TS").astype(float)
    o["drct"] = d["drct"].to_numpy()
    idx = pd.to_datetime(o["t"], unit="s")
    r = o.set_index(idx)
    roll = lambda c, w, f: getattr(r[c].rolling(w), f)().to_numpy()
    o["w_snow_3h"] = roll("w_snow", "3h", "sum")
    o["w_snow_12h"] = roll("w_snow", "12h", "sum")
    o["w_snow_48h"] = roll("w_snow", "48h", "sum")
    o["w_frz_6h"] = roll("w_frz", "6h", "sum")
    o["w_precip_3h"] = (r["w_rain"] + r["w_snow"]).rolling("3h").sum().to_numpy()
    o["w_ts_3h"] = roll("w_ts", "3h", "sum")
    o["w_ts_1h"] = roll("w_ts", "1h", "sum")
    o["w_tmin_12h"] = roll("w_temp", "12h", "min")
    o["w_vis_min_3h"] = roll("w_vis", "3h", "min")
    o["w_frost"] = ((o["w_temp"] <= 1) & (o["w_spread"] <= 2)).astype(float)
    return o


def features(metar_dir, apt, T, runway):
    """apt, runway: string arrays per departure; T: take-off epoch seconds. -> dict of float32 arrays."""
    out = {c: np.full(len(T), np.nan, dtype="float32") for c in COLS}
    rw = pd.Series(runway).astype(str).str.extract(r"^(\d{2})")[0]
    hdg = pd.to_numeric(rw, errors="coerce").to_numpy(dtype="float64") * 10.0
    for a in np.unique(apt):
        path = os.path.join(metar_dir, f"metar_{a}.csv")
        if not os.path.exists(path):
            print(f"  weather: no file for {a}; its weather features stay empty")
            continue
        o = station_table(path)
        t = o["t"].to_numpy()
        rows = np.flatnonzero(apt == a)
        i = np.searchsorted(t, T[rows], side="right") - 1
        ok = i >= 0
        age = np.where(ok, T[rows] - t[np.maximum(i, 0)], np.nan)
        ok &= age <= 3 * 3600
        j = np.maximum(i, 0)
        for c in COLS:
            if c in o.columns:
                out[c][rows] = np.where(ok, o[c].to_numpy()[j], np.nan)
        out["w_age"][rows] = np.where(ok, age, np.nan)
        k = np.clip(np.searchsorted(t, T[rows] + 3600, side="right") - 1, 0, len(t) - 1)
        out["w_ts_next1h"][rows] = np.where(ok, o["w_ts_1h"].to_numpy()[k], np.nan)
        ang = np.deg2rad(o["drct"].to_numpy()[j] - hdg[rows])
        spd = o["w_wind"].to_numpy()[j]
        out["w_headwind"][rows] = np.where(ok, spd * np.cos(ang), np.nan)
        out["w_crosswind"][rows] = np.where(ok, np.abs(spd * np.sin(ang)), np.nan)
    return out
