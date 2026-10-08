"""
Feature engineering beyond raw columns:
  - LOCAL_AIRPORT_mvt: which airport the taxi phase actually happens at
  - airport_congestion_30min: how busy that airport is in the same window
  - HistoricalAggEncoder: leak-free historical-average taxi time by group
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def add_local_airport(df: pd.DataFrame) -> pd.DataFrame:
    """
    ADEP_mvt is the departure airport, ADES_mvt the arrival airport —
    but taxi-out happens at ADEP and taxi-in happens at ADES. This
    collapses both into a single "where is this taxi phase happening"
    column so it can be used for grouping/congestion regardless of phase.
    """
    df = df.copy()
    df["LOCAL_AIRPORT_mvt"] = df["ADEP_mvt"].where(df["PHASE_mvt"] == "DEP", df["ADES_mvt"])
    return df


def add_congestion_feature(df: pd.DataFrame, window_minutes: int = 15) -> pd.DataFrame:
    """
    Count of other movements at the same LOCAL_AIRPORT_mvt within
    +/- window_minutes of each row's MVT_TIME_UTC_mvt (a
    2*window_minutes total window, so window_minutes=15 -> "30min").

    Vectorized per-airport via np.searchsorted instead of a row-by-row
    Python loop (which is both slow and was the source of a bug: it
    reset the index while sorting, then reindexed against the
    ORIGINAL differently-numbered index. Any original index label
    that didn't happen to fall inside the fresh 0..n-1 range came back
    as NaN — which, after a time-based train/val split, corrupted
    almost the entire validation frame including the target column.
    This version keeps the original index labels throughout, so
    `.loc[...]` assignment can never misalign like that.)
    """
    df = df.copy()
    delta = np.timedelta64(window_minutes, "m")
    counts = pd.Series(index=df.index, dtype="float64")

    for _, group in df.groupby("LOCAL_AIRPORT_mvt"):
        order = group["MVT_TIME_UTC_mvt"].sort_values()
        times = order.to_numpy()
        lo = np.searchsorted(times, times - delta, side="left")
        hi = np.searchsorted(times, times + delta, side="right")
        counts.loc[order.index] = (hi - lo).astype("float64")

    df["airport_congestion_30min"] = counts
    return df


class HistoricalAggEncoder:
    """
    Smoothed historical mean of TARGET_COL grouped by `group_cols`,
    fit on a training set and reused elsewhere.

    - fit(): computes per-group sum/count and the global mean from
      the training data.
    - transform_train(): leave-one-out mean for the SAME rows the
      encoder was fit on — each row's own target is excluded from its
      own group average, so the encoding doesn't leak the row's own
      answer back to itself.
    - transform(): plain smoothed group mean, for data the encoder was
      NOT fit on (validation rows, prediction rows). Groups unseen
      during fit fall back to the global mean.

    Smoothing pulls small-sample group means toward the global mean:
    smoothed = (count * group_mean + k * global_mean) / (count + k)
    """

    def __init__(self, group_cols: list[str], smoothing: float = None):
        self.group_cols = group_cols
        self.smoothing = smoothing if smoothing is not None else config.AGG_SMOOTHING
        self.stats_: pd.DataFrame | None = None
        self.global_mean_: float | None = None
        self.feature_name_ = "avg_taxi_" + "_".join(group_cols)

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "HistoricalAggEncoder":
        tmp = df[self.group_cols].copy()
        tmp["_y"] = y.values
        self.global_mean_ = float(tmp["_y"].mean())
        self.stats_ = (
            tmp.groupby(self.group_cols, dropna=False)["_y"]
            .agg(_sum="sum", _count="count")
            .reset_index()
        )
        return self

    def transform_train(self, df: pd.DataFrame, y: pd.Series) -> pd.Series:
        if self.stats_ is None:
            raise RuntimeError("Call fit() before transform_train().")
        merged = df[self.group_cols].merge(self.stats_, on=self.group_cols, how="left")
        loo_sum = merged["_sum"].to_numpy() - y.to_numpy()
        loo_count = merged["_count"].to_numpy() - 1
        loo_mean = pd.Series(loo_sum).divide(pd.Series(loo_count))
        loo_mean = loo_mean.mask(loo_count <= 0, self.global_mean_)
        loo_count_safe = pd.Series(loo_count).clip(lower=0)
        smoothed = (
            loo_count_safe * loo_mean.fillna(self.global_mean_)
            + self.smoothing * self.global_mean_
        ) / (loo_count_safe + self.smoothing)
        return smoothed.rename(self.feature_name_).reset_index(drop=True)

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.stats_ is None:
            raise RuntimeError("Call fit() before transform().")
        merged = df[self.group_cols].merge(self.stats_, on=self.group_cols, how="left")
        count = merged["_count"].fillna(0)
        mean = (merged["_sum"] / merged["_count"]).fillna(self.global_mean_)
        smoothed = (count * mean + self.smoothing * self.global_mean_) / (count + self.smoothing)
        return smoothed.rename(self.feature_name_).reset_index(drop=True)
