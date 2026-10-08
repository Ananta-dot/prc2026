"""
Loading and preprocessing shared by train.py and predict.py.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))
import config
from features import add_congestion_feature, add_local_airport


def load_training() -> pd.DataFrame:
    files = sorted(config.DATA_DIR.glob(config.TRAINING_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No files matching {config.TRAINING_GLOB} in {config.DATA_DIR}. "
            "Check config.DATA_DIR."
        )
    print(f"Loading {len(files)} training files...")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Combined training shape: {df.shape}")
    return df


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist — check config.py paths.")
    return pd.read_parquet(path)


def engineer_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converts each TIMESTAMP_COLS column to hour/dow/month, drops the raw column."""
    df = df.copy()
    for col in config.TIMESTAMP_COLS:
        if col not in df.columns:
            continue
        df[f"{col}_hour"] = df[col].dt.hour
        df[f"{col}_dow"] = df[col].dt.dayofweek
        df[f"{col}_month"] = df[col].dt.month
        df = df.drop(columns=[col])
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-engineering pass EXCEPT the historical-average
    encoders (those need fit/transform across train vs val/predict —
    see features.HistoricalAggEncoder and how train.py/predict.py use it).
    Order matters: local-airport and congestion need the raw
    MVT_TIME_UTC_mvt/ADEP/ADES/PHASE columns before datetime engineering
    drops the raw timestamp.
    """
    df = add_local_airport(df)
    df = add_congestion_feature(df)
    df = engineer_datetime_features(df)
    return df


def split_features_target(df: pd.DataFrame, categorical_dtypes: dict | None = None):
    """
    Drops ID/target/DROP_COLS, returns (X, y, categorical_cols, categorical_dtypes).

    categorical_dtypes lets you reuse the SAME category-code mapping
    across dataframes (e.g. fit on train, reapply to val/predict) —
    without this, casting each dataframe to "category" independently
    can silently assign different integer codes to the same category
    name in different frames, corrupting predictions. Pass the dtypes
    returned from processing the training set when processing val/predict.
    """
    drop = set(config.DROP_COLS) | {config.ID_COL, config.TARGET_COL}
    y = df[config.TARGET_COL] if config.TARGET_COL in df.columns else None
    X = df.drop(columns=[c for c in drop if c in df.columns]).copy()

    if categorical_dtypes is None:
        categorical_cols = [
            c
            for c in X.columns
            if c in config.CATEGORICAL_OVERRIDE
            or pd.api.types.is_object_dtype(X[c])
            or pd.api.types.is_string_dtype(X[c])
        ]
        for c in categorical_cols:
            X[c] = X[c].astype("category")
        categorical_dtypes = {c: X[c].dtype for c in categorical_cols}
    else:
        categorical_cols = list(categorical_dtypes.keys())
        for c in categorical_cols:
            if c in X.columns:
                X[c] = X[c].astype(categorical_dtypes[c])

    return X, y, categorical_cols, categorical_dtypes


def build_submission_frame() -> pd.DataFrame:
    """
    submitting.parquet only has (ID_COL, TARGET_COL) with TARGET_COL
    entirely null. The real features for those same rows live in
    ranking.parquet, on the rows where TARGET_COL is null (same count,
    same ids). This joins the two to get a full feature+id frame for
    the rows that need predicting, in submitting.parquet's row order.
    """
    submitting = load_parquet(config.SUBMITTING_FILE)
    ranking = load_parquet(config.RANKING_FILE)

    to_predict = ranking[ranking[config.TARGET_COL].isna()].drop(columns=[config.TARGET_COL])

    merged = submitting[[config.ID_COL]].merge(to_predict, on=config.ID_COL, how="left")

    other_cols = [c for c in merged.columns if c != config.ID_COL]
    unmatched = merged[other_cols].isna().all(axis=1).sum()
    if unmatched:
        print(
            f"Warning: {unmatched} ids from submitting.parquet had no matching "
            "feature row in ranking.parquet."
        )

    return merged
