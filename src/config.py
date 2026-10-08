"""
Central place for paths, column-name facts, and modeling config.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAINING_GLOB = "training_*.parquet"
RANKING_FILE = DATA_DIR / "ranking.parquet"
SUBMITTING_FILE = DATA_DIR / "submitting.parquet"

MODEL_OUT = OUTPUT_DIR / "model.txt"
ENCODERS_OUT = OUTPUT_DIR / "encoders.pkl"
SUBMISSION_OUT = OUTPUT_DIR / "submission.csv"

# --- Column facts (confirmed via inspect_data.py) -----------------------
ID_COL = "MVT_ID_mvt"
TARGET_COL = "TAXITIME_SEC_mvt"

# BLOCK_TIME_UTC_mvt is the ONE column excluded outright, and it's not
# a judgment call: TAXITIME_SEC_mvt is computed directly as
# |MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt|, and BLOCK_TIME_UTC_mvt is
# null on exactly the rows you need to predict (verified against
# ranking.parquet / submitting.parquet). Training on it would leak the
# answer, and it's simply absent at prediction time regardless.
#
# Everything else — including FLIGHT_ID_mvt / FLIGHT_mvt / CALLSIGN_flt —
# is left in as a feature; let the model's feature importance decide
# whether it's useful rather than pre-filtering.
DROP_COLS: list[str] = ["BLOCK_TIME_UTC_mvt"]

# Timestamp columns converted into hour/day-of-week/month features,
# then dropped in raw form — absolute timestamps from 2025 training
# data won't generalize to 2026 prediction data, but time-of-day/
# day-of-week/seasonality patterns should. MVT_TIME_UTC_mvt is also
# consumed by the congestion feature before it's dropped here.
TIMESTAMP_COLS: list[str] = [
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
]

CATEGORICAL_OVERRIDE: list[str] = []

# --- Feature engineering config ------------------------------------------
# Historical-average taxi time, computed at a few granularities.
# LOCAL_AIRPORT_mvt = ADEP_mvt for departures, ADES_mvt for arrivals —
# i.e. the airport where the taxi phase actually happens (added by
# features.add_local_airport). Each group becomes its own numeric
# feature; the model can weight the more/less specific ones as it likes.
AGG_GROUPS: list[list[str]] = [
    ["LOCAL_AIRPORT_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "PHASE_mvt"],
    ["LOCAL_AIRPORT_mvt", "STAND_mvt", "PHASE_mvt"],
    ["LOCAL_AIRPORT_mvt", "PHASE_mvt"],
]
AGG_SMOOTHING = 10.0  # higher = trust the global mean more for rare groups

# --- Validation strategy --------------------------------------------------
# Training data is 2025; the actual prediction task is 2026. A random
# split would overstate generalization, so validate on a held-out
# trailing window instead of a random sample.
VALIDATION_CUTOFF = "2025-11-01"
