"""
Run this FIRST. Prints schema, dtypes, null counts, and a sample of
rows for one training file plus ranking.parquet and submitting.parquet,
so you can correct config.py's ID_COL/TARGET_COL/DROP_COLS before
trusting train.py or predict.py.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))
import config


def inspect(path: Path, label: str):
    if not path.exists():
        print(f"\n[{label}] MISSING: {path}")
        return
    df = pd.read_parquet(path)
    print(f"\n{'=' * 70}\n[{label}] {path.name}  —  shape={df.shape}\n{'=' * 70}")
    print("\ndtypes:")
    print(df.dtypes)
    print("\nnull counts (top 15):")
    print(df.isna().sum().sort_values(ascending=False).head(15))
    print("\nsample rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(3))


if __name__ == "__main__":
    training_files = sorted(config.DATA_DIR.glob(config.TRAINING_GLOB))
    if not training_files:
        print(f"No files matching {config.TRAINING_GLOB} in {config.DATA_DIR}")
        print("Check config.DATA_DIR is set correctly.")
    else:
        inspect(training_files[0], "TRAINING (one month)")

    inspect(config.RANKING_FILE, "RANKING")
    inspect(config.SUBMITTING_FILE, "SUBMITTING")

    print(f"\n{'=' * 70}")
    print("Now check: does ID_COL / TARGET_COL in config.py match what you see above?")
    print("Does TARGET_COL actually appear in ranking.parquet (labeled) or not (unlabeled)?")
