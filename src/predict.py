"""
Loads the trained model + fitted encoders, builds the feature set for
the rows that need predicting (joining submitting.parquet's ids against
ranking.parquet's features), and writes submission.csv.

Before uploading: re-check the current submissions bucket name/naming
convention (`mc ls dc24`) since the 2026 layout has already diverged
from the 2024 docs in other ways.
"""
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.append(str(Path(__file__).parent))
import config
from data import build_submission_frame, preprocess, split_features_target


def main():
    if not config.MODEL_OUT.exists() or not config.ENCODERS_OUT.exists():
        raise FileNotFoundError(
            f"{config.MODEL_OUT} / {config.ENCODERS_OUT} not found — run train.py first."
        )

    model = lgb.Booster(model_file=str(config.MODEL_OUT))
    with open(config.ENCODERS_OUT, "rb") as f:
        saved = pickle.load(f)
    encoders = saved["encoders"]
    cat_dtypes = saved["categorical_dtypes"]

    raw = build_submission_frame()
    if config.ID_COL not in raw.columns:
        raise ValueError(
            f"ID_COL='{config.ID_COL}' not found in submission columns: "
            f"{list(raw.columns)}. Fix config.py after running inspect_data.py."
        )
    ids = raw[config.ID_COL].reset_index(drop=True)

    df = preprocess(raw)
    for enc in encoders:
        df[enc.feature_name_] = enc.transform(df).to_numpy()

    X, _, _, _ = split_features_target(df, categorical_dtypes=cat_dtypes)

    # Align columns to what the model was trained on, in case anything drifted.
    trained_features = model.feature_name()
    missing = [c for c in trained_features if c not in X.columns]
    if missing:
        print(f"Warning: prediction data is missing {missing}; filling with NaN.")
        for c in missing:
            X[c] = pd.NA
    X = X[trained_features]

    preds = model.predict(X)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({config.ID_COL: ids, config.TARGET_COL: preds})
    out.to_csv(config.SUBMISSION_OUT, index=False)
    print(f"Wrote {len(out)} predictions to {config.SUBMISSION_OUT}")
    print(out.head())


if __name__ == "__main__":
    main()
