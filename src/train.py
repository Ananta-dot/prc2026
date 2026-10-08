import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

sys.path.append(str(Path(__file__).parent))
import config
from data import load_training, preprocess, split_features_target
from features import HistoricalAggEncoder


def time_split(df, cutoff: str):
    train_mask = df["MVT_TIME_UTC_mvt"] < cutoff
    return df[train_mask].copy(), df[~train_mask].copy()


def main():
    raw = load_training()

    if config.TARGET_COL not in raw.columns:
        raise ValueError(f"TARGET_COL='{config.TARGET_COL}' not found in training columns.")

    # 1. Filter target outliers (e.g., negative values and > 90 mins)
    n_before = len(raw)
    raw = raw.dropna(subset=[config.TARGET_COL])
    raw = raw[(raw[config.TARGET_COL] >= 60) & (raw[config.TARGET_COL] <= 5400)]
    print(f"Cleaned target dataset size: {len(raw):,} (dropped {n_before - len(raw):,} rows)")

    train_raw, val_raw = time_split(raw, config.VALIDATION_CUTOFF)
    print(f"Train rows: {len(train_raw):,} | Val rows: {len(val_raw):,}")

    train_df = preprocess(train_raw)
    val_df = preprocess(val_raw)
    y_train_full = train_df[config.TARGET_COL]

    # 2. Target Encoding
    encoders = []
    for group_cols in config.AGG_GROUPS:
        enc = HistoricalAggEncoder(group_cols)
        enc.fit(train_df, y_train_full)
        train_df[enc.feature_name_] = enc.transform_train(train_df, y_train_full).to_numpy()
        val_df[enc.feature_name_] = enc.transform(val_df).to_numpy()
        encoders.append(enc)

    X_train, y_train, cat_cols, cat_dtypes = split_features_target(train_df)
    X_val, y_val, _, _ = split_features_target(val_df, categorical_dtypes=cat_dtypes)

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_set)

    # 3. Regularized Hyperparameters to prevent early overfitting
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"\n>>> Updated Validation RMSE: {rmse:,.2f} seconds ({rmse/60:.2f} mins) <<<")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(config.MODEL_OUT))
    with open(config.ENCODERS_OUT, "wb") as f:
        pickle.dump({"encoders": encoders, "categorical_dtypes": cat_dtypes}, f)


if __name__ == "__main__":
    main()