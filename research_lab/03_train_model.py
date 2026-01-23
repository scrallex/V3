#!/usr/bin/env python3
"""
03_train_model.py
Purpose: Phase 3 - Train Gold Standard Machine Learning Models.
Universality: Train pair-specific models using the universal "Safe Regime" definition.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score, recall_score

MODEL_DIR = Path("/sep/research_lab/models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("TrainLab")


def add_technicals(df):
    """Add standard technical features for context."""
    df["close"] = df["close"].astype(float)

    # RSI (14)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volatility
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["log_ret"].rolling(20).std()

    # Trend
    ema60 = df["close"].ewm(span=60).mean()
    df["dist_ema60"] = (df["close"] - ema60) / ema60

    df.fillna(0, inplace=True)
    return df


def train_instrument(instrument):
    path = f"/sep/research_lab/data/{instrument}_rich.parquet"
    if not Path(path).exists():
        logger.error(f"Missing data for {instrument}")
        return

    logger.info(f"=== Training {instrument} ===")
    df = pd.read_parquet(path)

    df = add_technicals(df)
    if "structure_entropy" in df.columns:
        df["entropy"] = df["structure_entropy"]

    future_ret = np.log(df["close"].shift(-60) / df["close"])

    # Universal Logic: Identify Low Hazard, Profitable Moments
    target_long = (df["lambda_hazard"] < 0.10) & (future_ret > 0.0001)

    features = [
        "stability",
        "entropy",
        "coherence",
        "lambda_hazard",
        "rsi",
        "volatility",
        "dist_ema60",
        "structure_reynolds_ratio",
        "structure_spectral_lowf_share",
    ]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = target_long.astype(int)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pos = y_train.sum()
    if pos < 10:
        logger.error(f"Not enough positive samples for {instrument}")
        return

    ratio = (len(y_train) - pos) / pos
    logger.info(f"Imbalance Ratio: {ratio:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    logger.info(f"Precision: {precision_score(y_test, preds):.2%}")
    logger.info(f"Recall: {recall_score(y_test, preds):.2%}")

    save_path = MODEL_DIR / f"model_{instrument}.json"
    model.save_model(save_path)
    logger.info(f"Saved {save_path}")


def main():
    instruments = ["USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]
    for inst in instruments:
        try:
            train_instrument(inst)
        except Exception as e:
            logger.error(f"Failed {inst}: {e}")


if __name__ == "__main__":
    main()
