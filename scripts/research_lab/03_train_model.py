import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
logger = logging.getLogger("ML_Trainer")

def load_data(filename="s5_rich_data_EUR_USD.csv"):
    path = f"/app/logs/{filename}" 
    logger.info(f"Loading {path}...")
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        exit(1)
    
    # Load and parse time
    df = pd.read_csv(path)
    
    # Robust timestamp handling
    if "timestamp_ns" in df.columns:
         df["datetime"] = pd.to_datetime(df["timestamp_ns"], unit="ns")
    else:
         df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
         
    df.set_index("datetime", inplace=True)
    return df

def prepare_features_labels(df, lookahead_mins=5):
    """
    Resample to M1, create features, and label targets.
    """
    # 1. Resample to M1
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "coherence": "last",
        "entropy": "last",
        "stability": "last"
    }
    df_m1 = df.resample("1min").agg(agg).dropna()
    
    # 2. Features
    # Rolling Physics Metrics
    df_m1["coh_mean_5"] = df_m1["coherence"].rolling(5).mean()
    df_m1["ent_mean_5"] = df_m1["entropy"].rolling(5).mean()
    df_m1["coh_std_15"] = df_m1["coherence"].rolling(15).std()
    
    # Price Features
    df_m1["returns"] = df_m1["close"].pct_change()
    df_m1["vol_5"] = df_m1["returns"].rolling(5).std()
    
    # Interaction
    df_m1["coh_x_vol"] = df_m1["coherence"] * df_m1["vol_5"]
    
    # 3. Labeling (The "Edge")
    # Target: Future Return > 2 pips (approx 0.0002)
    # We want to predict if Price(t+5) > Price(t)
    
    future_close = df_m1["close"].shift(-lookahead_mins)
    df_m1["fut_ret"] = (future_close - df_m1["close"]) / df_m1["close"]
    
    # Binary Target: 
    # Class 1: Return > +2 pips (Long)
    # Class 0: Else (Noise/Down)
    # Note: Balanced classes help training.
    
    THRESHOLD = 0.00015 # 1.5 pips target
    df_m1["target"] = (df_m1["fut_ret"] > THRESHOLD).astype(int)
    
    df_m1.dropna(inplace=True)
    return df_m1

def train_model(df):
    features = ["coherence", "entropy", "stability", "coh_mean_5", "ent_mean_5", "coh_std_15", "vol_5"]
    X = df[features]
    y = df["target"]
    
    # Split
    # Time-series split (Train first 70%, Test last 30%)
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    logger.info(f"Training on {len(X_train)} samples, Testing on {len(X_test)}...")
    logger.info(f"Target Rate (Train): {y_train.mean():.2%}")
    
    # XGBoost
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        eval_metric="logloss",
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Eval
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    
    print("\n" + "="*40)
    print("MODEL RESULTS (Out-of-Sample)")
    print("="*40)
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%} (True Positive / Predicted Positive)")
    print(f"Recall:    {rec:.2%} (True Positive / Actual Positive)")
    print("-" * 40)
    print("\nFeature Importance:")
    
    # Importance
    imps = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(imps)
    
    return model

if __name__ == "__main__":
    # 1. Load
    df = load_data()
    
    # 2. Prep
    df_labeled = prepare_features_labels(df)
    logger.info(f"Prepared {len(df_labeled)} labeled samples.")
    
    # 3. Train
    train_model(df_labeled)
