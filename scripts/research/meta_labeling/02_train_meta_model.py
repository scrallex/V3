
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report
import joblib
import joblib

DATA_PATH = "/tmp/meta_training_data.csv"
MODEL_OUT = "/tmp/meta_model.json"

def train():
    print("Loading Data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print(f"Loaded {len(df)} trade events.")
    print("Class Balance:")
    print(df["label"].value_counts())

    # Features
    feature_cols = ["prob", "volatility", "hazard", "rsi", "entropy"]
    X = df[feature_cols]
    y = df["label"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Train XGBoost
    print("Training XGBoost Meta-Model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    
    print(f"\nModel Accuracy: {acc:.4f}")
    print(f"Model Precision: {prec:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Feature Importance
    print("\nFeature Importance:")
    imps = model.feature_importances_
    for i, col in enumerate(feature_cols):
        print(f"{col:<12}: {imps[i]:.4f}")

    # Save
    model.save_model(MODEL_OUT)
    print(f"\nSaved Meta-Model to {MODEL_OUT}")

if __name__ == "__main__":
    train()
