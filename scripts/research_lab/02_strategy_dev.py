import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
logger = logging.getLogger("StrategyLab")

def load_data(filename="s5_rich_data_EUR_USD.csv"):
    # Look in logs dir
    path = f"/app/logs/{filename}"
    
    logger.info(f"Loading {path}...")
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        exit(1)
        
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["timestamp_ns"], unit="ns") if "timestamp_ns" in df.columns else pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("datetime", inplace=True)
    logger.info(f"Loaded {len(df)} rows.")
    return df

def resample_m1(df_s5):
    """
    Resample S5 rich data to M1.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "coherence": "last", # Take state at end of minute
        "entropy": "last",
        "stability": "last"
    }
    df_m1 = df_s5.resample("1min").agg(agg).dropna()
    logger.info(f"Resampled to {len(df_m1)} M1 bars.")
    return df_m1

def apply_strategy(df):
    """
    Define your trading logic here.
    """
    # Example Strategy: High Coherence Trend Following
    # No XGBoost -> Pure Physics
    
    # 1. Feature: Returns
    df["returns"] = df["close"].pct_change()
    
    # 2. Logic
    # Buy if Coherence is high (Strong Market Structure)
    # AND Entropy is low (Ordered State)
    # AND Stability is positive
    
    # Thresholds (to be tuned)
    COHERENCE_THRES = 0.60
    ENTROPY_THRES = 0.90
    
    df["signal"] = 0
    
    # Long Condition
    long_mask = (df["coherence"] > COHERENCE_THRES) & (df["entropy"] < ENTROPY_THRES)
    df.loc[long_mask, "signal"] = 1
    
    # Short Condition? (Maybe just long for now, or inverse)
    # df.loc[short_mask, "signal"] = -1
    
    return df

def simulate_pnl(df):
    """
    Vectorized Backtest
    """
    # Shift signal: We trade at close of candle i, so we capture return of candle i+1
    df["pos"] = df["signal"].shift(1).fillna(0)
    
    # Simple Returns
    df["strategy_return"] = df["pos"] * df["returns"]
    
    # Costs (1.5bps one way = 3bps round trip)
    # Cost incurred on Position Change
    cost_per_turn = 0.00015
    df["pos_change"] = df["pos"].diff().abs()
    df["cost"] = df["pos_change"] * cost_per_turn
    
    df["net_return"] = df["strategy_return"] - df["cost"]
    
    # Cumulative
    df["cum_pnl"] = df["net_return"].cumsum()
    
    return df

def report(df):
    total_trades = (df["pos_change"] > 0).sum()
    total_pnl = df["net_return"].sum()
    sharpe = (df["net_return"].mean() / df["net_return"].std()) * np.sqrt(1440*365) if df["net_return"].std() > 0 else 0
    
    print("="*40)
    print(f"STRATEGY RESULTS")
    print("="*40)
    print(f"Trades: {total_trades}")
    print(f"Net PnL: {total_pnl*100:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")
    print("-" * 40)
    print("Sample Metrics (Tail):")
    print(df[["close", "coherence", "signal", "cum_pnl"]].tail())

def run_sweep(df_m1):
    results = []
    
    # Sweep Coherence from 0.50 to 0.95
    coherence_range = np.arange(0.50, 0.96, 0.05)
    # Sweep Entropy from 0.30 to 0.90
    entropy_range = np.arange(0.30, 0.95, 0.10)
    
    # Calculate SMA for direction
    df_m1["sma"] = df_m1["close"].rolling(50).mean()
    
    for coh in coherence_range:
        for ent in entropy_range:
            # Vectorized logic
            df_test = df_m1.copy()
            
            # Base Filter: Strong Structure
            structure_mask = (df_test["coherence"] > coh) & (df_test["entropy"] < ent)
            
            df_test["signal"] = 0
            
            # Long: Structure + Price > SMA
            long_mask = structure_mask & (df_test["close"] > df_test["sma"])
            df_test.loc[long_mask, "signal"] = 1
            
            # Short: Structure + Price < SMA
            short_mask = structure_mask & (df_test["close"] < df_test["sma"])
            df_test.loc[short_mask, "signal"] = -1
            
            # Sim PnL
            df_test["pos"] = df_test["signal"].shift(1).fillna(0)
            df_test["returns"] = df_test["close"].pct_change()
            df_test["strategy_return"] = df_test["pos"] * df_test["returns"]
            
            # Costs (3bps roundtrip)
            cost_per_turn = 0.00015
            df_test["pos_change"] = df_test["pos"].diff().abs()
            df_test["cost"] = df_test["pos_change"] * cost_per_turn
            df_test["net_return"] = df_test["strategy_return"] - df_test["cost"]
            
            total_pnl = df_test["net_return"].sum()
            trades = (df_test["pos_change"] > 0).sum()
            
            if trades > 10: # Min Sample
                sharpe = (df_test["net_return"].mean() / df_test["net_return"].std()) * np.sqrt(1440*365)
                results.append({
                    "Coh": coh,
                    "Ent": ent,
                    "PnL": total_pnl,
                    "Trades": trades,
                    "Sharpe": sharpe
                })
                
    # Sort by Sharpe
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res.sort_values("Sharpe", ascending=False, inplace=True)
        print("\nTOP 10 CONFIGURATIONS (Long/Short):")
        print(df_res.head(10))
        
        # Best Config Details
        best = df_res.iloc[0]
        print(f"\nrunning detailed report for BEST: Coh>{best['Coh']:.2f}, Ent<{best['Ent']:.2f}")
        return best["Coh"], best["Ent"]
    else:
        print("No profitable configs found with >10 trades.")
        return 0.6, 0.9

if __name__ == "__main__":
    # 1. Load S5
    if os.path.exists("/app/logs/s5_rich_data_EUR_USD.csv"):
        df_s5 = load_data()
    else:
        logger.error("Data not found. Run 01_data_prep.py first.")
        exit(1)
        
    # 2. Resample
    df_m1 = resample_m1(df_s5)
    
    # 3. Sweep (Uncomment to run optimization)
    # best_coh, best_ent = run_sweep(df_m1)
    
    # Defaults from initial research
    best_coh = 0.55
    best_ent = 0.80
    
    # 4. Final Detail Report
    df_m1["sma"] = df_m1["close"].rolling(50).mean()
    df_m1["returns"] = df_m1["close"].pct_change()
    
    structure_mask = (df_m1["coherence"] > best_coh) & (df_m1["entropy"] < best_ent)
    df_m1["signal"] = 0
    
    long_mask = structure_mask & (df_m1["close"] > df_m1["sma"])
    df_m1.loc[long_mask, "signal"] = 1
    
    short_mask = structure_mask & (df_m1["close"] < df_m1["sma"])
    df_m1.loc[short_mask, "signal"] = -1
    
    df_sim = simulate_pnl(df_m1)
    report(df_sim)
