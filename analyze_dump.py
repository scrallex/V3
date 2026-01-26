import pandas as pd
import json

path = "/sep/logs/backend/EUR_USD_manifold_input.json"
try:
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records.")
    
    df = pd.DataFrame(data)
    print("\nHEAD:")
    print(df.head())
    
    print("\nTAIL:")
    print(df.tail())
    
    print("\nSTATS:")
    print(df.describe())
    
    # Check Timestamps
    df["diff"] = df["timestamp"].diff()
    print("\nTIME DIFF STATS:")
    print(df["diff"].describe())
    
    print("\nZEROS:")
    print((df == 0).sum())

    # Check for flat price
    print("\nPRICE CHANGE:")
    print(df["close"].pct_change().describe())

except Exception as e:
    print(e)
