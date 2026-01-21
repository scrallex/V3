import os
import json
import subprocess
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger("ManifoldLib")

class ManifoldRunner:
    def __init__(self, binary_path="/app/bin/manifold_generator"):
        self.binary_path = binary_path

    def run(self, df_s5: pd.DataFrame) -> list:
        """
        Runs the manifold generator on S5 DataFrame.
        Input DF must have: timestamp (seconds), open, high, low, close.
        Returns a list of signal dicts.
        """
        # 1. Prepare Input (The "Proven V3.1" Format)
        # Binary requires timestamp in SECONDS
        
        # Check columns
        required = ["timestamp", "open", "high", "low", "close"]
        if not all(col in df_s5.columns for col in required):
            # Try to fix it if 'timestamp_ns' exists
            if "timestamp_ns" in df_s5.columns and "timestamp" not in df_s5.columns:
                 df_s5 = df_s5.copy()
                 df_s5["timestamp"] = (df_s5["timestamp_ns"] / 1e9).astype(int)
            else:
                logger.error(f"Missing columns. Need {required}")
                return []

        # Create input struct
        # We send a list of candles
        candles = []
        for _, row in df_s5.iterrows():
            candles.append({
                "timestamp": int(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0))
            })
            
        json_input = json.dumps(candles)
        
        # 2. Run Binary (via Files)
        import tempfile
        
        input_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        input_file.write(json.dumps(candles))
        input_file.close() # Close so binary can read
        
        output_path = input_file.name + ".out.json"
        
        logger.info(f"Running manifold on {len(candles)} candles...")
        try:
            cmd = [self.binary_path, "--input", input_file.name, "--output", output_path]
            
            # Run
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                logger.error(f"Binary failed: {result.stderr}")
                return []
                
            # 3. Parse Output
            if not os.path.exists(output_path):
                 logger.error("Output file not created.")
                 return []
                 
            with open(output_path, "r") as f:
                res = json.load(f)
                
            # Clean up
            try:
                os.remove(input_file.name)
                os.remove(output_path)
            except:
                pass

            signals = res.get("signals", []) # Binary returns { "signals": [...] }
            logger.info(f"Generated {len(signals)} signals.")
            if signals:
                logger.info(f"Sample Signal: {json.dumps(signals[0])}")
            return signals

        except Exception as e:
            logger.exception("Failed to execute manifold binary")
            return []
