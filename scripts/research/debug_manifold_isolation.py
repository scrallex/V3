
import json
import subprocess
import os
import sys
import tempfile
import random

# Configuration
BIN_PATH = "/app/bin/manifold_generator"

def test_payload(name, payload):
    print(f"--- Testing Payload: {name} ---")
    
    # 1. Write Input
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_in:
        json.dump(payload, f_in, indent=2)
        input_path = f_in.name
    
    output_path = input_path + ".out.json"
    
    try:
        # 2. Run Binary
        cmd = [BIN_PATH, "--input", input_path, "--output", output_path]
        print(f"Command: {' '.join(cmd)}")
        
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Exit Code: {proc.returncode}")
        if proc.stdout: print(f"STDOUT: {proc.stdout.strip()}")
        if proc.stderr: print(f"STDERR: {proc.stderr.strip()}")
        
        # 3. Check Output
        if os.path.exists(output_path):
            with open(output_path, 'r') as f_out:
                try:
                    res = json.load(f_out)
                    signals = res.get("signals", [])
                    print(f"Generated {len(signals)} signals.")
                    
                    if len(signals) >= 3:
                        # Print variability check
                        s1 = signals[0].get("metrics", {}).get("coherence", 0)
                        s2 = signals[1].get("metrics", {}).get("coherence", 0)
                        s3 = signals[2].get("metrics", {}).get("coherence", 0)
                        print(f"Coherence Sample: {s1} -> {s2} -> {s3}")
                        
                        if s1 == s2 == s3:
                            print("!! WARNING: Metrics are STATIC (Input likely rejected/defaulted) !!")
                        else:
                            print("SUCCESS: Metrics are dynamic.")
                            
                        # Print full first signal
                        print("Sample Signal[0]:", json.dumps(signals[0], indent=2))
                    else:
                        print("Not enough signals to verify dynamics.")
                        
                except json.JSONDecodeError:
                    print(f"Output file is not valid JSON: {open(output_path).read()}")
        else:
            print("Output file NOT created.")
            
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        if os.path.exists(input_path): os.unlink(input_path)
        if os.path.exists(output_path): os.unlink(output_path)
    print("\n")

def main():
    # Base timestamp
    ts_base = 1769000000 
    
    # Generate Synthetic Random Walk
    candles = []
    price = 1.1000
    for i in range(60):
        price += (random.random() - 0.5) * 0.0005
        candles.append({
            "t_sec": ts_base + (i * 5),
            "t_ns": (ts_base + (i * 5)) * 1_000_000_000,
            "o": price,
            "h": price + 0.0002,
            "l": price - 0.0002,
            "c": price,
            "v": 100
        })

    # Case 1: Live Agent V3.2 Format (timestamp_ns + price)
    payload_v3_2 = [
        {"timestamp_ns": c["t_ns"], "price": c["c"]}
        for c in candles
    ]
    test_payload("Live Agent V3.2 (ns + price)", payload_v3_2)

    # Case 2: Proven V3.1 Format (timestamp seconds + OHLC)
    payload_v3_1 = [
        {
            "timestamp": c["t_sec"],
            "open": c["o"],
            "high": c["h"],
            "low": c["l"],
            "close": c["c"]
        }
        for c in candles
    ]
    test_payload("Proven V3.1 (sec + OHLC)", payload_v3_1)

    # Case 3: Hybrid (All Keys)
    payload_hybrid = [
        {
            "timestamp": c["t_sec"],
            "timestamp_ns": c["t_ns"],
            "price": c["c"],
            "open": c["o"],
            "high": c["h"],
            "low": c["l"],
            "close": c["c"],
            "volume": c["v"]
        }
        for c in candles
    ]
    test_payload("Hybrid (All Keys)", payload_hybrid)

if __name__ == "__main__":
    main()
