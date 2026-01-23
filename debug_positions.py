import requests
import json

print("--- /api/positions ---")
try:
    r = requests.get("http://sep-backend:8000/api/positions", timeout=3)
    pos = r.json()
    print(json.dumps(pos, indent=2))
except Exception as e:
    print(f"Error fetching positions: {e}")

print("\n--- /api/state (REDIS Direct Check) ---")
# Since we are in a container, we can't hit 'sep-frontend' easily if we lack certs/host routing for 'mxbikes.xyz'.
# But we can read Redis directly if we install redis or just trust the logs.
# better: Try hitting the frontend main.py port 8000 directly?
# The frontend container exposes 80/443. 
# Inside the network, we can hit `sep-frontend:80`.
try:
    r = requests.get("http://sep-frontend:80/api/state", timeout=3)
    state = r.json()
    # Print summary of metrics to check diversity
    print(f"{'Inst':<10} | {'Vol':<10} | {'RSI':<8} | {'Prob':<6} | {'Regime'}")
    for k, v in state.items():
        if "error" in v:
            print(f"{k:<10} | ERROR")
            continue
        vol = v.get("volatility", 0)
        rsi = v.get("rsi", 0)
        prob = v.get("prob", 0)
        reg = v.get("regime", "N/A")
        print(f"{k:<10} | {vol:.2e}   | {rsi:>6.2f}   | {prob:>6.2f} | {reg}")
except Exception as e:
    print(f"Error fetching state: {e}")
