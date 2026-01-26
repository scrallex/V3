
import os
import sys
import requests
import time

# Manual OANDA connection test
TOKEN = os.environ.get("OANDA_API_TOKEN") # Corrected from OANDA_TOKEN
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
API_URL = os.environ.get("OANDA_API_URL", "https://api-fxtrade.oanda.com")

if not TOKEN:
    print("ERROR: OANDA_TOKEN not set!")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

url = f"{API_URL}/v3/instruments/EUR_USD/candles?granularity=S5&count=5"
print(f"Testing connectivity to: {url}")

try:
    start = time.time()
    r = requests.get(url, headers=headers, timeout=10)
    elapsed = time.time() - start
    print(f"Status Code: {r.status_code}")
    print(f"Time: {elapsed:.2f}s")
    if r.status_code == 200:
        print("Success! Body snippet:")
        print(r.text[:200])
    else:
        print("Failed!")
        print(r.text)
except Exception as e:
    print(f"Exception: {e}")
