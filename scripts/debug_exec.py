
import os
import json
import redis
import requests
import sys

# Mock Env
os.environ["OANDA_API_URL"] = "https://api-fxtrade.oanda.com"
# Assuming these are set in the environment or need to be fetched?
# The agent runs in the container which has them. I will run this script IN THE CONTAINER.

def test_execution():
    r = redis.from_url("redis://sep-valkey:6379/0")
    inst = "AUD_USD"
    
    # 1. Check Signal
    raw = r.get(f"gate:last:{inst}")
    if not raw:
        print("No Redis Key Found")
        return
    
    data = json.loads(raw)
    print(f"Signal Data: {data}")
    
    # 2. Check Account
    token = os.environ.get("OANDA_API_KEY")
    account = os.environ.get("OANDA_ACCOUNT_ID")
    if not token or not account:
        print("Missing OANDA Creds in Env")
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    resp = requests.get(f"https://api-fxtrade.oanda.com/v3/accounts/{account}/summary", headers=headers)
    print(f"Account Summary: {resp.status_code} {resp.text}")
    
    if resp.status_code == 200:
        nav = float(resp.json()['account']['NAV'])
        print(f"NAV: {nav}")
        
        # 3. Size
        price = data.get("price", 0.69)
        # Logic from agent:
        # if inst.endswith("_USD"): units = alloc_usd / price
        # alloc = nav * 1.0
        units = int(nav / price)
        print(f"Calculated Units: {units}")
        
        # 4. Check Position
        p_resp = requests.get(f"https://api-fxtrade.oanda.com/v3/accounts/{account}/positions/{inst}", headers=headers)
        print(f"Position Check: {p_resp.status_code} {p_resp.text}")

if __name__ == "__main__":
    test_execution()
