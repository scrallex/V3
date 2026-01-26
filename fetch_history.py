import os
import requests
import json
import logging

# Config
OANDA_URL = os.getenv("OANDA_API_URL", "https://api-fxtrade.oanda.com")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
API_KEY = os.getenv("OANDA_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def fetch_trades():
    url = f"{OANDA_URL}/v3/accounts/{ACCOUNT_ID}/trades?count=50&state=ALL"
    print(f"Fetching trades from {url}...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            trades = resp.json().get("trades", [])
            print(f"Fetched {len(trades)} trades.")
            return trades
        else:
            print(f"Error: {resp.status_code} {resp.text}")
            return []
    except Exception as e:
        print(f"Exception: {e}")
        return []

def analyze_profitability(trades):
    total_pl = 0.0
    wins = 0
    losses = 0
    
    print("\n--- REAL EXECUTION LOG ---")
    print(f"{'ID':<10} {'Instrument':<10} {'Open':<20} {'Close':<20} {'PL':<10}")
    
    for t in trades:
        # Only closed trades have PnL
        if t["state"] == "CLOSED":
            pl = float(t["realizedPL"])
            total_pl += pl
            if pl > 0: wins += 1
            else: losses += 1
            
            # Times
            open_time = t.get("openTime", "N/A")
            close_time = t.get("closeTime", "N/A")
            
            print(f"{t['id']:<10} {t['instrument']:<10} {open_time:<20} {close_time:<20} {pl:<10}")

    print("-" * 60)
    print(f"Total Trades: {wins + losses}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {wins / (wins+losses) * 100:.2f}%" if (wins+losses) > 0 else "Win Rate: 0%")
    print(f"Total Realized PnL: {total_pl:.4f}")

if __name__ == "__main__":
    t = fetch_trades()
    analyze_profitability(t)
