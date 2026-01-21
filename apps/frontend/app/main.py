from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import redis.asyncio as redis
import json
import os
import asyncio
import httpx

app = FastAPI()

# Config
REDIS_URL = os.getenv("VALKEY_URL", "redis://valkey:6379/0")
PAIRS = os.getenv("HOTBAND_PAIRS", "EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD").split(",")

# Redis
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Templates & Static
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "pairs": PAIRS})

@app.get("/api/state")
async def get_state():
    """Fetch latest state for all pairs from Redis."""
    results = {}
    
    # Pipeline for efficiency
    async with redis_client.pipeline() as pipe:
        for pair in PAIRS:
            pipe.get(f"gate:last:{pair}")
        values = await pipe.execute()
    
    for pair, val in zip(PAIRS, values):
        if val:
            try:
                data = json.loads(val)
                # Normalize hazard for UI gauge (-1 to 1)
                hazard = data.get("hazard_norm", 0)
                data["hazard_pct"] = (max(min(hazard, 2.0), -2.0) + 2.0) / 4.0 * 100
                results[pair] = data
            except:
                results[pair] = {"error": "Invalid Data", "instrument": pair}
        else:
            results[pair] = {"status": "Waiting for Signal...", "instrument": pair}
            
    return results

@app.get("/api/account")
async def get_account():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://sep-backend:8000/api/account", timeout=5.0)
            return resp.json()
    except Exception as e:
        print(f"Error fetching account: {e}")
        return {"error": "backend_unavailable"}

@app.get("/api/positions")
async def get_positions():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://sep-backend:8000/api/positions", timeout=5.0)
            return resp.json()
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return {"positions": []}

@app.get("/health")
def health():
    return {"status": "ok"}
