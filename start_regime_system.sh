#!/bin/bash
# V3 Regime Adaptive System Launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 1. Environment
if [ -f "OANDA.env" ]; then
    export $(cat OANDA.env | sed 's/#.*//g' | xargs)
fi

export PYTHONPATH="$DIR:$PYTHONPATH"
export PATH="$DIR/bin:$PATH"

# 2. Redis Check
if ! nc -z localhost 6379; then
    echo "Warning: Redis not detected on localhost:6379. Ensure Redis is running."
    # We don't start redis here, we assume it's a service on the droplet
fi

# 3. Start Agents
echo "Starting Regime Adaptive Agent (7 Pairs)..."
/sep/.venv/bin/python scripts/trading/regime_agent.py \
    --pairs "EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD" \
    --redis "${VALKEY_URL:-redis://localhost:6379/0}" &

echo "Starting Execution Agent (NAV Sizing)..."
/sep/.venv/bin/python scripts/trading/execution_agent.py \
    --pairs "EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD" \
    --redis "${VALKEY_URL:-redis://localhost:6379/0}" &

# Wait for children
wait
