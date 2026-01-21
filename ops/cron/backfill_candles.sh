#!/usr/bin/env bash
set -euo pipefail
cd /sep
export PYTHONPATH=/sep
if [ -f /sep/OANDA.env ]; then
  set -a
  source /sep/OANDA.env
  set +a
fi
: "${VALKEY_URL:=redis://localhost:6379/0}"
/usr/bin/python3 /sep/scripts/tools/backfill_candles.py --lookback-hours 24 --granularity M1 --redis "$VALKEY_URL"
