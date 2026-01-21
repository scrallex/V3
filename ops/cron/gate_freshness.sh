#!/bin/bash
# Simple cron wrapper for the gate freshness checker.
# Logs to syslog on failure so external alerting can pick it up.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}
THRESHOLD=${GATE_FRESHNESS_THRESHOLD:-180}
REDIS_URL=${VALKEY_URL:-redis://localhost:6379/0}

OUTPUT="$($PYTHON_BIN "$ROOT_DIR/scripts/tools/check_gate_freshness.py" --redis "$REDIS_URL" --threshold "$THRESHOLD" 2>&1)"
STATUS=$?

if [ $STATUS -ne 0 ]; then
  logger -t sep-gate-freshness "Gate freshness check failed (status $STATUS): $OUTPUT"
  echo "$OUTPUT" >&2
  exit $STATUS
fi

echo "$OUTPUT"
exit 0
