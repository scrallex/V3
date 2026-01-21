#!/bin/bash
# Gate freshness checker with Slack alerting.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}
THRESHOLD=${GATE_FRESHNESS_THRESHOLD:-180}
REDIS_URL=${VALKEY_URL:-redis://localhost:6379/0}
SLACK_WEBHOOK=${SLACK_WEBHOOK:-}

OUTPUT="$($PYTHON_BIN "$ROOT_DIR/scripts/tools/check_gate_freshness.py" --redis "$REDIS_URL" --threshold "$THRESHOLD" 2>&1)"
STATUS=$?

if [ $STATUS -ne 0 ]; then
  logger -t sep-gate-freshness "FAILED status=$STATUS: $OUTPUT"
  if [ -n "$SLACK_WEBHOOK" ]; then
    payload=$(printf '{"text":":warning: SEP gate freshness failure (status %s)\n```
%s
```"}' "$STATUS" "$OUTPUT")
    curl -sf -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK" || true
  fi
  echo "$OUTPUT" >&2
  exit $STATUS
fi

echo "$OUTPUT"
exit 0
