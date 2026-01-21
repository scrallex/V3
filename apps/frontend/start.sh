#!/bin/bash
# Start Uvicorn in background
uvicorn app.main:app --host 127.0.0.1 --port 8000 &

# Start Nginx in foreground
nginx -g "daemon off;"
