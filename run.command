#!/bin/bash
cd "$(dirname "$0")"
PY="../.venv/bin/python"
if [ -x "./.venv/bin/python" ]; then PY="./.venv/bin/python"; fi
if [ ! -x "$PY" ]; then PY="python3"; fi
echo "Starting Curio Retrieval on http://127.0.0.1:8765"
exec "$PY" app.py --port 8765 --preload
