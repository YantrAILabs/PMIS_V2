#!/bin/bash
# Start all ProMe web services (macOS / Linux)
# Usage: ./start.sh
#
# Launches three servers in the background:
#   http://localhost:8100  -- ProMe API + Wiki (main dashboard)
#   http://localhost:8200  -- Ops Dashboard (health, diagnostics)
#   http://localhost:8000  -- Platform Portal (external integrations)
#
# Logs stream to the current terminal. Ctrl+C stops all three.

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_DIR/productivity-tracker/.venv/bin/python"

if [[ ! -f "$PYTHON" ]]; then
    echo "Error: Run ./install.sh first"
    exit 1
fi

echo ""
echo "  Starting ProMe services..."
echo "    http://localhost:8100/wiki/goals   -- Main dashboard"
echo "    http://localhost:8200/             -- Ops Monitor"
echo "    http://localhost:8000/             -- Platform Portal"
echo "  Ctrl+C stops all three."
echo ""

# Trap Ctrl+C so all children are killed
pids=()
trap 'echo ""; echo "Stopping..."; kill "${pids[@]}" 2>/dev/null; exit 0' INT TERM

( cd "$REPO_DIR/pmis_v2"                  && "$PYTHON" server.py            2>&1 | sed "s/^/[8100] /" ) &
pids+=($!)
( cd "$REPO_DIR/pmis_v2"                  && "$PYTHON" health_dashboard.py  2>&1 | sed "s/^/[8200] /" ) &
pids+=($!)
( cd "$REPO_DIR/memory_system/platform"   && "$PYTHON" server.py            2>&1 | sed "s/^/[8000] /" ) &
pids+=($!)

wait
