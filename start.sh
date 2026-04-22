#!/bin/bash
# Start the ProMe platform portal
# Usage: ./start.sh

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_DIR/productivity-tracker/.venv/bin/python"
SERVER="$REPO_DIR/memory_system/platform/server.py"

if [[ ! -f "$PYTHON" ]]; then
    echo "Error: Run ./install.sh first"
    exit 1
fi

echo ""
echo "  Starting ProMe..."
echo "  Portal:  http://localhost:8000"
echo "  API:     http://localhost:8000/docs"
echo "  Press Ctrl+C to stop"
echo ""

cd "$REPO_DIR/memory_system/platform"
exec "$PYTHON" "$SERVER"
