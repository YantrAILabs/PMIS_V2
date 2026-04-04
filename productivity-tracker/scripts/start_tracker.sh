#!/bin/bash
# Start the productivity tracker daemon and MCP server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate

echo "Starting Productivity Tracker..."

# Start tracker daemon in background
echo "  → Tracker daemon..."
python -m src.agent.tracker &
TRACKER_PID=$!
echo "    PID: $TRACKER_PID"

# Start MCP server (stays in foreground for Claude Desktop stdio)
echo "  → MCP server (stdio mode for Claude Desktop)..."
echo ""
echo "Tracker running. Use Claude Desktop to interact."
echo "Press Ctrl+C to stop."

# Trap Ctrl+C to kill tracker daemon
trap "echo ''; echo 'Stopping...'; kill $TRACKER_PID 2>/dev/null; exit 0" INT TERM

# Wait for tracker (blocks until killed)
wait $TRACKER_PID
