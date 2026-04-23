#!/usr/bin/env bash
# 30-min sync job — invoked by launchd every 1800s.
# Runs sync run on today's date, which triggers:
#   - segment ingest from tracker.db
#   - work_page clustering + LLM summary
#   - salience filter
#   - humanizer (Gemini with local fallback)
#   - narrator (1-4 daily stories)
#
# Keeps logs bounded by rotating at ~5 MB via tail inside this script.

set -u
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
LOG_DIR="$REPO_ROOT/logs/scheduler"
mkdir -p "$LOG_DIR"

PY="$(command -v python3 || echo /opt/homebrew/bin/python3)"

{
  echo "=== sync $(date -Iseconds) ==="
  "$PY" "$REPO_ROOT/cli.py" sync run --model qwen2.5:7b 2>&1 || echo "sync exit=$?"
} >> "$LOG_DIR/sync.log" 2>&1

# Poor-man's log rotation — keep last 20k lines.
tail -n 20000 "$LOG_DIR/sync.log" > "$LOG_DIR/sync.log.tmp" && mv "$LOG_DIR/sync.log.tmp" "$LOG_DIR/sync.log"
