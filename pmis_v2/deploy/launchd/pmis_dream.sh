#!/usr/bin/env bash
# Dream nightly job — invoked by launchd once per day.
# Runs the consolidation pipeline which includes:
#   - 13-phase nightly consolidation
#   - HGCN retraining
#   - project matching
#   - work_page auto-match (Phase 9b) with bootstrap gate
#
# Separate plist from sync so failures are isolated and logs are clean.

set -u
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
LOG_DIR="$REPO_ROOT/logs/scheduler"
mkdir -p "$LOG_DIR"

PY="$(command -v python3 || echo /opt/homebrew/bin/python3)"

{
  echo "=== dream $(date -Iseconds) ==="
  "$PY" "$REPO_ROOT/cli.py" consolidate --today 2>&1 || echo "consolidate exit=$?"
  echo "--- dream match-pages ---"
  "$PY" "$REPO_ROOT/cli.py" dream match-pages 2>&1 || echo "match-pages exit=$?"
} >> "$LOG_DIR/dream.log" 2>&1

tail -n 20000 "$LOG_DIR/dream.log" > "$LOG_DIR/dream.log.tmp" && mv "$LOG_DIR/dream.log.tmp" "$LOG_DIR/dream.log"
