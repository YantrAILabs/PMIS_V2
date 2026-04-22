#!/usr/bin/env bash
# Install the PMIS menu bar app as a LaunchAgent.
# Runs at login, restarts on failure, logs to pmis_v2/logs/.
set -euo pipefail

DESKTOP_DIR="$(cd "$(dirname "$0")" && pwd)"
PMIS_DIR="$(dirname "$DESKTOP_DIR")"
LOG_DIR="$PMIS_DIR/logs"
VENV_PY="$DESKTOP_DIR/.venv/bin/python3"
# Default to the popover app (Claude-widget style). Set PMIS_MENUBAR_MODE=list
# to run the legacy dropdown-menu app instead.
MODE="${PMIS_MENUBAR_MODE:-popover}"
if [[ "$MODE" == "popover" ]]; then
  SCRIPT="$DESKTOP_DIR/popover_app.py"
else
  SCRIPT="$DESKTOP_DIR/menubar_app.py"
fi

PLIST_SRC="$DESKTOP_DIR/com.pmis.menubar.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.pmis.menubar.plist"

mkdir -p "$LOG_DIR"
mkdir -p "$HOME/Library/LaunchAgents"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: venv not found at $VENV_PY"
  echo "Create it with:"
  echo "  python3 -m venv $DESKTOP_DIR/.venv"
  echo "  $DESKTOP_DIR/.venv/bin/pip install rumps pyobjc-framework-Cocoa"
  exit 1
fi

# Substitute placeholders
sed \
  -e "s|__VENV_PYTHON__|$VENV_PY|g" \
  -e "s|__SCRIPT_PATH__|$SCRIPT|g" \
  -e "s|__LOG_DIR__|$LOG_DIR|g" \
  "$PLIST_SRC" > "$PLIST_DST"

# Unload if already loaded, then load
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "Installed LaunchAgent: $PLIST_DST"
echo "Menu bar app should appear shortly as '∞' in the top-right."
echo "Logs: $LOG_DIR/menubar.{out,err}.log"
echo ""
echo "To uninstall:  launchctl unload $PLIST_DST && rm $PLIST_DST"
