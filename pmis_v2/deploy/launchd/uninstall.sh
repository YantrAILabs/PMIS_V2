#!/usr/bin/env bash
# Remove the PMIS launchd agents.

set -u

UID_BOOTSTRAP="gui/$(id -u)"
DEST="$HOME/Library/LaunchAgents"

for AGENT in com.pmis.sync com.pmis.dream; do
  TARGET="$DEST/$AGENT.plist"
  launchctl bootout "$UID_BOOTSTRAP/$AGENT" 2>/dev/null || true
  launchctl unload "$TARGET" 2>/dev/null || true
  if [ -f "$TARGET" ]; then
    rm -f "$TARGET"
    echo "→ removed $TARGET"
  fi
done

echo ""
echo "✓ Uninstalled. Current launchctl jobs matching pmis:"
launchctl list | grep pmis || echo "    (none — confirmed removed)"
