#!/usr/bin/env bash
# Installer for the PMIS launchd agents.
#   - 30-min sync job    (com.pmis.sync)  — RunAtLoad + every 1800s
#   - Daily dream job    (com.pmis.dream) — 18:00 local
#
# Writes expanded plists (with real paths substituted) into
# ~/Library/LaunchAgents, then launchctl bootstraps them so they
# start immediately and on every login.
#
# Re-runnable: unloads any existing agents first.
#
# Uninstall with ./uninstall.sh.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
DEST="$HOME/Library/LaunchAgents"

if [ ! -d "$DEST" ]; then
  mkdir -p "$DEST"
fi

mkdir -p "$REPO_ROOT/pmis_v2/logs/scheduler"
chmod +x "$HERE/pmis_sync.sh" "$HERE/pmis_dream.sh"

UID_BOOTSTRAP="gui/$(id -u)"

for AGENT in com.pmis.sync com.pmis.dream; do
  TARGET="$DEST/$AGENT.plist"
  echo "→ installing $AGENT → $TARGET"
  # Unload any previous version (best-effort).
  launchctl bootout "$UID_BOOTSTRAP/$AGENT" 2>/dev/null || true
  launchctl unload "$TARGET" 2>/dev/null || true
  # Substitute __REPO_ROOT__ with the actual path.
  sed "s|__REPO_ROOT__|$REPO_ROOT|g" "$HERE/$AGENT.plist" > "$TARGET"
  # Bootstrap into the user's domain.
  launchctl bootstrap "$UID_BOOTSTRAP" "$TARGET"
done

echo ""
echo "✓ Installed. Check status with:"
echo "    launchctl list | grep pmis"
echo ""
echo "Logs:"
echo "    tail -f $REPO_ROOT/pmis_v2/logs/scheduler/sync.log"
echo "    tail -f $REPO_ROOT/pmis_v2/logs/scheduler/dream.log"
