#!/bin/bash
# Install Productivity Daemon as a macOS LaunchDaemon (root-level, always-active)
# Requires: sudo
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$HOME/.productivity-tracker"
LOG_DIR="/var/log/productivity-daemon"
PLIST_NAME="com.yantra.productivity-daemon"
PLIST_DEST="/Library/LaunchDaemons/${PLIST_NAME}.plist"
CRED_FILE="${DATA_DIR}/daemon_credentials.json"

echo "==================================="
echo "  Productivity Daemon Installer"
echo "==================================="
echo ""

# Check for sudo
if [ "$EUID" -ne 0 ]; then
    echo "This installer requires root privileges."
    echo "Please run: sudo $0"
    exit 1
fi

# Create directories
echo "Creating directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"

# Set admin password
echo ""
read -sp "Set admin password for Pipeline Monitor: " ADMIN_PASS
echo ""
read -sp "Confirm admin password: " ADMIN_PASS2
echo ""

if [ "$ADMIN_PASS" != "$ADMIN_PASS2" ]; then
    echo "Passwords don't match. Aborting."
    exit 1
fi

# Hash the password (SHA-256)
ADMIN_HASH=$(echo -n "$ADMIN_PASS" | shasum -a 256 | awk '{print $1}')

# Store credentials
cat > "$CRED_FILE" <<EOF
{
    "admin_password_hash": "${ADMIN_HASH}",
    "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "project_dir": "${PROJECT_DIR}"
}
EOF
chmod 600 "$CRED_FILE"
echo "Admin credentials stored."

# Detect Python path
PYTHON_PATH="${PROJECT_DIR}/.venv/bin/python"
if [ ! -f "$PYTHON_PATH" ]; then
    PYTHON_PATH=$(which python3)
fi

# Create LaunchDaemon plist
echo "Installing LaunchDaemon..."
cat > "$PLIST_DEST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_PATH}</string>
        <string>${SCRIPT_DIR}/productivity_daemon.py</string>
        <string>${PROJECT_DIR}/config/settings.yaml</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
PLIST

# Set permissions
chmod 644 "$PLIST_DEST"
chown root:wheel "$PLIST_DEST"

# Load the daemon
echo "Loading daemon..."
launchctl load "$PLIST_DEST" 2>/dev/null || true
launchctl start "$PLIST_NAME" 2>/dev/null || true

echo ""
echo "==================================="
echo "  Installation Complete!"
echo "==================================="
echo ""
echo "  Daemon:    $PLIST_NAME"
echo "  Monitor:   http://localhost:8200"
echo "  Logs:      $LOG_DIR/"
echo "  Config:    $PROJECT_DIR/config/settings.yaml"
echo ""
echo "  Commands:"
echo "    sudo launchctl stop $PLIST_NAME    # Stop daemon"
echo "    sudo launchctl start $PLIST_NAME   # Start daemon"
echo "    sudo launchctl unload $PLIST_DEST  # Uninstall"
echo ""
