#!/bin/bash
# Productivity Tracker — Installation Script
set -e

echo "═══════════════════════════════════════════════"
echo "  Productivity Tracker — Setup"
echo "═══════════════════════════════════════════════"

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED="3.11"
if [ "$(printf '%s\n' "$REQUIRED" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED" ]; then
    echo "ERROR: Python >= 3.11 required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION"

# Create data directory
DATA_DIR="$HOME/.productivity-tracker"
mkdir -p "$DATA_DIR"
echo "✓ Data directory: $DATA_DIR"

# Create virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]" --quiet
echo "✓ Dependencies installed"

# Check for .env file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠ Created .env from template."
    echo "  Please edit .env and add your OPENAI_API_KEY"
    echo ""
fi

# Create screenshot temp directory
SCREENSHOT_DIR="/tmp/productivity-tracker/frames"
mkdir -p "$SCREENSHOT_DIR"
echo "✓ Screenshot directory: $SCREENSHOT_DIR"

# Initialize database
python3 -c "
from src.storage.db import Database
db = Database()
db.initialize()
print('✓ Database initialized')
"

# Load deliverables
python3 -c "
from src.matching.deliverables_loader import DeliverablesLoader
from src.storage.db import Database
import yaml
with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)
db = Database()
db.initialize()
loader = DeliverablesLoader(db, config)
loader.load_from_yaml()
print('✓ Deliverables loaded')
"

# Set up launchd plist for auto-start (optional)
PLIST_PATH="$HOME/Library/LaunchAgents/com.yantra.productivity-tracker.plist"
PROJECT_DIR="$(pwd)"

cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.yantra.productivity-tracker</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PROJECT_DIR}/.venv/bin/python</string>
        <string>-m</string>
        <string>src.agent.tracker</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>${DATA_DIR}/tracker-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${DATA_DIR}/tracker-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF
echo "✓ launchd plist created at $PLIST_PATH"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit .env with your OPENAI_API_KEY"
echo "  2. Grant permissions in System Settings:"
echo "     - Privacy → Screen Recording"
echo "     - Privacy → Accessibility"
echo "  3. Add MCP config to Claude Desktop:"
echo "     See claude_desktop_config.json"
echo "  4. Start tracking: ./scripts/start_tracker.sh"
echo "═══════════════════════════════════════════════"
