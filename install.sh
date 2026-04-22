#!/bin/bash
# ═══════════════════════════════════════════════════════
# YantrAI Memory System — One-Command Installer
# ═══════════════════════════════════════════════════════
# Usage: git clone <repo> && cd memory && ./install.sh
# Idempotent — safe to run multiple times.
set -e

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }
info() { echo -e "  ${BLUE}→${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
step() { echo -e "\n${BOLD}[$1/8] $2${NC}"; }

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACKER_DIR="$REPO_DIR/productivity-tracker"
PMIS_DIR="$REPO_DIR/pmis_v2"
PLATFORM_DIR="$REPO_DIR/memory_system/platform"
DATA_DIR="$HOME/.productivity-tracker"
VENV_DIR="$TRACKER_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
PLIST_NAME="com.yantra.productivity-tracker"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

echo ""
echo -e "${BOLD}═══════════════════════════════════════${NC}"
echo -e "${BOLD}  YantrAI Memory System Installer${NC}"
echo -e "${BOLD}═══════════════════════════════════════${NC}"
echo ""
echo "  Repo:     $REPO_DIR"
echo "  Data:     $DATA_DIR"
echo ""

# ══════════════════════════════════════
# STEP 1: Pre-flight checks
# ══════════════════════════════════════
step 1 "Pre-flight checks"

# macOS check
if [[ "$(uname)" != "Darwin" ]]; then
    fail "This installer is for macOS only."
fi
ok "macOS detected ($(sw_vers -productVersion))"

# Python check — try multiple locations (Homebrew, system, etc.)
PYTHON3=""
for candidate in /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$candidate" > /dev/null 2>&1; then
        PY_MINOR=$($candidate -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        PY_MAJOR=$($candidate -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
        if [[ "$PY_MAJOR" -ge 3 ]] && [[ "$PY_MINOR" -ge 11 ]]; then
            PYTHON3="$candidate"
            break
        fi
    fi
done
if [[ -z "$PYTHON3" ]]; then
    fail "Python >= 3.11 required. Install via: brew install python@3.14"
fi
PY_VER=$($PYTHON3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python $PY_VER ($PYTHON3)"

# pip check
$PYTHON3 -m pip --version > /dev/null 2>&1 || fail "pip not available for $PYTHON3"
ok "pip available"

VENV_PYTHON="$PYTHON3"

# Stop existing daemon if running
if launchctl list "$PLIST_NAME" > /dev/null 2>&1; then
    info "Stopping existing daemon..."
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    sleep 1
    ok "Existing daemon stopped"
else
    ok "No existing daemon running"
fi

# Check required directories exist in repo
[[ -d "$TRACKER_DIR" ]] || fail "productivity-tracker/ not found in repo"
[[ -d "$PMIS_DIR" ]] || fail "pmis_v2/ not found in repo"
ok "Repository structure valid"

# Pre-check macOS permissions (advisory)
PERM_ISSUES=0
if ! $PYTHON3 -c "
import subprocess, sys
try:
    r = subprocess.run(['screencapture', '-x', '/tmp/.pmis_perm_test.png'],
                       capture_output=True, timeout=5)
    if r.returncode != 0:
        sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    warn "Screen Recording permission not granted — daemon won't capture screenshots"
    warn "Grant it now: System Settings → Privacy & Security → Screen Recording → Terminal"
    PERM_ISSUES=1
else
    rm -f /tmp/.pmis_perm_test.png
    ok "Screen Recording permission granted"
fi

if [[ $PERM_ISSUES -eq 1 ]]; then
    echo ""
    echo -e "  ${YELLOW}You can continue installing — grant permissions after.${NC}"
    echo ""
fi

# ══════════════════════════════════════
# STEP 2: Python venv + dependencies
# ══════════════════════════════════════
step 2 "Python environment + dependencies"

# Auto-fix broken venv — no manual intervention needed
if [[ -d "$VENV_DIR" ]]; then
    if [[ -f "$PIP" ]] && [[ -f "$PYTHON" ]]; then
        info "Existing venv found, updating..."
    else
        info "Broken venv detected — rebuilding automatically..."
        rm -rf "$VENV_DIR"
        $VENV_PYTHON -m venv "$VENV_DIR" || fail "Failed to create venv — check Python installation"
    fi
else
    info "Creating virtual environment..."
    $VENV_PYTHON -m venv "$VENV_DIR" || fail "Failed to create venv — check Python installation"
fi
ok "Virtual environment at $VENV_DIR"

info "Upgrading pip + setuptools..."
$PIP install --upgrade pip setuptools wheel -q
ok "pip upgraded"

# Check for Xcode Command Line Tools (needed to compile C extensions)
if ! xcode-select -p > /dev/null 2>&1; then
    warn "Xcode Command Line Tools not found — some packages may fail to compile"
    warn "Install with: xcode-select --install"
fi

# ── Stage A: Core packages (must succeed) ──
info "Installing core dependencies..."
CORE_FAIL=0
for pkg in openai sqlalchemy numpy Pillow pyyaml python-dotenv fastapi uvicorn httpx psutil pyjwt bcrypt jinja2; do
    if $PIP install "$pkg" -q 2>/dev/null; then
        ok "  $pkg"
    else
        echo -e "  ${YELLOW}Retrying $pkg...${NC}"
        if $PIP install "$pkg" 2>&1 | tail -3; then
            ok "  $pkg (retry)"
        else
            fail "  Core package $pkg failed to install. Cannot continue."
            CORE_FAIL=1
        fi
    fi
done
[[ $CORE_FAIL -eq 1 ]] && fail "Core packages missing — fix errors above and re-run ./install.sh"
ok "Core packages installed"

# ── Stage B: Heavy/optional packages (track what installed vs skipped) ──
info "Installing optional packages (may take a few minutes)..."
OPT_INSTALLED=()
OPT_SKIPPED=()
for pkg in chromadb scikit-image scikit-learn scipy; do
    if $PIP install "$pkg" -q 2>/dev/null; then
        ok "  $pkg"
        OPT_INSTALLED+=("$pkg")
    else
        warn "  $pkg skipped"
        OPT_SKIPPED+=("$pkg")
    fi
done
for pkg in pyobjc-core pyobjc-framework-Cocoa pyobjc-framework-Quartz pyobjc-framework-ApplicationServices; do
    if $PIP install "$pkg" -q 2>/dev/null; then
        ok "  $pkg"
        OPT_INSTALLED+=("$pkg")
    else
        warn "  $pkg skipped"
        OPT_SKIPPED+=("$pkg")
    fi
done

# Summary of optional packages
if [[ ${#OPT_SKIPPED[@]} -gt 0 ]]; then
    echo ""
    warn "Optional packages skipped: ${OPT_SKIPPED[*]}"
    # Explain impact
    for pkg in "${OPT_SKIPPED[@]}"; do
        case "$pkg" in
            chromadb)        warn "  → No vector search (falls back to linear scan)" ;;
            scikit-image)    warn "  → No frame segmentation (screenshot dedup disabled)" ;;
            scikit-learn)    warn "  → Some ML consolidation features unavailable" ;;
            scipy)           warn "  → Matrix operations fall back to numpy" ;;
            pyobjc-core|pyobjc-framework-*) warn "  → Input monitor uses fallback (less accurate app detection)" ;;
        esac
    done
    echo ""
fi
ok "Optional packages done (${#OPT_INSTALLED[@]} installed, ${#OPT_SKIPPED[@]} skipped)"

# ── Stage C: Install tracker + ProMe packages ──
info "Installing productivity-tracker package..."
cd "$TRACKER_DIR"
$PIP install -e "." --no-deps -q 2>/dev/null || $PIP install -e "." --no-deps 2>&1 | tail -3
ok "Productivity tracker installed"

info "Installing ProMe requirements..."
if [[ -f "$PMIS_DIR/requirements.txt" ]]; then
    $PIP install -r "$PMIS_DIR/requirements.txt" -q 2>/dev/null || warn "Some ProMe deps already installed or skipped"
fi
ok "ProMe done"

info "Installing platform requirements..."
if [[ -f "$PLATFORM_DIR/requirements.txt" ]]; then
    $PIP install -r "$PLATFORM_DIR/requirements.txt" -q 2>/dev/null || warn "Some platform deps already installed or skipped"
fi
ok "Platform done"

# ── Verify critical imports ──
info "Verifying critical imports..."
IMPORT_RESULT=$($PYTHON -c "
ok, fail = [], []
for mod, name in [('openai','openai'), ('sqlalchemy','sqlalchemy'), ('fastapi','fastapi'),
                   ('numpy','numpy'), ('yaml','pyyaml'), ('PIL','Pillow'), ('dotenv','python-dotenv'),
                   ('uvicorn','uvicorn'), ('httpx','httpx'), ('psutil','psutil')]:
    try:
        __import__(mod)
        ok.append(name)
    except ImportError:
        fail.append(name)
print(f'{len(ok)} OK, {len(fail)} missing')
if fail:
    print('Missing: ' + ', '.join(fail))
" 2>&1)
echo "  $IMPORT_RESULT"
if echo "$IMPORT_RESULT" | grep -q "0 missing"; then
    ok "All critical imports verified"
else
    warn "Some packages missing — check errors above"
fi

# ══════════════════════════════════════
# STEP 3: Directory structure
# ══════════════════════════════════════
step 3 "Creating directory structure"

dirs=(
    "$DATA_DIR"
    "$DATA_DIR/chromadb"
    "$PMIS_DIR/data"
    "$PMIS_DIR/data/chroma"
    "$PLATFORM_DIR/data"
    "$PLATFORM_DIR/data/memories"
    "/tmp/productivity-tracker/frames"
)
for d in "${dirs[@]}"; do
    mkdir -p "$d"
done
ok "All directories created"

# ══════════════════════════════════════
# STEP 4: Environment configuration
# ══════════════════════════════════════
step 4 "Environment configuration"

ENV_FILE="$TRACKER_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    if grep -q "your-openai-api-key-here" "$ENV_FILE" 2>/dev/null; then
        warn "Existing .env has placeholder API key — will prompt for real key"
    else
        info "Existing .env found with API key, preserving..."
        ok ".env preserved"
        SKIP_KEY_PROMPT=true
    fi
fi

if [[ -z "${SKIP_KEY_PROMPT:-}" ]]; then
    echo ""
    echo -e "  ${YELLOW}OpenAI API key required for ChatGPT Vision (screenshot analysis).${NC}"
    echo -e "  ${YELLOW}Get one at: https://platform.openai.com/api-keys${NC}"
    echo ""

    API_KEY=""
    ATTEMPTS=0
    while [[ $ATTEMPTS -lt 2 ]]; do
        read -p "  Enter your OpenAI API key (sk-...), or press Enter to skip: " API_KEY
        if [[ -z "$API_KEY" ]]; then
            warn "No API key entered — daemon will fail on screenshot analysis."
            warn "Add it later: edit $ENV_FILE"
            API_KEY="your-openai-api-key-here"
            break
        elif [[ "$API_KEY" == sk-* ]] && [[ ${#API_KEY} -ge 20 ]]; then
            ok "API key accepted"
            break
        else
            warn "API key should start with 'sk-' and be at least 20 characters."
            ATTEMPTS=$((ATTEMPTS + 1))
            if [[ $ATTEMPTS -ge 2 ]]; then
                warn "Using entered value as-is. Edit $ENV_FILE later if needed."
            fi
        fi
    done

    cp "$TRACKER_DIR/.env.example" "$ENV_FILE"
    sed -i '' "s|your-openai-api-key-here|$API_KEY|g" "$ENV_FILE"
    sed -i '' "s|__HOME__|$HOME|g" "$ENV_FILE"
    sed -i '' "s|__REPO__|$REPO_DIR|g" "$ENV_FILE"
    ok ".env created"
fi

# ══════════════════════════════════════
# STEP 5: Database initialization
# ══════════════════════════════════════
step 5 "Initializing databases"

# Tracker DB (SQLAlchemy)
info "Initializing tracker database..."
cd "$TRACKER_DIR"
$PYTHON -c "
import os
os.environ['SQLITE_DB_PATH'] = '$DATA_DIR/tracker.db'
from src.storage.db import Database
db = Database('$DATA_DIR/tracker.db')
db.initialize()
db.close()
print('  Tables: context_1, context_2, hourly_memory, daily_memory, deliverables')
" 2>/dev/null || fail "Tracker DB init failed"
ok "Tracker DB initialized at $DATA_DIR/tracker.db"

# ProMe DB
info "Initializing ProMe database..."
cd "$PMIS_DIR"
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from db.manager import DBManager
db = DBManager(db_path='data/memory.db')
n = db.count_nodes()
print(f'  Nodes: {n}, Tables: memory_nodes, embeddings, relations, projects, deliverables, ...')
db.close()
" 2>/dev/null || fail "ProMe DB init failed"
ok "ProMe DB initialized at $PMIS_DIR/data/memory.db"

# Platform users DB (auto-init on import)
info "Initializing platform database..."
cd "$PLATFORM_DIR"
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from platform_db import get_users_db
conn = get_users_db()
conn.close()
print('  Tables: users, api_tokens, sharing_rules, access_requests, sync_log')
" 2>/dev/null || fail "Platform DB init failed"
ok "Platform DB initialized at $PLATFORM_DIR/data/users.db"

# ══════════════════════════════════════
# STEP 6: End-to-end verification
# ══════════════════════════════════════
step 6 "End-to-end verification"

cd "$REPO_DIR"
$PYTHON "$REPO_DIR/verify_install.py" 2>/dev/null
VERIFY_EXIT=$?
if [[ $VERIFY_EXIT -ne 0 ]]; then
    warn "E2E verification had issues (exit code $VERIFY_EXIT) — continuing anyway"
else
    ok "E2E verification passed"
fi

# ══════════════════════════════════════
# STEP 7: Install and start daemon
# ══════════════════════════════════════
step 7 "Installing daemon (LaunchAgent)"

cat > "$PLIST_PATH" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${VENV_DIR}/bin/python</string>
        <string>-m</string>
        <string>src.agent.tracker</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${TRACKER_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${DATA_DIR}/tracker-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${DATA_DIR}/tracker-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
PLISTEOF
ok "LaunchAgent plist written"

info "Loading daemon..."
launchctl load "$PLIST_PATH" 2>/dev/null || true
launchctl start "$PLIST_NAME" 2>/dev/null || true

# Poll for daemon startup (up to 5 seconds)
DAEMON_STARTED=false
for i in 1 2 3 4 5; do
    if launchctl list "$PLIST_NAME" > /dev/null 2>&1; then
        DAEMON_PID=$(launchctl list "$PLIST_NAME" 2>/dev/null | grep PID | awk '{print $NF}' | tr -d '";')
        if [[ -n "$DAEMON_PID" ]] && [[ "$DAEMON_PID" != "-" ]] && [[ "$DAEMON_PID" != "0" ]]; then
            ok "Daemon running (PID: $DAEMON_PID)"
            DAEMON_STARTED=true
            break
        fi
    fi
    sleep 1
done
if [[ "$DAEMON_STARTED" != "true" ]]; then
    warn "Daemon may not have started — check logs:"
    warn "  tail -f $DATA_DIR/tracker-stderr.log"
fi

# ══════════════════════════════════════
# STEP 7b: Generate .claude/launch.json
# ══════════════════════════════════════
info "Writing .claude/launch.json for dev servers..."
mkdir -p "$REPO_DIR/.claude"
cat > "$REPO_DIR/.claude/launch.json" << LAUNCHEOF
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "pmis-v2-server",
      "runtimeExecutable": "python3",
      "runtimeArgs": ["$PMIS_DIR/server.py"],
      "port": 8100
    }
  ]
}
LAUNCHEOF
ok ".claude/launch.json generated"

# Generate MCP config with correct paths
info "Writing pmis_v2/claude_mcp_config.json..."
cat > "$PMIS_DIR/claude_mcp_config.json" << MCPEOF
{
  "mcpServers": {
    "pmis-memory": {
      "command": "$VENV_DIR/bin/python",
      "args": ["$PMIS_DIR/mcp_server.py", "--transport", "stdio"],
      "env": {
        "PYTHONPATH": "$PMIS_DIR"
      }
    }
  }
}
MCPEOF
ok "MCP config generated"

# ══════════════════════════════════════
# STEP 8: Summary
# ══════════════════════════════════════
step 8 "Installation complete"

echo ""
echo -e "${BOLD}═══════════════════════════════════════${NC}"
echo -e "${BOLD}  Installation Complete!${NC}"
echo -e "${BOLD}═══════════════════════════════════════${NC}"
echo ""
echo -e "  ${GREEN}Daemon:${NC}     Running (captures screenshots every 10s)"
echo -e "  ${GREEN}Portal:${NC}     Start with: ./start.sh"
echo -e "  ${GREEN}Portal URL:${NC} http://localhost:8000"
echo -e "  ${GREEN}Data:${NC}       $DATA_DIR/"
echo -e "  ${GREEN}Logs:${NC}       $DATA_DIR/tracker-stderr.log"
echo ""
if [[ $PERM_ISSUES -eq 1 ]]; then
    echo -e "  ${RED}ACTION REQUIRED — Grant macOS permissions:${NC}"
    echo -e "  ${YELLOW}1.${NC} System Settings → Privacy & Security → Screen Recording → Enable Terminal/Python"
    echo -e "  ${YELLOW}2.${NC} System Settings → Privacy & Security → Accessibility → Enable Terminal/Python"
    echo ""
fi
echo -e "  ${BLUE}Useful commands:${NC}"
echo -e "    Check daemon:   launchctl list $PLIST_NAME"
echo -e "    Stop daemon:    launchctl unload $PLIST_PATH"
echo -e "    Start daemon:   launchctl load $PLIST_PATH"
echo -e "    View logs:      tail -f $DATA_DIR/tracker-stderr.log"
echo -e "    Start portal:   ./start.sh"
echo ""
