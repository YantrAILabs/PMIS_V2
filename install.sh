#!/bin/bash
# ═══════════════════════════════════════════════════════
#  ProMe — one-command installer (macOS)
# ═══════════════════════════════════════════════════════
# Thin wrapper that delegates to prome_installer/ (Python).
# All install logic lives in prome_installer/steps.py so the same code path
# runs on Windows via install.bat.
#
# Usage:  ./install.sh
# Idempotent — safe to run multiple times.

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Find a Python >= 3.11 to bootstrap the installer.
# Once the installer runs, everything else happens inside the venv.
PYTHON3=""
for candidate in /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$candidate" > /dev/null 2>&1; then
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
            PYTHON3="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON3" ]]; then
    echo "Error: Python >= 3.11 required."
    echo "Install via: brew install python@3.12"
    exit 1
fi

exec "$PYTHON3" -m prome_installer "$@"
