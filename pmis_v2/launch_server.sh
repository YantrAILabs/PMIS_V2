#!/bin/bash
# ProMe Server Launch Script
# Called by launchd to start the persistent memory server

set -e

PMIS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${PMIS_DIR}/logs"

mkdir -p "${LOG_DIR}"

export PYTHONPATH="${PMIS_DIR}:${PYTHONPATH}"
export PATH="/opt/homebrew/bin:${PATH}"

cd "${PMIS_DIR}"

PYTHON3=""
for candidate in /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$candidate" > /dev/null 2>&1; then
        PYTHON3="$candidate"
        break
    fi
done
[[ -z "$PYTHON3" ]] && { echo "Python3 not found"; exit 1; }

exec "$PYTHON3" "${PMIS_DIR}/server.py"
