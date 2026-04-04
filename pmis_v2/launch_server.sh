#!/bin/bash
# PMIS V2 Server Launch Script
# Called by launchd to start the persistent memory server

set -e

PMIS_DIR="/Users/rohitsingh/Desktop/memory/pmis_v2"
LOG_DIR="${PMIS_DIR}/logs"

mkdir -p "${LOG_DIR}"

export PYTHONPATH="${PMIS_DIR}:${PYTHONPATH}"
export PATH="/opt/homebrew/bin:${PATH}"

cd "${PMIS_DIR}"

exec /opt/homebrew/bin/python3 "${PMIS_DIR}/server.py"
