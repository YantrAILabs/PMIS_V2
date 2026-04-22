"""
Cross-platform path resolution.

Single source of truth for every directory and file the installer touches.
Importing this module computes paths once — no side effects, no I/O.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


# ── Repo structure (relative to this file) ─────────────────────────────────
# prome_installer/ sits at the repo root, so parent of __file__ is the repo.
REPO_DIR: Path = Path(__file__).resolve().parent.parent

TRACKER_DIR: Path = REPO_DIR / "productivity-tracker"
PMIS_DIR: Path = REPO_DIR / "pmis_v2"
PLATFORM_DIR: Path = REPO_DIR / "memory_system" / "platform"

# venv lives inside the tracker project (matches current install.sh)
VENV_DIR: Path = TRACKER_DIR / ".venv"

# Platform-appropriate venv Python binary path
if sys.platform == "win32":
    VENV_PYTHON: Path = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP: Path = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP = VENV_DIR / "bin" / "pip"


# ── User data directory ────────────────────────────────────────────────────
# Same on both platforms (~/.productivity-tracker/) — pathlib handles separator.
DATA_DIR: Path = Path.home() / ".productivity-tracker"


# ── Screenshot working directory ───────────────────────────────────────────
# /tmp/productivity-tracker/frames on macOS (preserves current default);
# %TEMP%\productivity-tracker\frames on Windows.
if sys.platform == "win32":
    SCREENSHOT_DIR: Path = Path(tempfile.gettempdir()) / "productivity-tracker" / "frames"
else:
    SCREENSHOT_DIR = Path("/tmp") / "productivity-tracker" / "frames"


# ── Service / daemon identifiers ───────────────────────────────────────────
# macOS LaunchAgent label + plist path
LAUNCH_AGENT_LABEL = "com.yantra.productivity-tracker"
LAUNCH_AGENT_PLIST: Path = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"

# Windows Task Scheduler task name
TASK_SCHEDULER_NAME = "ProMeTracker"


# ── Config files the installer writes ──────────────────────────────────────
CLAUDE_LAUNCH_JSON: Path = REPO_DIR / ".claude" / "launch.json"
MCP_CONFIG_JSON: Path = PMIS_DIR / "claude_mcp_config.json"

# Tracker env
ENV_FILE: Path = TRACKER_DIR / ".env"
ENV_EXAMPLE: Path = TRACKER_DIR / ".env.example"


def all_dirs_to_create() -> list[Path]:
    """Directories the installer ensures exist during step 3."""
    return [
        DATA_DIR,
        DATA_DIR / "chromadb",
        PMIS_DIR / "data",
        PMIS_DIR / "data" / "chroma",
        PLATFORM_DIR / "data",
        PLATFORM_DIR / "data" / "memories",
        SCREENSHOT_DIR,
    ]
