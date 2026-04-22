"""
Daemon installation — platform-branched.

macOS:   LaunchAgent at ~/Library/LaunchAgents/com.yantra.productivity-tracker.plist,
         loaded with `launchctl`. RunAtLoad + KeepAlive.

Windows: Task Scheduler task "ProMeTracker" registered via `schtasks.exe`.
         Trigger: at logon of current user. Restart on failure handled by
         the `/RI 1 /DU 9999:59 /Z` semantics — simpler alternative: let
         it run once per logon and rely on the tracker's internal
         resilience. For v0.1 we opt for the simpler "start at logon,
         auto-restart off" and revisit if crashes become a pattern.

Public API used by steps/daemon.py:
    stop_daemon()   -> bool (True if a previous instance was stopped)
    install_daemon() -> (started: bool, message: str)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from . import paths, ui


# ═══════════════════════════════════════════════════════════════════════════
# macOS implementation (LaunchAgent)
# ═══════════════════════════════════════════════════════════════════════════

_LAUNCH_AGENT_PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{venv_python}</string>
        <string>-m</string>
        <string>src.agent.tracker</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{tracker_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{stdout_log}</string>
    <key>StandardErrorPath</key>
    <string>{stderr_log}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
"""


def _macos_stop_if_running() -> bool:
    """Unload the LaunchAgent if it's currently loaded. Returns True if stopped."""
    try:
        result = subprocess.run(
            ["launchctl", "list", paths.LAUNCH_AGENT_LABEL],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False
    except Exception:
        return False

    try:
        subprocess.run(
            ["launchctl", "unload", str(paths.LAUNCH_AGENT_PLIST)],
            capture_output=True,
            timeout=5,
        )
        time.sleep(1)
        return True
    except Exception:
        return False


def _macos_install() -> tuple[bool, str]:
    plist_content = _LAUNCH_AGENT_PLIST_TEMPLATE.format(
        label=paths.LAUNCH_AGENT_LABEL,
        venv_python=paths.VENV_PYTHON,
        tracker_dir=paths.TRACKER_DIR,
        stdout_log=paths.DATA_DIR / "tracker-stdout.log",
        stderr_log=paths.DATA_DIR / "tracker-stderr.log",
    )
    paths.LAUNCH_AGENT_PLIST.parent.mkdir(parents=True, exist_ok=True)
    paths.LAUNCH_AGENT_PLIST.write_text(plist_content)

    subprocess.run(
        ["launchctl", "load", str(paths.LAUNCH_AGENT_PLIST)],
        capture_output=True,
    )
    subprocess.run(
        ["launchctl", "start", paths.LAUNCH_AGENT_LABEL],
        capture_output=True,
    )

    # Poll for up to 5 seconds for a valid PID
    for _ in range(5):
        result = subprocess.run(
            ["launchctl", "list", paths.LAUNCH_AGENT_LABEL],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Output is a plist-ish dict; we just need to confirm PID > 0
            for line in result.stdout.splitlines():
                if '"PID"' in line:
                    pid_str = line.split("=")[-1].strip().rstrip(";").strip('"')
                    if pid_str.isdigit() and int(pid_str) > 0:
                        return True, f"PID {pid_str}"
        time.sleep(1)
    return False, f"check logs at {paths.DATA_DIR / 'tracker-stderr.log'}"


# ═══════════════════════════════════════════════════════════════════════════
# Windows implementation (Task Scheduler)
# ═══════════════════════════════════════════════════════════════════════════

def _windows_stop_if_running() -> bool:
    """Stop + delete the existing scheduled task, if present."""
    try:
        query = subprocess.run(
            ["schtasks.exe", "/Query", "/TN", paths.TASK_SCHEDULER_NAME],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if query.returncode != 0:
            return False  # no existing task
    except Exception:
        return False

    # End any currently-running instance, then delete the task so /Create can recreate.
    subprocess.run(
        ["schtasks.exe", "/End", "/TN", paths.TASK_SCHEDULER_NAME],
        capture_output=True,
    )
    subprocess.run(
        ["schtasks.exe", "/Delete", "/TN", paths.TASK_SCHEDULER_NAME, "/F"],
        capture_output=True,
    )
    time.sleep(1)
    return True


def _windows_install() -> tuple[bool, str]:
    # Write a .bat wrapper into DATA_DIR. schtasks /TR points at this.
    # This handles:
    #   - cd to TRACKER_DIR so `-m src.agent.tracker` resolves
    #   - stdout/stderr redirection (Task Scheduler has no equivalent of
    #     launchd's StandardOutPath / StandardErrorPath keys)
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    wrapper_bat = paths.DATA_DIR / "tracker-run.bat"
    stdout_log = paths.DATA_DIR / "tracker-stdout.log"
    stderr_log = paths.DATA_DIR / "tracker-stderr.log"
    wrapper_bat.write_text(
        "@echo off\r\n"
        f'cd /d "{paths.TRACKER_DIR}"\r\n'
        f'"{paths.VENV_PYTHON}" -m src.agent.tracker >> "{stdout_log}" 2>> "{stderr_log}"\r\n'
    )

    create = subprocess.run(
        [
            "schtasks.exe", "/Create",
            "/SC", "ONLOGON",          # trigger: at user logon
            "/TN", paths.TASK_SCHEDULER_NAME,
            "/TR", f'"{wrapper_bat}"',
            "/RL", "LIMITED",          # run with user's standard privileges (no UAC)
            "/F",                      # force overwrite if exists
        ],
        capture_output=True,
        text=True,
    )
    if create.returncode != 0:
        return False, f"schtasks /Create failed: {create.stderr.strip() or create.stdout.strip()}"

    # Start immediately (so user doesn't have to log out+in).
    run = subprocess.run(
        ["schtasks.exe", "/Run", "/TN", paths.TASK_SCHEDULER_NAME],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        return False, f"schtasks /Run failed: {run.stderr.strip() or run.stdout.strip()}"

    # Poll /Query for "Running" status for up to 5 seconds.
    for _ in range(5):
        q = subprocess.run(
            ["schtasks.exe", "/Query", "/TN", paths.TASK_SCHEDULER_NAME, "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
        )
        if q.returncode == 0 and "Running" in q.stdout:
            return True, "Task Scheduler: ProMeTracker (Running)"
        time.sleep(1)
    # Not "Running" yet but scheduled — still a success; it'll run next logon.
    return True, "Task Scheduler: ProMeTracker (scheduled, not yet confirmed running)"


# ═══════════════════════════════════════════════════════════════════════════
# Public dispatchers
# ═══════════════════════════════════════════════════════════════════════════

def stop_daemon() -> bool:
    if sys.platform == "darwin":
        return _macos_stop_if_running()
    if sys.platform == "win32":
        return _windows_stop_if_running()
    return False


def install_daemon() -> tuple[bool, str]:
    if sys.platform == "darwin":
        return _macos_install()
    if sys.platform == "win32":
        return _windows_install()
    return False, f"Unsupported platform: {sys.platform}"


def summary_commands() -> list[tuple[str, str]]:
    """Return (label, command) pairs for the summary printout."""
    if sys.platform == "darwin":
        return [
            ("Check daemon", f"launchctl list {paths.LAUNCH_AGENT_LABEL}"),
            ("Stop daemon",  f"launchctl unload {paths.LAUNCH_AGENT_PLIST}"),
            ("Start daemon", f"launchctl load {paths.LAUNCH_AGENT_PLIST}"),
            ("View logs",    f"tail -f {paths.DATA_DIR}/tracker-stderr.log"),
        ]
    if sys.platform == "win32":
        tn = paths.TASK_SCHEDULER_NAME
        return [
            ("Check daemon", f'schtasks /Query /TN {tn}'),
            ("Stop daemon",  f'schtasks /End /TN {tn}'),
            ("Start daemon", f'schtasks /Run /TN {tn}'),
            ("View logs",    f'type "{paths.DATA_DIR}\\tracker-stderr.log"'),
        ]
    return []
