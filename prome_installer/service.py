"""
Daemon installation — platform-branched.

macOS:   LaunchAgent at ~/Library/LaunchAgents/com.yantra.productivity-tracker.plist,
         loaded with `launchctl`. RunAtLoad + KeepAlive.

Windows: Startup folder shortcut at
         %APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\ProMeTracker.lnk.
         Plus an immediate detached launch so the tracker is running NOW.
         We chose this over Task Scheduler because corporate-managed Windows
         machines commonly disable schtasks for non-admin users via Group
         Policy. Startup folder works on every Windows — no admin required.

Public API used by steps/daemon.py:
    stop_daemon()   -> bool (True if a previous instance was stopped)
    install_daemon() -> (started: bool, message: str)
"""

from __future__ import annotations

import os
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
    paths.LAUNCH_AGENT_PLIST.write_text(plist_content, encoding="utf-8")

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
# Windows implementation (Startup folder shortcut + immediate detached launch)
# ═══════════════════════════════════════════════════════════════════════════

def _windows_startup_shortcut_path() -> Path:
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "ProMeTracker.lnk"


def _windows_stop_if_running() -> bool:
    """Stop tracker + remove Startup shortcut + clean up any legacy schtasks entry."""
    stopped = False

    # 1. Kill any running tracker python processes (precise match via cmdline)
    try:
        ps_cmd = (
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -like '*src.agent.tracker*' } | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, timeout=10,
        )
        stopped = True
    except Exception:
        pass

    # 2. Remove Startup folder shortcut if present
    shortcut = _windows_startup_shortcut_path()
    if shortcut.exists():
        try:
            shortcut.unlink()
            stopped = True
        except OSError:
            pass

    # 3. Clean up any legacy Task Scheduler entry from older installs
    try:
        subprocess.run(
            ["schtasks.exe", "/Query", "/TN", paths.TASK_SCHEDULER_NAME],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["schtasks.exe", "/End", "/TN", paths.TASK_SCHEDULER_NAME],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["schtasks.exe", "/Delete", "/TN", paths.TASK_SCHEDULER_NAME, "/F"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass

    if stopped:
        time.sleep(1)
    return stopped


def _windows_install() -> tuple[bool, str]:
    """Install the Windows tracker daemon via Startup folder shortcut.

    Works on any Windows (including corporate-managed with Group Policy
    restrictions) because it requires no admin and no schtasks access.
    Also immediately launches the tracker as a detached background process
    so the user doesn't need to log out/in before data starts flowing.
    """
    # 1. Write the wrapper .bat — handles cwd + log redirection
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    wrapper_bat = paths.DATA_DIR / "tracker-run.bat"
    stdout_log = paths.DATA_DIR / "tracker-stdout.log"
    stderr_log = paths.DATA_DIR / "tracker-stderr.log"
    # Three orthogonal defenses baked into the wrapper:
    #  - PYTHONDONTWRITEBYTECODE=1  stops Python reading/writing .pyc; avoids
    #    the Windows bug where `git pull` leaves stale .pyc shadowing fresh
    #    .py (filesystem-mtime quirk), silently regressing settings.
    #  - chcp 65001 + PYTHONIOENCODING=utf-8 + PYTHONUTF8=1  keeps stdout/
    #    stderr on UTF-8 once redirected to a file. Otherwise cp1252 is the
    #    default and any non-ASCII print() raises UnicodeEncodeError and
    #    crashes the process (seen on pmis_v2/server.py startup).
    wrapper_bat.write_text(
        "@echo off\r\n"
        "chcp 65001 > nul\r\n"
        "set PYTHONDONTWRITEBYTECODE=1\r\n"
        "set PYTHONIOENCODING=utf-8\r\n"
        "set PYTHONUTF8=1\r\n"
        f'cd /d "{paths.TRACKER_DIR}"\r\n'
        f'"{paths.VENV_PYTHON}" -B -m src.agent.tracker >> "{stdout_log}" 2>> "{stderr_log}"\r\n',
        encoding="utf-8",
    )

    # 2. Create the Startup folder shortcut via WScript.Shell (no admin).
    #    PowerShell is the most reliable host for COM on any Windows edition.
    shortcut = _windows_startup_shortcut_path()
    shortcut.parent.mkdir(parents=True, exist_ok=True)

    ps_cmd = (
        f"$s=(New-Object -COM WScript.Shell).CreateShortcut('{shortcut}'); "
        f"$s.TargetPath='{wrapper_bat}'; "
        f"$s.WorkingDirectory='{paths.TRACKER_DIR}'; "
        f"$s.WindowStyle=7; "
        f"$s.Save()"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        return False, f"Startup shortcut creation failed: {result.stderr.strip() or result.stdout.strip()}"

    # 3. Launch the tracker NOW as a detached, window-less background process.
    #    -B flag + PYTHONDONTWRITEBYTECODE env guarantees no stale .pyc
    #    shadowing the fresh .py (same rationale as wrapper.bat above).
    CREATE_NO_WINDOW = 0x08000000
    DETACHED_PROCESS = 0x00000008
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        subprocess.Popen(
            [str(paths.VENV_PYTHON), "-B", "-m", "src.agent.tracker"],
            cwd=str(paths.TRACKER_DIR),
            stdout=open(stdout_log, "ab"),
            stderr=open(stderr_log, "ab"),
            stdin=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS,
            close_fds=True,
            env=env,
        )
    except Exception as e:
        return True, f"Startup shortcut installed but immediate launch failed: {e}. Tracker will start on next login."

    # 4. Verify a tracker process actually came up (poll up to 5s)
    for _ in range(5):
        time.sleep(1)
        check = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*src.agent.tracker*' } | "
             "Select-Object -First 1 -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=5,
        )
        pid = (check.stdout or "").strip()
        if pid.isdigit():
            return True, f"Tracker running (PID {pid}) — Startup shortcut at {shortcut.name}"
    return True, f"Startup shortcut at {shortcut.name} (will start on next login)"


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
        wrapper = paths.DATA_DIR / "tracker-run.bat"
        return [
            ("Check tracker", 'Get-Process python | Where-Object { $_.Path -like "*productivity-tracker*" }'),
            ("Stop tracker",  'Get-CimInstance Win32_Process -Filter "Name=\'python.exe\'" | Where-Object { $_.CommandLine -like "*src.agent.tracker*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }'),
            ("Start tracker", f'Start-Process "{wrapper}" -WindowStyle Hidden'),
            ("View logs",     f'Get-Content "{paths.DATA_DIR}\\tracker-stderr.log" -Tail 50'),
        ]
    return []
