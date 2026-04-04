#!/usr/bin/env python3
"""
PMIS V2 Desktop Agent — CLI entry point

Commands:
    python3 pmis_v2/agent.py start       # Start server via launchd
    python3 pmis_v2/agent.py stop        # Stop server
    python3 pmis_v2/agent.py restart     # Restart server
    python3 pmis_v2/agent.py status      # Show server + platform status
    python3 pmis_v2/agent.py connect <platform>  # Run setup wizard
"""

import sys
import os
import json
import subprocess
import urllib.request
import urllib.error

PMIS_DIR = os.path.dirname(os.path.abspath(__file__))
PLIST = os.path.expanduser("~/Library/LaunchAgents/com.pmis.memory-server.plist")
LABEL = "com.pmis.memory-server"
SERVER_URL = "http://localhost:8100"


def _server_running() -> bool:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _launchd_loaded() -> bool:
    result = subprocess.run(
        ["launchctl", "list"], capture_output=True, text=True
    )
    return LABEL in result.stdout


def _ollama_running() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def cmd_start():
    if _server_running():
        print("[PMIS] Server already running on port 8100")
        return

    if not os.path.exists(PLIST):
        print(f"[PMIS] LaunchAgent plist not found: {PLIST}")
        print("[PMIS] Starting server directly...")
        subprocess.Popen(
            ["/opt/homebrew/bin/python3", os.path.join(PMIS_DIR, "server.py")],
            cwd=PMIS_DIR,
            stdout=open(os.path.join(PMIS_DIR, "logs", "stdout.log"), "a"),
            stderr=open(os.path.join(PMIS_DIR, "logs", "stderr.log"), "a"),
        )
    else:
        if not _launchd_loaded():
            subprocess.run(["launchctl", "load", PLIST])
            print("[PMIS] LaunchAgent loaded")
        else:
            subprocess.run(["launchctl", "kickstart", f"gui/{os.getuid()}/{LABEL}"])
            print("[PMIS] LaunchAgent kicked")

    # Wait for server
    import time
    for _ in range(10):
        time.sleep(1)
        if _server_running():
            print("[PMIS] Server is running on http://localhost:8100")
            return
    print("[PMIS] Server did not start in time. Check logs/stderr.log")


def cmd_stop():
    if _launchd_loaded():
        subprocess.run(["launchctl", "unload", PLIST])
        print("[PMIS] LaunchAgent unloaded")
    else:
        # Kill process on port 8100
        result = subprocess.run(
            ["lsof", "-ti:8100"], capture_output=True, text=True
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                subprocess.run(["kill", pid])
        print("[PMIS] Server stopped")


def cmd_restart():
    cmd_stop()
    import time
    time.sleep(2)
    cmd_start()


def cmd_status():
    server_ok = _server_running()
    ollama_ok = _ollama_running()
    launchd_ok = _launchd_loaded()

    print(f"\n{'='*50}")
    print(f"  PMIS V2 Agent Status")
    print(f"{'='*50}")
    print(f"  Server:     {'RUNNING' if server_ok else 'STOPPED'}")
    print(f"  LaunchAgent: {'LOADED' if launchd_ok else 'NOT LOADED'}")
    print(f"  Ollama:     {'RUNNING' if ollama_ok else 'STOPPED'}")

    if server_ok:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/api/agent/status", timeout=5) as resp:
                data = json.loads(resp.read())

            uptime = data.get("server_uptime_seconds", 0)
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            print(f"  Uptime:     {hours}h {minutes}m")

            stats = data.get("memory_stats", {})
            print(f"\n  Memory: {stats.get('total_nodes', 0)} nodes "
                  f"({stats.get('super_contexts', 0)} SC, "
                  f"{stats.get('contexts', 0)} CTX, "
                  f"{stats.get('anchors', 0)} ANC)")

            platforms = data.get("platforms", [])
            if platforms:
                print(f"\n  Platforms ({len(platforms)}):")
                for p in platforms:
                    status_icon = {"active": "+", "idle": "~", "disconnected": "-"}.get(p.get("status", ""), "?")
                    print(f"    [{status_icon}] {p.get('name', p['id']):20s} "
                          f"turns={p.get('total_turns', 0):4d}  "
                          f"memories={p.get('total_memories', 0):4d}  "
                          f"last={p.get('last_seen', 'never')[:16]}")
            else:
                print("\n  No platforms connected yet.")
        except Exception as e:
            print(f"\n  (Could not fetch detailed status: {e})")

    print(f"\n  Dashboard: http://localhost:8100")
    print(f"  Integrations: http://localhost:8100/integrations")
    print(f"{'='*50}\n")


def cmd_connect(platform_id: str):
    sys.path.insert(0, PMIS_DIR)
    from agent.setup_wizard import run_cli_wizard
    run_cli_wizard(platform_id)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pmis_v2/agent.py <command>")
        print("Commands: start, stop, restart, status, connect <platform>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "start":
        cmd_start()
    elif cmd == "stop":
        cmd_stop()
    elif cmd == "restart":
        cmd_restart()
    elif cmd == "status":
        cmd_status()
    elif cmd == "connect" and len(sys.argv) >= 3:
        cmd_connect(sys.argv[2])
    elif cmd == "connect":
        print("Available platforms: claude-code, claude-web, claude-desktop, openai-gpt, cursor, custom")
        print("Usage: python3 pmis_v2/agent.py connect <platform>")
    else:
        print(f"Unknown command: {cmd}")
        print("Commands: start, stop, restart, status, connect <platform>")


if __name__ == "__main__":
    main()
