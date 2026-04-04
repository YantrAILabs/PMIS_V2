"""
Platform Health Checker

Runs as a background thread inside the server process.
Checks platform connectivity and dependency health every 60s.
"""

import threading
import time
import json
import urllib.request
from datetime import datetime
from typing import Callable, Optional


class HealthChecker:
    """Periodic health checker for platforms and dependencies."""

    def __init__(self, get_platforms: Callable, update_status: Callable, push_event: Callable):
        self._get_platforms = get_platforms
        self._update_status = update_status
        self._push_event = push_event
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.ollama_status = "unknown"
        self.last_check = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def _loop(self):
        while self._running:
            time.sleep(60)
            try:
                self._check_all()
            except Exception:
                pass

    def _check_all(self):
        self.last_check = datetime.now().isoformat()

        # Check Ollama
        self.ollama_status = "running" if self._check_ollama() else "stopped"

        # Check platforms
        platforms = self._get_platforms()
        for p in platforms:
            self._check_platform(p)

    def _check_ollama(self) -> bool:
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _check_platform(self, platform: dict):
        last_seen = platform.get("last_seen")
        if not last_seen:
            return

        try:
            seen_dt = datetime.fromisoformat(last_seen)
        except (ValueError, TypeError):
            return

        delta_minutes = (datetime.now() - seen_dt).total_seconds() / 60

        if delta_minutes < 5:
            new_status = "active"
        elif delta_minutes < 60:
            new_status = "idle"
        else:
            new_status = "disconnected"

        if platform.get("status") != new_status:
            self._update_status(platform["id"], new_status)
            self._push_event({
                "type": "status_change",
                "platform_id": platform["id"],
                "old_status": platform["status"],
                "new_status": new_status,
                "timestamp": datetime.now().isoformat(),
            })

    def get_status(self) -> dict:
        return {
            "running": self.is_running,
            "ollama": self.ollama_status,
            "last_check": self.last_check,
        }
