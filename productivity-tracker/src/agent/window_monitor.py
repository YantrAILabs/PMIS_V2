"""
Active window monitoring using macOS NSWorkspace + Accessibility APIs.
Detects current app, window title, and URL (for browsers).
"""

import logging
import subprocess
import json
from dataclasses import dataclass

logger = logging.getLogger("tracker.window")


@dataclass
class WindowInfo:
    app_name: str       # e.g., "Google Chrome", "Terminal"
    bundle_id: str      # e.g., "com.google.Chrome"
    title: str          # Window title (includes URL for browsers)
    pid: int

    def to_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "bundle_id": self.bundle_id,
            "title": self.title,
            "pid": self.pid,
        }


class WindowMonitor:
    """Monitors the active (frontmost) window on macOS."""

    # Browser bundle IDs — window title usually contains page title / URL
    BROWSER_BUNDLES = {
        "com.google.Chrome",
        "com.apple.Safari",
        "org.mozilla.firefox",
        "com.brave.Browser",
        "com.microsoft.edgemac",
        "company.thebrowser.Browser",  # Arc
    }

    def __init__(self):
        self._last_window: WindowInfo | None = None
        self._running = False

    def start(self):
        self._running = True
        logger.info("Window monitor started.")

    def stop(self):
        self._running = False
        logger.info("Window monitor stopped.")

    def get_active_window(self) -> dict | None:
        """
        Get the currently active window info.
        Returns dict with app_name, bundle_id, title, pid.
        """
        try:
            window = self._get_frontmost_window()
            if window:
                self._last_window = window
                return window.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get active window: {e}")
            return None

    def did_window_change(self, current: dict) -> bool:
        """Check if the active window has changed from last capture."""
        if self._last_window is None:
            return True
        return (
            current["bundle_id"] != self._last_window.bundle_id
            or current["title"] != self._last_window.title
        )

    def is_browser(self, window_info: dict) -> bool:
        """Check if the current window is a web browser."""
        return window_info.get("bundle_id", "") in self.BROWSER_BUNDLES

    def extract_platform(self, window_info: dict) -> str:
        """
        Extract the platform/website name from window info.
        For browsers: extract domain from title.
        For apps: use app name.
        """
        if self.is_browser(window_info):
            title = window_info.get("title", "")
            # Browser titles often end with " - App Name" or " — App Name"
            # Common patterns: "Page Title - Google Chrome"
            for sep in [" - ", " — ", " – "]:
                if sep in title:
                    page_title = title.rsplit(sep, 1)[0].strip()
                    # Try to identify platform from page title
                    platform_hints = {
                        "ChatGPT": "ChatGPT",
                        "Claude": "Claude",
                        "GitHub": "GitHub",
                        "Google Docs": "Google Docs",
                        "Notion": "Notion",
                        "Figma": "Figma",
                        "Slack": "Slack",
                        "Gmail": "Gmail",
                        "YouTube": "YouTube",
                        "Stack Overflow": "Stack Overflow",
                        "Asana": "Asana",
                        "Linear": "Linear",
                    }
                    for hint, name in platform_hints.items():
                        if hint.lower() in page_title.lower():
                            return name
                    return page_title[:50]  # Truncate long titles
            return title[:50]
        else:
            return window_info.get("app_name", "Unknown")

    def _get_frontmost_window(self) -> WindowInfo | None:
        """Get frontmost app info using AppleScript (reliable, no accessibility needed)."""
        script = '''
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set appName to name of frontApp
            set bundleID to bundle identifier of frontApp
            set appPID to unix id of frontApp
            try
                set winTitle to name of front window of frontApp
            on error
                set winTitle to ""
            end try
            return appName & "|" & bundleID & "|" & winTitle & "|" & appPID
        end tell
        '''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode != 0:
                return None

            parts = result.stdout.strip().split("|")
            if len(parts) >= 4:
                return WindowInfo(
                    app_name=parts[0],
                    bundle_id=parts[1],
                    title=parts[2],
                    pid=int(parts[3]) if parts[3].isdigit() else 0,
                )
        except Exception as e:
            logger.error(f"AppleScript window detection failed: {e}")
        return None
