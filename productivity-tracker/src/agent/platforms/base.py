"""
Shared base classes and types for cross-platform agent modules.

Platform-specific modules (_macos.py, _windows.py) subclass these and fill in
the native-API hooks (`_capture_raw_png`, `_get_frontmost_window`).
"""

import json
import logging
import os
import sqlite3
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

logger = logging.getLogger("tracker.platforms")


@dataclass
class WindowInfo:
    app_name: str       # e.g., "Google Chrome", "Terminal", "Code.exe"
    bundle_id: str      # macOS bundle ID (com.google.Chrome) or Windows process name (chrome.exe)
    title: str          # Window title (includes page title / URL for browsers)
    pid: int

    def to_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "bundle_id": self.bundle_id,
            "title": self.title,
            "pid": self.pid,
        }


# Union of browser identifiers: macOS bundle IDs + Windows process names.
# Each platform populates `bundle_id` with its native form, and
# `is_browser()` matches against this set regardless of platform.
BROWSER_IDENTIFIERS = frozenset({
    # macOS bundle IDs
    "com.google.Chrome",
    "com.apple.Safari",
    "org.mozilla.firefox",
    "com.brave.Browser",
    "com.microsoft.edgemac",
    "company.thebrowser.Browser",  # Arc
    # Windows process names (lowercase)
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "brave.exe",
    "iexplore.exe",
    "opera.exe",
    "vivaldi.exe",
    "arc.exe",
})


class ScreenshotBase(ABC):
    """
    Cross-platform screenshot capture with shared resize/thumbnail/cleanup logic.

    Subclasses implement `_capture_raw_png(path)` to write a PIL-readable image
    to the given path. The base class handles resize, JPEG conversion, and
    lifecycle bookkeeping.
    """

    def __init__(self, config: dict):
        self.max_width = config["tracking"]["screenshot_max_width"]
        # Preserve existing macOS default (/tmp/...); use tempfile-based path on Windows
        # where /tmp does not exist.
        default_dir = (
            Path(tempfile.gettempdir()) / "productivity-tracker" / "frames"
            if sys.platform == "win32"
            else Path("/tmp") / "productivity-tracker" / "frames"
        )
        self.screenshot_dir = Path(os.environ.get("SCREENSHOT_DIR", str(default_dir)))
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _capture_raw_png(self, path: Path) -> bool:
        """
        Write a PIL-readable image (PNG or JPEG) to `path`. Return True on success.
        This is the one method each platform must implement.
        """
        ...

    def capture(self) -> str | None:
        """
        Capture a screenshot, resize it, save as JPEG.
        Returns the final JPEG path, or None on failure.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = self.screenshot_dir / f"raw_{timestamp}.png"
        final_path = self.screenshot_dir / f"frame_{timestamp}.jpg"

        try:
            if not self._capture_raw_png(raw_path) or not raw_path.exists():
                logger.warning("Raw capture failed")
                return None

            img = Image.open(raw_path)
            w, h = img.size
            if w > self.max_width:
                ratio = self.max_width / w
                img = img.resize((self.max_width, int(h * ratio)), Image.LANCZOS)
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(final_path, "JPEG", quality=85)

            raw_path.unlink(missing_ok=True)
            return str(final_path)

        except Exception as e:
            logger.error(f"Screenshot pipeline failed: {e}")
            raw_path.unlink(missing_ok=True)
            return None

    def generate_thumbnail(self, jpeg_path: str, target_width: int = 256) -> str | None:
        """Write a small thumbnail next to the full JPEG (`.thumb.jpg`).
        Thumbnails are retained permanently so an audit trail survives even after
        the full JPEG is reclaimed post-sync. Returns the thumbnail path or None.
        """
        p = Path(jpeg_path)
        if not p.exists() or p.suffix.lower() != ".jpg":
            return None
        thumb = p.with_suffix(".thumb.jpg")
        if thumb.exists():
            return str(thumb)
        try:
            img = Image.open(p)
            w, h = img.size
            if w > target_width:
                ratio = target_width / w
                img = img.resize((target_width, int(h * ratio)), Image.LANCZOS)
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(thumb, "JPEG", quality=50, optimize=True)
            return str(thumb)
        except Exception as e:
            logger.warning(f"thumbnail generation failed for {p.name}: {e}")
            return None

    def cleanup_synced(self, db) -> int:
        """Delete full JPEGs for segments that are synced to memory AND
        tagged to a project. Thumbnails are always retained. Returns the
        number of JPEGs reclaimed.
        """
        db_path = str(db.engine.url).replace("sqlite:///", "")
        removed = 0
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT frame_paths_json FROM context_1 "
                "WHERE synced_to_memory = 1 AND project_id != '' "
                "  AND frame_paths_json IS NOT NULL AND frame_paths_json != ''"
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.warning(f"cleanup_synced query failed: {e}")
            return 0

        for (paths_json,) in rows:
            try:
                for path in json.loads(paths_json):
                    f = Path(path)
                    if f.exists() and f.suffix.lower() == ".jpg" and ".thumb." not in f.name:
                        f.unlink(missing_ok=True)
                        removed += 1
            except Exception:
                continue
        return removed


class WindowMonitorBase(ABC):
    """
    Cross-platform active-window monitor.

    Subclasses implement `_get_frontmost_window()` to return a `WindowInfo`
    using native OS APIs. The base class handles caching, diff detection,
    and browser/platform extraction.
    """

    BROWSER_BUNDLES = BROWSER_IDENTIFIERS  # kept for backward compat with existing call sites

    def __init__(self):
        self._last_window: WindowInfo | None = None
        self._running = False

    def start(self):
        self._running = True
        logger.info("Window monitor started.")

    def stop(self):
        self._running = False
        logger.info("Window monitor stopped.")

    @abstractmethod
    def _get_frontmost_window(self) -> WindowInfo | None:
        """Native-API hook. Returns the frontmost window's info, or None."""
        ...

    def get_active_window(self) -> dict | None:
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
        if self._last_window is None:
            return True
        return (
            current["bundle_id"] != self._last_window.bundle_id
            or current["title"] != self._last_window.title
        )

    def is_browser(self, window_info: dict) -> bool:
        return window_info.get("bundle_id", "") in BROWSER_IDENTIFIERS

    def extract_platform(self, window_info: dict) -> str:
        """
        Extract the platform/website name from window info.
        For browsers: derive from page title.
        For native apps: use app name.
        """
        if self.is_browser(window_info):
            title = window_info.get("title", "")
            for sep in [" - ", " — ", " – "]:
                if sep in title:
                    page_title = title.rsplit(sep, 1)[0].strip()
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
                    return page_title[:50]
            return title[:50]
        return window_info.get("app_name", "Unknown")
