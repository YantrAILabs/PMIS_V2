"""
Windows platform implementations.

- ScreenshotCapture: `mss` for fast multi-monitor primary-display capture.
- InputMonitor: `pynput` event listeners for separate kb/mouse counts,
  `GetLastInputInfo` fallback for combined-activity detection.
- WindowMonitor: `pywin32` foreground-window APIs + `psutil` for process name.

All native deps are marked `sys_platform == "win32"` in pyproject.toml, so
this module is safe to import only from the dispatcher (__init__.py).
"""

import logging
import time
from pathlib import Path
from threading import Thread, Lock

from .base import ScreenshotBase, WindowInfo, WindowMonitorBase

logger = logging.getLogger("tracker.platforms.windows")

# Screenshots via mss (pure-Python, works out of the box).
try:
    import mss  # type: ignore
    _HAS_MSS = True
except ImportError:
    _HAS_MSS = False
    logger.warning("mss not available — screenshots will fail on Windows")

# Event listeners via pynput (separate kb vs mouse counts, like Quartz CGEventTap on Mac).
try:
    from pynput import keyboard as _pynput_keyboard  # type: ignore
    from pynput import mouse as _pynput_mouse  # type: ignore
    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False
    logger.info("pynput not available — using GetLastInputInfo (combined kb+mouse)")

# pywin32 for active window + idle time.
try:
    import win32gui  # type: ignore
    import win32process  # type: ignore
    import ctypes
    from ctypes import wintypes
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False
    logger.warning("pywin32 not available — active window detection disabled")

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class ScreenshotCapture(ScreenshotBase):
    """Windows screenshot capture via mss (all monitors or primary)."""

    def _capture_raw_png(self, path: Path) -> bool:
        if not _HAS_MSS:
            logger.error("mss not installed — cannot capture screenshots on Windows")
            return False
        try:
            with mss.mss() as sct:
                # Monitor index 0 = virtual bounding box of all monitors;
                # index 1 = primary display. We use primary for parity with
                # macOS screencapture default behavior.
                monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                sct_img = sct.grab(monitor)
                # mss writes PNG via its own encoder; base class handles resize + JPEG.
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(path))
            return path.exists()
        except Exception as e:
            logger.error(f"mss screenshot failed: {e}")
            return False


class InputMonitor:
    """
    Tracks keyboard and mouse activity between frame captures on Windows.

    Uses pynput listeners when available (separate kb/mouse counts);
    falls back to GetLastInputInfo polling (combined activity only).
    """

    def __init__(self, config: dict):
        self._interval = config["tracking"].get("screenshot_interval_active", 10)
        self._running = False
        self._lock = Lock()

        self._keyboard_events = 0
        self._mouse_events = 0

        # Fallback state
        self._last_input_tick = None
        self._last_check_time = time.time()

        self._kb_listener = None
        self._mouse_listener = None
        self._poll_thread: Thread | None = None
        self._use_pynput = _HAS_PYNPUT

    # ── Public API (mirrors macOS InputMonitor) ────────────────────────────

    def start(self):
        self._running = True
        if self._use_pynput:
            try:
                self._kb_listener = _pynput_keyboard.Listener(
                    on_press=self._on_key,
                    on_release=None,
                )
                self._mouse_listener = _pynput_mouse.Listener(
                    on_move=self._on_mouse,
                    on_click=self._on_mouse,
                    on_scroll=self._on_mouse,
                )
                self._kb_listener.start()
                self._mouse_listener.start()
                logger.info("Input monitor started (pynput — separate kb/mouse)")
                return
            except Exception as e:
                logger.warning(f"pynput listener failed: {e}, falling back to GetLastInputInfo")
                self._use_pynput = False

        # Fallback: poll GetLastInputInfo
        self._poll_thread = Thread(target=self._poll_last_input, daemon=True)
        self._poll_thread.start()
        logger.info("Input monitor started (GetLastInputInfo fallback — combined activity)")

    def stop(self):
        self._running = False
        if self._kb_listener:
            try:
                self._kb_listener.stop()
            except Exception:
                pass
        if self._mouse_listener:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
        if self._poll_thread:
            self._poll_thread.join(timeout=3)
        logger.info("Input monitor stopped.")

    def get_activity_since_last_check(self) -> dict:
        with self._lock:
            if self._use_pynput:
                kb = self._keyboard_events > 0
                mouse = self._mouse_events > 0
                self._keyboard_events = 0
                self._mouse_events = 0
                return {"keyboard": kb, "mouse": mouse}
            has_activity = self._check_last_input_activity()
            return {"keyboard": has_activity, "mouse": has_activity}

    # ── pynput callbacks ───────────────────────────────────────────────────
    # Privacy: we count events but never record key codes or coordinates.

    def _on_key(self, key):
        with self._lock:
            self._keyboard_events += 1

    def _on_mouse(self, *args, **kwargs):
        with self._lock:
            self._mouse_events += 1

    # ── Fallback polling via GetLastInputInfo ──────────────────────────────

    def _poll_last_input(self):
        """Poll GetLastInputInfo to detect combined kb+mouse activity."""
        if not _HAS_WIN32:
            logger.error("pywin32 not available — cannot poll GetLastInputInfo")
            return

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        user32 = ctypes.windll.user32

        while self._running:
            try:
                if user32.GetLastInputInfo(ctypes.byref(lii)):
                    # dwTime is GetTickCount ms; compute idle = current - last.
                    current_tick = ctypes.windll.kernel32.GetTickCount()
                    idle_ms = current_tick - lii.dwTime
                    with self._lock:
                        self._last_input_tick = idle_ms
            except Exception as e:
                logger.debug(f"GetLastInputInfo poll failed: {e}")
            time.sleep(1)

    def _check_last_input_activity(self) -> bool:
        if self._last_input_tick is None:
            return False
        idle_secs = self._last_input_tick / 1000
        return idle_secs < self._interval


class WindowMonitor(WindowMonitorBase):
    """Active-window monitor via pywin32 (GetForegroundWindow + GetWindowText)."""

    def _get_frontmost_window(self) -> WindowInfo | None:
        if not _HAS_WIN32:
            return None
        try:
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None

            title = win32gui.GetWindowText(hwnd) or ""
            _, pid = win32process.GetWindowThreadProcessId(hwnd)

            # Resolve exe name (used as bundle_id on Windows).
            exe_name = "unknown.exe"
            app_name = "Unknown"
            if _HAS_PSUTIL and pid:
                try:
                    proc = psutil.Process(pid)
                    exe_name = (proc.name() or "unknown.exe").lower()
                    # Friendly app name: strip .exe, title-case.
                    app_name = exe_name.removesuffix(".exe").title()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            return WindowInfo(
                app_name=app_name,
                bundle_id=exe_name,
                title=title,
                pid=int(pid) if pid else 0,
            )
        except Exception as e:
            logger.error(f"Windows window detection failed: {e}")
            return None
