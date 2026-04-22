"""
macOS platform implementations.

- ScreenshotCapture: screencapture CLI (fast) with Quartz PyObjC fallback.
- InputMonitor: Quartz CGEventTap for separate kb/mouse counts, IOKit idle-time fallback.
- WindowMonitor: AppleScript via `osascript` + System Events.
"""

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock

from .base import ScreenshotBase, WindowInfo, WindowMonitorBase

logger = logging.getLogger("tracker.platforms.macos")

try:
    from Quartz import (
        CGEventTapCreate,
        CGEventTapEnable,
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionListenOnly,
        CGEventMaskBit,
        kCGEventKeyDown,
        kCGEventKeyUp,
        kCGEventMouseMoved,
        kCGEventLeftMouseDown,
        kCGEventLeftMouseUp,
        kCGEventRightMouseDown,
        kCGEventScrollWheel,
        kCGEventLeftMouseDragged,
        kCGEventFlagsChanged,
        CFMachPortCreateRunLoopSource,
        CFRunLoopGetCurrent,
        CFRunLoopAddSource,
        CFRunLoopRunInMode,
        kCFRunLoopDefaultMode,
    )
    _HAS_QUARTZ = True
except ImportError:
    _HAS_QUARTZ = False
    logger.info("Quartz not available — using IOKit-only input detection (combined kb+mouse)")


class ScreenshotCapture(ScreenshotBase):
    """macOS screenshot capture via `screencapture` CLI with PyObjC fallback."""

    def _capture_raw_png(self, path: Path) -> bool:
        # Fast path: screencapture CLI (no accessibility permission needed for screen).
        try:
            result = subprocess.run(
                ["screencapture", "-x", "-t", "png", str(path)],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0 and path.exists():
                return True
            logger.warning(f"screencapture failed: {result.stderr.decode(errors='ignore')}")
        except subprocess.TimeoutExpired:
            logger.error("screencapture timed out")
        except Exception as e:
            logger.error(f"screencapture error: {e}")

        # Fallback: PyObjC CGWindowListCreateImage.
        return self._capture_with_pyobjc(path)

    def _capture_with_pyobjc(self, path: Path) -> bool:
        try:
            import Quartz

            image = Quartz.CGWindowListCreateImage(
                Quartz.CGRectInfinite,
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
                Quartz.kCGWindowImageDefault,
            )
            if image is None:
                logger.error("CGWindowListCreateImage returned None")
                return False

            bitmap = Quartz.NSBitmapImageRep.alloc().initWithCGImage_(image)
            # Write PNG so the base class's PIL pipeline handles resize + JPEG conversion.
            png_data = bitmap.representationUsingType_properties_(
                Quartz.NSBitmapImageFileTypePNG,
                None,
            )
            png_data.writeToFile_atomically_(str(path), True)
            return path.exists()

        except ImportError:
            logger.error("PyObjC not available — screencapture CLI is the only option")
            return False
        except Exception as e:
            logger.error(f"PyObjC screenshot failed: {e}")
            return False


class InputMonitor:
    """
    Tracks keyboard and mouse activity between frame captures on macOS.

    Uses Quartz CGEventTap when available (gives separate kb/mouse counts);
    falls back to IOKit HIDIdleTime polling (combined activity only).
    """

    def __init__(self, config: dict):
        self._interval = config["tracking"].get("screenshot_interval_active", 10)
        self._running = False
        self._lock = Lock()

        self._keyboard_events = 0
        self._mouse_events = 0

        self._last_idle_ns = None
        self._last_check_time = time.time()

        self._thread: Thread | None = None
        self._use_quartz = _HAS_QUARTZ

    def start(self):
        self._running = True
        if self._use_quartz:
            self._thread = Thread(target=self._run_event_tap, daemon=True)
            self._thread.start()
            logger.info("Input monitor started (Quartz CGEvent tap — separate kb/mouse)")
        else:
            self._thread = Thread(target=self._poll_iokit, daemon=True)
            self._thread.start()
            logger.info("Input monitor started (IOKit fallback — combined activity)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Input monitor stopped.")

    def get_activity_since_last_check(self) -> dict:
        with self._lock:
            if self._use_quartz:
                kb = self._keyboard_events > 0
                mouse = self._mouse_events > 0
                self._keyboard_events = 0
                self._mouse_events = 0
                return {"keyboard": kb, "mouse": mouse}
            has_activity = self._check_iokit_activity()
            return {"keyboard": has_activity, "mouse": has_activity}

    def _run_event_tap(self):
        keyboard_mask = (
            CGEventMaskBit(kCGEventKeyDown)
            | CGEventMaskBit(kCGEventKeyUp)
            | CGEventMaskBit(kCGEventFlagsChanged)
        )
        mouse_mask = (
            CGEventMaskBit(kCGEventMouseMoved)
            | CGEventMaskBit(kCGEventLeftMouseDown)
            | CGEventMaskBit(kCGEventLeftMouseUp)
            | CGEventMaskBit(kCGEventRightMouseDown)
            | CGEventMaskBit(kCGEventScrollWheel)
            | CGEventMaskBit(kCGEventLeftMouseDragged)
        )
        all_mask = keyboard_mask | mouse_mask

        def callback(proxy, event_type, event, refcon):
            with self._lock:
                if event_type in (kCGEventKeyDown, kCGEventKeyUp, kCGEventFlagsChanged):
                    self._keyboard_events += 1
                else:
                    self._mouse_events += 1
            return event

        try:
            tap = CGEventTapCreate(
                kCGSessionEventTap,
                kCGHeadInsertEventTap,
                kCGEventTapOptionListenOnly,
                all_mask,
                callback,
                None,
            )
            if tap is None:
                logger.warning("Failed to create CGEvent tap (accessibility permissions needed?)")
                logger.info("Falling back to IOKit polling")
                self._use_quartz = False
                self._poll_iokit()
                return

            CGEventTapEnable(tap, True)
            source = CFMachPortCreateRunLoopSource(None, tap, 0)
            loop = CFRunLoopGetCurrent()
            CFRunLoopAddSource(loop, source, kCFRunLoopDefaultMode)

            while self._running:
                CFRunLoopRunInMode(kCFRunLoopDefaultMode, 1.0, False)

        except Exception as e:
            logger.warning(f"CGEvent tap failed: {e}, falling back to IOKit")
            self._use_quartz = False
            self._poll_iokit()

    def _poll_iokit(self):
        while self._running:
            try:
                result = subprocess.run(
                    ["ioreg", "-c", "IOHIDSystem", "-d", "4", "-S"],
                    capture_output=True, text=True, timeout=3,
                )
                for line in result.stdout.split("\n"):
                    if "HIDIdleTime" in line:
                        parts = line.split("=")
                        if len(parts) >= 2:
                            ns = int(parts[-1].strip())
                            with self._lock:
                                self._last_idle_ns = ns
                        break
            except Exception as e:
                logger.debug(f"IOKit poll failed: {e}")
            time.sleep(1)

    def _check_iokit_activity(self) -> bool:
        if self._last_idle_ns is None:
            return False
        idle_secs = self._last_idle_ns / 1_000_000_000
        return idle_secs < self._interval


class WindowMonitor(WindowMonitorBase):
    """Active-window monitor via AppleScript + System Events (no accessibility API needed)."""

    def _get_frontmost_window(self) -> WindowInfo | None:
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
