"""
Input monitor — detects keyboard and mouse activity per frame interval.

Uses macOS IOKit HIDIdleTime to detect combined input activity,
and Quartz CGEvent to distinguish keyboard vs mouse events.

Privacy: No keystrokes or mouse positions are logged. Only boolean
activity flags (keyboard active? mouse active?) per frame interval.
"""

import logging
import subprocess
import time
from threading import Thread, Lock

logger = logging.getLogger("tracker.input")

# Try to import Quartz for event monitoring (optional)
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


class InputMonitor:
    """
    Tracks keyboard and mouse activity between frame captures.

    Call `get_activity_since_last_check()` at each frame interval
    to get boolean flags for keyboard and mouse activity.
    """

    def __init__(self, config: dict):
        self._interval = config["tracking"].get("screenshot_interval_active", 10)
        self._running = False
        self._lock = Lock()

        # Activity counters (reset on each check)
        self._keyboard_events = 0
        self._mouse_events = 0

        # Fallback: IOKit-based combined activity detection
        self._last_idle_ns = None
        self._last_check_time = time.time()

        self._thread: Thread | None = None
        self._use_quartz = _HAS_QUARTZ

    def start(self):
        """Start input monitoring."""
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
        """
        Returns keyboard/mouse activity since last call.
        Resets counters after reading.

        Returns:
            {"keyboard": bool, "mouse": bool}
        """
        with self._lock:
            if self._use_quartz:
                kb = self._keyboard_events > 0
                mouse = self._mouse_events > 0
                self._keyboard_events = 0
                self._mouse_events = 0
                return {"keyboard": kb, "mouse": mouse}
            else:
                # IOKit fallback: can only detect combined activity
                has_activity = self._check_iokit_activity()
                return {"keyboard": has_activity, "mouse": has_activity}

    def _run_event_tap(self):
        """Run a Quartz CGEvent tap to count keyboard and mouse events."""
        keyboard_mask = (
            CGEventMaskBit(kCGEventKeyDown) |
            CGEventMaskBit(kCGEventKeyUp) |
            CGEventMaskBit(kCGEventFlagsChanged)
        )
        mouse_mask = (
            CGEventMaskBit(kCGEventMouseMoved) |
            CGEventMaskBit(kCGEventLeftMouseDown) |
            CGEventMaskBit(kCGEventLeftMouseUp) |
            CGEventMaskBit(kCGEventRightMouseDown) |
            CGEventMaskBit(kCGEventScrollWheel) |
            CGEventMaskBit(kCGEventLeftMouseDragged)
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
                None
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
        """Fallback: poll IOKit HIDIdleTime for combined activity detection."""
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
        """
        Check if there was input activity since last check using IOKit idle time.
        If HIDIdleTime < interval, user was active.
        """
        if self._last_idle_ns is None:
            return False
        idle_secs = self._last_idle_ns / 1_000_000_000
        # If idle time is less than our check interval, there was activity
        return idle_secs < self._interval
