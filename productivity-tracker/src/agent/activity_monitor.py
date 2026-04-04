"""
Activity monitor — tracks keystroke and mouse activity density for idle detection.
Does NOT log actual keystrokes (privacy). Only measures activity level.
"""

import logging
import subprocess
import time
from threading import Thread

logger = logging.getLogger("tracker.activity")


class ActivityMonitor:
    """Monitors user input activity for idle detection."""

    def __init__(self, config: dict):
        self.idle_threshold = config["tracking"]["idle_threshold_secs"]
        self._last_activity_time = time.time()
        self._running = False
        self._thread: Thread | None = None

    def start(self):
        """Start monitoring activity via IOKit idle time."""
        self._running = True
        self._thread = Thread(target=self._poll_idle_time, daemon=True)
        self._thread.start()
        logger.info("Activity monitor started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Activity monitor stopped.")

    def is_idle(self) -> bool:
        """Check if user has been idle longer than threshold."""
        return self.idle_duration() > self.idle_threshold

    def idle_duration(self) -> float:
        """Get seconds since last user activity."""
        return time.time() - self._last_activity_time

    def _poll_idle_time(self):
        """
        Poll macOS system idle time using ioreg.
        This measures HID (Human Interface Device) idle time —
        time since last keyboard/mouse/trackpad input.
        """
        while self._running:
            try:
                # ioreg reports idle time in nanoseconds
                result = subprocess.run(
                    [
                        "ioreg", "-c", "IOHIDSystem",
                        "-d", "4", "-S",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                output = result.stdout
                for line in output.split("\n"):
                    if "HIDIdleTime" in line:
                        # Extract nanoseconds value
                        parts = line.split("=")
                        if len(parts) >= 2:
                            ns = int(parts[-1].strip())
                            idle_secs = ns / 1_000_000_000
                            if idle_secs < 2:
                                # User active within last 2 seconds
                                self._last_activity_time = time.time()
                        break
            except Exception as e:
                logger.debug(f"Idle time poll failed: {e}")

            time.sleep(1)  # Poll every second
