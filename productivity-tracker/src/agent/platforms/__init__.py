"""
Platform dispatcher for native-API agent modules.

Selects the correct implementation (macOS / Windows) based on `sys.platform`
at import time. Top-level modules (screenshot.py, input_monitor.py,
window_monitor.py) re-export from here, so call sites elsewhere in the
codebase (e.g. src/agent/tracker.py) remain unchanged.
"""

import sys

from .base import WindowInfo, BROWSER_IDENTIFIERS

if sys.platform == "darwin":
    from ._macos import ScreenshotCapture, InputMonitor, WindowMonitor
elif sys.platform == "win32":
    from ._windows import ScreenshotCapture, InputMonitor, WindowMonitor
else:
    raise ImportError(
        f"Unsupported platform: {sys.platform!r}. "
        "ProMe productivity-tracker supports 'darwin' (macOS) and 'win32' (Windows). "
        "Linux support is not implemented."
    )

__all__ = [
    "ScreenshotCapture",
    "InputMonitor",
    "WindowMonitor",
    "WindowInfo",
    "BROWSER_IDENTIFIERS",
]
