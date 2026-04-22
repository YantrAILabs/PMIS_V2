"""
Active-window monitor — platform dispatcher.

Re-exports the concrete `WindowMonitor` for the current OS from
`src.agent.platforms`. macOS uses AppleScript (System Events); Windows uses
pywin32 (GetForegroundWindow + GetWindowText). Shared logic — did_window_change,
is_browser, extract_platform — lives in `platforms.base.WindowMonitorBase`.

The `WindowInfo` dataclass is re-exported for any consumer that
constructs or inspects it directly.
"""

from src.agent.platforms import WindowMonitor, WindowInfo

__all__ = ["WindowMonitor", "WindowInfo"]
