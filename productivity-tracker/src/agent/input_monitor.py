"""
Input monitor — platform dispatcher.

Re-exports the concrete `InputMonitor` for the current OS from
`src.agent.platforms`. macOS uses Quartz CGEventTap; Windows uses pynput.
Both expose the same interface: start(), stop(),
get_activity_since_last_check() -> {"keyboard": bool, "mouse": bool}.

Privacy: no keystrokes or mouse positions are logged — only boolean
activity flags per frame interval.
"""

from src.agent.platforms import InputMonitor

__all__ = ["InputMonitor"]
