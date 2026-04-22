"""
Screenshot capture — platform dispatcher.

Re-exports the concrete `ScreenshotCapture` for the current OS from
`src.agent.platforms`. All shared logic (resize, thumbnail, cleanup_synced)
lives in `platforms.base.ScreenshotBase`; platform-specific frame grabbing
lives in `_macos.py` / `_windows.py`.
"""

from src.agent.platforms import ScreenshotCapture

__all__ = ["ScreenshotCapture"]
