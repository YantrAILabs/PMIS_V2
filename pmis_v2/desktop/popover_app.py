"""
PMIS Popover Menu Bar App — Claude-widget style.

A status-bar icon that opens a styled popover panel (instead of a native
OS dropdown) when clicked. The panel hosts a WKWebView pointed at
http://localhost:8100/widget/floating, so all UI lives in the HTML and
can be iterated on without rebuilding the native app.

Behaviour:
  • Click ∞ → popover opens below the status item (transient — auto-closes
    when you click anywhere else).
  • Popover hosts a WKWebView sized 360×560 px.
  • On each open, JS is nudged to re-fetch state via window.faReload().

Run:
  pmis_v2/desktop/.venv/bin/python3 pmis_v2/desktop/popover_app.py
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import objc
from AppKit import (
    NSApplication, NSApplicationActivationPolicyAccessory,
    NSStatusBar, NSVariableStatusItemLength,
    NSPopover, NSPopoverBehaviorTransient,
    NSViewController, NSView,
    NSMinYEdge, NSMaxYEdge,
    NSRect, NSMakeRect, NSMakeSize, NSMakePoint,
    NSObject, NSEvent, NSScreen,
    NSWindow, NSWindowStyleMaskBorderless,
    NSBackingStoreBuffered,
    NSFloatingWindowLevel,
    NSColor, NSImage, NSBezierPath, NSGradient,
    NSGraphicsContext,
)
from Foundation import NSURL, NSURLRequest, NSBundle, NSTimer, NSDate
from WebKit import WKWebView, WKWebViewConfiguration

# Quartz — CGEventSourceSecondsSinceLastEventType for idle detection.
# If unavailable (fresh install missing pyobjc-framework-Quartz), idle-return
# trigger is silently disabled; click + hot-corner still work.
try:
    from Quartz import (
        CGEventSourceSecondsSinceLastEventType,
        kCGEventSourceStateHIDSystemState,
        kCGAnyInputEventType,
    )
    _QUARTZ_AVAILABLE = True
except Exception:
    _QUARTZ_AVAILABLE = False

PMIS_WIDGET_URL = "http://localhost:8100/widget/floating"
PMIS_API_BASE = "http://localhost:8100"
POPOVER_SIZE = (360, 560)
ICON_GLYPH = "∞"

# SF Symbol → NSImage (macOS Big Sur+). We try to apply a hierarchical
# color configuration for the "active" / "drifted" / "down" states so the
# menu bar icon is visually distinct from the idle monochrome.
try:
    from AppKit import NSImageSymbolConfiguration
except Exception:
    NSImageSymbolConfiguration = None  # Fallback to text glyphs

# State → (SF-symbol-name, optional-color-factory, text-fallback-glyph).
# The "bound" row uses a custom-drawn rainbow circle (see _build_rainbow_dot),
# not an SF Symbol — red felt too alarming; rainbow reads as "live + friendly".
ICON_STATES = {
    "down":      ("exclamationmark.circle",       lambda: NSColor.systemGrayColor(),   "∞?"),
    "idle":      ("infinity",                      None,                                "∞"),
    "observing": ("infinity.circle",               lambda: NSColor.systemBlueColor(),  "∞·"),
    "bound":     (None,                            None,                                "●"),  # custom-drawn
    "drifted":   ("exclamationmark.triangle.fill", lambda: NSColor.systemOrangeColor(),"▲"),
}

# Menu-bar symbols look thin at default weight. A slightly larger point
# size + bold weight makes a red recording dot legible without making the
# icon feel heavy.
MENUBAR_SYMBOL_POINT_SIZE = 15.0

STATE_POLL_INTERVAL = 5.0    # how often the menu bar icon re-syncs to session state
STATE_TIMEOUT_SECS  = 3.0    # per-request timeout — keep short so UI doesn't stall

# Auto-trigger behavior
HOTCORNER_POLL_INTERVAL = 0.25         # seconds between mouse checks
HOTEDGE_WIDTH_PX        = 6            # right-edge strip width
HOTEDGE_VSKIP_FRAC      = 0.15         # skip top 15% and bottom 15% of the edge
HOTCORNER_DWELL_TICKS   = 2            # ≥2 ticks in zone = ~500ms dwell
HOTCORNER_COOLDOWN_SECS = 2.5          # after close, ignore retrigger for this long

# Idle-return trigger
IDLE_THRESHOLD_SECS     = 300          # away ≥5 min = "stepped away"
IDLE_RETURN_REARM_SECS  = 30           # input resumes within 30s = "came back"
IDLE_POLL_INTERVAL      = 10.0         # idle is cheap to sample; every 10s plenty

# Boot trigger — auto-open once if system uptime < this on app start
BOOT_UPTIME_THRESHOLD_SECS = 120

# Prefer the NonactivatingPanel style when we can — keeps keyboard focus
# in the user's current app. Fallback handled at runtime.
try:
    from AppKit import NSWindowStyleMaskNonactivatingPanel  # type: ignore
except Exception:
    NSWindowStyleMaskNonactivatingPanel = 0

logger = logging.getLogger("pmis.popover")


class StatusController(NSObject):
    """Target-action receiver + popover lifecycle."""

    def initWithApp_(self, app_ref):
        self = objc.super(StatusController, self).init()
        if self is None:
            return None
        self._app = app_ref
        self._status_item = None
        self._popover = None
        self._webview = None
        self._anchor_window = None
        self._anchor_view = None
        self._hot_dwell = 0
        self._last_close_time = 0.0
        self._poll_timer = None
        self._state_timer = None
        self._idle_timer = None
        self._prev_idle_secs = 0.0          # last sampled idle seconds — detect the drop
        self._was_away = False              # true once idle crossed IDLE_THRESHOLD_SECS
        self._current_icon_state = None
        self._build_status_item()
        self._build_popover()
        self._build_hot_corner_anchor()
        self._start_hot_corner_poll()
        self._start_state_poll()
        self._start_idle_poll()
        self._fetch_state_async()
        # Boot auto-open: if the machine just started, open the widget proactively.
        self._maybe_trigger_boot_open()
        return self

    def _build_status_item(self):
        bar = NSStatusBar.systemStatusBar()
        self._status_item = bar.statusItemWithLength_(NSVariableStatusItemLength)
        button = self._status_item.button()
        button.setTarget_(self)
        button.setAction_("toggle:")
        self._apply_icon_state("idle")   # initial appearance

    def _apply_icon_state(self, state: str) -> None:
        """Update the status item appearance for a logical session state.
        state ∈ {'down','idle','observing','bound','drifted'}."""
        if state == self._current_icon_state:
            return
        self._current_icon_state = state
        button = self._status_item.button()
        if button is None:
            return
        symbol_name, color_factory, fallback_text = ICON_STATES.get(
            state, ICON_STATES["idle"]
        )

        # "bound" uses a custom-drawn rainbow dot — the user wanted recording
        # feedback that's colorful but not a red alarm.
        if state == "bound":
            img = _build_rainbow_dot(size=18)
            if img is not None:
                img.setTemplate_(False)
                button.setImage_(img)
                button.setTitle_("")
                return
            # fall through to fallback_text if something went wrong

        image = None
        try:
            image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                symbol_name, f"PMIS {state}",
            ) if symbol_name else None
        except Exception:
            image = None

        if image is not None and color_factory is not None and NSImageSymbolConfiguration:
            # Palette color = full saturation (unlike hierarchical which
            # dims secondary layers and caused the washed-out look).
            # Combine with a point-size config so the dot is legible in
            # the menu bar.
            try:
                color = color_factory()
                # palette config — one color per "layer"; single-color
                # symbols just use the first entry for the whole glyph.
                try:
                    palette = NSImageSymbolConfiguration.configurationWithPaletteColors_([color])
                except Exception:
                    palette = NSImageSymbolConfiguration.configurationWithHierarchicalColor_(color)
                # size config — slight bump + bold weight for visibility
                try:
                    from AppKit import NSFontWeightSemibold
                    size_cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                        MENUBAR_SYMBOL_POINT_SIZE, NSFontWeightSemibold,
                    )
                    combined = palette.configurationByApplyingConfiguration_(size_cfg)
                except Exception:
                    combined = palette
                colored = image.imageWithSymbolConfiguration_(combined)
                if colored is not None:
                    colored.setTemplate_(False)
                    button.setImage_(colored)
                    button.setTitle_("")
                    return
            except Exception:
                pass

        if image is not None:
            # Monochrome template — adapts to menu-bar dark/light
            image.setTemplate_(True)
            try:
                size_cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                    MENUBAR_SYMBOL_POINT_SIZE, 0,
                )
                sized = image.imageWithSymbolConfiguration_(size_cfg)
                if sized is not None:
                    sized.setTemplate_(True)
                    image = sized
            except Exception:
                pass
            button.setImage_(image)
            button.setTitle_("")
            return

        # Last-resort: text fallback (old macOS or SF Symbols unavailable)
        button.setImage_(None)
        button.setTitle_(fallback_text)

    def _build_popover(self):
        vc = NSViewController.alloc().init()
        container = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, POPOVER_SIZE[0], POPOVER_SIZE[1])
        )

        config = WKWebViewConfiguration.alloc().init()
        webview = WKWebView.alloc().initWithFrame_configuration_(
            NSMakeRect(0, 0, POPOVER_SIZE[0], POPOVER_SIZE[1]),
            config,
        )
        webview.setAutoresizingMask_(0x02 | 0x10)  # width+height resize
        # Fill the full container
        container.addSubview_(webview)
        vc.setView_(container)
        self._webview = webview

        # Kick off the first load
        url = NSURL.URLWithString_(PMIS_WIDGET_URL)
        req = NSURLRequest.requestWithURL_(url)
        webview.loadRequest_(req)

        popover = NSPopover.alloc().init()
        popover.setContentSize_(NSMakeSize(POPOVER_SIZE[0], POPOVER_SIZE[1]))
        popover.setContentViewController_(vc)
        popover.setBehavior_(NSPopoverBehaviorTransient)
        popover.setAnimates_(True)
        self._popover = popover

    # ── Actions ──────────────────────────────────────────────────────

    def toggle_(self, sender):
        """Toggle popover open/closed on each status-item click."""
        if self._popover.isShown():
            self._popover.performClose_(sender)
            self._last_close_time = NSDate.date().timeIntervalSince1970()
            return
        button = self._status_item.button()
        self._popover.showRelativeToRect_ofView_preferredEdge_(
            button.bounds(), button, NSMinYEdge,
        )
        self._nudge_refresh()

    def _nudge_refresh(self):
        try:
            self._webview.evaluateJavaScript_completionHandler_(
                "window.faReload && window.faReload();", None,
            )
        except Exception:
            pass

    # ── Hot-corner ───────────────────────────────────────────────────

    def _build_hot_corner_anchor(self):
        """Invisible 1×1 anchor window positioned at the right edge, vertically
        centered. Popover edges to the left from this anchor so it hugs the
        right side of the screen where the user's cursor just arrived."""
        screen = NSScreen.mainScreen() or NSScreen.screens()[0]
        frame = screen.frame()
        center_y = frame.origin.y + frame.size.height * 0.5
        rect = NSMakeRect(frame.size.width - 1, center_y, 1, 1)
        style = NSWindowStyleMaskBorderless
        if NSWindowStyleMaskNonactivatingPanel:
            style |= NSWindowStyleMaskNonactivatingPanel
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False,
        )
        win.setOpaque_(False)
        win.setBackgroundColor_(NSColor.clearColor())
        win.setLevel_(NSFloatingWindowLevel)
        win.setIgnoresMouseEvents_(True)
        view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 1, 1))
        win.setContentView_(view)
        win.orderFrontRegardless()
        self._anchor_window = win
        self._anchor_view = view

    def _start_hot_corner_poll(self):
        self._poll_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            HOTCORNER_POLL_INTERVAL, self, "pollMouse:", None, True,
        )

    def _start_state_poll(self):
        """Separate timer that re-fetches session state every 5s and
        updates the menu bar icon accordingly."""
        self._state_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            STATE_POLL_INTERVAL, self, "pollState:", None, True,
        )

    def _start_idle_poll(self):
        """Idle-return trigger. Tracks system-wide input idleness via
        CGEventSourceSecondsSinceLastEventType; fires when the idle time had
        crossed IDLE_THRESHOLD_SECS ("away") and then drops back near zero
        ("returned"). Silently disabled if pyobjc-framework-Quartz is missing."""
        if not _QUARTZ_AVAILABLE:
            logger.info("Quartz idle detection unavailable; idle-return trigger disabled.")
            return
        self._idle_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            IDLE_POLL_INTERVAL, self, "pollIdle:", None, True,
        )

    def pollIdle_(self, _timer):
        if not _QUARTZ_AVAILABLE:
            return
        try:
            idle = float(CGEventSourceSecondsSinceLastEventType(
                kCGEventSourceStateHIDSystemState, kCGAnyInputEventType,
            ))
        except Exception:
            return

        # Mark "was away" the first time idle crosses the threshold
        if idle >= IDLE_THRESHOLD_SECS:
            self._was_away = True
        # Fire "welcome back" when idle has dropped below the rearm window
        # AND we previously registered an away state.
        elif self._was_away and idle <= IDLE_RETURN_REARM_SECS:
            self._was_away = False
            # Respect the same cooldown as hot-corner so we don't flash the
            # popover when user clicks-through immediately.
            now = NSDate.date().timeIntervalSince1970()
            if now - self._last_close_time >= HOTCORNER_COOLDOWN_SECS:
                self._show_at_hot_corner()
        self._prev_idle_secs = idle

    def _maybe_trigger_boot_open(self):
        """Auto-open the popover once if the system just booted (uptime
        < BOOT_UPTIME_THRESHOLD_SECS). Gives user a 'welcome back' at
        start-of-day without a manual click."""
        try:
            import subprocess, re, time
            out = subprocess.run(["sysctl", "-n", "kern.boottime"],
                                  capture_output=True, text=True, timeout=2).stdout
            m = re.search(r"sec\s*=\s*(\d+)", out)
            if not m:
                return
            boot_ts = int(m.group(1))
            uptime = time.time() - boot_ts
            if uptime < BOOT_UPTIME_THRESHOLD_SECS:
                logger.info("Boot trigger: uptime %.1fs (< %ds), opening widget.",
                              uptime, BOOT_UPTIME_THRESHOLD_SECS)
                # Small delay so status item is fully rendered before we pop.
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    1.5, self, "firstBootOpen:", None, False,
                )
        except Exception as e:
            logger.debug("Boot detection skipped: %s", e)

    def firstBootOpen_(self, _timer):
        """Fires ~1.5s after app start if the boot-trigger criterion was met."""
        if not self._popover.isShown():
            self._show_at_hot_corner()

    def pollState_(self, _timer):
        self._fetch_state_async()

    def _fetch_state_async(self):
        """Kick the HTTP request on a worker thread; hop back to the main
        thread for the actual icon update so AppKit stays happy."""
        t = threading.Thread(target=self._fetch_state_worker, daemon=True)
        t.start()

    def _fetch_state_worker(self):
        state = self._derive_state()
        # Schedule the UI update on the main thread. Use a 0-second
        # scheduled timer to trampoline into the AppKit run loop.
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "applyState:", state, False,
        )

    def applyState_(self, state):
        self._apply_icon_state(state)

    def _derive_state(self) -> str:
        """Hit /api/work/current and map to an ICON_STATES key."""
        try:
            req = urllib.request.Request(
                PMIS_API_BASE + "/api/work/current",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=STATE_TIMEOUT_SECS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return "down"

        if not data.get("active"):
            return "idle"
        drift = data.get("drift") or {}
        if drift.get("drifted"):
            return "drifted"
        session = data.get("session") or {}
        if session.get("deliverable_id"):
            return "bound"
        return "observing"

    def pollMouse_(self, _timer):
        if self._popover.isShown():
            self._hot_dwell = 0
            return

        # Honor cooldown after a close so flicking the cursor away (which
        # is how NSPopoverBehaviorTransient closes) doesn't immediately re-open.
        now = NSDate.date().timeIntervalSince1970()
        if now - self._last_close_time < HOTCORNER_COOLDOWN_SECS:
            self._hot_dwell = 0
            return

        loc = NSEvent.mouseLocation()  # screen coords, origin = bottom-left
        # The screen containing the cursor (multi-display safe)
        scr = None
        for s in NSScreen.screens() or []:
            f = s.frame()
            if (f.origin.x <= loc.x < f.origin.x + f.size.width and
                f.origin.y <= loc.y < f.origin.y + f.size.height):
                scr = s
                break
        if scr is None:
            self._hot_dwell = 0
            return

        frame = scr.frame()
        # Right edge strip — x within the rightmost HOTEDGE_WIDTH_PX pixels,
        # y in the middle band (skip top/bottom 15% so menu-bar + Dock
        # interactions never collide).
        y_min = frame.origin.y + frame.size.height * HOTEDGE_VSKIP_FRAC
        y_max = frame.origin.y + frame.size.height * (1 - HOTEDGE_VSKIP_FRAC)
        in_zone = (
            loc.x >= frame.origin.x + frame.size.width - HOTEDGE_WIDTH_PX
            and y_min <= loc.y <= y_max
        )

        if in_zone:
            self._hot_dwell += 1
            if self._hot_dwell >= HOTCORNER_DWELL_TICKS:
                self._show_at_hot_corner()
                self._hot_dwell = 0
        else:
            self._hot_dwell = 0

    def _show_at_hot_corner(self):
        """Open the popover pinned to the right-edge anchor (vertically
        centered). Popover edges to the left so it hugs the right side of
        the screen where the user's cursor arrived."""
        if self._anchor_view is None or self._popover.isShown():
            return
        # Reposition anchor on the CURRENT main screen each time — handles
        # display changes + fixes the earlier bug where we pushed y=0 (bottom).
        try:
            scr = NSScreen.mainScreen()
            frame = scr.frame()
            center_y = frame.origin.y + frame.size.height * 0.5
            self._anchor_window.setFrame_display_(
                NSMakeRect(frame.origin.x + frame.size.width - 1,
                           center_y, 1, 1),
                True,
            )
        except Exception:
            pass

        # NSMinXEdge pushes the popover to the LEFT of the anchor, which is
        # what we want at the right edge of the screen.
        from AppKit import NSMinXEdge
        self._popover.showRelativeToRect_ofView_preferredEdge_(
            self._anchor_view.bounds(),
            self._anchor_view,
            NSMinXEdge,
        )
        self._nudge_refresh()


def _build_rainbow_dot(size: int = 18):
    """Render a small filled circle with a 6-stop rainbow linear gradient.

    Used for the 'bound' / recording state: playful, colorful, yet small
    enough to stay subtle in the menu bar. Always returns a non-template
    NSImage so the colors survive menu-bar tint.
    """
    try:
        img = NSImage.alloc().initWithSize_(NSMakeSize(size, size))
        img.lockFocus()
        try:
            inset = max(2, size // 6)  # ~3px inset at 18px → 12px dot
            rect = NSMakeRect(inset, inset, size - 2 * inset, size - 2 * inset)
            path = NSBezierPath.bezierPathWithOvalInRect_(rect)

            # 6-stop rainbow (ROYGBV). Slight saturation reduction on the
            # extremes keeps it from feeling neon.
            colors = [
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.92, 0.26, 0.21, 1.0),  # red
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.96, 0.55, 0.16, 1.0),  # orange
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.96, 0.82, 0.18, 1.0),  # yellow
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.32, 0.78, 0.35, 1.0),  # green
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.21, 0.55, 0.92, 1.0),  # blue
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.58, 0.33, 0.83, 1.0),  # violet
            ]
            gradient = NSGradient.alloc().initWithColors_(colors)
            # 45° diagonal sweep reads as a rainbow band across the dot
            gradient.drawInBezierPath_angle_(path, 45.0)
        finally:
            img.unlockFocus()
        return img
    except Exception:
        return None


_CONTROLLER_REF = None  # module-global keeps controller alive past main()


def main():
    global _CONTROLLER_REF
    logging.basicConfig(level=logging.INFO)

    app = NSApplication.sharedApplication()
    # Accessory mode = no Dock icon, menu-bar only
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    _CONTROLLER_REF = StatusController.alloc().initWithApp_(app)

    # Ctrl-C works when running in foreground
    signal.signal(signal.SIGINT, lambda *a: app.terminate_(None))

    app.run()


if __name__ == "__main__":
    main()
