"""
PMIS Menu Bar App — Option A.

A native macOS menu bar icon (∞) that mirrors the infinite-loop web widget.
Clicks run synchronously on the main thread; network uses urllib with a
5-second timeout so a flaky server can't wedge the UI. The whole surface is
just a thin client over the existing /api/work/* and /api/harness/* endpoints.

Run:
  pmis_v2/desktop/.venv/bin/python3 pmis_v2/desktop/menubar_app.py

Auto-start:
  ./pmis_v2/desktop/install_launch_agent.sh
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import Any, Dict, List, Optional

import rumps

logger = logging.getLogger("pmis.menubar")

PMIS_BASE = "http://localhost:8100"
POLL_INTERVAL_SECS = 30
REQUEST_TIMEOUT_SECS = 5

ICON_IDLE  = "∞"
ICON_DRIFT = "∞⚠"
ICON_DOWN  = "∞?"


# ──────────────────────────────────────────────────────────────────────
# Tiny HTTP client (stdlib only)
# ──────────────────────────────────────────────────────────────────────

def _request(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    url = PMIS_BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.debug("Request %s %s failed: %s", method, path, e)
        return None


# ──────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────

class PMISMenubar(rumps.App):
    """All state lives on `self`. All menu mutations happen on the main
    thread inside the rumps timer callback or a user-initiated click."""

    # Stable dict keys for each menu item we need to address later
    K_STATUS    = "status_row"
    K_DELIV     = "deliv_row"
    K_START     = "start_action"
    K_END       = "end_action"
    K_HARNESSES = "harnesses_submenu"
    K_BUILD     = "build_harness"
    K_TRAINING  = "training_row"

    def __init__(self) -> None:
        super().__init__("PMIS", title=ICON_IDLE, quit_button=None)
        self._state: Dict[str, Any] = {}
        self._deliverables: List[Dict[str, Any]] = []
        self._harnesses: List[Dict[str, Any]] = []
        self._training: Dict[str, int] = {}
        self._server_up: bool = False
        self._build_menu()
        # First refresh is synchronous so the menu has content when opened
        self._refresh()

    # ── Menu scaffold ─────────────────────────────────────────────────

    def _build_menu(self) -> None:
        status = rumps.MenuItem("No active session")
        deliv  = rumps.MenuItem("Deliverable: —")
        start_action = rumps.MenuItem("Start observing session",
                                        callback=self._on_click_start_or_end)
        end_action   = rumps.MenuItem("End session",
                                        callback=self._on_click_start_or_end)
        start_on = rumps.MenuItem("Start on deliverable")   # parent submenu
        confirm_on = rumps.MenuItem("Confirm deliverable")  # parent submenu
        harnesses = rumps.MenuItem("Harnesses")             # parent submenu
        build = rumps.MenuItem("Build harness (template)",
                                 callback=self._on_build_harness)
        training = rumps.MenuItem("Training: —")

        # Register each item against a stable key in self.menu
        self.menu = [
            (self.K_STATUS,    status),
            (self.K_DELIV,     deliv),
            None,
            (self.K_START,     start_action),
            ("start_on",        start_on),
            ("confirm_on",      confirm_on),
            (self.K_END,       end_action),
            None,
            (self.K_HARNESSES, harnesses),
            (self.K_BUILD,     build),
            None,
            rumps.MenuItem("Open Goals", callback=self._on_open_goals),
            rumps.MenuItem("Open Wiki",  callback=self._on_open_wiki),
            None,
            (self.K_TRAINING,  training),
            rumps.MenuItem("Export training corpus", callback=self._on_export_training),
            None,
            rumps.MenuItem("Refresh now", callback=lambda _: self._refresh()),
            rumps.MenuItem("Quit", callback=rumps.quit_application),
        ]

    # ── Polling (main thread, per rumps.timer contract) ───────────────

    @rumps.timer(POLL_INTERVAL_SECS)
    def _tick(self, _sender) -> None:
        self._refresh()

    def _refresh(self) -> None:
        """Pull all state + re-render. Synchronous; blocks up to a few
        seconds if the server is slow, but rumps.timer runs on its own
        main-thread cadence so this does not freeze clicks."""
        state = _request("GET", "/api/work/current")
        self._server_up = state is not None
        self._state = state or {}

        # Deliverables change rarely; refetch only if list is empty (first run)
        # or on demand via Refresh now.
        if not self._deliverables or not state or not state.get("active"):
            d = _request("GET", "/api/work/deliverables") or {}
            self._deliverables = d.get("deliverables", []) or []

        t = _request("GET", "/api/training/counts") or {}
        self._training = t if isinstance(t, dict) else {}

        did = (self._state.get("session") or {}).get("deliverable_id")
        if did:
            h = _request("GET", f"/api/harness?deliverable_id={urllib.parse.quote(did)}") or {}
            self._harnesses = h.get("harnesses", []) or []
        else:
            self._harnesses = []

        self._render()

    # ── Render ────────────────────────────────────────────────────────

    def _render(self) -> None:
        active = bool(self._state.get("active"))
        session = self._state.get("session") or {}
        bound = self._state.get("bound_deliverable") or {}
        drift = self._state.get("drift") or {}
        seg = self._state.get("latest_segment") or {}

        # 1. Title icon in menu bar
        if not self._server_up:
            self.title = ICON_DOWN
        elif active and drift.get("drifted"):
            self.title = ICON_DRIFT
        else:
            self.title = ICON_IDLE

        # 2. Status + deliverable rows
        if not self._server_up:
            self.menu[self.K_STATUS].title  = "Server unreachable (localhost:8100)"
            self.menu[self.K_DELIV].title   = "Deliverable: —"
        elif not active:
            self.menu[self.K_STATUS].title  = "No active session"
            self.menu[self.K_DELIV].title   = "Deliverable: —"
        elif session.get("deliverable_id"):
            name = bound.get("name") or session.get("deliverable_id")
            proj = bound.get("project_name") or ""
            self.menu[self.K_STATUS].title = "● Active"
            tag = ""
            if drift.get("drifted"):
                tag = f"  ⚠ drift {int(drift.get('similarity', 0)*100)}%"
            self.menu[self.K_DELIV].title  = f"Deliverable: {proj} · {name[:40]}{tag}"
        else:
            self.menu[self.K_STATUS].title = "● Observing (no deliverable bound)"
            self.menu[self.K_DELIV].title  = "Deliverable: —"

        # 3. Start vs End swap — the Start row becomes the End row when
        # a session is active, so the user never sees an illegal action.
        if active:
            self.menu[self.K_START].title = "Start session"
            self.menu[self.K_START].set_callback(None)
            self.menu[self.K_END].set_callback(self._on_click_start_or_end)
        else:
            self.menu[self.K_START].title = "Start observing session"
            self.menu[self.K_START].set_callback(self._on_click_start_or_end)
            self.menu[self.K_END].set_callback(None)

        # 4. Rebuild "Start on…" and "Confirm deliverable…" submenus
        self._rebuild_deliverable_submenu("start_on", self._on_start_on,
                                            enabled=not active)
        self._rebuild_deliverable_submenu("confirm_on", self._on_confirm_deliverable,
                                            enabled=active)

        # 5. Harness submenu + Build button gating
        bound_did = session.get("deliverable_id") if active else None
        self._rebuild_harnesses_submenu(enabled=bool(bound_did))
        if bound_did:
            self.menu[self.K_BUILD].set_callback(self._on_build_harness)
        else:
            self.menu[self.K_BUILD].set_callback(None)

        # 6. Training footer
        if self._training:
            parts = " · ".join(f"{k}:{v}" for k, v in self._training.items())
            self.menu[self.K_TRAINING].title = f"Training: {parts}"
        else:
            self.menu[self.K_TRAINING].title = "Training: —"

        # 7. If a session is active, append the latest segment as a third
        # info row below Status/Deliverable. Keep it short.
        if active and seg:
            mins = int((seg.get("length_secs") or 0) / 60)
            self.menu[self.K_DELIV].title += f"  · seg {mins} min on {seg.get('platform') or '—'}"

    # ── Submenu rebuild helpers ───────────────────────────────────────

    def _rebuild_deliverable_submenu(self, key: str, click_cb, enabled: bool) -> None:
        parent = self.menu.get(key)
        if parent is None:
            return
        parent.clear()
        # Parent is clickable only as a hover — we never want to fire an action
        # at the parent itself. The submenu is where clicks happen.
        if not self._deliverables:
            parent.add(rumps.MenuItem("(no deliverables synced)", callback=None))
            return
        for d in self._deliverables[:40]:
            label = f"{d.get('project_name','—')} · {d.get('name','—')[:40]}"
            mi = rumps.MenuItem(label,
                                 callback=click_cb if enabled else None)
            mi._pmis_deliverable_id = d.get("deliverable_id")
            parent.add(mi)

    def _rebuild_harnesses_submenu(self, enabled: bool) -> None:
        parent = self.menu.get(self.K_HARNESSES)
        if parent is None:
            return
        parent.clear()
        if not enabled or not self._harnesses:
            parent.add(rumps.MenuItem("(bind a deliverable first)"
                                        if not enabled else
                                        "(none for this deliverable)",
                                        callback=None))
            return
        for h in self._harnesses[:10]:
            runs = h.get("run_count") or 0
            rate = h.get("success_rate") or 0
            label = f"{(h.get('title') or h.get('id'))[:36]} · {runs}r · {int(rate*100)}%"
            mi = rumps.MenuItem(label)
            copy_item = rumps.MenuItem("Copy /pmis-harness <id>", callback=self._on_copy_harness)
            copy_item._pmis_harness_id = h.get("id")
            up_item = rumps.MenuItem("Thumb up", callback=self._on_thumb_up)
            up_item._pmis_harness_id = h.get("id")
            dn_item = rumps.MenuItem("Thumb down", callback=self._on_thumb_down)
            dn_item._pmis_harness_id = h.get("id")
            mi.add(copy_item)
            mi.add(up_item)
            mi.add(dn_item)
            parent.add(mi)

    # ── Callbacks (all main thread) ───────────────────────────────────

    def _on_click_start_or_end(self, sender) -> None:
        """Unified handler for the Start / End action row. Looks at current
        state to decide, so the click is always safe."""
        if self._state.get("active"):
            r = _request("POST", "/api/work/end", body={})
            if r and r.get("segments_bound") is not None:
                rumps.notification(
                    "PMIS session ended", "",
                    f"{r.get('segments_bound', 0)} segment(s) bound",
                )
        else:
            _request("POST", "/api/work/start", body={})
        self._refresh()

    def _on_start_on(self, sender) -> None:
        did = getattr(sender, "_pmis_deliverable_id", None)
        if not did:
            return
        _request("POST", "/api/work/start", body={"deliverable_id": did})
        self._refresh()

    def _on_confirm_deliverable(self, sender) -> None:
        did = getattr(sender, "_pmis_deliverable_id", None)
        if not did:
            return
        _request("POST", "/api/work/confirm",
                  body={"deliverable_id": did, "auto_assigned": 0})
        self._refresh()

    def _on_build_harness(self, _sender) -> None:
        did = (self._state.get("session") or {}).get("deliverable_id")
        if not did:
            return
        r = _request("POST", "/api/harness/build",
                      body={"deliverable_id": did, "use_llm": False})
        if r and r.get("id"):
            rumps.notification("Harness built", "", f"{r['id']} — {r.get('mode','')}")
        self._refresh()

    def _on_copy_harness(self, sender) -> None:
        hid = getattr(sender, "_pmis_harness_id", None)
        if not hid:
            return
        self._copy_to_clipboard(f"/pmis-harness {hid}")
        rumps.notification("Copied", "", f"/pmis-harness {hid}")

    def _on_thumb_up(self, sender) -> None:
        self._record_thumb(sender, "up")

    def _on_thumb_down(self, sender) -> None:
        self._record_thumb(sender, "down")

    def _record_thumb(self, sender, thumb: str) -> None:
        hid = getattr(sender, "_pmis_harness_id", None)
        if not hid:
            return
        _request("POST", f"/api/harness/{hid}/record-run",
                  body={"thumb": thumb, "notes": "menubar"})
        self._refresh()

    def _on_export_training(self, _sender) -> None:
        r = _request("POST", "/api/training/export", body={}) or {}
        total = r.get("total_events", 0)
        rumps.notification("Training exported", "", f"{total} events")
        self._refresh()

    def _on_open_goals(self, _sender) -> None:
        webbrowser.open(f"{PMIS_BASE}/wiki/goals")

    def _on_open_wiki(self, _sender) -> None:
        webbrowser.open(f"{PMIS_BASE}/wiki/")

    # ── Clipboard ─────────────────────────────────────────────────────

    def _copy_to_clipboard(self, text: str) -> None:
        try:
            from AppKit import NSPasteboard, NSStringPboardType
            pb = NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb.setString_forType_(text, NSStringPboardType)
        except Exception:
            import subprocess
            subprocess.run(["pbcopy"], input=text.encode(), check=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                          format="%(asctime)s %(name)s %(levelname)s %(message)s")
    PMISMenubar().run()


if __name__ == "__main__":
    main()
