"""Tests for Phase D4 — load_deliverable_links + project_link_rollup +
links slot auto-fill + PATCH /api/links/bindings."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def renderer(tmp_path):
    """A real WikiRenderer over a sqlite DB carrying the four tables D4
    reads (links, link_bindings, deliverable_sections, review_proposals)."""
    from wiki_renderer import WikiRenderer

    db_path = tmp_path / "pm.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE links (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL,
            title TEXT DEFAULT ''
        );
        CREATE TABLE link_bindings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            contributed INTEGER NOT NULL DEFAULT 1,
            dwell_frames INTEGER DEFAULT 0,
            UNIQUE(link_id, scope, scope_id)
        );
        CREATE TABLE deliverable_sections (
            deliverable_id TEXT NOT NULL,
            slot TEXT NOT NULL,
            body_md TEXT NOT NULL DEFAULT '',
            updated_at TEXT DEFAULT '',
            source TEXT DEFAULT 'auto',
            PRIMARY KEY (deliverable_id, slot)
        );
        CREATE TABLE review_proposals (
            id TEXT PRIMARY KEY,
            status TEXT,
            user_assigned_deliverable_id TEXT DEFAULT '',
            auto_attached_to_deliverable_id TEXT DEFAULT '',
            tree_json TEXT,
            confirmed_at TEXT
        );
    """)
    conn.commit()

    db = MagicMock()
    db.db_path = str(db_path)
    db._conn = conn
    yield WikiRenderer(db), conn
    conn.close()


def _seed_link(conn, link_id, url, kind="web"):
    conn.execute(
        "INSERT INTO links (id, url, kind) VALUES (?, ?, ?)",
        (link_id, url, kind),
    )
    conn.commit()


def _seed_binding(
    conn, link_id, scope_id, contributed=1, dwell=2, scope="deliverable",
):
    conn.execute(
        "INSERT INTO link_bindings "
        "(link_id, scope, scope_id, contributed, dwell_frames) "
        "VALUES (?, ?, ?, ?, ?)",
        (link_id, scope, scope_id, contributed, dwell),
    )
    conn.commit()


# ─── load_deliverable_links ───────────────────────────────────────────

class TestLoadDeliverableLinks:
    def test_empty_returns_empty_list(self, renderer):
        wr, _ = renderer
        assert wr.load_deliverable_links("D-1") == []

    def test_single_binding(self, renderer):
        wr, conn = renderer
        _seed_link(conn, "L-1", "https://figma.com/file/abc", "design")
        _seed_binding(conn, "L-1", "D-1", contributed=1, dwell=4)
        out = wr.load_deliverable_links("D-1")
        assert out == [{
            "link_id": "L-1",
            "url": "https://figma.com/file/abc",
            "kind": "design",
            "contributed": 1,
            "dwell_frames": 4,
        }]

    def test_contributed_first_then_dwell_desc(self, renderer):
        wr, conn = renderer
        _seed_link(conn, "L-low", "https://low.com", "web")
        _seed_link(conn, "L-mid", "https://mid.com", "web")
        _seed_link(conn, "L-hi", "https://hi.com", "web")
        # Two contributed + one not — contributed=1 sorts first
        # regardless of dwell.
        _seed_binding(conn, "L-hi", "D-1", contributed=1, dwell=10)
        _seed_binding(conn, "L-mid", "D-1", contributed=1, dwell=2)
        _seed_binding(conn, "L-low", "D-1", contributed=0, dwell=99)
        urls = [r["url"] for r in wr.load_deliverable_links("D-1")]
        assert urls == [
            "https://hi.com", "https://mid.com", "https://low.com",
        ]

    def test_other_deliverable_ignored(self, renderer):
        wr, conn = renderer
        _seed_link(conn, "L-1", "https://other.com")
        _seed_binding(conn, "L-1", "D-other")
        assert wr.load_deliverable_links("D-1") == []

    def test_other_scope_ignored(self, renderer):
        """A binding tagged scope='project' shouldn't surface on the
        deliverable view."""
        wr, conn = renderer
        _seed_link(conn, "L-1", "https://x.com")
        _seed_binding(conn, "L-1", "D-1", scope="project")
        assert wr.load_deliverable_links("D-1") == []


# ─── project_link_rollup ──────────────────────────────────────────────

class TestProjectLinkRollup:
    def _stub_pm(self, wr, deliverable_ids):
        wr._render_pm_projects = lambda: {  # type: ignore
            "goals": [{
                "id": "G-1",
                "projects": [{
                    "id": "P-1",
                    "deliverables": [{"id": did} for did in deliverable_ids],
                }],
            }],
        }

    def test_unknown_project_returns_empty(self, renderer):
        wr, _ = renderer
        wr._render_pm_projects = lambda: {"goals": []}  # type: ignore
        assert wr.project_link_rollup("P-missing") == []

    def test_aggregates_across_deliverables(self, renderer):
        wr, conn = renderer
        self._stub_pm(wr, ["D-1", "D-2"])
        _seed_link(conn, "L-shared", "https://shared.com", "web")
        _seed_link(conn, "L-d1", "https://d1.com", "code")
        _seed_binding(conn, "L-shared", "D-1", contributed=1, dwell=3)
        _seed_binding(conn, "L-shared", "D-2", contributed=1, dwell=4)
        _seed_binding(conn, "L-d1", "D-1", contributed=1, dwell=2)
        out = wr.project_link_rollup("P-1")
        # Sums dwell across the two deliverables for the shared URL.
        shared = next(r for r in out if r["url"] == "https://shared.com")
        d1only = next(r for r in out if r["url"] == "https://d1.com")
        assert shared["dwell_total"] == 7
        assert d1only["dwell_total"] == 2
        # Sorted by dwell desc.
        assert out[0]["url"] == "https://shared.com"

    def test_only_contributed_counted(self, renderer):
        wr, conn = renderer
        self._stub_pm(wr, ["D-1"])
        _seed_link(conn, "L-hidden", "https://hidden.com")
        _seed_binding(conn, "L-hidden", "D-1", contributed=0, dwell=10)
        assert wr.project_link_rollup("P-1") == []

    def test_top_n_respected(self, renderer):
        wr, conn = renderer
        self._stub_pm(wr, ["D-1"])
        for i in range(8):
            _seed_link(conn, f"L-{i}", f"https://x{i}.com")
            _seed_binding(conn, f"L-{i}", "D-1", contributed=1, dwell=10 - i)
        out = wr.project_link_rollup("P-1", limit=3)
        assert len(out) == 3
        assert [r["url"] for r in out] == [
            "https://x0.com", "https://x1.com", "https://x2.com",
        ]


# ─── Auto-fill: links slot on load_deliverable_sections ───────────────

class TestLinksSlotAutoFill:
    def test_links_slot_fills_with_link_rows(self, renderer):
        wr, conn = renderer
        _seed_link(conn, "L-1", "https://figma.com/x", "design")
        _seed_binding(conn, "L-1", "D-1", contributed=1, dwell=3)
        out = wr.load_deliverable_sections("D-1")
        slot = out["links"]
        assert slot["link_rows"]
        assert slot["link_rows"][0]["url"] == "https://figma.com/x"
        assert slot["auto_filled"] is True
        assert slot["tagged_count"] == 1
        assert slot["is_empty"] is False  # has link_rows even though body_md empty

    def test_user_saved_links_slot_not_overridden(self, renderer):
        wr, conn = renderer
        _seed_link(conn, "L-1", "https://x.com")
        _seed_binding(conn, "L-1", "D-1")
        conn.execute(
            "INSERT INTO deliverable_sections "
            "(deliverable_id, slot, body_md, source, updated_at) "
            "VALUES ('D-1', 'links', 'my own list', 'user', '2026-04-25')"
        )
        conn.commit()
        out = wr.load_deliverable_sections("D-1")
        slot = out["links"]
        assert slot["body_md"] == "my own list"
        assert slot["auto_filled"] is False
        assert slot["link_rows"] == []  # user mode → no auto

    def test_no_bindings_slot_stays_empty(self, renderer):
        wr, _ = renderer
        slot = wr.load_deliverable_sections("D-1")["links"]
        assert slot["link_rows"] == []
        assert slot["is_empty"] is True
        assert slot["auto_filled"] is False


# ─── PATCH endpoint ───────────────────────────────────────────────────

class TestLinkBindingToggleEndpoint:
    def test_endpoint_registered(self):
        from server import app
        path = "/api/links/bindings"
        methods: set = set()
        for r in app.routes:
            if getattr(r, "path", None) == path:
                methods |= set(getattr(r, "methods", set()) or set())
        assert "PATCH" in methods

    def test_payload_model(self):
        from server import LinkBindingToggle
        m = LinkBindingToggle(
            link_id="L-1", scope="deliverable", scope_id="D-1",
            contributed=1,
        )
        assert m.contributed == 1

    def test_payload_requires_all_fields(self):
        from server import LinkBindingToggle
        with pytest.raises(Exception):
            LinkBindingToggle()  # type: ignore[call-arg]
