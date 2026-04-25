"""Tests for Phase C2 — deliverable_sections reader + PUT endpoint."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── load_deliverable_sections ────────────────────────────────────────

@pytest.fixture
def renderer_with_sections(tmp_path):
    """A real WikiRenderer wired to a sqlite DB carrying just the
    deliverable_sections table — enough for the reader under test."""
    from wiki_renderer import WikiRenderer

    db_path = tmp_path / "pm.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE deliverable_sections (
            deliverable_id TEXT NOT NULL,
            slot TEXT NOT NULL,
            body_md TEXT NOT NULL DEFAULT '',
            updated_at TEXT DEFAULT '',
            source TEXT DEFAULT 'auto',
            PRIMARY KEY (deliverable_id, slot)
        );
    """)
    conn.commit()

    db = MagicMock()
    db.db_path = str(db_path)
    db._conn = conn
    yield WikiRenderer(db), conn
    conn.close()


def _seed(conn, deliverable_id, slot, body_md, source="user", updated_at="2026-04-25"):
    conn.execute(
        "INSERT OR REPLACE INTO deliverable_sections "
        "(deliverable_id, slot, body_md, source, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (deliverable_id, slot, body_md, source, updated_at),
    )
    conn.commit()


class TestLoadDeliverableSections:
    def test_returns_all_six_slots_for_unseen_deliverable(self, renderer_with_sections):
        wr, _ = renderer_with_sections
        out = wr.load_deliverable_sections("D-new")
        assert set(out.keys()) == {
            "overview", "progress", "decisions", "questions", "risks", "links",
        }

    def test_unseen_deliverable_has_empty_bodies(self, renderer_with_sections):
        wr, _ = renderer_with_sections
        out = wr.load_deliverable_sections("D-empty")
        for slot, data in out.items():
            assert data["body_md"] == ""
            assert data["body_html"] == ""
            assert data["is_empty"] is True

    def test_each_slot_carries_its_label(self, renderer_with_sections):
        wr, _ = renderer_with_sections
        out = wr.load_deliverable_sections("D-x")
        assert out["overview"]["label"] == "Overview"
        assert out["progress"]["label"] == "Progress"
        assert out["decisions"]["label"] == "Decisions"
        assert out["questions"]["label"] == "Open questions"
        assert out["risks"]["label"] == "Risks"
        assert out["links"]["label"] == "Contributing links"

    def test_seeded_slot_returned(self, renderer_with_sections):
        wr, conn = renderer_with_sections
        _seed(conn, "D-1", "overview", "## Hello\n\nbody", source="user")
        out = wr.load_deliverable_sections("D-1")
        assert out["overview"]["body_md"] == "## Hello\n\nbody"
        assert out["overview"]["source"] == "user"
        assert out["overview"]["is_empty"] is False
        # Other slots remain empty.
        assert out["progress"]["body_md"] == ""
        assert out["progress"]["is_empty"] is True

    def test_body_html_is_rendered(self, renderer_with_sections):
        wr, conn = renderer_with_sections
        _seed(conn, "D-html", "overview", "## Title\n\nA paragraph here.")
        out = wr.load_deliverable_sections("D-html")
        html = out["overview"]["body_html"]
        # The shared _markdown_to_html turns ## into <h2 id=...>
        assert "<h2" in html
        assert "Title</h2>" in html
        assert "<p>" in html

    def test_unknown_slot_in_db_is_ignored(self, renderer_with_sections):
        """If somehow an out-of-enum slot leaked into the DB, the reader
        skips it rather than surfacing it as a 7th section."""
        wr, conn = renderer_with_sections
        _seed(conn, "D-bad", "bogus_slot", "something")
        out = wr.load_deliverable_sections("D-bad")
        assert "bogus_slot" not in out
        assert len(out) == 6

    def test_only_target_deliverable_loaded(self, renderer_with_sections):
        wr, conn = renderer_with_sections
        _seed(conn, "D-1", "overview", "for D-1")
        _seed(conn, "D-2", "overview", "for D-2")
        out = wr.load_deliverable_sections("D-1")
        assert out["overview"]["body_md"] == "for D-1"


# ─── render_project_detail wiring ─────────────────────────────────────

class TestRenderProjectDetailSections:
    """Verifies the C2 payload extension: when a deliverable is selected,
    sections + section_order come back in the dict; when not, they're
    empty."""

    def test_no_selection_returns_empty_sections(self):
        from wiki_renderer import WikiRenderer
        db = MagicMock()
        wr = WikiRenderer(db)
        wr._render_pm_projects = lambda: {  # type: ignore
            "goals": [{
                "id": "G1", "title": "G",
                "projects": [{
                    "id": "P1", "title": "P", "deliverables": [],
                }],
            }],
        }
        out = wr.render_project_detail("P1")
        assert out["sections"] == {}
        assert out["section_order"] == []

    def test_selection_includes_sections(self, tmp_path):
        from wiki_renderer import WikiRenderer
        db_path = tmp_path / "pm.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE deliverable_sections (
                deliverable_id TEXT NOT NULL,
                slot TEXT NOT NULL,
                body_md TEXT NOT NULL DEFAULT '',
                updated_at TEXT DEFAULT '',
                source TEXT DEFAULT 'auto',
                PRIMARY KEY (deliverable_id, slot)
            );
        """)
        conn.commit()

        db = MagicMock()
        db.db_path = str(db_path)
        db._conn = conn

        wr = WikiRenderer(db)
        wr._render_pm_projects = lambda: {  # type: ignore
            "goals": [{
                "id": "G1", "title": "G",
                "projects": [{
                    "id": "P1", "title": "P",
                    "deliverables": [{"id": "D1", "name": "Build it"}],
                }],
            }],
        }
        out = wr.render_project_detail("P1", "D1")
        assert set(out["sections"].keys()) == {
            "overview", "progress", "decisions", "questions", "risks", "links",
        }
        assert out["section_order"] == [
            "overview", "progress", "decisions", "questions", "risks", "links",
        ]
        conn.close()


# ─── PUT /api/pm/deliverables/{did}/sections/{slot} ───────────────────

class TestSectionUpdateEndpoint:
    def test_endpoint_registered(self):
        from server import app
        path = "/api/pm/deliverables/{deliverable_id}/sections/{slot}"
        methods: set = set()
        for r in app.routes:
            if getattr(r, "path", None) == path:
                methods |= set(getattr(r, "methods", set()) or set())
        assert "PUT" in methods, f"PUT not on {path}; methods seen: {methods}"

    def test_payload_model_defaults(self):
        from server import PMDeliverableSectionUpdate
        m = PMDeliverableSectionUpdate()
        assert m.body_md == ""
        assert m.source == "user"

    def test_slot_allowlist(self):
        """Module-level allowlist matches the CHECK constraint and the
        renderer's slot order."""
        from server import _DELIVERABLE_SLOTS
        from wiki_renderer import WikiRenderer
        assert set(_DELIVERABLE_SLOTS) == set(WikiRenderer.DELIVERABLE_SLOT_ORDER)
        assert len(_DELIVERABLE_SLOTS) == 6

    def test_source_allowlist(self):
        from server import _DELIVERABLE_SOURCES
        assert set(_DELIVERABLE_SOURCES) == {"user", "auto", "llm"}
