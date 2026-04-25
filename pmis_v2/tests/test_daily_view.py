"""Tests for Phase E — load_daily_view + work_match action endpoints
+ daily feedback endpoint."""

from __future__ import annotations

import sqlite3
import sys
from datetime import date as _date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def renderer(tmp_path):
    """A real WikiRenderer over a sqlite DB carrying just the tables E
    reads (project_work_match_log, daily_summaries, link_bindings,
    links)."""
    from wiki_renderer import WikiRenderer

    db_path = tmp_path / "pm.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE project_work_match_log (
            id TEXT PRIMARY KEY,
            segment_id TEXT,
            project_id TEXT,
            deliverable_id TEXT,
            work_description TEXT,
            time_mins REAL,
            matched_at TEXT,
            is_correct INTEGER
        );
        CREATE TABLE daily_summaries (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            deliverable_id TEXT,
            date TEXT,
            body_md TEXT,
            status TEXT,
            composed_at TEXT
        );
        CREATE TABLE links (
            id TEXT PRIMARY KEY, url TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL
        );
        CREATE TABLE link_bindings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link_id TEXT, scope TEXT, scope_id TEXT,
            contributed INTEGER DEFAULT 1, dwell_frames INTEGER DEFAULT 0,
            UNIQUE(link_id, scope, scope_id)
        );
        CREATE TABLE daily_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            daily_summary_id TEXT, feedback_text TEXT,
            applied INTEGER DEFAULT 0
        );
    """)
    conn.commit()

    db = MagicMock()
    db.db_path = str(db_path)
    db._conn = conn
    yield WikiRenderer(db), conn
    conn.close()


def _seed_match(conn, **kw):
    defaults = {
        "id": "m-1", "segment_id": "S-1", "project_id": "P-1",
        "deliverable_id": "D-1", "work_description": "Wrote test",
        "time_mins": 5.0, "matched_at": "2026-04-25 10:00:00",
        "is_correct": 1,
    }
    defaults.update(kw)
    conn.execute(
        """INSERT INTO project_work_match_log
           (id, segment_id, project_id, deliverable_id, work_description,
            time_mins, matched_at, is_correct)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (defaults["id"], defaults["segment_id"], defaults["project_id"],
         defaults["deliverable_id"], defaults["work_description"],
         defaults["time_mins"], defaults["matched_at"],
         defaults["is_correct"]),
    )
    conn.commit()


def _seed_daily(conn, **kw):
    defaults = {
        "id": "ds-1", "project_id": "P-1", "deliverable_id": "D-1",
        "date": "2026-04-25", "body_md": "## Hello\n\nbody",
        "status": "auto", "composed_at": "2026-04-25 23:00:00",
    }
    defaults.update(kw)
    conn.execute(
        """INSERT INTO daily_summaries
           (id, project_id, deliverable_id, date, body_md, status, composed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (defaults["id"], defaults["project_id"], defaults["deliverable_id"],
         defaults["date"], defaults["body_md"], defaults["status"],
         defaults["composed_at"]),
    )
    conn.commit()


# ─── load_daily_view ──────────────────────────────────────────────────

class TestLoadDailyView:
    def test_empty_returns_skeleton(self, renderer):
        wr, _ = renderer
        out = wr.load_daily_view("D-1", "2026-04-25")
        assert out["tagged_items"] == []
        assert out["daily_summary"] is None
        assert out["total_minutes"] == 0.0
        assert out["date"] == "2026-04-25"

    def test_tagged_items_for_matching_date(self, renderer):
        wr, conn = renderer
        _seed_match(conn, id="m-today", matched_at="2026-04-25 10:00:00")
        _seed_match(conn, id="m-other", matched_at="2026-04-24 10:00:00")
        out = wr.load_daily_view("D-1", "2026-04-25")
        assert [i["match_id"] for i in out["tagged_items"]] == ["m-today"]

    def test_other_deliverable_filtered_out(self, renderer):
        wr, conn = renderer
        _seed_match(conn, id="m-d1", deliverable_id="D-1")
        _seed_match(conn, id="m-d2", deliverable_id="D-2")
        out = wr.load_daily_view("D-1", "2026-04-25")
        assert {i["match_id"] for i in out["tagged_items"]} == {"m-d1"}

    def test_total_minutes_summed(self, renderer):
        wr, conn = renderer
        _seed_match(conn, id="m-1", time_mins=2.0)
        _seed_match(conn, id="m-2", time_mins=3.5)
        out = wr.load_daily_view("D-1", "2026-04-25")
        assert out["total_minutes"] == 5.5

    def test_daily_summary_returned(self, renderer):
        wr, conn = renderer
        _seed_daily(conn)
        out = wr.load_daily_view("D-1", "2026-04-25")
        assert out["daily_summary"]["id"] == "ds-1"
        assert "<h2" in out["daily_summary"]["body_html"]
        assert out["daily_summary"]["status"] == "auto"

    def test_daily_summary_none_when_missing(self, renderer):
        wr, _ = renderer
        out = wr.load_daily_view("D-1", "2026-04-25")
        assert out["daily_summary"] is None

    def test_can_feedback_true_for_past_date(self, renderer):
        wr, _ = renderer
        out = wr.load_daily_view("D-1", "1990-01-01")
        assert out["can_feedback"] is True

    def test_can_feedback_false_for_today(self, renderer):
        wr, _ = renderer
        out = wr.load_daily_view("D-1", _date.today().isoformat())
        assert out["can_feedback"] is False

    def test_contributing_links_filtered_to_contributed(self, renderer):
        wr, conn = renderer
        conn.execute(
            "INSERT INTO links (id, url, kind) VALUES "
            "('l1', 'https://a.com', 'web'), "
            "('l2', 'https://b.com', 'code')"
        )
        conn.execute(
            "INSERT INTO link_bindings "
            "(link_id, scope, scope_id, contributed, dwell_frames) VALUES "
            "('l1', 'deliverable', 'D-1', 1, 5), "
            "('l2', 'deliverable', 'D-1', 0, 1)"
        )
        conn.commit()
        out = wr.load_daily_view("D-1", "2026-04-25")
        urls = [r["url"] for r in out["contributing_links"]]
        assert urls == ["https://a.com"]


# ─── Endpoint registration + payload models ───────────────────────────

class TestEndpointShape:
    @pytest.mark.parametrize("path", [
        "/api/work_match/{match_id}/confirm",
        "/api/work_match/{match_id}/remove",
        "/api/work_match/{match_id}/reassign",
        "/api/daily/{daily_summary_id}/feedback",
    ])
    def test_endpoint_registered(self, path):
        from server import app
        methods: set = set()
        for r in app.routes:
            if getattr(r, "path", None) == path:
                methods |= set(getattr(r, "methods", set()) or set())
        assert "POST" in methods, f"POST missing on {path}"

    def test_reassign_payload_model(self):
        from server import WorkMatchReassignPayload
        m = WorkMatchReassignPayload(project_id="P-1", deliverable_id="D-1")
        assert m.project_id == "P-1"
        assert m.deliverable_id == "D-1"
        # deliverable_id is optional (defaults to "")
        m2 = WorkMatchReassignPayload(project_id="P-1")
        assert m2.deliverable_id == ""

    def test_feedback_payload_model(self):
        from server import DailyFeedbackPayload
        m = DailyFeedbackPayload(feedback_text="needs more detail")
        assert m.feedback_text == "needs more detail"
