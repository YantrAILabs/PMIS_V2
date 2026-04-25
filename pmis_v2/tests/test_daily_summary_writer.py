"""Tests for Phase G — compose_missing_daily_summaries +
apply_pending_feedback."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sync.daily_summary_writer import (  # noqa: E402
    apply_pending_feedback,
    compose_missing_daily_summaries,
)


@pytest.fixture
def db(tmp_path):
    """Real sqlite DB carrying just the tables Phase G reads/writes."""
    path = tmp_path / "pmis.db"
    c = sqlite3.connect(path)
    c.executescript("""
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
            composed_at TEXT,
            UNIQUE(project_id, deliverable_id, date)
        );
        CREATE TABLE daily_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            daily_summary_id TEXT,
            feedback_text TEXT,
            created_at TEXT DEFAULT '2026-04-25 12:00:00',
            applied INTEGER DEFAULT 0,
            applied_at TEXT
        );
        CREATE TABLE review_proposals (
            id TEXT PRIMARY KEY,
            status TEXT,
            user_assigned_deliverable_id TEXT DEFAULT '',
            auto_attached_to_deliverable_id TEXT DEFAULT '',
            tree_json TEXT,
            confirmed_at TEXT
        );
        CREATE TABLE links (
            id TEXT PRIMARY KEY, url TEXT NOT NULL UNIQUE, kind TEXT NOT NULL
        );
        CREATE TABLE link_bindings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link_id TEXT, scope TEXT, scope_id TEXT,
            contributed INTEGER DEFAULT 1, dwell_frames INTEGER DEFAULT 0,
            UNIQUE(link_id, scope, scope_id)
        );
    """)
    c.commit()
    c.close()
    return str(path)


def _seed_match(path, **kw):
    defaults = {
        "id": "m-1", "segment_id": "s-1",
        "project_id": "P-1", "deliverable_id": "D-1",
        "work_description": "Wrote a thing", "time_mins": 5.0,
        "matched_at": "2026-04-25 10:00:00", "is_correct": 1,
    }
    defaults.update(kw)
    c = sqlite3.connect(path)
    c.execute(
        """INSERT INTO project_work_match_log
           (id, segment_id, project_id, deliverable_id, work_description,
            time_mins, matched_at, is_correct)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        tuple(defaults[k] for k in [
            "id", "segment_id", "project_id", "deliverable_id",
            "work_description", "time_mins", "matched_at", "is_correct",
        ]),
    )
    c.commit()
    c.close()


def _seed_proposal(path, **kw):
    defaults = {
        "id": "rp-1", "status": "auto_attached",
        "user_assigned_deliverable_id": "",
        "auto_attached_to_deliverable_id": "D-1",
        "tree_json": json.dumps({
            "sc_title": "Captured work",
            "sc_summary": "summary",
            "anchors": [{"title": "a", "body": "body"}],
            "segment_ids": ["s1"], "duration_mins": 5.0,
        }),
        "confirmed_at": "2026-04-25 10:00:00",
    }
    defaults.update(kw)
    c = sqlite3.connect(path)
    c.execute(
        """INSERT INTO review_proposals
           (id, status, user_assigned_deliverable_id,
            auto_attached_to_deliverable_id, tree_json, confirmed_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        tuple(defaults[k] for k in [
            "id", "status", "user_assigned_deliverable_id",
            "auto_attached_to_deliverable_id", "tree_json", "confirmed_at",
        ]),
    )
    c.commit()
    c.close()


def _seed_link(path, link_id, url, kind, scope_id, dwell=4, contributed=1):
    c = sqlite3.connect(path)
    c.execute("INSERT OR IGNORE INTO links (id, url, kind) VALUES (?, ?, ?)",
              (link_id, url, kind))
    c.execute(
        """INSERT INTO link_bindings
           (link_id, scope, scope_id, contributed, dwell_frames)
           VALUES (?, 'deliverable', ?, ?, ?)""",
        (link_id, scope_id, contributed, dwell),
    )
    c.commit()
    c.close()


def _read_summary(path, deliverable_id, date):
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    row = c.execute(
        "SELECT * FROM daily_summaries WHERE deliverable_id = ? AND date = ?",
        (deliverable_id, date),
    ).fetchone()
    c.close()
    return dict(row) if row else None


# ─── compose_missing_daily_summaries ──────────────────────────────────

class TestComposeMissing:
    def test_empty_match_log_writes_nothing(self, db):
        out = compose_missing_daily_summaries(db, date="2026-04-25")
        assert out == {"composed": 0, "skipped": 0, "scanned": 0}

    def test_one_match_writes_one_summary(self, db):
        _seed_match(db)
        out = compose_missing_daily_summaries(db, date="2026-04-25")
        assert out["composed"] == 1
        row = _read_summary(db, "D-1", "2026-04-25")
        assert row is not None
        assert row["status"] == "auto"
        assert "## D-1 · 2026-04-25" in row["body_md"]
        assert "What you worked on" in row["body_md"]
        assert "Wrote a thing" in row["body_md"]
        assert "5.0 minutes total" in row["body_md"]

    def test_existing_summary_skipped(self, db):
        _seed_match(db)
        # Pre-existing summary on the same (deliverable, date).
        c = sqlite3.connect(db)
        c.execute(
            "INSERT INTO daily_summaries "
            "(id, project_id, deliverable_id, date, body_md, status, composed_at) "
            "VALUES ('ds-pre', 'P-1', 'D-1', '2026-04-25', 'pre-written', "
            "'manual', '2026-04-25 09:00')"
        )
        c.commit()
        c.close()
        out = compose_missing_daily_summaries(db, date="2026-04-25")
        assert out["composed"] == 0
        assert out["skipped"] == 1
        # Pre-existing body untouched.
        row = _read_summary(db, "D-1", "2026-04-25")
        assert row["body_md"] == "pre-written"

    def test_pending_match_ignored(self, db):
        _seed_match(db, is_correct=-1)
        out = compose_missing_daily_summaries(db, date="2026-04-25")
        assert out["composed"] == 0

    def test_other_deliverable_isolated(self, db):
        _seed_match(db, id="m-1", deliverable_id="D-1", time_mins=5.0)
        _seed_match(db, id="m-2", deliverable_id="D-2", time_mins=8.0)
        compose_missing_daily_summaries(db, date="2026-04-25")
        d1 = _read_summary(db, "D-1", "2026-04-25")
        d2 = _read_summary(db, "D-2", "2026-04-25")
        assert "5.0 minutes total" in d1["body_md"]
        assert "8.0 minutes total" in d2["body_md"]

    def test_total_minutes_sums_correctly(self, db):
        _seed_match(db, id="m-1", time_mins=2.5)
        _seed_match(db, id="m-2", time_mins=3.5,
                    work_description="other thing")
        compose_missing_daily_summaries(db, date="2026-04-25")
        row = _read_summary(db, "D-1", "2026-04-25")
        assert "6.0 minutes total" in row["body_md"]
        assert "2 work items" in row["body_md"]

    def test_captured_progress_block(self, db):
        _seed_match(db)
        _seed_proposal(db)
        compose_missing_daily_summaries(db, date="2026-04-25")
        row = _read_summary(db, "D-1", "2026-04-25")
        assert "Captured progress" in row["body_md"]
        assert "Captured work" in row["body_md"]
        assert "*auto-attached*" in row["body_md"]

    def test_contributing_links_block(self, db):
        _seed_match(db)
        _seed_link(db, "L-1", "https://figma.com/file/abc", "design",
                   "D-1", dwell=4, contributed=1)
        compose_missing_daily_summaries(db, date="2026-04-25")
        row = _read_summary(db, "D-1", "2026-04-25")
        assert "Contributing links" in row["body_md"]
        assert "[design]" in row["body_md"]
        assert "https://figma.com/file/abc" in row["body_md"]

    def test_non_contributed_links_excluded(self, db):
        _seed_match(db)
        _seed_link(db, "L-1", "https://x.com", "web",
                   "D-1", dwell=1, contributed=0)
        compose_missing_daily_summaries(db, date="2026-04-25")
        row = _read_summary(db, "D-1", "2026-04-25")
        assert "Contributing links" not in row["body_md"]


# ─── apply_pending_feedback ───────────────────────────────────────────

class TestApplyFeedback:
    def _seed_summary(self, db, ds_id="ds-1", body="## Original"):
        c = sqlite3.connect(db)
        c.execute(
            """INSERT INTO daily_summaries
               (id, project_id, deliverable_id, date, body_md, status,
                composed_at)
               VALUES (?, 'P-1', 'D-1', '2026-04-25', ?, 'auto',
                       '2026-04-25 23:00:00')""",
            (ds_id, body),
        )
        c.commit()
        c.close()

    def _seed_feedback(self, db, ds_id="ds-1", text="please fix the second sentence"):
        c = sqlite3.connect(db)
        c.execute(
            "INSERT INTO daily_feedback (daily_summary_id, feedback_text) "
            "VALUES (?, ?)",
            (ds_id, text),
        )
        c.commit()
        c.close()

    def test_no_feedback_is_noop(self, db):
        out = apply_pending_feedback(db)
        assert out == {"applied": 0, "orphan": 0, "scanned": 0}

    def test_pending_feedback_applied(self, db):
        _seed_match(db)
        self._seed_summary(db)
        self._seed_feedback(db, text="add a section about open risks")
        out = apply_pending_feedback(db)
        assert out["applied"] == 1
        row = _read_summary(db, "D-1", "2026-04-25")
        assert row["status"] == "edited"
        assert "**Feedback applied**" in row["body_md"]
        assert "open risks" in row["body_md"]
        # Re-composed body includes the work item too.
        assert "Wrote a thing" in row["body_md"]
        c = sqlite3.connect(db)
        applied_flag = c.execute(
            "SELECT applied FROM daily_feedback"
        ).fetchone()[0]
        c.close()
        assert applied_flag == 1

    def test_already_applied_not_reprocessed(self, db):
        self._seed_summary(db)
        c = sqlite3.connect(db)
        c.execute(
            "INSERT INTO daily_feedback "
            "(daily_summary_id, feedback_text, applied) "
            "VALUES ('ds-1', 'old', 1)"
        )
        c.commit()
        c.close()
        out = apply_pending_feedback(db)
        assert out == {"applied": 0, "orphan": 0, "scanned": 0}

    def test_orphan_feedback_left_unapplied(self, db):
        # Feedback targets a daily_summary that doesn't exist.
        self._seed_feedback(db, ds_id="ds-missing", text="anything")
        out = apply_pending_feedback(db)
        assert out["orphan"] == 1
        assert out["applied"] == 0
        c = sqlite3.connect(db)
        applied_flag = c.execute(
            "SELECT applied FROM daily_feedback"
        ).fetchone()[0]
        c.close()
        assert applied_flag == 0  # orphan stays in the queue

    def test_status_flips_to_edited(self, db):
        _seed_match(db)
        self._seed_summary(db)
        self._seed_feedback(db)
        apply_pending_feedback(db)
        row = _read_summary(db, "D-1", "2026-04-25")
        assert row["status"] == "edited"
