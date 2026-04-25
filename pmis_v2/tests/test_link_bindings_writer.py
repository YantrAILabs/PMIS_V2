"""Tests for Phase D3 — bind_recent_matches.

Two databases are mocked: PMIS (links + link_bindings + match log)
and tracker (context_1.segment_links). Both real sqlite, no mocks.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sync.link_bindings_writer import bind_recent_matches  # noqa: E402


@pytest.fixture
def dbs(tmp_path):
    """A pair of (pmis, tracker) DBs with the tables D3 reads/writes."""
    pmis_path = tmp_path / "pmis.db"
    tracker_path = tmp_path / "tracker.db"

    pc = sqlite3.connect(pmis_path)
    pc.executescript("""
        CREATE TABLE links (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL,
            title TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE link_bindings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            contributed INTEGER NOT NULL DEFAULT 1,
            dwell_frames INTEGER DEFAULT 0,
            added_at TEXT DEFAULT (datetime('now')),
            UNIQUE(link_id, scope, scope_id)
        );
        CREATE TABLE project_work_match_log (
            id TEXT PRIMARY KEY,
            segment_id TEXT,
            project_id TEXT,
            deliverable_id TEXT,
            is_correct INTEGER,
            matched_at TEXT,
            link_bindings_written INTEGER DEFAULT 0
        );
    """)
    pc.commit()
    pc.close()

    tc = sqlite3.connect(tracker_path)
    tc.execute("""
        CREATE TABLE context_1 (
            id TEXT PRIMARY KEY,
            segment_links TEXT
        );
    """)
    tc.commit()
    tc.close()

    return str(pmis_path), str(tracker_path)


def _seed_match(pmis_path, **kw):
    defaults = {
        "id": "m-1", "segment_id": "S-1", "project_id": "P-1",
        "deliverable_id": "D-1", "is_correct": 1,
        "matched_at": "2026-04-25 10:00:00",
    }
    defaults.update(kw)
    pc = sqlite3.connect(pmis_path)
    pc.execute(
        """INSERT INTO project_work_match_log
           (id, segment_id, project_id, deliverable_id, is_correct, matched_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (defaults["id"], defaults["segment_id"], defaults["project_id"],
         defaults["deliverable_id"], defaults["is_correct"],
         defaults["matched_at"]),
    )
    pc.commit()
    pc.close()


def _seed_segment_links(tracker_path, segment_id, links):
    tc = sqlite3.connect(tracker_path)
    tc.execute(
        "INSERT INTO context_1 (id, segment_links) VALUES (?, ?)",
        (segment_id, json.dumps(links)),
    )
    tc.commit()
    tc.close()


def _bindings(pmis_path):
    pc = sqlite3.connect(pmis_path)
    pc.row_factory = sqlite3.Row
    rows = pc.execute(
        "SELECT link_id, scope, scope_id, contributed, dwell_frames "
        "FROM link_bindings ORDER BY scope_id, link_id"
    ).fetchall()
    pc.close()
    return [dict(r) for r in rows]


def _links(pmis_path):
    pc = sqlite3.connect(pmis_path)
    rows = pc.execute("SELECT url, kind FROM links ORDER BY url").fetchall()
    pc.close()
    return rows


# ─── Tests ────────────────────────────────────────────────────────────

class TestBindRecentMatches:
    def test_empty_match_log(self, dbs):
        pmis, tracker = dbs
        out = bind_recent_matches(pmis, tracker)
        assert out == {
            "matches_processed": 0, "bindings_written": 0,
            "links_created": 0,
        }

    def test_match_with_no_segment_links(self, dbs):
        pmis, tracker = dbs
        _seed_match(pmis)
        _seed_segment_links(tracker, "S-1", [])
        out = bind_recent_matches(pmis, tracker)
        assert out["matches_processed"] == 1
        assert out["bindings_written"] == 0
        assert out["links_created"] == 0
        # Match flagged as processed.
        pc = sqlite3.connect(pmis)
        flag = pc.execute(
            "SELECT link_bindings_written FROM project_work_match_log"
        ).fetchone()[0]
        pc.close()
        assert flag == 1

    def test_one_link_writes_one_binding(self, dbs):
        pmis, tracker = dbs
        _seed_match(pmis)
        _seed_segment_links(tracker, "S-1", [
            {"url": "https://github.com/x/y", "kind": "code", "dwell_frames": 5},
        ])
        out = bind_recent_matches(pmis, tracker)
        assert out["matches_processed"] == 1
        assert out["bindings_written"] == 1
        assert out["links_created"] == 1
        bindings = _bindings(pmis)
        assert len(bindings) == 1
        assert bindings[0]["scope"] == "deliverable"
        assert bindings[0]["scope_id"] == "D-1"
        assert bindings[0]["contributed"] == 1  # dwell 5 >= 2
        assert bindings[0]["dwell_frames"] == 5

    def test_dwell_below_threshold_marks_not_contributed(self, dbs):
        pmis, tracker = dbs
        _seed_match(pmis)
        _seed_segment_links(tracker, "S-1", [
            {"url": "https://x.com", "kind": "web", "dwell_frames": 1},
        ])
        bind_recent_matches(pmis, tracker, min_dwell_for_contributed=2)
        bindings = _bindings(pmis)
        assert bindings[0]["contributed"] == 0
        assert bindings[0]["dwell_frames"] == 1

    def test_custom_threshold_respected(self, dbs):
        pmis, tracker = dbs
        _seed_match(pmis)
        _seed_segment_links(tracker, "S-1", [
            {"url": "https://x.com", "kind": "web", "dwell_frames": 5},
        ])
        bind_recent_matches(pmis, tracker, min_dwell_for_contributed=10)
        bindings = _bindings(pmis)
        assert bindings[0]["contributed"] == 0  # 5 < 10

    def test_same_url_two_deliverables_one_link_two_bindings(self, dbs):
        pmis, tracker = dbs
        _seed_match(pmis, id="m-1", segment_id="S-1", deliverable_id="D-A")
        _seed_match(pmis, id="m-2", segment_id="S-2", deliverable_id="D-B")
        _seed_segment_links(tracker, "S-1", [
            {"url": "https://figma.com/file/abc", "kind": "design", "dwell_frames": 3},
        ])
        _seed_segment_links(tracker, "S-2", [
            {"url": "https://figma.com/file/abc", "kind": "design", "dwell_frames": 4},
        ])
        bind_recent_matches(pmis, tracker)
        assert len(_links(pmis)) == 1  # de-duped at the links table
        bindings = _bindings(pmis)
        assert len(bindings) == 2
        scope_ids = {b["scope_id"] for b in bindings}
        assert scope_ids == {"D-A", "D-B"}

    def test_repeat_run_is_idempotent(self, dbs):
        pmis, tracker = dbs
        _seed_match(pmis)
        _seed_segment_links(tracker, "S-1", [
            {"url": "https://x.com", "kind": "web", "dwell_frames": 5},
        ])
        first = bind_recent_matches(pmis, tracker)
        second = bind_recent_matches(pmis, tracker)
        assert first["matches_processed"] == 1
        assert second["matches_processed"] == 0  # already flagged

    def test_pending_match_skipped(self, dbs):
        """is_correct=-1 means awaiting user review — D3 doesn't bind."""
        pmis, tracker = dbs
        _seed_match(pmis, is_correct=-1)
        _seed_segment_links(tracker, "S-1", [
            {"url": "https://x.com", "kind": "web", "dwell_frames": 5},
        ])
        out = bind_recent_matches(pmis, tracker)
        assert out["matches_processed"] == 0

    def test_match_without_deliverable_skipped(self, dbs):
        """deliverable_id is the binding scope; nothing useful to write
        when it's empty (project-only match)."""
        pmis, tracker = dbs
        _seed_match(pmis, deliverable_id="")
        _seed_segment_links(tracker, "S-1", [
            {"url": "https://x.com", "kind": "web", "dwell_frames": 5},
        ])
        out = bind_recent_matches(pmis, tracker)
        assert out["matches_processed"] == 0

    def test_same_url_same_deliverable_two_matches_upserts(self, dbs):
        """Second match for the same (url, deliverable) UPSERTs the
        binding — latest match's dwell wins."""
        pmis, tracker = dbs
        _seed_match(pmis, id="m-1", segment_id="S-1", deliverable_id="D-A",
                    matched_at="2026-04-24 10:00")
        _seed_match(pmis, id="m-2", segment_id="S-2", deliverable_id="D-A",
                    matched_at="2026-04-25 10:00")
        _seed_segment_links(tracker, "S-1", [
            {"url": "https://x.com", "kind": "web", "dwell_frames": 2},
        ])
        _seed_segment_links(tracker, "S-2", [
            {"url": "https://x.com", "kind": "web", "dwell_frames": 7},
        ])
        bind_recent_matches(pmis, tracker)
        bindings = _bindings(pmis)
        # ORDER BY matched_at means S-1 processed first, then S-2 wins
        # the upsert.
        assert len(bindings) == 1
        assert bindings[0]["dwell_frames"] == 7
