"""Tests for Phase D2 — populate_extracted_links + rollup_segment_links."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sync.links_writer import (  # noqa: E402
    populate_extracted_links,
    rollup_segment_links,
    run_links_pass,
)


@pytest.fixture
def tracker_db(tmp_path):
    """A miniature tracker DB with the two tables D2 reads/writes,
    matching the real tracker.db column set we care about."""
    db_path = tmp_path / "tracker.db"
    c = sqlite3.connect(db_path)
    c.executescript("""
        CREATE TABLE context_1 (
            id TEXT PRIMARY KEY,
            timestamp_start TEXT,
            window_name TEXT,
            segment_links TEXT
        );
        CREATE TABLE context_2 (
            id TEXT PRIMARY KEY,
            target_segment_id TEXT,
            frame_timestamp TEXT,
            raw_text TEXT,
            extracted_links TEXT
        );
    """)
    c.commit()
    c.close()
    return str(db_path)


def _seed_segment(db_path, seg_id, window_name="", ts="2026-04-25 10:00:00"):
    c = sqlite3.connect(db_path)
    c.execute(
        "INSERT INTO context_1 (id, timestamp_start, window_name) "
        "VALUES (?, ?, ?)",
        (seg_id, ts, window_name),
    )
    c.commit()
    c.close()


def _seed_frame(db_path, frame_id, segment_id, raw_text="", ts="2026-04-25 10:00:00"):
    c = sqlite3.connect(db_path)
    c.execute(
        "INSERT INTO context_2 "
        "(id, target_segment_id, frame_timestamp, raw_text) "
        "VALUES (?, ?, ?, ?)",
        (frame_id, segment_id, ts, raw_text),
    )
    c.commit()
    c.close()


def _read_extracted(db_path, frame_id):
    c = sqlite3.connect(db_path)
    row = c.execute(
        "SELECT extracted_links FROM context_2 WHERE id = ?", (frame_id,),
    ).fetchone()
    c.close()
    return row[0] if row else None


def _read_segment_links(db_path, seg_id):
    c = sqlite3.connect(db_path)
    row = c.execute(
        "SELECT segment_links FROM context_1 WHERE id = ?", (seg_id,),
    ).fetchone()
    c.close()
    return row[0] if row else None


# ─── populate_extracted_links ─────────────────────────────────────────

class TestPopulateExtractedLinks:
    def test_empty_db(self, tracker_db):
        out = populate_extracted_links(tracker_db)
        assert out == {
            "frames_scanned": 0, "frames_with_links": 0,
            "total_links_written": 0,
        }

    def test_one_frame_with_url(self, tracker_db):
        _seed_segment(tracker_db, "S-1", window_name="")
        _seed_frame(
            tracker_db, "F-1", "S-1",
            raw_text="check https://github.com/x/y for the change",
        )
        out = populate_extracted_links(tracker_db)
        assert out["frames_scanned"] == 1
        assert out["frames_with_links"] == 1
        assert out["total_links_written"] == 1
        stored = json.loads(_read_extracted(tracker_db, "F-1"))
        assert len(stored) == 1
        assert stored[0]["url"] == "https://github.com/x/y"
        assert stored[0]["kind"] == "code"

    def test_frame_with_no_links_writes_empty_array(self, tracker_db):
        _seed_segment(tracker_db, "S-1")
        _seed_frame(tracker_db, "F-1", "S-1", raw_text="just plain prose")
        populate_extracted_links(tracker_db)
        assert _read_extracted(tracker_db, "F-1") == "[]"

    def test_already_populated_frame_skipped(self, tracker_db):
        _seed_segment(tracker_db, "S-1")
        _seed_frame(tracker_db, "F-1", "S-1", raw_text="https://example.com")
        # First run scans + writes.
        populate_extracted_links(tracker_db)
        # Mutate manually so we'd notice if the second run overwrote.
        c = sqlite3.connect(tracker_db)
        c.execute(
            "UPDATE context_2 SET extracted_links = ? WHERE id='F-1'",
            (json.dumps([{"url": "sentinel", "kind": "other", "source": "ocr"}]),),
        )
        c.commit()
        c.close()
        # Second run should NOT touch the (non-empty) row.
        out2 = populate_extracted_links(tracker_db)
        assert out2["frames_scanned"] == 0
        stored = json.loads(_read_extracted(tracker_db, "F-1"))
        assert stored[0]["url"] == "sentinel"

    def test_window_name_picks_up_url(self, tracker_db):
        _seed_segment(
            tracker_db, "S-1",
            window_name="Anthropic - Chrome - https://www.anthropic.com",
        )
        _seed_frame(tracker_db, "F-1", "S-1", raw_text="")
        populate_extracted_links(tracker_db)
        stored = json.loads(_read_extracted(tracker_db, "F-1"))
        assert len(stored) == 1
        assert stored[0]["url"] == "https://www.anthropic.com"
        assert stored[0]["source"] == "window"

    def test_since_filter_respects_timestamp(self, tracker_db):
        _seed_segment(tracker_db, "S-1")
        _seed_frame(tracker_db, "F-old", "S-1",
                    raw_text="https://x.com",
                    ts="2025-01-01 10:00:00")
        _seed_frame(tracker_db, "F-new", "S-1",
                    raw_text="https://y.com",
                    ts="2026-04-25 10:00:00")
        populate_extracted_links(tracker_db, since="2026-01-01 00:00:00")
        # Only the recent one was scanned.
        assert _read_extracted(tracker_db, "F-new") is not None
        assert _read_extracted(tracker_db, "F-new") != ""
        assert _read_extracted(tracker_db, "F-old") is None

    def test_missing_extracted_links_column_handled(self, tmp_path):
        """If the Phase A migration hasn't been applied, return a clean
        error rather than crashing."""
        db_path = tmp_path / "missing.db"
        c = sqlite3.connect(db_path)
        c.execute(
            "CREATE TABLE context_2 (id TEXT, raw_text TEXT, "
            "target_segment_id TEXT, frame_timestamp TEXT)"
        )
        c.commit()
        c.close()
        out = populate_extracted_links(str(db_path))
        assert "error" in out
        assert out["frames_scanned"] == 0


# ─── rollup_segment_links ─────────────────────────────────────────────

class TestRollupSegmentLinks:
    def test_dwell_counts_aggregate(self, tracker_db):
        _seed_segment(tracker_db, "S-1")
        # Same URL appears in 3 frames → dwell_frames=3.
        for i in range(3):
            _seed_frame(tracker_db, f"F-{i}", "S-1",
                        raw_text=f"saw https://figma.com/file/abc step {i}")
        # And one frame with a different URL.
        _seed_frame(tracker_db, "F-3", "S-1",
                    raw_text="and https://github.com/x/y")
        populate_extracted_links(tracker_db)
        rollup_segment_links(tracker_db)

        rolled = json.loads(_read_segment_links(tracker_db, "S-1"))
        # Sorted by dwell desc.
        assert rolled[0]["url"] == "https://figma.com/file/abc"
        assert rolled[0]["dwell_frames"] == 3
        assert rolled[0]["kind"] == "design"
        assert rolled[1]["url"] == "https://github.com/x/y"
        assert rolled[1]["dwell_frames"] == 1

    def test_segment_with_no_links_writes_empty_array(self, tracker_db):
        _seed_segment(tracker_db, "S-1")
        _seed_frame(tracker_db, "F-1", "S-1", raw_text="plain")
        populate_extracted_links(tracker_db)
        rollup_segment_links(tracker_db)
        assert _read_segment_links(tracker_db, "S-1") == "[]"

    def test_segments_with_existing_rollup_skipped(self, tracker_db):
        _seed_segment(tracker_db, "S-1")
        _seed_frame(tracker_db, "F-1", "S-1", raw_text="https://a.com")
        # Seed a sentinel rollup directly.
        c = sqlite3.connect(tracker_db)
        c.execute(
            "UPDATE context_1 SET segment_links = ? WHERE id='S-1'",
            (json.dumps([{"url": "sentinel"}]),),
        )
        c.commit()
        c.close()
        populate_extracted_links(tracker_db)
        rollup_segment_links(tracker_db)
        rolled = json.loads(_read_segment_links(tracker_db, "S-1"))
        assert rolled[0]["url"] == "sentinel"

    def test_sources_collected(self, tracker_db):
        _seed_segment(
            tracker_db, "S-1",
            window_name="https://figma.com/file/abc",
        )
        _seed_frame(tracker_db, "F-1", "S-1",
                    raw_text="working in https://figma.com/file/abc")
        populate_extracted_links(tracker_db)
        rollup_segment_links(tracker_db)
        rolled = json.loads(_read_segment_links(tracker_db, "S-1"))
        # In a single frame we'd dedup window vs ocr to one entry, but
        # the source set captures whichever survived (window wins).
        assert rolled[0]["sources"]
        assert "window" in rolled[0]["sources"]


# ─── run_links_pass wrapper ───────────────────────────────────────────

class TestRunLinksPass:
    def test_combined_wrapper(self, tracker_db):
        _seed_segment(tracker_db, "S-1")
        _seed_frame(tracker_db, "F-1", "S-1",
                    raw_text="https://github.com/x/y")
        out = run_links_pass(tracker_db)
        assert out["extracted"]["frames_scanned"] == 1
        assert out["rolled_up"]["segments_scanned"] == 1
        assert out["rolled_up"]["total_unique_links"] == 1
