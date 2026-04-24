"""Tests for F1 claims helper — is_segment_claimed / is_workpage_claimed."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from consolidation.claims import (  # noqa: E402
    all_claimed_segment_ids,
    claimed_segment_ids,
    fully_consolidated_page_ids,
    is_segment_claimed,
    is_workpage_claimed,
)


@pytest.fixture
def conn(tmp_path):
    """Real sqlite DB populated with just the tables claims.py touches."""
    db_path = tmp_path / "pm.db"
    c = sqlite3.connect(db_path)
    c.executescript("""
        CREATE TABLE activity_time_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id TEXT,
            memory_node_id TEXT,
            duration_seconds INTEGER,
            match_source TEXT
        );
        CREATE TABLE review_proposals (
            id TEXT PRIMARY KEY,
            status TEXT,
            segment_ids_json TEXT,
            confirmed_at TEXT,
            processed_by_nightly_at TEXT
        );
        CREATE TABLE work_pages (
            id TEXT PRIMARY KEY,
            state TEXT,
            tag_state TEXT
        );
        CREATE TABLE work_page_anchors (
            page_id TEXT,
            segment_id TEXT,
            weight REAL
        );
    """)
    c.commit()
    yield c
    c.close()


class TestIsSegmentClaimed:
    def test_unseen_segment_not_claimed(self, conn):
        claimed, reason = is_segment_claimed(conn, "s-unknown")
        assert claimed is False
        assert reason == ""

    def test_activity_time_log_claims(self, conn):
        conn.execute(
            "INSERT INTO activity_time_log (segment_id, match_source) VALUES (?, ?)",
            ("s-1", "nightly"),
        )
        conn.commit()
        claimed, reason = is_segment_claimed(conn, "s-1")
        assert claimed is True
        assert reason == "activity_time_log"

    @pytest.mark.parametrize("status", ["draft", "confirmed", "auto_attached"])
    def test_claiming_proposal_statuses(self, conn, status):
        conn.execute(
            "INSERT INTO review_proposals (id, status, segment_ids_json) "
            "VALUES (?, ?, ?)",
            (f"rp-{status}", status, json.dumps(["s-2"])),
        )
        conn.commit()
        claimed, reason = is_segment_claimed(conn, "s-2")
        assert claimed is True
        assert reason == "review_proposals"

    @pytest.mark.parametrize("status", ["rejected", "superseded"])
    def test_non_claiming_statuses_release_segment(self, conn, status):
        conn.execute(
            "INSERT INTO review_proposals (id, status, segment_ids_json) "
            "VALUES (?, ?, ?)",
            (f"rp-{status}", status, json.dumps(["s-3"])),
        )
        conn.commit()
        claimed, _ = is_segment_claimed(conn, "s-3")
        assert claimed is False

    def test_malformed_segment_json_is_skipped(self, conn):
        conn.execute(
            "INSERT INTO review_proposals (id, status, segment_ids_json) "
            "VALUES ('rp-bad', 'draft', 'not-json')"
        )
        conn.commit()
        claimed, _ = is_segment_claimed(conn, "s-4")
        assert claimed is False

    def test_empty_segment_id(self, conn):
        claimed, _ = is_segment_claimed(conn, "")
        assert claimed is False

    def test_activity_log_wins_over_proposal(self, conn):
        """If both claim the segment, activity_time_log is reported first."""
        conn.execute(
            "INSERT INTO activity_time_log (segment_id, match_source) VALUES (?, ?)",
            ("s-5", "nightly"),
        )
        conn.execute(
            "INSERT INTO review_proposals (id, status, segment_ids_json) "
            "VALUES ('rp-dup', 'draft', ?)",
            (json.dumps(["s-5"]),),
        )
        conn.commit()
        claimed, reason = is_segment_claimed(conn, "s-5")
        assert claimed is True
        assert reason == "activity_time_log"


class TestIsWorkpageClaimed:
    def test_open_untagged_page_not_claimed(self, conn):
        conn.execute(
            "INSERT INTO work_pages (id, state, tag_state) "
            "VALUES ('p-1', 'open', '')"
        )
        conn.commit()
        claimed, _ = is_workpage_claimed(conn, "p-1")
        assert claimed is False

    def test_tag_state_confirmed_claims(self, conn):
        conn.execute(
            "INSERT INTO work_pages (id, state, tag_state) "
            "VALUES ('p-2', 'open', 'confirmed')"
        )
        conn.commit()
        claimed, reason = is_workpage_claimed(conn, "p-2")
        assert claimed is True
        assert reason == "tag_state=confirmed"

    @pytest.mark.parametrize("state", ["tagged", "archived"])
    def test_terminal_state_claims(self, conn, state):
        conn.execute(
            "INSERT INTO work_pages (id, state, tag_state) VALUES (?, ?, '')",
            (f"p-{state}", state),
        )
        conn.commit()
        claimed, reason = is_workpage_claimed(conn, f"p-{state}")
        assert claimed is True
        assert reason == f"state={state}"

    def test_missing_page_not_claimed(self, conn):
        claimed, _ = is_workpage_claimed(conn, "p-missing")
        assert claimed is False

    def test_fully_consolidated_set_claims_open_page(self, conn):
        """Page state=open/tag_state='' but every segment is claimed."""
        conn.execute(
            "INSERT INTO work_pages (id, state, tag_state) VALUES ('p-fc', 'open', '')"
        )
        conn.commit()
        claimed, reason = is_workpage_claimed(
            conn, "p-fc", fully_consolidated={"p-fc"},
        )
        assert claimed is True
        assert reason == "segments_fully_consolidated"

    def test_fully_consolidated_set_ignored_when_not_passed(self, conn):
        """Without the set, the segment-coverage check is skipped."""
        conn.execute(
            "INSERT INTO work_pages (id, state, tag_state) VALUES ('p-no', 'open', '')"
        )
        conn.commit()
        claimed, _ = is_workpage_claimed(conn, "p-no")
        assert claimed is False


class TestFullyConsolidatedPageIds:
    def test_page_with_all_segments_claimed(self, conn):
        conn.execute(
            "INSERT INTO work_pages (id, state, tag_state) VALUES ('p-all', 'open', '')"
        )
        conn.execute(
            "INSERT INTO work_page_anchors (page_id, segment_id) VALUES ('p-all', 's-1')"
        )
        conn.execute(
            "INSERT INTO work_page_anchors (page_id, segment_id) VALUES ('p-all', 's-2')"
        )
        conn.execute(
            "INSERT INTO activity_time_log (segment_id, match_source) VALUES ('s-1', 'nightly')"
        )
        conn.execute(
            "INSERT INTO activity_time_log (segment_id, match_source) VALUES ('s-2', 'nightly')"
        )
        conn.commit()
        assert fully_consolidated_page_ids(conn) == {"p-all"}

    def test_page_with_partial_coverage_not_returned(self, conn):
        conn.execute(
            "INSERT INTO work_pages (id, state, tag_state) VALUES ('p-part', 'open', '')"
        )
        conn.execute(
            "INSERT INTO work_page_anchors (page_id, segment_id) VALUES ('p-part', 's-10')"
        )
        conn.execute(
            "INSERT INTO work_page_anchors (page_id, segment_id) VALUES ('p-part', 's-11')"
        )
        conn.execute(
            "INSERT INTO activity_time_log (segment_id, match_source) VALUES ('s-10', 'nightly')"
        )
        conn.commit()
        assert fully_consolidated_page_ids(conn) == set()

    def test_page_with_no_anchors_not_returned(self, conn):
        conn.execute(
            "INSERT INTO work_pages (id, state, tag_state) VALUES ('p-zero', 'open', '')"
        )
        conn.commit()
        assert fully_consolidated_page_ids(conn) == set()

    def test_only_open_pages_considered(self, conn):
        conn.execute(
            "INSERT INTO work_pages (id, state, tag_state) VALUES ('p-tagged', 'tagged', 'confirmed')"
        )
        conn.execute(
            "INSERT INTO work_page_anchors (page_id, segment_id) VALUES ('p-tagged', 's-t1')"
        )
        conn.execute(
            "INSERT INTO activity_time_log (segment_id, match_source) VALUES ('s-t1', 'nightly')"
        )
        conn.commit()
        # 'p-tagged' is terminal — caught by state/tag_state checks, not here.
        assert fully_consolidated_page_ids(conn) == set()


class TestClaimedSegmentIds:
    def test_bulk_returns_only_claimed(self, conn):
        conn.execute(
            "INSERT INTO activity_time_log (segment_id, match_source) VALUES (?, ?)",
            ("s-10", "nightly"),
        )
        conn.execute(
            "INSERT INTO review_proposals (id, status, segment_ids_json) "
            "VALUES ('rp-10', 'draft', ?)",
            (json.dumps(["s-11", "s-12"]),),
        )
        conn.commit()
        out = claimed_segment_ids(conn, ["s-10", "s-11", "s-13"])
        assert out == {
            "s-10": "activity_time_log",
            "s-11": "review_proposals",
        }

    def test_empty_input(self, conn):
        assert claimed_segment_ids(conn, []) == {}

    def test_rejected_proposal_does_not_claim(self, conn):
        conn.execute(
            "INSERT INTO review_proposals (id, status, segment_ids_json) "
            "VALUES ('rp-r', 'rejected', ?)",
            (json.dumps(["s-20"]),),
        )
        conn.commit()
        assert claimed_segment_ids(conn, ["s-20"]) == {}


class TestAllClaimedSegmentIds:
    def test_collects_from_every_source(self, conn):
        conn.execute(
            "INSERT INTO activity_time_log (segment_id, match_source) VALUES (?, ?)",
            ("s-a", "nightly"),
        )
        conn.execute(
            "INSERT INTO review_proposals (id, status, segment_ids_json) "
            "VALUES ('rp-d', 'draft', ?)",
            (json.dumps(["s-b"]),),
        )
        conn.execute(
            "INSERT INTO review_proposals (id, status, segment_ids_json) "
            "VALUES ('rp-c', 'confirmed', ?)",
            (json.dumps(["s-c"]),),
        )
        conn.execute(
            "INSERT INTO review_proposals (id, status, segment_ids_json) "
            "VALUES ('rp-aa', 'auto_attached', ?)",
            (json.dumps(["s-d"]),),
        )
        conn.execute(
            "INSERT INTO review_proposals (id, status, segment_ids_json) "
            "VALUES ('rp-r', 'rejected', ?)",
            (json.dumps(["s-e"]),),
        )
        conn.execute(
            "INSERT INTO review_proposals (id, status, segment_ids_json) "
            "VALUES ('rp-s', 'superseded', ?)",
            (json.dumps(["s-f"]),),
        )
        conn.commit()
        out = all_claimed_segment_ids(conn)
        assert out == {
            "s-a": "activity_time_log",
            "s-b": "review_proposals",
            "s-c": "review_proposals",
            "s-d": "review_proposals",
        }

    def test_empty_db(self, conn):
        assert all_claimed_segment_ids(conn) == {}
