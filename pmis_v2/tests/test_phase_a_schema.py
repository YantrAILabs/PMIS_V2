"""Tests for Phase A — schema plumbing for F2+ features.

Runs DBManager.__init__ against a fresh sqlite path and asserts every
Phase A table/column/index exists, CHECK constraints fire on invalid
input, and the migration is idempotent across multiple init calls.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.manager import DBManager  # noqa: E402


@pytest.fixture
def fresh_db(tmp_path):
    """A fully-initialized DBManager against a temp sqlite file."""
    db_path = tmp_path / "phase_a.db"
    yield DBManager(str(db_path))


def _table_cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _index_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        r[1] for r in conn.execute(f"PRAGMA index_list({table})").fetchall()
    }


class TestLinksTable:
    def test_columns(self, fresh_db):
        assert _table_cols(fresh_db._conn, "links") == {
            "id", "url", "kind", "title", "created_at",
        }

    def test_kind_check_constraint(self, fresh_db):
        fresh_db._conn.execute(
            "INSERT INTO links (id, url, kind) VALUES ('l1', 'https://a.com', 'web')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db._conn.execute(
                "INSERT INTO links (id, url, kind) VALUES ('l2', 'https://b.com', 'bogus')"
            )

    def test_url_uniqueness(self, fresh_db):
        fresh_db._conn.execute(
            "INSERT INTO links (id, url, kind) VALUES ('l1', 'https://a.com', 'web')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db._conn.execute(
                "INSERT INTO links (id, url, kind) VALUES ('l2', 'https://a.com', 'web')"
            )

    def test_kind_index_exists(self, fresh_db):
        assert "idx_links_kind" in _index_names(fresh_db._conn, "links")


class TestLinkBindings:
    def test_columns(self, fresh_db):
        assert _table_cols(fresh_db._conn, "link_bindings") == {
            "id", "link_id", "scope", "scope_id", "contributed",
            "dwell_frames", "added_at",
        }

    def test_scope_check_constraint(self, fresh_db):
        fresh_db._conn.execute(
            "INSERT INTO links (id, url, kind) VALUES ('l1', 'x', 'web')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db._conn.execute(
                "INSERT INTO link_bindings (link_id, scope, scope_id) "
                "VALUES ('l1', 'universe', 'p1')"
            )

    def test_unique_triple_per_link_scope(self, fresh_db):
        fresh_db._conn.execute(
            "INSERT INTO links (id, url, kind) VALUES ('l1', 'x', 'web')"
        )
        fresh_db._conn.execute(
            "INSERT INTO link_bindings (link_id, scope, scope_id) "
            "VALUES ('l1', 'project', 'p1')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db._conn.execute(
                "INSERT INTO link_bindings (link_id, scope, scope_id) "
                "VALUES ('l1', 'project', 'p1')"
            )


class TestDailySummaries:
    def test_columns(self, fresh_db):
        assert _table_cols(fresh_db._conn, "daily_summaries") == {
            "id", "project_id", "deliverable_id", "date", "body_md",
            "status", "composed_at",
        }

    def test_status_check_constraint(self, fresh_db):
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db._conn.execute(
                "INSERT INTO daily_summaries (id, project_id, date, status) "
                "VALUES ('d1', 'p1', '2026-04-25', 'weird')"
            )

    def test_unique_per_project_deliverable_date(self, fresh_db):
        fresh_db._conn.execute(
            "INSERT INTO daily_summaries (id, project_id, deliverable_id, date) "
            "VALUES ('d1', 'p1', 'dv1', '2026-04-25')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db._conn.execute(
                "INSERT INTO daily_summaries (id, project_id, deliverable_id, date) "
                "VALUES ('d2', 'p1', 'dv1', '2026-04-25')"
            )


class TestDailyFeedback:
    def test_columns(self, fresh_db):
        assert _table_cols(fresh_db._conn, "daily_feedback") == {
            "id", "daily_summary_id", "feedback_text", "created_at",
            "applied", "applied_at",
        }

    def test_applied_defaults_to_zero(self, fresh_db):
        fresh_db._conn.execute(
            "INSERT INTO daily_summaries (id, project_id, date) "
            "VALUES ('d1', 'p1', '2026-04-25')"
        )
        fresh_db._conn.execute(
            "INSERT INTO daily_feedback (daily_summary_id, feedback_text) "
            "VALUES ('d1', 'fix link attribution')"
        )
        row = fresh_db._conn.execute(
            "SELECT applied FROM daily_feedback"
        ).fetchone()
        assert row[0] == 0


class TestDeliverableSections:
    def test_columns(self, fresh_db):
        assert _table_cols(fresh_db._conn, "deliverable_sections") == {
            "deliverable_id", "slot", "body_md", "updated_at", "source",
        }

    @pytest.mark.parametrize(
        "slot",
        ["overview", "progress", "decisions", "questions", "risks", "links"],
    )
    def test_all_six_slots_valid(self, fresh_db, slot):
        fresh_db._conn.execute(
            "INSERT INTO deliverable_sections (deliverable_id, slot) "
            "VALUES (?, ?)",
            (f"dv-{slot}", slot),
        )

    def test_invalid_slot_rejected(self, fresh_db):
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db._conn.execute(
                "INSERT INTO deliverable_sections (deliverable_id, slot) "
                "VALUES ('dv1', 'summary')"
            )

    def test_pk_prevents_double_slot(self, fresh_db):
        fresh_db._conn.execute(
            "INSERT INTO deliverable_sections (deliverable_id, slot, body_md) "
            "VALUES ('dv1', 'overview', 'first')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db._conn.execute(
                "INSERT INTO deliverable_sections (deliverable_id, slot, body_md) "
                "VALUES ('dv1', 'overview', 'dup')"
            )


class TestRejectionFingerprints:
    def test_columns(self, fresh_db):
        assert _table_cols(fresh_db._conn, "tree_rejection_fingerprints") == {
            "id", "centroid_embedding", "reason",
            "example_segment_ids", "created_at",
        }


class TestReviewProposalsAdditions:
    def test_tree_json_column_present(self, fresh_db):
        # review_proposals may not exist on a brand-new DB until some
        # other code path creates it. The migration only runs the
        # ensure_column if the table exists, so a fresh DB may legitimately
        # lack it. Create it to prove the column-add path works.
        fresh_db._conn.execute("""
            CREATE TABLE IF NOT EXISTS review_proposals (
                id TEXT PRIMARY KEY, status TEXT
            )
        """)
        # Re-run the ensure path:
        fresh_db._ensure_column("review_proposals", "tree_json", "TEXT DEFAULT ''")
        fresh_db._ensure_column(
            "review_proposals", "auto_attached_to_deliverable_id",
            "TEXT DEFAULT ''",
        )
        cols = _table_cols(fresh_db._conn, "review_proposals")
        assert "tree_json" in cols
        assert "auto_attached_to_deliverable_id" in cols


class TestIdempotence:
    def test_second_init_is_noop(self, tmp_path):
        db_path = tmp_path / "idem.db"
        DBManager(str(db_path))
        # Re-init against the same path — should not raise.
        db2 = DBManager(str(db_path))
        # All Phase A tables still present.
        for t in (
            "links", "link_bindings", "daily_summaries", "daily_feedback",
            "deliverable_sections", "tree_rejection_fingerprints",
        ):
            row = db2._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (t,),
            ).fetchone()
            assert row is not None, f"{t} missing after re-init"
