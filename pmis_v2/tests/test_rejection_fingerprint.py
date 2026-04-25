"""Tests for F4 — rejection-fingerprint learning.

Two surfaces:
  reject()                       — writes a fingerprint on rejection
                                   when the proposal's tree_json carries
                                   a centroid.
  _matches_rejection_fingerprint — pre-consolidation filter that drops
                                   clusters resembling prior rejections.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def rp(tmp_path):
    """ReviewProposals wired to a real sqlite DB carrying only the
    tables F4 touches."""
    from review.proposals import ReviewProposals
    db_path = tmp_path / "pm.db"
    c = sqlite3.connect(db_path)
    c.executescript("""
        CREATE TABLE review_proposals (
            id TEXT PRIMARY KEY,
            target_date TEXT DEFAULT '',
            author TEXT DEFAULT '',
            status TEXT,
            proposed_content TEXT DEFAULT '',
            segment_ids_json TEXT DEFAULT '[]',
            project_probs_json TEXT DEFAULT '[]',
            tree_json TEXT DEFAULT '',
            user_assigned_project_id TEXT DEFAULT '',
            user_assigned_deliverable_id TEXT DEFAULT '',
            auto_attached_to_deliverable_id TEXT DEFAULT '',
            anchor_node_id TEXT DEFAULT '',
            confirmed_at TEXT,
            processed_by_nightly_at TEXT
        );
        CREATE TABLE tree_rejection_fingerprints (
            id TEXT PRIMARY KEY,
            centroid_embedding BLOB,
            reason TEXT DEFAULT '',
            example_segment_ids TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    c.commit()
    c.close()

    db = MagicMock()
    db.db_path = str(db_path)
    yield ReviewProposals(db, {}), db_path


def _insert_draft(db_path, pid, tree_json="", segs=None):
    c = sqlite3.connect(db_path)
    c.execute(
        "INSERT INTO review_proposals "
        "(id, status, tree_json, segment_ids_json) VALUES (?, 'draft', ?, ?)",
        (pid, tree_json, json.dumps(segs or [])),
    )
    c.commit()
    c.close()


class TestRejectWritesFingerprint:
    def test_centroid_present_writes_row(self, rp):
        rev, db_path = rp
        centroid = [0.1] * 32
        tree_json = json.dumps({"centroid_embedding": centroid})
        _insert_draft(db_path, "rp-1", tree_json, ["s1", "s2"])

        out = rev.reject("rp-1", reason="noise")
        assert out["ok"] is True
        assert out["fingerprinted"] is True

        c = sqlite3.connect(db_path)
        row = c.execute(
            "SELECT reason, example_segment_ids, centroid_embedding "
            "FROM tree_rejection_fingerprints"
        ).fetchone()
        c.close()
        assert row is not None
        assert row[0] == "noise"
        assert json.loads(row[1]) == ["s1", "s2"]
        stored = np.frombuffer(row[2], dtype=np.float32)
        assert stored.shape == (32,)
        assert np.allclose(stored, np.asarray(centroid, dtype=np.float32))

    def test_missing_centroid_skips_write(self, rp):
        rev, db_path = rp
        _insert_draft(db_path, "rp-2", json.dumps({}), ["s1"])
        out = rev.reject("rp-2")
        assert out["ok"] is True
        assert out["fingerprinted"] is False
        c = sqlite3.connect(db_path)
        count = c.execute(
            "SELECT COUNT(*) FROM tree_rejection_fingerprints"
        ).fetchone()[0]
        c.close()
        assert count == 0

    def test_null_centroid_skips_write(self, rp):
        rev, db_path = rp
        tree_json = json.dumps({"centroid_embedding": None})
        _insert_draft(db_path, "rp-3", tree_json, ["s1"])
        out = rev.reject("rp-3")
        assert out["fingerprinted"] is False

    def test_empty_tree_json_skips_write(self, rp):
        rev, db_path = rp
        _insert_draft(db_path, "rp-4", "", ["s1"])
        out = rev.reject("rp-4")
        assert out["fingerprinted"] is False

    def test_reason_truncated_at_500_chars(self, rp):
        rev, db_path = rp
        tree_json = json.dumps({"centroid_embedding": [0.2] * 16})
        _insert_draft(db_path, "rp-5", tree_json)
        long_reason = "x" * 1000
        rev.reject("rp-5", reason=long_reason)
        c = sqlite3.connect(db_path)
        stored_reason = c.execute(
            "SELECT reason FROM tree_rejection_fingerprints"
        ).fetchone()[0]
        c.close()
        assert len(stored_reason) == 500

    def test_example_segment_ids_capped_at_10(self, rp):
        rev, db_path = rp
        tree_json = json.dumps({"centroid_embedding": [0.3] * 8})
        segs = [f"s{i}" for i in range(25)]
        _insert_draft(db_path, "rp-6", tree_json, segs)
        rev.reject("rp-6")
        c = sqlite3.connect(db_path)
        stored = json.loads(c.execute(
            "SELECT example_segment_ids FROM tree_rejection_fingerprints"
        ).fetchone()[0])
        c.close()
        assert len(stored) == 10
        assert stored == segs[:10]

    def test_status_flipped_even_when_fingerprint_skipped(self, rp):
        rev, db_path = rp
        _insert_draft(db_path, "rp-7", "", ["s1"])
        rev.reject("rp-7")
        c = sqlite3.connect(db_path)
        status = c.execute(
            "SELECT status FROM review_proposals WHERE id='rp-7'"
        ).fetchone()[0]
        c.close()
        assert status == "rejected"

    def test_non_draft_cannot_be_rejected(self, rp):
        rev, db_path = rp
        c = sqlite3.connect(db_path)
        c.execute(
            "INSERT INTO review_proposals (id, status) "
            "VALUES ('rp-done', 'confirmed')"
        )
        c.commit()
        c.close()
        out = rev.reject("rp-done")
        assert out["ok"] is False
        assert "not_draft" in out.get("error", "")

    def test_not_found(self, rp):
        rev, _ = rp
        out = rev.reject("rp-missing")
        assert out["ok"] is False
        assert out["error"] == "not_found"


class TestMatchesRejectionFingerprint:
    def _seed_fp(self, db_path, centroid):
        c = sqlite3.connect(db_path)
        arr = np.asarray(centroid, dtype=np.float32).tobytes()
        c.execute(
            "INSERT INTO tree_rejection_fingerprints "
            "(id, centroid_embedding) VALUES ('trf-1', ?)",
            (arr,),
        )
        c.commit()
        c.close()

    def test_empty_store_is_no_match(self, rp):
        rev, _ = rp
        tree = SimpleNamespace(centroid_embedding=[0.1] * 16)
        assert rev._matches_rejection_fingerprint(tree) is False

    def test_tree_without_centroid_bypasses_filter(self, rp):
        rev, db_path = rp
        self._seed_fp(db_path, [0.1] * 16)
        tree = SimpleNamespace(centroid_embedding=None)
        assert rev._matches_rejection_fingerprint(tree) is False

    def test_near_match_triggers(self, rp):
        rev, db_path = rp
        centroid = [0.5] * 16
        self._seed_fp(db_path, centroid)
        # Exact same vector → cosine distance = 0 → matches.
        tree = SimpleNamespace(centroid_embedding=centroid)
        assert rev._matches_rejection_fingerprint(tree) is True

    def test_far_match_does_not_trigger(self, rp):
        rev, db_path = rp
        # Orthogonal vectors have cosine similarity ~0 → distance ~1
        # which exceeds the 0.30 threshold.
        self._seed_fp(db_path, [1.0] + [0.0] * 15)
        tree = SimpleNamespace(centroid_embedding=[0.0] * 15 + [1.0])
        assert rev._matches_rejection_fingerprint(tree) is False

    def test_shape_mismatch_is_skipped(self, rp):
        rev, db_path = rp
        self._seed_fp(db_path, [0.5] * 16)
        # Even an "identical" value on a different dim is ignored.
        tree = SimpleNamespace(centroid_embedding=[0.5] * 32)
        assert rev._matches_rejection_fingerprint(tree) is False

    def test_zero_vector_query_no_match(self, rp):
        rev, db_path = rp
        self._seed_fp(db_path, [0.5] * 16)
        tree = SimpleNamespace(centroid_embedding=[0.0] * 16)
        assert rev._matches_rejection_fingerprint(tree) is False

    def test_custom_threshold_respected(self, rp):
        rev, db_path = rp
        # Two vectors with cosine distance ≈ 0.134 (30-deg apart).
        import math
        a = [1.0] + [0.0] * 15
        b = [math.cos(math.radians(30)), math.sin(math.radians(30))] + [0.0] * 14
        self._seed_fp(db_path, a)

        # Default threshold 0.30 — 0.134 is inside → match.
        tree = SimpleNamespace(centroid_embedding=b)
        assert rev._matches_rejection_fingerprint(tree) is True

        # Tighten threshold to 0.10 — 0.134 is outside → no match.
        rev.hp = {"rejection_fingerprint_match_threshold": 0.10}
        assert rev._matches_rejection_fingerprint(tree) is False

    def test_mixed_store_any_hit_wins(self, rp):
        """A far-away fingerprint and a near one — the near one should
        still produce a match."""
        rev, db_path = rp
        c = sqlite3.connect(db_path)
        far = np.asarray([1.0] + [0.0] * 15, dtype=np.float32).tobytes()
        near = np.asarray([0.5] * 16, dtype=np.float32).tobytes()
        c.executemany(
            "INSERT INTO tree_rejection_fingerprints "
            "(id, centroid_embedding) VALUES (?, ?)",
            [("trf-far", far), ("trf-near", near)],
        )
        c.commit()
        c.close()
        tree = SimpleNamespace(centroid_embedding=[0.5] * 16)
        assert rev._matches_rejection_fingerprint(tree) is True
