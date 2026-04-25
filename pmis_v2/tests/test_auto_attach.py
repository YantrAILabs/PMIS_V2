"""Tests for F2b auto-attach: confidence gate, deliverable picker,
and the soft-review listing.

Full end-to-end auto_attach (Embedder + memory_node creation) is
exercised via manual smoke test — these unit tests cover the logic
that actually has branching decisions."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def _proposals_with_hp(hp: dict):
    """A ReviewProposals instance wired to a MagicMock db — enough for
    the pure-Python helpers under test."""
    from review.proposals import ReviewProposals
    db = MagicMock()
    db.db_path = ":memory:"
    return ReviewProposals(db, hp)


class TestShouldAutoAttach:
    def test_empty_probs_is_false(self):
        rp = _proposals_with_hp({})
        assert rp._should_auto_attach([]) is False

    def test_top_score_below_threshold_is_false(self):
        rp = _proposals_with_hp({})
        probs = [{"project_id": "P1", "score": 0.65}]
        assert rp._should_auto_attach(probs) is False

    def test_top_score_at_threshold_is_true(self):
        rp = _proposals_with_hp({})
        probs = [{"project_id": "P1", "score": 0.70}]
        assert rp._should_auto_attach(probs) is True

    def test_top_score_above_threshold_is_true(self):
        rp = _proposals_with_hp({})
        probs = [{"project_id": "P1", "score": 0.95}]
        assert rp._should_auto_attach(probs) is True

    def test_custom_threshold_respected(self):
        rp = _proposals_with_hp({"auto_attach_confidence_threshold": 0.9})
        probs = [{"project_id": "P1", "score": 0.85}]
        assert rp._should_auto_attach(probs) is False
        probs = [{"project_id": "P1", "score": 0.92}]
        assert rp._should_auto_attach(probs) is True

    def test_only_top_prob_considered(self):
        """Secondary probs shouldn't trigger auto-attach even if they
        somehow exceed the threshold."""
        rp = _proposals_with_hp({})
        probs = [
            {"project_id": "P1", "score": 0.4},
            {"project_id": "P2", "score": 0.95},  # ignored — not top
        ]
        assert rp._should_auto_attach(probs) is False


class TestPickDeliverable:
    @pytest.fixture
    def goals_yaml(self, tmp_path, monkeypatch):
        """Stage a goals.yaml at the path the method reads from."""
        # The method reads a hard-coded path:
        # ~/Desktop/memory/productivity-tracker/config/goals.yaml
        # We redirect Path.home() so the file lands where it expects.
        home = tmp_path / "home"
        cfg_dir = home / "Desktop" / "memory" / "productivity-tracker" / "config"
        cfg_dir.mkdir(parents=True)
        goals_file = cfg_dir / "goals.yaml"
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        return goals_file

    def _write(self, path, data):
        path.write_text(yaml.safe_dump(data))

    def test_no_goals_file_returns_empty(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / "Desktop" / "memory" / "productivity-tracker" / "config").mkdir(parents=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        rp = _proposals_with_hp({})
        assert rp._pick_deliverable_for_project("P1", "anything") == ""

    def test_unknown_project_returns_empty(self, goals_yaml):
        self._write(goals_yaml, {
            "goals": [{"id": "G1", "projects": [{"id": "P-known"}]}],
        })
        rp = _proposals_with_hp({})
        assert rp._pick_deliverable_for_project("P-unknown", "txt") == ""

    def test_project_without_deliverable_patterns_returns_empty(self, goals_yaml):
        self._write(goals_yaml, {
            "goals": [{"id": "G1", "projects": [{"id": "P1"}]}],
        })
        rp = _proposals_with_hp({})
        assert rp._pick_deliverable_for_project("P1", "txt") == ""

    def test_single_pattern_match_returns_its_deliverable(self, goals_yaml):
        self._write(goals_yaml, {
            "goals": [{
                "id": "G1", "projects": [{
                    "id": "P1",
                    "deliverable_patterns": {
                        "D-alpha": ["alpha-keyword"],
                        "D-beta": ["beta-keyword"],
                    },
                }],
            }],
        })
        rp = _proposals_with_hp({})
        assert rp._pick_deliverable_for_project(
            "P1", "working on alpha-keyword features",
        ) == "D-alpha"

    def test_higher_hit_count_wins(self, goals_yaml):
        self._write(goals_yaml, {
            "goals": [{
                "id": "G1", "projects": [{
                    "id": "P1",
                    "deliverable_patterns": {
                        "D-alpha": ["alpha"],
                        "D-beta": ["beta", "beta2"],
                    },
                }],
            }],
        })
        rp = _proposals_with_hp({})
        # text has 1 alpha hit, 2 beta hits → D-beta wins
        text = "alpha segment followed by beta work and beta2 cleanup"
        assert rp._pick_deliverable_for_project("P1", text) == "D-beta"

    def test_no_hits_returns_empty_even_with_patterns(self, goals_yaml):
        self._write(goals_yaml, {
            "goals": [{
                "id": "P1",
                "projects": [{
                    "id": "P1",
                    "deliverable_patterns": {"D1": ["cursor"]},
                }],
            }],
        })
        rp = _proposals_with_hp({})
        assert rp._pick_deliverable_for_project("P1", "nothing relevant") == ""

    def test_case_insensitive_matching(self, goals_yaml):
        self._write(goals_yaml, {
            "goals": [{
                "id": "G1", "projects": [{
                    "id": "P1",
                    "deliverable_patterns": {"D-alpha": ["PMIS"]},
                }],
            }],
        })
        rp = _proposals_with_hp({})
        assert rp._pick_deliverable_for_project("P1", "pmis work") == "D-alpha"


class TestListAutoAttachedPendingReview:
    @pytest.fixture
    def rp_with_proposals(self, tmp_path):
        """ReviewProposals wired to a real sqlite DB with just the
        review_proposals table seeded for this test."""
        from review.proposals import ReviewProposals
        db_path = tmp_path / "pm.db"
        c = sqlite3.connect(db_path)
        c.executescript("""
            CREATE TABLE review_proposals (
                id TEXT PRIMARY KEY,
                target_date TEXT,
                author TEXT,
                status TEXT,
                proposed_content TEXT,
                segment_ids_json TEXT,
                project_probs_json TEXT,
                tree_json TEXT,
                user_assigned_project_id TEXT,
                user_assigned_deliverable_id TEXT,
                auto_attached_to_deliverable_id TEXT,
                anchor_node_id TEXT,
                confirmed_at TEXT,
                processed_by_nightly_at TEXT
            );
        """)
        c.commit()
        c.close()

        db = MagicMock()
        db.db_path = str(db_path)
        yield ReviewProposals(db, {}), db_path

    def _insert(self, db_path, **kwargs):
        defaults = {
            "status": "auto_attached",
            "proposed_content": "",
            "segment_ids_json": "[]",
            "project_probs_json": "[]",
            "tree_json": "",
            "user_assigned_project_id": "",
            "auto_attached_to_deliverable_id": "",
            "anchor_node_id": "",
        }
        defaults.update(kwargs)
        c = sqlite3.connect(db_path)
        c.execute(f"""
            INSERT INTO review_proposals
            (id, target_date, author, status, proposed_content,
             segment_ids_json, project_probs_json, tree_json,
             user_assigned_project_id, auto_attached_to_deliverable_id,
             anchor_node_id, confirmed_at)
            VALUES (:id, '', 'auto', :status, :proposed_content,
                    :segment_ids_json, :project_probs_json, :tree_json,
                    :user_assigned_project_id,
                    :auto_attached_to_deliverable_id,
                    :anchor_node_id, :confirmed_at)
        """, defaults)
        c.commit()
        c.close()

    def test_within_window_included(self, rp_with_proposals):
        rp, db_path = rp_with_proposals
        self._insert(
            db_path, id="rp-recent",
            confirmed_at="2099-01-01 00:00:00",  # far future → always in window
        )
        out = rp.list_auto_attached_pending_review(days=14)
        assert len(out) == 1
        assert out[0]["id"] == "rp-recent"
        assert out[0]["status"] == "auto_attached"

    def test_outside_window_excluded(self, rp_with_proposals):
        rp, db_path = rp_with_proposals
        self._insert(
            db_path, id="rp-old",
            confirmed_at="1990-01-01 00:00:00",  # far past → outside window
        )
        out = rp.list_auto_attached_pending_review(days=14)
        assert out == []

    def test_wrong_status_excluded(self, rp_with_proposals):
        rp, db_path = rp_with_proposals
        self._insert(
            db_path, id="rp-draft", status="draft",
            confirmed_at="2099-01-01 00:00:00",
        )
        self._insert(
            db_path, id="rp-confirmed", status="confirmed",
            confirmed_at="2099-01-01 00:00:00",
        )
        out = rp.list_auto_attached_pending_review(days=14)
        assert out == []

    def test_null_confirmed_at_excluded(self, rp_with_proposals):
        rp, db_path = rp_with_proposals
        self._insert(
            db_path, id="rp-no-ts",
            confirmed_at=None,
        )
        out = rp.list_auto_attached_pending_review(days=14)
        assert out == []

    def test_ordered_confirmed_at_desc(self, rp_with_proposals):
        rp, db_path = rp_with_proposals
        self._insert(db_path, id="rp-older", confirmed_at="2099-01-01 00:00:00")
        self._insert(db_path, id="rp-newer", confirmed_at="2099-06-01 00:00:00")
        out = rp.list_auto_attached_pending_review(days=365 * 365)
        assert [r["id"] for r in out] == ["rp-newer", "rp-older"]

    def test_payload_fields_present(self, rp_with_proposals):
        rp, db_path = rp_with_proposals
        self._insert(
            db_path, id="rp-1",
            user_assigned_project_id="P1",
            auto_attached_to_deliverable_id="D-alpha",
            anchor_node_id="anc-123",
            segment_ids_json=json.dumps(["s1", "s2"]),
            proposed_content="sc title",
            project_probs_json=json.dumps([{"project_id": "P1", "score": 0.9}]),
            confirmed_at="2099-01-01 00:00:00",
        )
        out = rp.list_auto_attached_pending_review(days=14)
        assert out[0]["project_id"] == "P1"
        assert out[0]["deliverable_id"] == "D-alpha"
        assert out[0]["anchor_node_id"] == "anc-123"
        assert out[0]["segment_count"] == 2
