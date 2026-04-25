"""Tests for F5a wiki_tree_prose — renders a tree as wiki markdown
WITHOUT exposing SC / CTX / ANC scaffolding labels."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from consolidation.tree_builder import build_tree_candidate  # noqa: E402
from wiki_tree_prose import (  # noqa: E402
    render_proposal_as_prose,
    render_tree_as_prose,
)


# Substrings that must NEVER appear in rendered output — the scaffolding
# is an implementation detail, not a reader-facing concept.
_FORBIDDEN = [
    "Super-Context",
    "Super Context",
    "Super-context",
    "Context:",
    "CTX:",
    "Anchor:",
    "ANC:",
    "sc_title",
    "ctx_title",
]


def _seg(sid: str, window: str, title: str = "", body: str = "") -> dict:
    return {
        "id": sid, "window": window,
        "short_title": title, "summary": title or body,
        "detailed_summary": body,
        "duration_secs": 60,
    }


class TestLabelsAreHidden:
    def test_empty_tree_returns_empty(self):
        assert render_tree_as_prose(None) == ""
        assert render_tree_as_prose({}) == ""

    def test_missing_sc_title_returns_empty(self):
        tree = {"sc_title": "", "anchors": [{"title": "x", "body": "y"}]}
        assert render_tree_as_prose(tree) == ""

    @pytest.mark.parametrize("label", _FORBIDDEN)
    def test_no_scaffolding_labels_in_simple_tree(self, label):
        tree = {
            "sc_title": "Schema migration",
            "sc_summary": "Added Phase A tables",
            "ctx_title": "Internal scaffolding (hidden)",
            "ctx_summary": "should not appear",
            "anchors": [{"title": "Cursor work", "body": "wrote ALTER"}],
            "segment_ids": ["s1"],
            "duration_mins": 5.0,
        }
        out = render_tree_as_prose(tree)
        assert label not in out, f"unwanted label {label!r} in output"

    def test_ctx_title_not_rendered(self):
        """ctx_title is internal scaffolding and must stay invisible
        even though it's present in the dict."""
        tree = {
            "sc_title": "Work",
            "ctx_title": "THIS-STRING-MUST-NOT-APPEAR",
            "ctx_summary": "also-hidden",
            "anchors": [],
            "segment_ids": [],
            "duration_mins": 0,
        }
        out = render_tree_as_prose(tree)
        assert "THIS-STRING-MUST-NOT-APPEAR" not in out
        assert "also-hidden" not in out


class TestHeadingShape:
    def test_sc_title_is_h2(self):
        tree = {
            "sc_title": "Migration",
            "anchors": [],
            "segment_ids": [],
            "duration_mins": 0,
        }
        out = render_tree_as_prose(tree)
        assert out.startswith("## Migration")

    def test_anchors_are_h3(self):
        tree = {
            "sc_title": "Work",
            "anchors": [
                {"title": "Part A", "body": "did a"},
                {"title": "Part B", "body": "did b"},
            ],
            "segment_ids": [],
            "duration_mins": 0,
        }
        out = render_tree_as_prose(tree)
        assert "### Part A" in out
        assert "### Part B" in out
        assert out.count("### ") == 2

    def test_empty_body_renders_heading_only(self):
        tree = {
            "sc_title": "Work",
            "anchors": [{"title": "Just a heading", "body": ""}],
            "segment_ids": [],
            "duration_mins": 0,
        }
        out = render_tree_as_prose(tree)
        assert "### Just a heading" in out
        assert "\n\n\n" not in out  # no stray empty paragraph

    def test_anchor_with_no_title_and_no_body_skipped(self):
        tree = {
            "sc_title": "Work",
            "anchors": [
                {"title": "", "body": ""},
                {"title": "Real one", "body": "real"},
            ],
            "segment_ids": [],
            "duration_mins": 0,
        }
        out = render_tree_as_prose(tree)
        assert out.count("### ") == 1
        assert "Real one" in out


class TestStatsFooter:
    def test_footer_shows_segment_count_and_duration(self):
        tree = {
            "sc_title": "Work",
            "anchors": [{"title": "t", "body": "b", "window": "Cursor"}],
            "segment_ids": ["s1", "s2", "s3"],
            "duration_mins": 12.5,
        }
        out = render_tree_as_prose(tree)
        assert "3 segments" in out
        assert "12.5 minutes" in out
        assert "1 app" in out  # one window, singular

    def test_singular_segment(self):
        tree = {
            "sc_title": "Work",
            "anchors": [{"title": "t", "body": "b", "window": "Cursor"}],
            "segment_ids": ["s1"],
            "duration_mins": 1.0,
        }
        out = render_tree_as_prose(tree)
        assert "1 segment " in out or out.rstrip().endswith("1 segment") or "1 segment*" in out
        assert "1 segments" not in out  # no double-pluralization

    def test_no_footer_when_no_stats(self):
        tree = {
            "sc_title": "Work",
            "anchors": [{"title": "t", "body": "b"}],
            "segment_ids": [],
            "duration_mins": 0,
        }
        out = render_tree_as_prose(tree)
        assert "*" not in out  # no italic footer line


class TestBuilderRoundtrip:
    def test_tree_builder_output_renders_cleanly(self):
        """Take a real TreeCandidate, round-trip through JSON, render,
        verify no forbidden labels appear."""
        cluster = [
            _seg("s1", "Cursor", title="Wrote ALTER"),
            _seg("s2", "Cursor", title="Ran migration"),
            _seg("s3", "Slack", title="Discussed"),
            _seg("s4", "Cursor", title="Tested"),
            _seg("s5", "Chrome", title="Read docs"),
        ]
        tree = build_tree_candidate(cluster, sc_title="Phase A")
        parsed = json.loads(tree.to_json())
        out = render_tree_as_prose(parsed)
        assert out.startswith("## Phase A")
        for label in _FORBIDDEN:
            assert label not in out

    def test_flattened_tree_still_renders_cleanly(self):
        cluster = [_seg("s1", "Cursor"), _seg("s2", "Cursor")]
        tree = build_tree_candidate(cluster, sc_title="Tiny")
        parsed = json.loads(tree.to_json())
        assert parsed["ctx_title"] is None
        out = render_tree_as_prose(parsed)
        assert out.startswith("## Tiny")
        # No stray "None" appearing from the None ctx.
        assert "None" not in out


class TestRenderProposalAsProse:
    @pytest.fixture
    def db_with_proposals(self, tmp_path):
        db_path = tmp_path / "pm.db"
        c = sqlite3.connect(db_path)
        c.executescript("""
            CREATE TABLE review_proposals (
                id TEXT PRIMARY KEY,
                status TEXT,
                tree_json TEXT DEFAULT '',
                user_assigned_project_id TEXT DEFAULT '',
                user_assigned_deliverable_id TEXT DEFAULT '',
                auto_attached_to_deliverable_id TEXT DEFAULT '',
                anchor_node_id TEXT DEFAULT '',
                confirmed_at TEXT
            );
        """)
        c.commit()
        c.close()
        db = MagicMock()
        db.db_path = str(db_path)
        yield db, db_path

    def _insert(self, db_path, **kw):
        c = sqlite3.connect(db_path)
        c.execute("""
            INSERT INTO review_proposals
            (id, status, tree_json, user_assigned_project_id,
             user_assigned_deliverable_id,
             auto_attached_to_deliverable_id, anchor_node_id,
             confirmed_at)
            VALUES (:id, :status, :tree_json, :pid, :did, :adid,
                    :anchor_id, :confirmed_at)
        """, {
            "status": "draft", "tree_json": "", "pid": "",
            "did": "", "adid": "", "anchor_id": "", "confirmed_at": "",
            **kw,
        })
        c.commit()
        c.close()

    def test_unknown_id_returns_none(self, db_with_proposals):
        db, _ = db_with_proposals
        assert render_proposal_as_prose(db, "rp-missing") is None

    def test_empty_tree_json_returns_none(self, db_with_proposals):
        db, db_path = db_with_proposals
        self._insert(db_path, id="rp-empty", tree_json="")
        assert render_proposal_as_prose(db, "rp-empty") is None

    def test_malformed_tree_json_returns_none(self, db_with_proposals):
        db, db_path = db_with_proposals
        self._insert(db_path, id="rp-bad", tree_json="{not json")
        assert render_proposal_as_prose(db, "rp-bad") is None

    def test_happy_path_returns_prose_and_metadata(self, db_with_proposals):
        db, db_path = db_with_proposals
        tree = {
            "sc_title": "Shipping F5",
            "sc_summary": "Render tree as wiki prose",
            "anchors": [{"title": "Done", "body": "renderer + tests"}],
            "segment_ids": ["s1", "s2"],
            "duration_mins": 5.0,
        }
        self._insert(
            db_path, id="rp-good", status="confirmed",
            tree_json=json.dumps(tree), pid="P-1", did="D-2",
            anchor_id="anc-9",
            confirmed_at="2026-04-25 10:00:00",
        )
        out = render_proposal_as_prose(db, "rp-good")
        assert out is not None
        assert out["id"] == "rp-good"
        assert out["status"] == "confirmed"
        assert out["project_id"] == "P-1"
        assert out["deliverable_id"] == "D-2"
        assert out["anchor_node_id"] == "anc-9"
        assert out["prose_md"].startswith("## Shipping F5")
        assert "### Done" in out["prose_md"]

    def test_auto_attached_deliverable_fallback(self, db_with_proposals):
        """When user_assigned_deliverable_id is empty but auto_attached
        is set, the renderer returns the auto_attached one."""
        db, db_path = db_with_proposals
        tree = {
            "sc_title": "Auto", "anchors": [],
            "segment_ids": [], "duration_mins": 0,
        }
        self._insert(
            db_path, id="rp-aa", status="auto_attached",
            tree_json=json.dumps(tree),
            pid="P-1", did="",  # no user deliverable
            adid="D-auto",
        )
        out = render_proposal_as_prose(db, "rp-aa")
        assert out["deliverable_id"] == "D-auto"
