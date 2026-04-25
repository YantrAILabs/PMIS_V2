"""Tests for F2a tree_builder — deterministic tree shape + JSON roundtrip.

The centroid path is tested separately on the "graceful-None" case so
these tests don't depend on a running embedder (Ollama, OpenAI, etc.).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from consolidation.tree_builder import (  # noqa: E402
    FLATTEN_THRESHOLD,
    Anchor,
    TreeCandidate,
    build_tree_candidate,
)


def _seg(
    sid: str, window: str, summary: str = "",
    short_title: str = "", duration_secs: int = 60,
) -> dict:
    return {
        "id": sid,
        "window": window,
        "summary": summary or short_title or f"activity {sid}",
        "short_title": short_title,
        "detailed_summary": "",
        "duration_secs": duration_secs,
    }


@pytest.fixture(autouse=True)
def _no_embedder():
    """Every test in this module runs with centroid disabled so results
    are deterministic and don't depend on a local embedder."""
    with patch(
        "consolidation.tree_builder._compute_centroid",
        return_value=None,
    ):
        yield


class TestShape:
    def test_raises_on_empty_cluster(self):
        with pytest.raises(ValueError):
            build_tree_candidate([], sc_title="x")

    def test_tiny_cluster_skips_ctx(self):
        cluster = [_seg("s1", "Cursor"), _seg("s2", "Cursor")]
        tree = build_tree_candidate(cluster, sc_title="Coding")
        assert tree.ctx_title is None
        assert tree.ctx_summary == ""
        assert len(tree.anchors) == 1  # one window → one anchor
        assert tree.anchors[0].window == "Cursor"

    def test_large_cluster_keeps_ctx(self):
        cluster = [
            _seg("s1", "Cursor"),
            _seg("s2", "Cursor"),
            _seg("s3", "Slack"),
            _seg("s4", "Slack"),
            _seg("s5", "Chrome"),
        ]
        tree = build_tree_candidate(cluster, sc_title="Mixed work")
        assert tree.ctx_title is not None
        assert "3" in tree.ctx_title  # three contexts → "Across 3 contexts"
        assert "5 segments" in tree.ctx_summary

    def test_two_windows_uses_slash_join_for_ctx(self):
        cluster = [_seg(f"s{i}", "Cursor") for i in range(3)]
        cluster += [_seg(f"t{i}", "Chrome") for i in range(2)]
        tree = build_tree_candidate(cluster, sc_title="Work")
        assert tree.ctx_title == "Chrome / Cursor"

    def test_missing_window_becomes_unknown(self):
        cluster = [_seg("s1", ""), _seg("s2", "")]
        tree = build_tree_candidate(cluster, sc_title="Unknown")
        assert len(tree.anchors) == 1
        assert tree.anchors[0].window == "Unknown"

    def test_anchors_grouped_by_window(self):
        cluster = [
            _seg("s1", "Cursor"),
            _seg("s2", "Cursor"),
            _seg("s3", "Chrome"),
            _seg("s4", "Slack"),
            _seg("s5", "Slack"),
        ]
        tree = build_tree_candidate(cluster, sc_title="X")
        windows = {a.window for a in tree.anchors}
        assert windows == {"Cursor", "Chrome", "Slack"}

    def test_single_segment_anchor_title_uses_summary(self):
        cluster = [
            _seg("s1", "Cursor", short_title="Refactored auth"),
            _seg("s2", "Slack"),
            _seg("s3", "Cursor"),
            _seg("s4", "Cursor"),
            _seg("s5", "Cursor"),
        ]
        tree = build_tree_candidate(cluster, sc_title="X")
        slack = next(a for a in tree.anchors if a.window == "Slack")
        assert slack.title != ""
        # single-segment anchor gets its short_title-derived title, not
        # the "N segments in W" form.
        assert "segments in" not in slack.title.lower()


class TestSegmentPropagation:
    def test_segment_ids_propagate_to_top_level_and_anchors(self):
        cluster = [_seg("a", "Cursor"), _seg("b", "Cursor"), _seg("c", "Chrome")]
        tree = build_tree_candidate(cluster, sc_title="X")
        assert set(tree.segment_ids) == {"a", "b", "c"}
        all_anchor_ids = [sid for a in tree.anchors for sid in a.segment_ids]
        assert set(all_anchor_ids) == {"a", "b", "c"}


class TestDurationMath:
    def test_total_duration_sums_correctly(self):
        cluster = [
            _seg("s1", "C", duration_secs=120),
            _seg("s2", "C", duration_secs=60),
        ]
        tree = build_tree_candidate(cluster, sc_title="X")
        assert tree.duration_mins == 3.0

    def test_anchor_duration_per_window(self):
        cluster = [
            _seg("s1", "Cursor", duration_secs=60),
            _seg("s2", "Cursor", duration_secs=60),
            _seg("s3", "Slack", duration_secs=30),
        ]
        tree = build_tree_candidate(cluster, sc_title="X")
        cursor = next(a for a in tree.anchors if a.window == "Cursor")
        slack = next(a for a in tree.anchors if a.window == "Slack")
        assert cursor.duration_mins == 2.0
        assert slack.duration_mins == 0.5


class TestJsonRoundtrip:
    def test_tree_serializes_and_restores_shape(self):
        cluster = [_seg(f"s{i}", "Cursor") for i in range(5)]
        cluster[0]["window"] = "Chrome"
        tree = build_tree_candidate(cluster, sc_title="Work")
        payload = json.loads(tree.to_json())
        assert payload["sc_title"] == "Work"
        assert payload["ctx_title"] is not None  # 5 segments > FLATTEN_THRESHOLD
        assert payload["centroid_embedding"] is None
        assert len(payload["anchors"]) == 2
        assert {a["window"] for a in payload["anchors"]} == {"Cursor", "Chrome"}

    def test_missing_centroid_becomes_json_null(self):
        cluster = [_seg("s1", "Cursor"), _seg("s2", "Cursor")]
        tree = build_tree_candidate(cluster, sc_title="X")
        payload = json.loads(tree.to_json())
        assert payload["centroid_embedding"] is None


class TestCentroidFallback:
    def test_centroid_is_none_when_no_summaries(self):
        """When no segment has any text, the embedder has nothing to call
        on — centroid should come back None rather than raising."""
        # Bypass the autouse patch to test the real _compute_centroid path.
        from consolidation.tree_builder import _compute_centroid
        segs = [{"id": "s1", "summary": "", "short_title": "",
                 "detailed_summary": "", "window": "X"}]
        assert _compute_centroid(segs, {}) is None


class TestFlattenThresholdConstant:
    """Protect the external contract: clusters of exactly FLATTEN_THRESHOLD
    stay flat; FLATTEN_THRESHOLD + 1 gains a CTX. Prevents off-by-one drift
    if somebody 'simplifies' the condition later."""

    def test_exactly_at_threshold_flattens(self):
        cluster = [_seg(f"s{i}", "Cursor") for i in range(FLATTEN_THRESHOLD)]
        tree = build_tree_candidate(cluster, sc_title="X")
        assert tree.ctx_title is None

    def test_one_past_threshold_keeps_ctx(self):
        cluster = [_seg(f"s{i}", "Cursor") for i in range(FLATTEN_THRESHOLD + 1)]
        tree = build_tree_candidate(cluster, sc_title="X")
        assert tree.ctx_title is not None
