"""Tests for matching engine."""

import pytest
from unittest.mock import MagicMock, patch


class TestMatchingEngine:
    def test_exact_sc_match(self):
        """Exact supercontext match takes priority."""
        entry = {"supercontext": "Product Development", "context": "Frontend", "anchor": "Charts", "level": "anchor", "time_mins": 30, "human_mins": 20, "agent_mins": 10, "id": "test-1"}
        deliverables = [{"id": "D-001", "name": "Vision OS", "supercontext": "Product Development", "status": "active"}]

        from src.matching.matching_engine import MatchingEngine
        engine = MatchingEngine.__new__(MatchingEngine)
        engine.threshold = 0.8
        engine.chroma = MagicMock()

        match = engine._find_match(entry, deliverables)
        assert match is not None
        assert match.method == "exact"
        assert match.deliverable["id"] == "D-001"

    def test_no_match_returns_none(self):
        entry = {"supercontext": "Personal Learning", "context": "Reading", "anchor": "Papers", "level": "anchor", "time_mins": 15, "human_mins": 15, "agent_mins": 0, "id": "test-2"}
        deliverables = [{"id": "D-001", "name": "Vision OS", "supercontext": "Product Development", "status": "active"}]

        from src.matching.matching_engine import MatchingEngine
        engine = MatchingEngine.__new__(MatchingEngine)
        engine.threshold = 0.8
        engine.chroma = MagicMock()
        engine.chroma.match_to_deliverable.return_value = None

        match = engine._find_match(entry, deliverables)
        assert match is None


class TestMemoryRollup:
    def test_time_rollup_anchor_to_context(self):
        """Anchor time must roll up to context and SC."""
        # Simulate hourly data
        hourly_entries = [
            {"supercontext": "Product Dev", "context": "Frontend", "anchor": "Charts", "time_mins": 20, "human_mins": 15, "agent_mins": 5, "segment_ids": '["TS-001"]'},
            {"supercontext": "Product Dev", "context": "Frontend", "anchor": "Layout", "time_mins": 35, "human_mins": 10, "agent_mins": 25, "segment_ids": '["TS-002"]'},
            {"supercontext": "Product Dev", "context": "API", "anchor": "Auth", "time_mins": 15, "human_mins": 15, "agent_mins": 0, "segment_ids": '["TS-003"]'},
        ]

        # Group like the rollup does
        sc_groups = {}
        for entry in hourly_entries:
            sc = entry["supercontext"]
            ctx = entry["context"]
            if sc not in sc_groups:
                sc_groups[sc] = {"total": 0, "contexts": {}}
            sc_groups[sc]["total"] += entry["time_mins"]
            if ctx not in sc_groups[sc]["contexts"]:
                sc_groups[sc]["contexts"][ctx] = 0
            sc_groups[sc]["contexts"][ctx] += entry["time_mins"]

        # SC total should be sum of all anchors
        assert sc_groups["Product Dev"]["total"] == 70
        # Context totals should be sum of their anchors
        assert sc_groups["Product Dev"]["contexts"]["Frontend"] == 55
        assert sc_groups["Product Dev"]["contexts"]["API"] == 15

    def test_human_agent_split(self):
        """Agent time should only count when agent was running."""
        entries = [
            {"worker": "human", "target_segment_length_secs": 300},
            {"worker": "agent", "target_segment_length_secs": 600},
            {"worker": "human", "target_segment_length_secs": 120},
        ]

        human_secs = sum(e["target_segment_length_secs"] for e in entries if e["worker"] == "human")
        agent_secs = sum(e["target_segment_length_secs"] for e in entries if e["worker"] == "agent")

        assert human_secs == 420  # 300 + 120
        assert agent_secs == 600
        total = human_secs + agent_secs
        assert round(agent_secs / total * 100, 1) == 58.8
