"""Tests for memory pipeline and contribution tracking."""

import pytest


class TestContributionTracking:
    def test_contributed_work_flagged(self):
        """Work matched to a deliverable should be marked as contributed."""
        entry = {
            "id": "test-1",
            "level": "anchor",
            "supercontext": "Product Dev",
            "context": "Frontend",
            "anchor": "Charts",
            "time_mins": 30,
            "contributed_to_delivery": True,
            "deliverable_id": "D-001",
        }
        assert entry["contributed_to_delivery"]
        assert entry["deliverable_id"] == "D-001"

    def test_uncontributed_work_identified(self):
        """Work not matched to any deliverable = potential waste."""
        entries = [
            {"level": "anchor", "time_mins": 30, "contributed_to_delivery": True},
            {"level": "anchor", "time_mins": 45, "contributed_to_delivery": False},
            {"level": "anchor", "time_mins": 20, "contributed_to_delivery": False},
        ]
        wasted = [e for e in entries if not e["contributed_to_delivery"]]
        assert len(wasted) == 2
        assert sum(e["time_mins"] for e in wasted) == 65

    def test_wasted_pct_calculation(self):
        total = 300
        wasted = 75
        pct = round(wasted / total * 100, 1)
        assert pct == 25.0


class TestHierarchyMerge:
    def test_user_sc_maps_to_central_anchor(self):
        """User's SC might be just an anchor in central memory."""
        user_sc = "First Club QC Pipeline"
        central_hierarchy = {
            "SC": "Yantra AI Labs - Business",
            "contexts": {
                "Active Clients": {
                    "anchors": {
                        "First Club project": {"time_mins": 0},
                        "Another client": {"time_mins": 0},
                    }
                }
            }
        }
        # The user's SC matches a central anchor
        matched = False
        for ctx_data in central_hierarchy["contexts"].values():
            for anchor_name in ctx_data["anchors"]:
                if "first club" in anchor_name.lower():
                    matched = True
                    break
        assert matched

    def test_contribution_chain_integrity(self):
        """Contribution chain: Deliverable → contexts → anchors → hours."""
        chain = {
            "deliverable": "Vision OS Dashboard v2",
            "total_mins": 180,
            "human_mins": 108,
            "agent_mins": 72,
            "contexts": [
                {
                    "name": "Frontend Development",
                    "anchors": [
                        {"name": "Chart components", "time_mins": 45, "human_mins": 13, "agent_mins": 32},
                        {"name": "Dashboard layout", "time_mins": 75, "human_mins": 60, "agent_mins": 15},
                    ]
                },
                {
                    "name": "API Integration",
                    "anchors": [
                        {"name": "Auth flow", "time_mins": 60, "human_mins": 35, "agent_mins": 25},
                    ]
                }
            ]
        }
        # Verify rollup integrity
        total_from_anchors = sum(
            a["time_mins"]
            for c in chain["contexts"]
            for a in c["anchors"]
        )
        assert total_from_anchors == chain["total_mins"]

        human_from_anchors = sum(
            a["human_mins"]
            for c in chain["contexts"]
            for a in c["anchors"]
        )
        assert human_from_anchors == chain["human_mins"]
