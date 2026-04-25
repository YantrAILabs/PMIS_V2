"""Tests for Phase C1 — render_project_detail + the route shapes.

The endpoint itself imports the global _orch (full PMIS init) which
costs an embedder spin-up — too heavy for unit tests. We verify:
  - render_project_detail logic on a fake _render_pm_projects payload
  - both routes are registered with the right paths and methods
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_renderer(pm_payload):
    """A WikiRenderer instance with _render_pm_projects stubbed to
    return `pm_payload`. Just enough to drive render_project_detail."""
    from wiki_renderer import WikiRenderer
    db = MagicMock()
    db.db_path = ":memory:"
    wr = WikiRenderer(db)
    wr._render_pm_projects = lambda: pm_payload  # type: ignore[assignment]
    return wr


_PM_FIXTURE = {
    "goals": [
        {
            "id": "G-1",
            "title": "First goal",
            "projects": [
                {
                    "id": "P-001",
                    "title": "Alpha",
                    "status": "active",
                    "target_week": "2026-W18",
                    "pulse_mins_7d": 12.5,
                    "days_since_pulse": 0,
                    "column": "alive",
                    "deliverables": [
                        {"id": "D-001", "name": "Build engine", "alive": True, "mins_7d": 8},
                        {"id": "D-002", "name": "Ship docs", "alive": False, "mins_7d": 0},
                    ],
                },
                {
                    "id": "P-002",
                    "title": "Beta",
                    "status": "active",
                    "deliverables": [],
                },
            ],
        },
    ],
}


class TestRenderProjectDetail:
    def test_unknown_project_returns_none(self):
        wr = _make_renderer(_PM_FIXTURE)
        assert wr.render_project_detail("P-missing") is None

    def test_known_project_returns_payload(self):
        wr = _make_renderer(_PM_FIXTURE)
        out = wr.render_project_detail("P-001")
        assert out is not None
        assert out["project"]["id"] == "P-001"
        assert out["project"]["title"] == "Alpha"
        assert len(out["deliverables"]) == 2

    def test_breadcrumb_shape(self):
        wr = _make_renderer(_PM_FIXTURE)
        out = wr.render_project_detail("P-001")
        assert [c["label"] for c in out["breadcrumb"]] == [
            "Goals", "First goal", "Alpha",
        ]
        # Last crumb (the project itself) has no href.
        assert out["breadcrumb"][-1]["href"] == ""

    def test_parent_goal_attached(self):
        wr = _make_renderer(_PM_FIXTURE)
        out = wr.render_project_detail("P-001")
        assert out["parent_goal"] == {"id": "G-1", "title": "First goal"}

    def test_no_deliverable_selected_by_default(self):
        wr = _make_renderer(_PM_FIXTURE)
        out = wr.render_project_detail("P-001")
        assert out["selected_deliverable"] is None
        assert out["selected_deliverable_id"] == ""

    def test_known_deliverable_id_selects_it(self):
        wr = _make_renderer(_PM_FIXTURE)
        out = wr.render_project_detail("P-001", "D-002")
        assert out["selected_deliverable"]["id"] == "D-002"
        assert out["selected_deliverable"]["name"] == "Ship docs"
        assert out["selected_deliverable_id"] == "D-002"

    def test_unknown_deliverable_id_returns_none_selection(self):
        """Caller (route handler) should 404; this method just signals
        'not found here' via selected_deliverable=None."""
        wr = _make_renderer(_PM_FIXTURE)
        out = wr.render_project_detail("P-001", "D-bogus")
        assert out is not None  # project exists
        assert out["selected_deliverable"] is None

    def test_deliverable_lookup_does_not_cross_projects(self):
        """A deliverable id from a different project shouldn't match —
        this would be a real bug if it did."""
        import copy
        pm = copy.deepcopy(_PM_FIXTURE)
        pm["goals"][0]["projects"][1]["deliverables"] = [
            {"id": "D-other", "name": "Other one", "alive": False},
        ]
        wr = _make_renderer(pm)
        out = wr.render_project_detail("P-001", "D-other")
        assert out["selected_deliverable"] is None

    def test_project_with_no_deliverables(self):
        wr = _make_renderer(_PM_FIXTURE)
        out = wr.render_project_detail("P-002")
        assert out["deliverables"] == []
        assert out["selected_deliverable"] is None


class TestRouteRegistration:
    def test_project_detail_route_registered(self):
        from server import app
        path = "/wiki/goals/p/{project_id}"
        methods: set = set()
        for r in app.routes:
            if getattr(r, "path", None) == path:
                methods |= set(getattr(r, "methods", set()) or set())
        assert "GET" in methods

    def test_deliverable_route_registered(self):
        from server import app
        path = "/wiki/goals/p/{project_id}/d/{deliverable_id}"
        methods: set = set()
        for r in app.routes:
            if getattr(r, "path", None) == path:
                methods |= set(getattr(r, "methods", set()) or set())
        assert "GET" in methods
