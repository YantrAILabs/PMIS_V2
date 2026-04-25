"""Tests for PUT /api/pm/projects/{project_id} (Phase B inline rename).

Exercises the goals.yaml mutation directly — the endpoint's logic is
"find by id, mutate title, write back, sync". The handler imports
sync_pm_yaml_to_db at call time and uses the global _orch, so we test
the YAML mutation directly via a small extracted helper that mirrors
the endpoint body."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def _rename_project_in_yaml(
    raw: Dict[str, Any], project_id: str, new_title: str,
) -> Optional[str]:
    """The pure-data half of the PUT handler — find the project by id
    across all goals, update its title, return the goal_id that owns
    it. Returns None if not found."""
    for g in raw.get("goals", []) or []:
        for p in g.get("projects", []) or []:
            if p.get("id") == project_id:
                p["title"] = new_title
                return g.get("id")
    return None


@pytest.fixture
def sample_yaml(tmp_path):
    """Stage a multi-goal goals.yaml on disk so the rename tests can
    exercise both the find-and-mutate logic and the file roundtrip."""
    path = tmp_path / "goals.yaml"
    raw = {
        "goals": [
            {
                "id": "G-1",
                "title": "First goal",
                "projects": [
                    {"id": "P-001", "title": "Original alpha", "status": "active"},
                    {"id": "P-002", "title": "Original beta", "status": "active"},
                ],
            },
            {
                "id": "G-2",
                "title": "Second goal",
                "projects": [
                    {"id": "P-003", "title": "Original gamma", "status": "active"},
                ],
            },
        ],
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


class TestRenameLogic:
    def test_finds_and_renames_in_first_goal(self, sample_yaml):
        raw = yaml.safe_load(sample_yaml.read_text())
        gid = _rename_project_in_yaml(raw, "P-001", "Renamed alpha")
        assert gid == "G-1"
        assert raw["goals"][0]["projects"][0]["title"] == "Renamed alpha"

    def test_finds_and_renames_in_second_goal(self, sample_yaml):
        raw = yaml.safe_load(sample_yaml.read_text())
        gid = _rename_project_in_yaml(raw, "P-003", "Renamed gamma")
        assert gid == "G-2"
        assert raw["goals"][1]["projects"][0]["title"] == "Renamed gamma"

    def test_unknown_id_returns_none(self, sample_yaml):
        raw = yaml.safe_load(sample_yaml.read_text())
        assert _rename_project_in_yaml(raw, "P-missing", "x") is None

    def test_other_projects_unchanged(self, sample_yaml):
        raw = yaml.safe_load(sample_yaml.read_text())
        _rename_project_in_yaml(raw, "P-001", "Renamed alpha")
        assert raw["goals"][0]["projects"][1]["title"] == "Original beta"
        assert raw["goals"][1]["projects"][0]["title"] == "Original gamma"

    def test_other_goals_unchanged(self, sample_yaml):
        raw = yaml.safe_load(sample_yaml.read_text())
        _rename_project_in_yaml(raw, "P-001", "Renamed alpha")
        assert raw["goals"][1]["title"] == "Second goal"
        assert raw["goals"][1]["id"] == "G-2"

    def test_only_title_field_mutated(self, sample_yaml):
        """Other project fields (status, etc.) survive the rename."""
        raw = yaml.safe_load(sample_yaml.read_text())
        _rename_project_in_yaml(raw, "P-001", "Renamed alpha")
        assert raw["goals"][0]["projects"][0]["status"] == "active"
        assert raw["goals"][0]["projects"][0]["id"] == "P-001"

    def test_yaml_roundtrip_preserves_structure(self, sample_yaml):
        """Write + read produces a structurally-identical document."""
        raw = yaml.safe_load(sample_yaml.read_text())
        _rename_project_in_yaml(raw, "P-002", "Beta v2")
        sample_yaml.write_text(yaml.safe_dump(raw, sort_keys=False))
        reloaded = yaml.safe_load(sample_yaml.read_text())
        assert reloaded["goals"][0]["projects"][1]["title"] == "Beta v2"
        assert len(reloaded["goals"]) == 2
        assert len(reloaded["goals"][0]["projects"]) == 2

    def test_empty_goals_returns_none(self):
        assert _rename_project_in_yaml({"goals": []}, "P-001", "x") is None

    def test_no_goals_key_returns_none(self):
        assert _rename_project_in_yaml({}, "P-001", "x") is None


class TestEndpointShape:
    """The endpoint itself imports the global _orch, which requires a
    full PMIS init. We instead assert the handler exists and its
    request model has the right field, which is enough to catch
    accidental signature regressions."""

    def test_endpoint_registered(self):
        from server import app
        path = "/api/pm/projects/{project_id}"
        methods_for_path: set = set()
        for r in app.routes:
            if getattr(r, "path", None) == path:
                methods_for_path |= set(getattr(r, "methods", set()) or set())
        assert "PUT" in methods_for_path, (
            f"PUT not registered on {path}; saw methods: {methods_for_path}"
        )

    def test_payload_model_has_title(self):
        from server import PMProjectUpdate
        model = PMProjectUpdate(title="X")
        assert model.title == "X"
        with pytest.raises(Exception):
            PMProjectUpdate()  # title required
