"""
YAML → DB sync for projects + deliverables.

The Goals UI edits productivity-tracker/config/goals.yaml and deliverables.yaml
directly. The live_matcher and /api/work/deliverables read the SQL tables
(projects, deliverables). This module keeps the two in sync so the infinite-loop
widget picker actually shows the user's deliverables.

Idempotent — safe to run on every startup and on manual trigger. Uses YAML id
(P-001, D-001, ...) as the DB primary key so the sync is naturally upsert.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("pmis.sync.yaml_to_db")


def _config_paths() -> Tuple[Path, Path]:
    """Resolve YAML paths via the productivity-tracker repo layout."""
    pt_root = Path(__file__).resolve().parents[2] / "productivity-tracker"
    return pt_root / "config" / "goals.yaml", pt_root / "config" / "deliverables.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _find_sc_node_id(conn: sqlite3.Connection, supercontext_name: str) -> str:
    """Best-effort text match — a deliverable's supercontext string against
    existing SC-level memory_nodes. Returns '' if no match."""
    if not supercontext_name:
        return ""
    row = conn.execute(
        """SELECT id FROM memory_nodes
           WHERE level = 'SC' AND is_deleted = 0
             AND LOWER(content) = LOWER(?)
           LIMIT 1""",
        (supercontext_name,),
    ).fetchone()
    if row:
        return row[0]
    # Fuzzy: contains match
    row = conn.execute(
        """SELECT id FROM memory_nodes
           WHERE level = 'SC' AND is_deleted = 0
             AND LOWER(content) LIKE LOWER(?)
           LIMIT 1""",
        (f"%{supercontext_name}%",),
    ).fetchone()
    return row[0] if row else ""


def _pick_project_for_deliverable(
    deliv_id: str, goals_doc: Dict[str, Any]
) -> Tuple[str, str]:
    """Walk goals.yaml → projects → deliverable_patterns. Return (project_id,
    project_title) ONLY for an explicit link. Returns ('','') if unlinked —
    caller should skip rather than misattribute.
    """
    for goal in goals_doc.get("goals") or []:
        for proj in goal.get("projects") or []:
            pid = proj.get("id") or ""
            ptitle = proj.get("title") or ""
            patterns = proj.get("deliverable_patterns") or {}
            if deliv_id in patterns:
                return pid, ptitle
    return ("", "")


def sync_pm_yaml_to_db(db, embedder=None, hyperparams=None) -> Dict[str, int]:
    """Sync goals.yaml projects + deliverables.yaml into projects/deliverables
    SQL tables. Returns counts.

    `db` is a pmis_v2.db.manager.DBManager (needs `_connect()` context manager).

    If `embedder` is supplied, each deliverable that lacks a `context_node_id`
    gets an auto-embedded synthetic CTX memory_node created for it, so the
    LiveMatcher / BriefComposer / ProjectMatcher have a real semantic anchor
    to score incoming segments against. This is Fix 2 of the suggestion
    pipeline.
    """
    goals_path, deliv_path = _config_paths()
    goals_doc = _load_yaml(goals_path)
    deliv_doc = _load_yaml(deliv_path)

    counts = {"projects_upserted": 0, "deliverables_upserted": 0,
               "sc_resolved": 0, "project_fallbacks": 0,
               "ctx_nodes_created": 0, "ctx_nodes_reused": 0}

    with db._connect() as conn:
        # ─── Projects ────────────────────────────────────────────
        for goal in goals_doc.get("goals") or []:
            for proj in goal.get("projects") or []:
                pid = proj.get("id")
                if not pid:
                    continue
                conn.execute(
                    """INSERT INTO projects (id, name, description, status, source, source_id, updated_at)
                       VALUES (?, ?, ?, ?, 'yaml', ?, datetime('now'))
                       ON CONFLICT(id) DO UPDATE SET
                         name = excluded.name,
                         status = excluded.status,
                         updated_at = datetime('now')""",
                    (
                        pid,
                        proj.get("title") or pid,
                        goal.get("why") or goal.get("title") or "",
                        proj.get("status") or "active",
                        goal.get("id") or "",
                    ),
                )
                counts["projects_upserted"] += 1

        # ─── Deliverables ────────────────────────────────────────
        for deliv in deliv_doc.get("deliverables") or []:
            did = deliv.get("id")
            if not did:
                continue
            project_id, _ptitle = _pick_project_for_deliverable(did, goals_doc)
            if not project_id:
                counts["project_fallbacks"] += 1
                continue  # nothing to link — skip rather than orphan

            sc_node_id = _find_sc_node_id(conn, deliv.get("supercontext") or "")
            if sc_node_id:
                counts["sc_resolved"] += 1

            # Update projects.sc_node_id if we found one and it's not set
            if sc_node_id:
                conn.execute(
                    """UPDATE projects SET sc_node_id = ?
                       WHERE id = ? AND (sc_node_id IS NULL OR sc_node_id = '')""",
                    (sc_node_id, project_id),
                )

            conn.execute(
                """INSERT INTO deliverables
                   (id, project_id, name, description, status, deadline,
                    expected_hours, source, source_id, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'yaml', ?, datetime('now'))
                   ON CONFLICT(id) DO UPDATE SET
                     project_id = excluded.project_id,
                     name = excluded.name,
                     description = excluded.description,
                     status = excluded.status,
                     deadline = excluded.deadline,
                     updated_at = datetime('now')""",
                (
                    did,
                    project_id,
                    deliv.get("name") or did,
                    "; ".join(deliv.get("expected_contexts") or []),
                    deliv.get("status") or "active",
                    str(deliv.get("deadline") or ""),
                    float(deliv.get("expected_hours") or 0),
                    did,
                ),
            )
            counts["deliverables_upserted"] += 1

    # ─── Fix 2: Auto-embed deliverables into synthetic CTX memory_nodes ──
    # Runs outside the first connect block so create_node can use its own
    # transaction. Skipped when embedder is None (e.g. standalone CLI run).
    if embedder is not None:
        created, reused = _auto_embed_deliverables(
            db, embedder, deliv_doc, goals_doc, hyperparams or {}
        )
        counts["ctx_nodes_created"] = created
        counts["ctx_nodes_reused"] = reused

    logger.info("YAML sync: %s", counts)
    return counts


def _auto_embed_deliverables(
    db, embedder, deliv_doc: Dict[str, Any], goals_doc: Dict[str, Any],
    hyperparams: Dict[str, Any],
) -> tuple[int, int]:
    """For each deliverable lacking a context_node_id, create a synthetic CTX
    memory_node from its name + description + expected_contexts and wire it
    back as the deliverable's context_node_id. Idempotent — re-syncs reuse
    existing nodes by a deterministic hash marker."""
    from core.memory_node import MemoryNode, MemoryLevel
    from core.poincare import assign_hyperbolic_coords, ProjectionManager
    from core.temporal import temporal_encode
    from datetime import datetime

    created = 0
    reused = 0

    # Lazy build projection manager — matches the existing embedding dim
    input_dim = 768
    try:
        with db._connect() as conn:
            row = conn.execute(
                "SELECT euclidean FROM embeddings WHERE euclidean IS NOT NULL LIMIT 1"
            ).fetchone()
            if row and row["euclidean"]:
                import numpy as np
                input_dim = len(np.frombuffer(row["euclidean"], dtype=np.float32))
    except Exception:
        pass

    pm = ProjectionManager(input_dim=input_dim)

    for deliv in deliv_doc.get("deliverables") or []:
        did = deliv.get("id")
        if not did:
            continue
        # Only embed if the deliverable was actually inserted (linked)
        with db._connect() as conn:
            row = conn.execute(
                "SELECT context_node_id FROM deliverables WHERE id = ?", (did,)
            ).fetchone()
            if not row:
                continue
            # Already linked to a CTX node — skip (re-sync is cheap)
            if row["context_node_id"]:
                reused += 1
                continue

        # Compose the embeddable text
        name = deliv.get("name") or did
        desc = deliv.get("description") or ""
        expected = deliv.get("expected_contexts") or []
        parts = [name]
        if desc:
            parts.append(desc)
        if expected:
            parts.append("Expected contexts: " + ", ".join(expected))
        # Add pattern keywords from goals.yaml so the embedding captures them
        patterns = _collect_deliverable_patterns(did, goals_doc)
        if patterns:
            parts.append("Keywords: " + ", ".join(patterns))
        text = ". ".join(parts)

        # Deterministic source tag so re-syncs find existing node
        src_tag = f"deliverable_sync:{did}"

        # Look for existing CTX node with this source tag
        existing_id = None
        try:
            with db._connect() as conn:
                r = conn.execute(
                    """SELECT id FROM memory_nodes
                       WHERE source_conversation_id = ? AND level = 'CTX'
                         AND is_deleted = 0 LIMIT 1""",
                    (src_tag,),
                ).fetchone()
                if r:
                    existing_id = r["id"]
        except Exception:
            pass

        if existing_id:
            node_id = existing_id
            reused += 1
        else:
            # Embed + create the node
            try:
                eu = embedder.embed_text(text)
                hyp = assign_hyperbolic_coords(eu, "CTX", pm, hyperparams=hyperparams)
                temp = temporal_encode(datetime.now())
                node = MemoryNode.create(
                    content=text[:500],
                    level=MemoryLevel.CONTEXT,
                    euclidean_embedding=eu,
                    hyperbolic_coords=hyp,
                    temporal_embedding=temp,
                    source_conversation_id=src_tag,
                    surprise=0.2,
                    precision=0.6,
                )
                node_id = db.create_node(node)
                created += 1
            except Exception as e:
                logger.warning("auto-embed failed for %s: %s", did, e)
                continue

        # Wire deliverable.context_node_id → synthetic CTX node
        with db._connect() as conn:
            conn.execute(
                "UPDATE deliverables SET context_node_id = ? WHERE id = ?",
                (node_id, did),
            )

    return created, reused


def _collect_deliverable_patterns(deliv_id: str, goals_doc: Dict[str, Any]) -> List[str]:
    """Walk goals.yaml → all patterns relevant to this deliverable."""
    patterns: List[str] = []
    for goal in goals_doc.get("goals") or []:
        for proj in goal.get("projects") or []:
            dp = proj.get("deliverable_patterns") or {}
            if deliv_id in dp:
                patterns.extend(dp[deliv_id] or [])
            # Include project-level as a weaker signal
            patterns.extend(proj.get("match_patterns") or [])
    # Dedup, preserve order
    seen = set()
    ordered: List[str] = []
    for p in patterns:
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


if __name__ == "__main__":
    # Standalone — for manual runs
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from db.manager import DBManager

    pmis_db_path = str(Path(__file__).resolve().parents[1] / "data" / "memory.db")
    db = DBManager(db_path=pmis_db_path)
    print(sync_pm_yaml_to_db(db))
