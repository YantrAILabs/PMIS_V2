"""F5a — render a tree_builder.TreeCandidate (or its JSON form) as
markdown that reads like a wiki page.

Hard rule from the user: the output must NOT expose the SC/CTX/ANC
scaffolding. No "Super-Context:", "Context:", "Anchor:" strings, no
hierarchy levels shown. Just flowing prose with H2 for the domain,
H3 for each sub-topic, and a small footer with stats.

This is the display layer for confirmed / auto_attached proposals. It
pulls from `review_proposals.tree_json` (written by F2a) and produces
markdown that the existing wiki templates can render verbatim.

No LLM — deterministic. F5b will add an optional LLM pass that refines
each H3 block into paragraphs; for now the anchor bodies come through
verbatim so we don't pay tokens until the shape is settled.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pmis.wiki_tree_prose")


def render_tree_as_prose(tree_dict: Optional[Dict[str, Any]]) -> str:
    """Return markdown for a tree dict. Empty input → empty string.

    Layout:
      ## sc_title
      sc_summary (as paragraph)
      ### anchor[0].title
      anchor[0].body
      ### anchor[1].title
      anchor[1].body
      ...
      *N segments across M apps · X.X minutes*

    ctx_title is deliberately not rendered — it's internal scaffolding
    and the user's rule is that the reader shouldn't see hierarchy.
    Anchors with empty bodies render as just the heading.
    """
    if not tree_dict:
        return ""

    sc_title = (tree_dict.get("sc_title") or "").strip()
    if not sc_title:
        return ""

    parts: List[str] = [f"## {sc_title}"]

    sc_summary = (tree_dict.get("sc_summary") or "").strip()
    if sc_summary:
        parts.append(sc_summary)

    for anchor in tree_dict.get("anchors") or []:
        title = (anchor.get("title") or "").strip()
        body = (anchor.get("body") or "").strip()
        if not title and not body:
            continue
        if title:
            parts.append(f"### {title}")
        if body:
            parts.append(body)

    segment_count = len(tree_dict.get("segment_ids") or [])
    window_count = len({
        a.get("window", "")
        for a in (tree_dict.get("anchors") or [])
        if a.get("window")
    })
    duration = float(tree_dict.get("duration_mins") or 0.0)

    if segment_count or duration:
        stats = []
        if segment_count:
            stats.append(
                f"{segment_count} segment" + ("s" if segment_count != 1 else "")
            )
        if window_count:
            stats.append(
                f"{window_count} app" + ("s" if window_count != 1 else "")
            )
        if duration:
            stats.append(f"{duration:.1f} minutes")
        parts.append(f"*{' · '.join(stats)}*")

    return "\n\n".join(parts)


def render_proposal_as_prose(
    db, proposal_id: str,
) -> Optional[Dict[str, Any]]:
    """Load a review_proposals row and return its rendered markdown
    plus the routing metadata the UI needs to link back to a project /
    deliverable. Returns None if the proposal is missing or has no
    tree_json to render from.
    """
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """SELECT id, status, tree_json, user_assigned_project_id,
                      user_assigned_deliverable_id,
                      auto_attached_to_deliverable_id,
                      anchor_node_id, confirmed_at
               FROM review_proposals WHERE id = ?""",
            (proposal_id,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None
    raw = row["tree_json"] or ""
    if not raw.strip():
        return None

    try:
        tree_dict = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("tree_json not parseable for proposal %s", proposal_id)
        return None

    prose = render_tree_as_prose(tree_dict)
    if not prose:
        return None

    deliverable_id = (
        row["user_assigned_deliverable_id"]
        or row["auto_attached_to_deliverable_id"]
        or ""
    )
    return {
        "id": row["id"],
        "status": row["status"],
        "prose_md": prose,
        "project_id": row["user_assigned_project_id"] or "",
        "deliverable_id": deliverable_id,
        "anchor_node_id": row["anchor_node_id"] or "",
        "confirmed_at": row["confirmed_at"] or "",
    }
