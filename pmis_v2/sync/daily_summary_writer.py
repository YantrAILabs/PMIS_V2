"""Phase G — nightly auto-compose missing daily summaries and apply
queued daily_feedback rows.

Deterministic composer (no LLM) so the pipeline runs in CI and offline.
The LLM-refined version can layer on later by replacing
_compose_body_md with a richer renderer; the call sites stay the same.

Call sites:
  - consolidation/nightly.py runs both passes once per night.
  - CLI `python3 pmis_v2/cli.py daily compose|apply-feedback` for ad-hoc.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import uuid
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make wiki_tree_prose importable for "Captured progress" rendering.
_PMIS_ROOT = Path(__file__).resolve().parent.parent
if str(_PMIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PMIS_ROOT))

logger = logging.getLogger("pmis.sync.daily_summary_writer")


# ─── Composition ──────────────────────────────────────────────────────

def _format_minutes(mins: float) -> str:
    return f"{mins:.1f}".rstrip("0").rstrip(".") + "m"


def _compose_body_md(
    pmis_conn: sqlite3.Connection,
    project_id: str,
    deliverable_id: str,
    date: str,
    deliverable_name: str = "",
) -> Tuple[str, int, float]:
    """Deterministic markdown composer for a single (project,
    deliverable, date). Returns (body_md, item_count, total_minutes).
    Empty string when there's nothing to summarize."""

    # ── Today's tagged work ────────────────────────────────────────
    items = pmis_conn.execute(
        """SELECT work_description, time_mins, matched_at
           FROM project_work_match_log
           WHERE deliverable_id = ?
             AND DATE(matched_at) = ?
             AND is_correct = 1
           ORDER BY matched_at""",
        (deliverable_id, date),
    ).fetchall()

    if not items:
        return "", 0, 0.0

    total_mins = 0.0
    item_lines: List[str] = []
    for desc, mins, when in items:
        m = float(mins or 0.0)
        total_mins += m
        hhmm = (when or "")[11:16] or "—"
        text = (desc or "").strip()
        if len(text) > 160:
            text = text[:160].rstrip() + "…"
        if not text:
            text = "(no description)"
        item_lines.append(f"- {hhmm} · {_format_minutes(m)} — {text}")

    parts: List[str] = []
    title = (deliverable_name or deliverable_id) or "Daily summary"
    parts.append(f"## {title} · {date}")
    parts.append(f"*{len(items)} work item{'s' if len(items) != 1 else ''} · "
                 f"{total_mins:.1f} minutes total*")
    parts.append("### What you worked on")
    parts.append("\n".join(item_lines))

    # ── Captured progress (F5a render of confirmed proposals on date) ─
    try:
        from wiki_tree_prose import render_tree_as_prose
    except Exception:
        render_tree_as_prose = None  # type: ignore[assignment]

    if render_tree_as_prose is not None:
        prop_rows = pmis_conn.execute(
            """SELECT tree_json, status, confirmed_at
               FROM review_proposals
               WHERE (user_assigned_deliverable_id = ?
                      OR auto_attached_to_deliverable_id = ?)
                 AND status IN ('confirmed', 'auto_attached')
                 AND tree_json IS NOT NULL AND tree_json != ''
                 AND DATE(confirmed_at) = ?
               ORDER BY confirmed_at""",
            (deliverable_id, deliverable_id, date),
        ).fetchall()

        chunks: List[str] = []
        for tree_json, status, _ in prop_rows:
            try:
                tree = json.loads(tree_json)
            except Exception:
                continue
            prose = render_tree_as_prose(tree)
            if prose:
                badge = ("*auto-attached*" if status == "auto_attached"
                         else "*confirmed*")
                chunks.append(f"{badge}\n\n{prose}")
        if chunks:
            parts.append("### Captured progress")
            parts.append("\n\n---\n\n".join(chunks))

    # ── Contributing links ─────────────────────────────────────────
    link_rows = pmis_conn.execute(
        """SELECT l.url, l.kind, lb.dwell_frames
           FROM link_bindings lb
           JOIN links l ON l.id = lb.link_id
           WHERE lb.scope = 'deliverable' AND lb.scope_id = ?
             AND lb.contributed = 1
           ORDER BY lb.dwell_frames DESC, l.url""",
        (deliverable_id,),
    ).fetchall()
    if link_rows:
        parts.append("### Contributing links")
        link_lines = [
            f"- [{kind or 'other'}] {url} · {int(dwell or 0)} frames"
            for (url, kind, dwell) in link_rows
        ]
        parts.append("\n".join(link_lines))

    return "\n\n".join(parts).strip(), len(items), round(total_mins, 1)


# ─── Public passes ────────────────────────────────────────────────────

def compose_missing_daily_summaries(
    pmis_db_path: str,
    date: Optional[str] = None,
) -> Dict[str, int]:
    """Compose daily_summaries rows for any (deliverable, date) that
    has at least one is_correct=1 match-log row but no row yet.

    Args:
        pmis_db_path: PMIS DB path.
        date: ISO date. None defaults to yesterday — nightly's natural
              window since today is still in flight.
    """
    if not os.path.exists(pmis_db_path):
        return {"composed": 0, "skipped": 0, "scanned": 0}

    target_date = date or (_date.today() - timedelta(days=1)).isoformat()
    target_date = target_date[:10]

    conn = sqlite3.connect(pmis_db_path)
    composed = 0
    scanned = 0
    skipped_existing = 0
    try:
        # Distinct (project, deliverable) pairs with confirmed work
        # on the target date.
        pairs = conn.execute(
            """SELECT DISTINCT project_id, deliverable_id
               FROM project_work_match_log
               WHERE DATE(matched_at) = ?
                 AND is_correct = 1
                 AND deliverable_id != ''
                 AND deliverable_id IS NOT NULL""",
            (target_date,),
        ).fetchall()

        for project_id, deliverable_id in pairs:
            scanned += 1
            existing = conn.execute(
                """SELECT id FROM daily_summaries
                   WHERE deliverable_id = ? AND date = ?""",
                (deliverable_id, target_date),
            ).fetchone()
            if existing:
                skipped_existing += 1
                continue

            body, item_count, total_mins = _compose_body_md(
                conn, project_id, deliverable_id, target_date,
            )
            if not body:
                continue

            ds_id = f"ds-{uuid.uuid4().hex[:12]}"
            conn.execute(
                """INSERT INTO daily_summaries
                   (id, project_id, deliverable_id, date, body_md,
                    status, composed_at)
                   VALUES (?, ?, ?, ?, ?, 'auto', datetime('now'))""",
                (ds_id, project_id, deliverable_id, target_date, body),
            )
            composed += 1

        conn.commit()
    finally:
        conn.close()

    return {"composed": composed, "skipped": skipped_existing, "scanned": scanned}


def apply_pending_feedback(pmis_db_path: str) -> Dict[str, int]:
    """Process daily_feedback rows where applied=0:
      - Re-compose the target summary's body
      - Append "**Feedback applied** ({applied_at}): {text}"
      - UPDATE daily_summaries.status='edited' + body_md
      - Flip daily_feedback.applied=1, applied_at=now()

    Orphan feedback (target row missing) is left applied=0 with a log
    warning — preserves the signal that something went wrong upstream
    rather than silently flagging it processed.
    """
    if not os.path.exists(pmis_db_path):
        return {"applied": 0, "orphan": 0, "scanned": 0}

    conn = sqlite3.connect(pmis_db_path)
    conn.row_factory = sqlite3.Row
    applied = 0
    orphan = 0
    scanned = 0
    try:
        rows = conn.execute(
            """SELECT id, daily_summary_id, feedback_text, created_at
               FROM daily_feedback
               WHERE COALESCE(applied, 0) = 0
               ORDER BY created_at"""
        ).fetchall()

        for fb in rows:
            scanned += 1
            ds = conn.execute(
                """SELECT id, project_id, deliverable_id, date
                   FROM daily_summaries WHERE id = ?""",
                (fb["daily_summary_id"],),
            ).fetchone()
            if not ds:
                orphan += 1
                logger.warning(
                    "daily_feedback %s targets missing daily_summary %s; "
                    "leaving applied=0",
                    fb["id"], fb["daily_summary_id"],
                )
                continue

            new_body, _, _ = _compose_body_md(
                conn, ds["project_id"], ds["deliverable_id"], ds["date"],
            )
            text = (fb["feedback_text"] or "").strip()
            note = (
                f"\n\n---\n\n**Feedback applied** "
                f"({fb['created_at'][:10] if fb['created_at'] else 'today'}):"
                f" {text}"
            )
            final_body = (new_body or "") + note

            conn.execute(
                """UPDATE daily_summaries
                   SET body_md = ?, status = 'edited',
                       composed_at = datetime('now')
                   WHERE id = ?""",
                (final_body, ds["id"]),
            )
            conn.execute(
                """UPDATE daily_feedback
                   SET applied = 1, applied_at = datetime('now')
                   WHERE id = ?""",
                (fb["id"],),
            )
            applied += 1

        conn.commit()
    finally:
        conn.close()

    return {"applied": applied, "orphan": orphan, "scanned": scanned}


def run_daily_pass(
    pmis_db_path: str,
    date: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for nightly + CLI: compose then apply."""
    composed = compose_missing_daily_summaries(pmis_db_path, date=date)
    applied = apply_pending_feedback(pmis_db_path)
    return {"composed": composed, "feedback_applied": applied}
