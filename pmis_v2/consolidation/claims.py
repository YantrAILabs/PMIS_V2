"""Single source of truth for 'has this segment / work_page already been
consolidated?' Used by Review, Goals Unassigned, and Goals Recent-days
surfaces so they stay in sync.

Each helper returns (claimed: bool, reason: str). The reason names the
claim-table that owns the item — useful for debug tooltips when we surface
'already tagged' state in the UI.

Claim sources:
  1. activity_time_log.segment_id          — nightly time_assignment or
                                              user Review-confirm
  2. review_proposals (status in
     {'draft', 'confirmed', 'auto_attached'}) — manual consolidation
  3. work_pages.tag_state = 'confirmed'    — user tagged from Unassigned
  4. work_pages.state in {'tagged',
     'archived'}                           — terminal work_page states

'rejected' and 'superseded' proposals do NOT claim their segments — by
design, those segments drop back into the pool for re-clustering.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Dict, Iterable, Optional, Set, Tuple


_CLAIMING_PROPOSAL_STATUSES: Tuple[str, ...] = (
    "draft", "confirmed", "auto_attached",
)


def is_segment_claimed(
    pm_conn: sqlite3.Connection, segment_id: str,
) -> Tuple[bool, str]:
    """True when `segment_id` is already accounted for in the PMIS DB."""
    if not segment_id:
        return False, ""

    row = pm_conn.execute(
        "SELECT 1 FROM activity_time_log WHERE segment_id = ? LIMIT 1",
        (segment_id,),
    ).fetchone()
    if row:
        return True, "activity_time_log"

    placeholders = ",".join("?" * len(_CLAIMING_PROPOSAL_STATUSES))
    rows = pm_conn.execute(
        f"SELECT segment_ids_json FROM review_proposals "
        f"WHERE status IN ({placeholders})",
        _CLAIMING_PROPOSAL_STATUSES,
    ).fetchall()
    for (sids_json,) in rows:
        try:
            if segment_id in json.loads(sids_json or "[]"):
                return True, "review_proposals"
        except (json.JSONDecodeError, TypeError):
            continue

    return False, ""


def is_workpage_claimed(
    pm_conn: sqlite3.Connection, page_id: str,
    fully_consolidated: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """True when `page_id` is already tagged, terminal, or has every
    constituent segment already claimed by activity_time_log (i.e. nightly
    has fully consumed it).

    Pass `fully_consolidated` from `fully_consolidated_page_ids()` to avoid
    an N+1 query when checking a batch of pages. Without it, the segment-
    coverage check is skipped — suitable for single-page lookups where the
    state/tag_state signals are enough.
    """
    if not page_id:
        return False, ""
    row = pm_conn.execute(
        "SELECT state, tag_state FROM work_pages WHERE id = ?",
        (page_id,),
    ).fetchone()
    if not row:
        return False, ""
    state, tag_state = row
    if tag_state == "confirmed":
        return True, "tag_state=confirmed"
    if state in ("tagged", "archived"):
        return True, f"state={state}"
    if fully_consolidated is not None and page_id in fully_consolidated:
        return True, "segments_fully_consolidated"
    return False, ""


def fully_consolidated_page_ids(pm_conn: sqlite3.Connection) -> Set[str]:
    """work_pages (state='open') where every anchor segment is already in
    activity_time_log. These are pages nightly has already consumed — they
    shouldn't sit in the Unassigned lane anymore.

    A page with zero anchors is NOT considered consolidated (could be brand
    new). A page with some claimed and some unclaimed segments is also not
    considered consolidated — conservative, leaves ambiguous cases visible.
    """
    rows = pm_conn.execute(
        """
        SELECT wp.id FROM work_pages wp
        WHERE wp.state = 'open'
          AND EXISTS (
            SELECT 1 FROM work_page_anchors wpa WHERE wpa.page_id = wp.id
          )
          AND NOT EXISTS (
            SELECT 1 FROM work_page_anchors wpa
            WHERE wpa.page_id = wp.id
              AND wpa.segment_id NOT IN (
                SELECT segment_id FROM activity_time_log
                WHERE segment_id IS NOT NULL AND segment_id != ''
              )
          )
        """
    ).fetchall()
    return {r[0] for r in rows}


def all_claimed_segment_ids(pm_conn: sqlite3.Connection) -> Dict[str, str]:
    """Every segment_id currently claimed by any source. Cheap enough to call
    per request — activity_time_log is a few hundred rows, and the proposal
    scan is bounded by live proposal count."""
    out: Dict[str, str] = {}
    for (sid,) in pm_conn.execute(
        "SELECT DISTINCT segment_id FROM activity_time_log "
        "WHERE segment_id IS NOT NULL AND segment_id != ''"
    ).fetchall():
        out[sid] = "activity_time_log"

    status_placeholders = ",".join("?" * len(_CLAIMING_PROPOSAL_STATUSES))
    rows = pm_conn.execute(
        f"SELECT segment_ids_json FROM review_proposals "
        f"WHERE status IN ({status_placeholders})",
        _CLAIMING_PROPOSAL_STATUSES,
    ).fetchall()
    for (sids_json,) in rows:
        try:
            for sid in json.loads(sids_json or "[]"):
                if sid and sid not in out:
                    out[sid] = "review_proposals"
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def claimed_segment_ids(
    pm_conn: sqlite3.Connection, segment_ids: Iterable[str],
) -> Dict[str, str]:
    """Bulk variant — returns {segment_id: reason} for every id that's
    claimed. Faster than N individual calls when filtering a whole batch
    (e.g. Review's list_unconsolidated)."""
    ids = [sid for sid in segment_ids if sid]
    if not ids:
        return {}

    out: Dict[str, str] = {}
    placeholders = ",".join("?" * len(ids))
    for (sid,) in pm_conn.execute(
        f"SELECT DISTINCT segment_id FROM activity_time_log "
        f"WHERE segment_id IN ({placeholders})",
        ids,
    ).fetchall():
        out[sid] = "activity_time_log"

    remaining = {sid for sid in ids if sid not in out}
    if not remaining:
        return out

    status_placeholders = ",".join("?" * len(_CLAIMING_PROPOSAL_STATUSES))
    rows = pm_conn.execute(
        f"SELECT segment_ids_json FROM review_proposals "
        f"WHERE status IN ({status_placeholders})",
        _CLAIMING_PROPOSAL_STATUSES,
    ).fetchall()
    for (sids_json,) in rows:
        try:
            for sid in json.loads(sids_json or "[]"):
                if sid in remaining and sid not in out:
                    out[sid] = "review_proposals"
        except (json.JSONDecodeError, TypeError):
            continue
    return out
