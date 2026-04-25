"""Phase D3 — propagate segment_links into link_bindings whenever a
project_work_match_log row lands a segment on a deliverable.

Architecture: stays out of project_matcher.py. The matcher writes its
log rows as before; this pass runs after, finds rows with
deliverable_id set + is_correct=1 + link_bindings_written=0, and
upserts links + link_bindings for the segment's URLs.

Idempotency:
  - links table is UNIQUE on url; INSERT OR IGNORE returns the
    existing row's id when the URL is already cataloged.
  - link_bindings is UNIQUE on (link_id, scope, scope_id); we use
    INSERT ... ON CONFLICT to upsert dwell + contributed.
  - project_work_match_log gains link_bindings_written=1 once
    processed; subsequent runs skip those rows.

Contribution flag:
  - dwell_frames >= link_contribution_min_dwell → contributed=1
  - Default threshold: 2 (configurable via hyperparams).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pmis.sync.link_bindings_writer")


_DEFAULT_TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")


def _resolve_tracker(p: Optional[str]) -> str:
    return p or _DEFAULT_TRACKER_DB


def _classify_kind_safe(url: str, fallback: str) -> str:
    """Best-effort kind classification — falls back to whatever the
    segment_links payload carried if our extractor isn't importable
    (shouldn't happen in production, but robust)."""
    try:
        from links_extractor import classify_kind
        return classify_kind(url) or fallback or "other"
    except Exception:
        return fallback or "other"


def _upsert_link(conn: sqlite3.Connection, url: str, kind: str) -> str:
    """INSERT OR IGNORE then SELECT — returns the link's id."""
    new_id = f"lnk-{uuid.uuid4().hex[:12]}"
    conn.execute(
        "INSERT OR IGNORE INTO links (id, url, kind) VALUES (?, ?, ?)",
        (new_id, url, kind),
    )
    row = conn.execute(
        "SELECT id FROM links WHERE url = ?", (url,),
    ).fetchone()
    return row[0] if row else new_id


def _upsert_binding(
    conn: sqlite3.Connection,
    link_id: str,
    scope: str,
    scope_id: str,
    dwell_frames: int,
    contributed: int,
) -> None:
    """UPSERT keyed on the UNIQUE(link_id, scope, scope_id) index.
    Latest dwell + contributed wins on conflict — fresh data is
    more truthful than stale."""
    conn.execute(
        """INSERT INTO link_bindings
           (link_id, scope, scope_id, contributed, dwell_frames, added_at)
           VALUES (?, ?, ?, ?, ?, datetime('now'))
           ON CONFLICT(link_id, scope, scope_id) DO UPDATE SET
               contributed = excluded.contributed,
               dwell_frames = excluded.dwell_frames""",
        (link_id, scope, scope_id, contributed, dwell_frames),
    )


def bind_recent_matches(
    pmis_db_path: str,
    tracker_db_path: Optional[str] = None,
    since: Optional[str] = None,
    min_dwell_for_contributed: int = 2,
    batch: int = 200,
) -> Dict[str, int]:
    """Process unprocessed project_work_match_log rows and write the
    corresponding links + link_bindings.

    Args:
        pmis_db_path: path to the PMIS DB (links + link_bindings live here).
        tracker_db_path: path to the tracker DB (segment_links source).
        since: ISO timestamp filter on matched_at; None backfills all.
        min_dwell_for_contributed: dwell threshold for contributed=1.
        batch: process at most this many rows per round.

    Returns:
        {matches_processed, bindings_written, links_created}
    """
    if not os.path.exists(pmis_db_path):
        return {"matches_processed": 0, "bindings_written": 0, "links_created": 0}
    tracker_path = _resolve_tracker(tracker_db_path)
    if not os.path.exists(tracker_path):
        return {
            "matches_processed": 0, "bindings_written": 0,
            "links_created": 0, "error": "tracker_db missing",
        }

    pconn = sqlite3.connect(pmis_db_path)
    pconn.row_factory = sqlite3.Row
    tconn = sqlite3.connect(tracker_path)
    tconn.row_factory = sqlite3.Row
    try:
        # Probe for the link_bindings_written column — falls back to a
        # one-shot ALTER if migration hasn't run yet.
        cols = {
            r[1] for r in pconn.execute(
                "PRAGMA table_info(project_work_match_log)"
            ).fetchall()
        }
        if "link_bindings_written" not in cols:
            try:
                pconn.execute(
                    "ALTER TABLE project_work_match_log "
                    "ADD COLUMN link_bindings_written INTEGER DEFAULT 0"
                )
                pconn.commit()
            except sqlite3.Error:
                return {
                    "matches_processed": 0, "bindings_written": 0,
                    "links_created": 0,
                    "error": "link_bindings_written column missing",
                }

        params: List[Any] = []
        where = (
            "deliverable_id IS NOT NULL AND deliverable_id != '' "
            "AND is_correct = 1 "
            "AND COALESCE(link_bindings_written, 0) = 0"
        )
        if since:
            where += " AND matched_at >= ?"
            params.append(since)

        matches_processed = 0
        bindings_written = 0
        links_before = pconn.execute(
            "SELECT COUNT(*) FROM links"
        ).fetchone()[0]

        while True:
            rows = pconn.execute(
                f"SELECT id, segment_id, deliverable_id "
                f"FROM project_work_match_log WHERE {where} "
                f"ORDER BY matched_at LIMIT ?",
                params + [int(batch)],
            ).fetchall()
            if not rows:
                break

            for r in rows:
                seg_id = r["segment_id"]
                did = r["deliverable_id"]
                if not seg_id or not did:
                    pconn.execute(
                        "UPDATE project_work_match_log "
                        "SET link_bindings_written = 1 WHERE id = ?",
                        (r["id"],),
                    )
                    matches_processed += 1
                    continue

                seg_row = tconn.execute(
                    "SELECT segment_links FROM context_1 WHERE id = ?",
                    (seg_id,),
                ).fetchone()
                raw = (seg_row["segment_links"] if seg_row else "") or ""
                try:
                    items = json.loads(raw) if raw else []
                except Exception:
                    items = []

                for item in items or []:
                    url = (item.get("url") or "").strip()
                    if not url:
                        continue
                    dwell = int(item.get("dwell_frames") or 0)
                    kind = _classify_kind_safe(
                        url, item.get("kind", "other"),
                    )
                    contributed = 1 if dwell >= min_dwell_for_contributed else 0
                    link_id = _upsert_link(pconn, url, kind)
                    _upsert_binding(
                        pconn, link_id, "deliverable", did,
                        dwell, contributed,
                    )
                    bindings_written += 1

                pconn.execute(
                    "UPDATE project_work_match_log "
                    "SET link_bindings_written = 1 WHERE id = ?",
                    (r["id"],),
                )
                matches_processed += 1

            pconn.commit()
            if len(rows) < batch:
                break

        links_after = pconn.execute(
            "SELECT COUNT(*) FROM links"
        ).fetchone()[0]
        return {
            "matches_processed": matches_processed,
            "bindings_written": bindings_written,
            "links_created": links_after - links_before,
        }
    finally:
        pconn.close()
        tconn.close()
