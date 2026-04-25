"""Phase D2 — populate context_2.extracted_links from frame OCR +
window names, then roll up to context_1.segment_links.

Sits in pmis_v2/sync/ rather than productivity-tracker/ to avoid
cross-codebase coupling. Tracker keeps writing raw frames; this
writer (run on demand or by nightly) backfills the link metadata
columns added in Phase A.

Idempotency: an empty result for a row writes the JSON literal '[]'
so subsequent runs skip it. Forces a clear "scanned" signal.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the extractor importable regardless of where this file is run from.
_PMIS_ROOT = Path(__file__).resolve().parent.parent
if str(_PMIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PMIS_ROOT))

from links_extractor import extract_all  # noqa: E402

logger = logging.getLogger("pmis.sync.links_writer")


_DEFAULT_TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")


def _resolve_path(p: Optional[str]) -> str:
    return p or _DEFAULT_TRACKER_DB


def populate_extracted_links(
    tracker_db_path: Optional[str] = None,
    since: Optional[str] = None,
    batch: int = 500,
) -> Dict[str, int]:
    """Fill context_2.extracted_links for any frame that hasn't been
    scanned yet. Empty results are written as '[]' so re-runs skip
    them. Joins each frame to its parent context_1 row for window_name.

    `since` is an ISO-format timestamp filter on `frame_timestamp` —
    nightly passes only the recent window; ad-hoc CLI runs leave it
    None to backfill the whole table.

    Returns counts:
      {frames_scanned, frames_with_links, total_links_written}
    """
    path = _resolve_path(tracker_db_path)
    if not os.path.exists(path):
        return {"frames_scanned": 0, "frames_with_links": 0, "total_links_written": 0}

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        # Make sure the column exists (Phase A migration). If the tracker
        # DB hasn't seen the migration yet, give a clean error rather
        # than crashing somewhere weirder.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(context_2)").fetchall()}
        if "extracted_links" not in cols:
            return {
                "frames_scanned": 0, "frames_with_links": 0,
                "total_links_written": 0, "error": "extracted_links column missing",
            }

        params: List[Any] = []
        where = (
            "(extracted_links IS NULL OR extracted_links = '' "
            "OR extracted_links = '[]')"
        )
        if since:
            where += " AND frame_timestamp >= ?"
            params.append(since)

        # Cache window_name lookups across frames in the same segment.
        win_cache: Dict[str, str] = {}

        scanned = 0
        with_links = 0
        total_written = 0

        while True:
            rows = conn.execute(
                f"SELECT id, target_segment_id, raw_text "
                f"FROM context_2 WHERE {where} "
                f"ORDER BY frame_timestamp DESC LIMIT ?",
                params + [int(batch)],
            ).fetchall()
            if not rows:
                break

            for r in rows:
                segment_id = r["target_segment_id"]
                if segment_id not in win_cache:
                    seg = conn.execute(
                        "SELECT window_name FROM context_1 WHERE id = ?",
                        (segment_id,),
                    ).fetchone()
                    win_cache[segment_id] = (seg["window_name"] if seg else "") or ""
                links = extract_all(
                    r["raw_text"] or "",
                    win_cache.get(segment_id, ""),
                )
                conn.execute(
                    "UPDATE context_2 SET extracted_links = ? WHERE id = ?",
                    (json.dumps(links), r["id"]),
                )
                scanned += 1
                if links:
                    with_links += 1
                    total_written += len(links)

            conn.commit()
            # If we got fewer than batch, we're done — the WHERE clause
            # filters to unscanned only, and we just scanned them.
            if len(rows) < batch:
                break

        return {
            "frames_scanned": scanned,
            "frames_with_links": with_links,
            "total_links_written": total_written,
        }
    finally:
        conn.close()


def rollup_segment_links(
    tracker_db_path: Optional[str] = None,
    since: Optional[str] = None,
    batch: int = 200,
) -> Dict[str, int]:
    """For each context_1 segment whose segment_links is empty/null,
    aggregate its child frames' extracted_links and write a deduped
    list with dwell_frames counts.

    Output JSON shape per segment:
      [{url, kind, dwell_frames, sources: [...]}, ...]
    """
    path = _resolve_path(tracker_db_path)
    if not os.path.exists(path):
        return {"segments_scanned": 0, "segments_with_links": 0, "total_unique_links": 0}

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(context_1)").fetchall()}
        if "segment_links" not in cols:
            return {
                "segments_scanned": 0, "segments_with_links": 0,
                "total_unique_links": 0, "error": "segment_links column missing",
            }

        params: List[Any] = []
        where = (
            "(segment_links IS NULL OR segment_links = '' "
            "OR segment_links = '[]')"
        )
        if since:
            where += " AND timestamp_start >= ?"
            params.append(since)

        scanned = 0
        with_links = 0
        total_unique = 0

        while True:
            rows = conn.execute(
                f"SELECT id FROM context_1 WHERE {where} "
                f"ORDER BY timestamp_start DESC LIMIT ?",
                params + [int(batch)],
            ).fetchall()
            if not rows:
                break

            for r in rows:
                seg_id = r["id"]
                frames = conn.execute(
                    "SELECT extracted_links FROM context_2 "
                    "WHERE target_segment_id = ?",
                    (seg_id,),
                ).fetchall()

                # Aggregate: url -> {kind, dwell_frames, sources: set()}
                agg: Dict[str, Dict[str, Any]] = {}
                for f in frames:
                    raw = (f["extracted_links"] or "").strip()
                    if not raw:
                        continue
                    try:
                        items = json.loads(raw)
                    except Exception:
                        continue
                    for item in items or []:
                        url = item.get("url", "")
                        if not url:
                            continue
                        slot = agg.setdefault(url, {
                            "url": url,
                            "kind": item.get("kind", "other"),
                            "dwell_frames": 0,
                            "sources": set(),
                        })
                        slot["dwell_frames"] += 1
                        if item.get("source"):
                            slot["sources"].add(item["source"])

                serialized: List[Dict[str, Any]] = []
                for v in agg.values():
                    v["sources"] = sorted(v["sources"])
                    serialized.append(v)
                # Deterministic order: dwell desc, url asc.
                serialized.sort(key=lambda x: (-x["dwell_frames"], x["url"]))

                conn.execute(
                    "UPDATE context_1 SET segment_links = ? WHERE id = ?",
                    (json.dumps(serialized), seg_id),
                )
                scanned += 1
                if serialized:
                    with_links += 1
                    total_unique += len(serialized)

            conn.commit()
            if len(rows) < batch:
                break

        return {
            "segments_scanned": scanned,
            "segments_with_links": with_links,
            "total_unique_links": total_unique,
        }
    finally:
        conn.close()


def run_links_pass(
    tracker_db_path: Optional[str] = None,
    since: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for nightly + CLI: populate then roll up."""
    extracted = populate_extracted_links(tracker_db_path, since=since)
    rolled = rollup_segment_links(tracker_db_path, since=since)
    return {"extracted": extracted, "rolled_up": rolled}
