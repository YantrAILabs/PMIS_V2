"""
Recent-work aggregator — fuels the "Resume where you left off" panel.

Reads the tracker's context_1 segments over a recent window (default 2 h),
clusters them by a normalized topic key, and returns the top-K clusters by
total time spent. One line per cluster is exactly the kind of thing the user
wants to see when they return to the machine:

    "Researching memory topics — 34 min · Chrome, Claude"

Clustering priority (first non-empty wins, per segment):
  1. `anchor_node_id` — if pipeline_sync has already resolved it to a PMIS
     memory node, that's the cleanest topic identity.
  2. `anchor`          — tracker-segmenter's mid-level topic string.
  3. `context`         — broader topic string.
  4. `window_name` root — first 3 tokens as last-resort fallback.

Each cluster is annotated with:
  - total_minutes, segment_count, distinct_days (always 1 in the window but kept for parity)
  - platforms set (e.g. ["Chrome", "Claude"])
  - best display title (anchor text preferred over window root)
  - linked project_id / deliverable_id if segments have one (majority vote)
  - project_color key (hash of project_id) so widget can colorize the ribbon
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pmis.retrieval.recent_work")

TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")
_NON_WORD = re.compile(r"[^a-zA-Z0-9\s]+")
_COLLAPSE_WS = re.compile(r"\s+")


def _window_root(name: str) -> str:
    s = _NON_WORD.sub(" ", name or "")
    s = _COLLAPSE_WS.sub(" ", s).strip()
    return " ".join(s.split()[:3])


def _cluster_key(seg: Dict[str, Any]) -> str:
    """First non-empty identifier wins — so a real anchor beats a window root."""
    for field in ("anchor_node_id", "anchor", "context"):
        v = (seg.get(field) or "").strip()
        if v:
            return f"{field}:{v}"
    return f"window:{_window_root(seg.get('window_name') or '')}"


def get_recent_work(
    lookback_minutes: int = 120,
    limit: int = 3,
    tracker_db_path: str = TRACKER_DB,
) -> List[Dict[str, Any]]:
    """Return the top-K task clusters from the last N minutes of tracker data."""
    if not os.path.exists(tracker_db_path):
        return []

    cutoff = (datetime.now() - timedelta(minutes=lookback_minutes)).isoformat()
    conn = sqlite3.connect(tracker_db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT target_segment_id, window_name, platform, context, supercontext,
                      anchor, anchor_node_id, context_node_id, project_id, deliverable_id,
                      detailed_summary, target_segment_length_secs, timestamp_start
               FROM context_1
               WHERE timestamp_start >= ?
                 AND COALESCE(target_segment_length_secs, 0) >= 5
               ORDER BY timestamp_start DESC""",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    clusters: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        seg = dict(r)
        key = _cluster_key(seg)
        bucket = clusters.setdefault(key, {
            "key": key,
            "total_secs": 0,
            "segment_count": 0,
            "platforms": Counter(),
            "projects": Counter(),
            "deliverables": Counter(),
            "title_candidates": [],
            "summary_candidates": [],
            "last_seen": seg.get("timestamp_start") or "",
            "first_seen": seg.get("timestamp_start") or "",
        })
        secs = int(seg.get("target_segment_length_secs") or 0)
        bucket["total_secs"] += secs
        bucket["segment_count"] += 1
        if seg.get("platform"):
            bucket["platforms"][seg["platform"]] += secs
        if seg.get("project_id"):
            bucket["projects"][seg["project_id"]] += secs
        if seg.get("deliverable_id"):
            bucket["deliverables"][seg["deliverable_id"]] += secs
        for f in ("anchor", "context", "supercontext"):
            v = (seg.get(f) or "").strip()
            if v and v.lower() != "unclassified":
                bucket["title_candidates"].append(v)
        summ = (seg.get("detailed_summary") or "").strip()
        if summ and "Classification failed" not in summ:
            bucket["summary_candidates"].append(summ)
        ts = seg.get("timestamp_start") or ""
        if ts < bucket["first_seen"] or not bucket["first_seen"]:
            bucket["first_seen"] = ts
        if ts > bucket["last_seen"]:
            bucket["last_seen"] = ts

    # Pick display title — shortest-but-specific anchor text wins; fall back to
    # window root if nothing else is populated.
    out: List[Dict[str, Any]] = []
    for key, b in clusters.items():
        if b["title_candidates"]:
            # Most frequent, then shortest — tight titles read better
            title_counter = Counter(b["title_candidates"])
            title = sorted(title_counter.most_common(),
                            key=lambda kv: (-kv[1], len(kv[0])))[0][0]
        else:
            title = key.split(":", 1)[1] or "(untitled)"
            title = title.title() if title else "(untitled)"
        platforms_top = [p for p, _ in b["platforms"].most_common(3)]
        project_id = b["projects"].most_common(1)[0][0] if b["projects"] else ""
        deliverable_id = b["deliverables"].most_common(1)[0][0] if b["deliverables"] else ""
        preview = (b["summary_candidates"][0] if b["summary_candidates"] else "")[:160]
        out.append({
            "title": title,
            "total_seconds": b["total_secs"],
            "total_minutes": round(b["total_secs"] / 60.0, 1),
            "segment_count": b["segment_count"],
            "platforms": platforms_top,
            "project_id": project_id,
            "deliverable_id": deliverable_id,
            "preview": preview,
            "last_seen": b["last_seen"],
            "first_seen": b["first_seen"],
        })

    out.sort(key=lambda c: c["total_seconds"], reverse=True)
    return out[:limit]
