"""
Phase 4b — PatternMiner.

Scans tracker segments to surface recurring task shapes per deliverable.
When a shape crosses a repetition threshold AND its associated anchors have
sufficient value_score, it becomes a *harness candidate* — a suggestion for
PMIS to spin up an automation bundle.

A "task shape" is identified by:
  (platform, normalized_window_root, anchor_cluster)

where `normalized_window_root` is the first 2–3 meaningful tokens of the
window title (strips app suffixes, trailing separators, user names).

Rules (tunable via hyperparams):
  - min_reps (default 5) — how many segment matches before we suggest
  - min_total_minutes (default 10) — prevents single-glance false positives
  - min_anchor_value_score (default 0.05) — skip low-value clusters
  - lookback_days (default 14) — recent-only signal
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("pmis.harness.pattern_miner")

TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")

# Tokens to strip from window titles when computing the normalized root.
_STRIP_TRAILING = re.compile(r"\s*[-–—|·]\s*(?:Google\s*Chrome|Safari|PMIS Wiki|Your Chrome|Rohit).*$", re.IGNORECASE)
_NON_WORD = re.compile(r"[^a-zA-Z0-9\s]+")
_COLLAPSE_WS = re.compile(r"\s+")


@dataclass
class PatternCandidate:
    deliverable_id: str
    platform: str
    window_root: str
    anchor_node_ids: List[str] = field(default_factory=list)
    reps: int = 0
    total_seconds: int = 0
    distinct_days: int = 0
    avg_value_score: float = 0.0
    max_value_score: float = 0.0
    suggested_title: str = ""
    pattern_signature: str = ""
    sample_window_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["total_minutes"] = round(self.total_seconds / 60.0, 1)
        return out


class PatternMiner:
    def __init__(self, db, hyperparams: Optional[Dict[str, Any]] = None,
                 tracker_db_path: str = TRACKER_DB):
        self.db = db
        self.hp = hyperparams or {}
        self.tracker_db_path = tracker_db_path
        self.min_reps = int(self.hp.get("pattern_miner_min_reps", 5))
        self.min_total_minutes = float(self.hp.get("pattern_miner_min_total_minutes", 10))
        self.min_value_score = float(self.hp.get("pattern_miner_min_value_score", 0.05))
        self.lookback_days = int(self.hp.get("pattern_miner_lookback_days", 14))
        self.sample_limit = int(self.hp.get("pattern_miner_sample_limit", 3))

    # ------------------------------------------------------------------

    def find_candidates(
        self,
        deliverable_id: Optional[str] = None,
    ) -> List[PatternCandidate]:
        """Return pattern candidates. If deliverable_id is provided, scope to
        that deliverable's anchor universe. Otherwise scan all active
        deliverables."""
        if deliverable_id:
            scopes = [(deliverable_id, self._anchor_universe(deliverable_id))]
        else:
            with self.db._connect() as conn:
                rows = conn.execute(
                    "SELECT id FROM deliverables WHERE status='active'"
                ).fetchall()
            scopes = [(r["id"], self._anchor_universe(r["id"])) for r in rows]

        if not scopes:
            return []

        segments = self._load_recent_segments()
        if not segments:
            return []

        # Build value_score lookup for all anchors mentioned
        all_anchor_ids = {s["anchor_node_id"] for s in segments if s.get("anchor_node_id")}
        value_map = self._fetch_value_scores(all_anchor_ids)

        results: List[PatternCandidate] = []
        for did, anchor_universe in scopes:
            if not anchor_universe:
                continue
            # Filter segments to ones whose anchor is in this deliverable's universe
            relevant = [s for s in segments if s.get("anchor_node_id") in anchor_universe]
            if len(relevant) < self.min_reps:
                continue
            results.extend(self._group_and_rank(did, relevant, value_map))

        results.sort(key=lambda c: (c.reps * c.avg_value_score), reverse=True)
        return results

    # ------------------------------------------------------------------

    def _anchor_universe(self, deliverable_id: str) -> set:
        """All memory_node ids reachable from this deliverable: its
        context_node_id, listed anchor_node_ids, and anchors that are children
        of that context."""
        universe: set = set()
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT context_node_id, anchor_node_ids FROM deliverables WHERE id = ?",
                (deliverable_id,),
            ).fetchone()
            if not row:
                return universe
            ctx_id = row["context_node_id"] or ""
            if ctx_id:
                universe.add(ctx_id)
                # children of this context
                child_rows = conn.execute(
                    """SELECT source_id FROM relations
                       WHERE relation_type='child_of' AND target_id = ?""",
                    (ctx_id,),
                ).fetchall()
                universe.update(r["source_id"] for r in child_rows)
            try:
                import json as _json
                for aid in _json.loads(row["anchor_node_ids"] or "[]"):
                    if aid:
                        universe.add(aid)
            except Exception:
                pass
        return universe

    def _load_recent_segments(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.tracker_db_path):
            return []
        cutoff = (datetime.now() - timedelta(days=self.lookback_days)).isoformat()
        conn = sqlite3.connect(self.tracker_db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """SELECT target_segment_id, window_name, platform, anchor_node_id,
                          context_node_id, target_segment_length_secs, timestamp_start
                   FROM context_1
                   WHERE timestamp_start >= ? AND anchor_node_id != ''
                   ORDER BY timestamp_start DESC""",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def _fetch_value_scores(self, node_ids) -> Dict[str, float]:
        if not node_ids:
            return {}
        ids_list = list(node_ids)
        result: Dict[str, float] = {}
        with self.db._connect() as conn:
            CHUNK = 500
            for i in range(0, len(ids_list), CHUNK):
                chunk = ids_list[i:i + CHUNK]
                ph = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT id, value_score FROM memory_nodes WHERE id IN ({ph})",
                    chunk,
                ).fetchall()
                for r in rows:
                    result[r["id"]] = float(r["value_score"] or 0.0)
        return result

    def _group_and_rank(
        self,
        deliverable_id: str,
        segments: List[Dict[str, Any]],
        value_map: Dict[str, float],
    ) -> List[PatternCandidate]:
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for seg in segments:
            platform = (seg.get("platform") or "").strip()
            root = _normalize_window(seg.get("window_name") or "")
            if not platform and not root:
                continue
            groups[(platform, root)].append(seg)

        candidates: List[PatternCandidate] = []
        for (platform, root), segs in groups.items():
            if len(segs) < self.min_reps:
                continue
            total_secs = sum(int(s.get("target_segment_length_secs") or 0) for s in segs)
            if total_secs < self.min_total_minutes * 60:
                continue
            anchor_ids = [s["anchor_node_id"] for s in segs if s.get("anchor_node_id")]
            value_scores = [value_map.get(a, 0.0) for a in anchor_ids]
            avg_v = sum(value_scores) / len(value_scores) if value_scores else 0.0
            max_v = max(value_scores) if value_scores else 0.0
            if avg_v < self.min_value_score:
                continue

            distinct_days = len({(s.get("timestamp_start") or "")[:10] for s in segs})
            top_anchors = [a for a, _ in Counter(anchor_ids).most_common(3)]
            sample_windows = [s.get("window_name", "") for s in segs[: self.sample_limit]]
            signature = _pattern_signature(platform, root, top_anchors)
            title = _suggest_title(platform, root)

            candidates.append(PatternCandidate(
                deliverable_id=deliverable_id,
                platform=platform,
                window_root=root,
                anchor_node_ids=top_anchors,
                reps=len(segs),
                total_seconds=total_secs,
                distinct_days=distinct_days,
                avg_value_score=round(avg_v, 3),
                max_value_score=round(max_v, 3),
                suggested_title=title,
                pattern_signature=signature,
                sample_window_names=sample_windows,
            ))
        return candidates


# ─── helpers ──────────────────────────────────────────────────────────

def _normalize_window(name: str) -> str:
    s = _STRIP_TRAILING.sub("", name or "")
    s = _NON_WORD.sub(" ", s)
    s = _COLLAPSE_WS.sub(" ", s).strip()
    # Keep first 3 tokens to form a stable root
    tokens = s.split(" ")[:3]
    return " ".join(tokens).lower()


def _pattern_signature(platform: str, root: str, top_anchors: List[str]) -> str:
    anchor_frag = "+".join(a[:6] for a in top_anchors[:2]) if top_anchors else "noanc"
    return f"{platform.lower().replace(' ','_')}|{root.replace(' ','_')}|{anchor_frag}"


def _suggest_title(platform: str, root: str) -> str:
    root_clean = (root or "").title().strip()
    if not root_clean:
        return f"Automate recurring {platform} tasks"
    return f"Automate recurring {platform} · {root_clean}"
