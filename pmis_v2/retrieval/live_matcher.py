"""
Live Matcher — Phase 1 realtime deliverable suggestion.

Semantic-only (cosine on Euclidean embedding of `detailed_summary`) so:
  - latency stays sub-200ms
  - fresh segments don't need stable Poincare positions (HGCN trains nightly)

Triangulated scoring (semantic × hyperbolic × temporal) remains the job of the
nightly `ProjectMatcher`. This module handles:

  1. `suggest_for_latest_segment()` — top-K deliverables for the most recent
     closed segment that passed a minimum-duration gate.
  2. `check_drift()` — for an active session bound to deliverable D, return
     True when the latest segment's semantic distance to D exceeds threshold
     AND the segment has accrued >= MIN_DRIFT_SECS of time.

No DB writes — the server endpoint decides what to do with the result.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger("pmis.live_matcher")

TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")
DEFAULT_MIN_SEGMENT_SECS = 300   # suggestion + drift both gated by 5 min
DEFAULT_DRIFT_THRESHOLD = 0.45   # cosine below this = drifted (tunable)
DEFAULT_TOP_K = 3
# Fix 1: pattern fast-path bypasses the 5-min gate — keyword hits fire
# immediately and the per-candidate score is capped at this ceiling so a
# future semantic match with real embeddings can still win.
PATTERN_MATCH_SCORE_CEIL = 0.85
PATTERN_MIN_SEGMENT_SECS = 30    # 30s sanity floor; below this, tracker noise
GOALS_YAML_REL = "productivity-tracker/config/goals.yaml"


@dataclass
class SegmentView:
    segment_id: str
    summary: str
    supercontext: str
    context: str
    anchor: str
    window_name: str
    platform: str
    length_secs: int
    timestamp_start: str
    timestamp_end: Optional[str]


@dataclass
class Suggestion:
    deliverable_id: str
    project_id: str
    deliverable_name: str
    score: float
    sc_node_id: str
    context_node_id: str
    source: str = "semantic"          # 'semantic' | 'pattern' | 'hybrid'
    matched_patterns: Optional[List[str]] = None


class LiveMatcher:
    def __init__(
        self,
        db,
        embedder,
        hyperparams: Optional[Dict[str, Any]] = None,
        tracker_db_path: str = TRACKER_DB,
    ):
        self.db = db
        self.embedder = embedder
        self.hp = hyperparams or {}
        self.tracker_db_path = tracker_db_path

        self.min_segment_secs = int(
            self.hp.get("live_matcher_min_segment_secs", DEFAULT_MIN_SEGMENT_SECS)
        )
        self.drift_threshold = float(
            self.hp.get("live_matcher_drift_threshold", DEFAULT_DRIFT_THRESHOLD)
        )
        self.top_k = int(self.hp.get("live_matcher_top_k", DEFAULT_TOP_K))
        self.pattern_min_secs = int(
            self.hp.get("live_matcher_pattern_min_segment_secs", PATTERN_MIN_SEGMENT_SECS)
        )
        self.pattern_score_ceil = float(
            self.hp.get("live_matcher_pattern_score_ceil", PATTERN_MATCH_SCORE_CEIL)
        )
        # goals.yaml patterns cached + reloaded on mtime change
        self._patterns_cache: Optional[Dict[str, List[str]]] = None
        self._patterns_mtime: float = 0.0

    # ------------------------------------------------------------------
    # Tracker DB reads (read-only)
    # ------------------------------------------------------------------

    def _tracker_conn(self) -> Optional[sqlite3.Connection]:
        if not Path(self.tracker_db_path).exists():
            return None
        conn = sqlite3.connect(self.tracker_db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def get_latest_segment(self, after_iso: Optional[str] = None) -> Optional[SegmentView]:
        """Most recent segment. If after_iso is given, restrict to segments
        whose timestamp_start is >= after_iso (i.e. started during the session).
        """
        conn = self._tracker_conn()
        if conn is None:
            return None
        try:
            if after_iso:
                row = conn.execute(
                    """SELECT target_segment_id, detailed_summary, supercontext, context,
                              anchor, window_name, platform, target_segment_length_secs,
                              timestamp_start, timestamp_end
                       FROM context_1
                       WHERE timestamp_start >= ?
                       ORDER BY timestamp_start DESC LIMIT 1""",
                    (after_iso,),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT target_segment_id, detailed_summary, supercontext, context,
                              anchor, window_name, platform, target_segment_length_secs,
                              timestamp_start, timestamp_end
                       FROM context_1
                       ORDER BY timestamp_start DESC LIMIT 1"""
                ).fetchone()
            if not row:
                return None
            return SegmentView(
                segment_id=row["target_segment_id"] or "",
                summary=row["detailed_summary"] or "",
                supercontext=row["supercontext"] or "",
                context=row["context"] or "",
                anchor=row["anchor"] or "",
                window_name=row["window_name"] or "",
                platform=row["platform"] or "",
                length_secs=int(row["target_segment_length_secs"] or 0),
                timestamp_start=row["timestamp_start"] or "",
                timestamp_end=row["timestamp_end"],
            )
        finally:
            conn.close()

    def get_segments_in_window(self, start_iso: str, end_iso: Optional[str]) -> List[SegmentView]:
        conn = self._tracker_conn()
        if conn is None:
            return []
        try:
            if end_iso:
                rows = conn.execute(
                    """SELECT target_segment_id, detailed_summary, supercontext, context,
                              anchor, window_name, platform, target_segment_length_secs,
                              timestamp_start, timestamp_end
                       FROM context_1
                       WHERE timestamp_start >= ? AND timestamp_start <= ?
                       ORDER BY timestamp_start ASC""",
                    (start_iso, end_iso),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT target_segment_id, detailed_summary, supercontext, context,
                              anchor, window_name, platform, target_segment_length_secs,
                              timestamp_start, timestamp_end
                       FROM context_1
                       WHERE timestamp_start >= ?
                       ORDER BY timestamp_start ASC""",
                    (start_iso,),
                ).fetchall()
            return [
                SegmentView(
                    segment_id=r["target_segment_id"] or "",
                    summary=r["detailed_summary"] or "",
                    supercontext=r["supercontext"] or "",
                    context=r["context"] or "",
                    anchor=r["anchor"] or "",
                    window_name=r["window_name"] or "",
                    platform=r["platform"] or "",
                    length_secs=int(r["target_segment_length_secs"] or 0),
                    timestamp_start=r["timestamp_start"] or "",
                    timestamp_end=r["timestamp_end"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_segment(self, seg: SegmentView) -> Optional[np.ndarray]:
        text = seg.summary or f"{seg.anchor}. {seg.context}. {seg.supercontext}"
        text = text.strip()
        if not text:
            return None
        try:
            return self.embedder.embed_text(text)
        except Exception as e:
            logger.warning("Embed failed for segment %s: %s", seg.segment_id, e)
            return None

    def _candidate_embedding(self, cand: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prefer anchor_node, fall back to context_node, then sc_node."""
        for key in ("anchor_node_id", "context_node_id", "sc_node_id"):
            nid = cand.get(key)
            if not nid:
                continue
            try:
                embs = self.db.get_embeddings(nid)
                eu = embs.get("euclidean") if embs else None
                if eu is not None and getattr(eu, "size", 0) > 0:
                    return eu
            except Exception:
                continue
        return None

    @staticmethod
    def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        if a.shape != b.shape:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        cos = float(np.dot(a, b) / (na * nb))
        return max(0.0, min(1.0, cos))

    # ------------------------------------------------------------------
    # Public — suggestion
    # ------------------------------------------------------------------

    def suggest(
        self,
        after_iso: Optional[str] = None,
        min_secs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return top-K deliverable suggestions for the most recent segment.

        Two-stage pipeline (Fix 1 + Fix 2 combined):
          1. **Pattern fast-path** — keyword-match the segment's window_name
             + detailed_summary against goals.yaml match_patterns. Fires
             after 30s (not 5 min) because deterministic keyword hits are
             high-confidence signals. Score capped at PATTERN_MATCH_SCORE_CEIL
             so a great semantic match can still beat a weak pattern hit.
          2. **Semantic pass** — cosine on embedder(summary) vs deliverable
             CTX/ANC embeddings. Gated by 5 min as before. Blends with
             pattern scores; when both hit, source becomes 'hybrid'.

        Response shape:
            {"ready": bool, "reason": str, "segment": {...} | None,
             "suggestions": [{deliverable_id, ..., source, matched_patterns}]}
        """
        semantic_min = min_secs if min_secs is not None else self.min_segment_secs
        seg = self.get_latest_segment(after_iso=after_iso)
        if seg is None:
            return {"ready": False, "reason": "no_segments",
                     "segment": None, "suggestions": []}

        if seg.length_secs < self.pattern_min_secs:
            return {
                "ready": False,
                "reason": f"segment_too_short ({seg.length_secs}s < {self.pattern_min_secs}s)",
                "segment": _seg_to_dict(seg),
                "suggestions": [],
            }

        candidates = self.db.get_active_deliverable_candidates()
        if not candidates:
            return {
                "ready": False,
                "reason": "no_active_deliverables",
                "segment": _seg_to_dict(seg),
                "suggestions": [],
            }

        # Pattern fast-path — always available as soon as pattern_min_secs is met
        pattern_hits = self._pattern_match(seg, candidates)

        # Semantic path — only kicks in once the segment crosses the 5-min gate
        semantic_hits: Dict[str, Suggestion] = {}
        if seg.length_secs >= semantic_min:
            seg_emb = self._embed_segment(seg)
            if seg_emb is not None:
                semantic_hits = self._semantic_match(seg_emb, candidates)

        # Merge — same deliverable from both sources becomes 'hybrid' and
        # keeps the max score (bounded below the ceiling only for pure pattern).
        merged: Dict[str, Suggestion] = {}
        for did, p in pattern_hits.items():
            merged[did] = p
        for did, s in semantic_hits.items():
            existing = merged.get(did)
            if existing is None:
                merged[did] = s
            else:
                new_score = max(existing.score, s.score)
                merged[did] = Suggestion(
                    deliverable_id=did,
                    project_id=existing.project_id or s.project_id,
                    deliverable_name=existing.deliverable_name or s.deliverable_name,
                    score=new_score,
                    sc_node_id=existing.sc_node_id or s.sc_node_id,
                    context_node_id=existing.context_node_id or s.context_node_id,
                    source="hybrid",
                    matched_patterns=existing.matched_patterns,
                )

        ranked = sorted(merged.values(), key=lambda s: s.score, reverse=True)[: self.top_k]

        if not ranked:
            reason = ("waiting_for_semantic_gate"
                       if seg.length_secs < semantic_min else "no_matches")
            return {
                "ready": False,
                "reason": reason,
                "segment": _seg_to_dict(seg),
                "suggestions": [],
            }

        return {
            "ready": True,
            "reason": "ok",
            "segment": _seg_to_dict(seg),
            "suggestions": [s.__dict__ for s in ranked],
        }

    # ------------------------------------------------------------------
    # Pattern fast-path
    # ------------------------------------------------------------------

    def _pattern_match(
        self, seg: SegmentView, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Suggestion]:
        """Match goals.yaml keywords against segment title + summary.
        Returns {deliverable_id: Suggestion} — one best entry per deliverable."""
        patterns_by_deliv = self._load_patterns()
        if not patterns_by_deliv:
            return {}

        haystack = " ".join([
            seg.window_name or "", seg.summary or "",
            seg.context or "", seg.anchor or "",
        ]).lower()
        if not haystack.strip():
            return {}

        best: Dict[str, Suggestion] = {}
        # Build a deliverable → candidate-metadata lookup so we can attach
        # names/project_ids without extra DB queries
        cand_ix: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            did = c.get("deliverable_id", "")
            if did and did not in cand_ix:
                cand_ix[did] = c

        for did, patterns in patterns_by_deliv.items():
            if did not in cand_ix:
                continue  # deliverable not in active DB set
            matched: List[str] = []
            for pat in patterns:
                if not pat:
                    continue
                # Word-boundary match against lowercased pattern
                needle = pat.lower()
                if re.search(r"\b" + re.escape(needle) + r"\b", haystack):
                    matched.append(pat)
            if not matched:
                continue
            # Score scales with fraction of patterns hit, capped at ceiling
            score = min(
                self.pattern_score_ceil,
                0.55 + 0.10 * min(len(matched), 3),
            )
            cand = cand_ix[did]
            best[did] = Suggestion(
                deliverable_id=did,
                project_id=cand.get("project_id", ""),
                deliverable_name=cand.get("name", ""),
                score=score,
                sc_node_id=cand.get("sc_node_id", ""),
                context_node_id=cand.get("context_node_id", ""),
                source="pattern",
                matched_patterns=matched[:5],
            )
        return best

    def _semantic_match(
        self, seg_emb: np.ndarray, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Suggestion]:
        """Cosine similarity path (extracted from the old suggest body)."""
        best: Dict[str, Suggestion] = {}
        for cand in candidates:
            emb = self._candidate_embedding(cand)
            if emb is None:
                continue
            score = self._cosine(seg_emb, emb)
            did = cand.get("deliverable_id", "")
            if not did:
                continue
            existing = best.get(did)
            if existing is None or score > existing.score:
                best[did] = Suggestion(
                    deliverable_id=did,
                    project_id=cand.get("project_id", ""),
                    deliverable_name=cand.get("name", ""),
                    score=score,
                    sc_node_id=cand.get("sc_node_id", ""),
                    context_node_id=cand.get("context_node_id", ""),
                    source="semantic",
                )
        return best

    def _load_patterns(self) -> Dict[str, List[str]]:
        """Load goals.yaml → {deliverable_id: [keyword, ...]}. Reloads if the
        file's mtime has changed. Merges project-level match_patterns into
        every deliverable under that project."""
        path = Path(__file__).resolve().parents[2] / GOALS_YAML_REL
        if not path.exists():
            return {}
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return self._patterns_cache or {}
        if self._patterns_cache is not None and mtime == self._patterns_mtime:
            return self._patterns_cache

        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except Exception as e:
            logger.warning("Failed to load goals.yaml: %s", e)
            return self._patterns_cache or {}

        result: Dict[str, List[str]] = {}
        for goal in doc.get("goals") or []:
            for proj in goal.get("projects") or []:
                proj_pats = proj.get("match_patterns") or []
                dp = proj.get("deliverable_patterns") or {}
                for did, dp_list in dp.items():
                    # Merge per-deliverable patterns with project-level as a
                    # weaker fallback. Dedup while preserving order.
                    combined = list(dp_list or []) + list(proj_pats)
                    seen = set()
                    ordered: List[str] = []
                    for p in combined:
                        if p and p not in seen:
                            seen.add(p)
                            ordered.append(p)
                    result[did] = ordered

        self._patterns_cache = result
        self._patterns_mtime = mtime
        return result

    # ------------------------------------------------------------------
    # Public — drift
    # ------------------------------------------------------------------

    def check_drift(
        self,
        bound_deliverable_id: str,
        session_start_iso: str,
    ) -> Dict[str, Any]:
        """Return drift status for an active session.

        Response shape:
            {"drifted": bool, "similarity": float, "latest_segment": {...} | None,
             "drift_threshold": float, "segment_long_enough": bool}
        """
        seg = self.get_latest_segment(after_iso=session_start_iso)
        if seg is None:
            return {
                "drifted": False,
                "similarity": 1.0,
                "latest_segment": None,
                "drift_threshold": self.drift_threshold,
                "segment_long_enough": False,
            }

        if seg.length_secs < self.min_segment_secs:
            return {
                "drifted": False,
                "similarity": 1.0,
                "latest_segment": _seg_to_dict(seg),
                "drift_threshold": self.drift_threshold,
                "segment_long_enough": False,
            }

        deliverable = self.db.get_deliverable(bound_deliverable_id)
        if not deliverable:
            return {
                "drifted": False,
                "similarity": 0.0,
                "latest_segment": _seg_to_dict(seg),
                "drift_threshold": self.drift_threshold,
                "segment_long_enough": True,
                "reason": "deliverable_not_found",
            }

        cand = {
            "anchor_node_id": "",
            "context_node_id": deliverable.get("context_node_id", ""),
            "sc_node_id": deliverable.get("project_sc_node_id", ""),
        }
        cand_emb = self._candidate_embedding(cand)
        seg_emb = self._embed_segment(seg)
        sim = self._cosine(seg_emb, cand_emb) if cand_emb is not None and seg_emb is not None else 0.0

        return {
            "drifted": sim < self.drift_threshold,
            "similarity": sim,
            "latest_segment": _seg_to_dict(seg),
            "drift_threshold": self.drift_threshold,
            "segment_long_enough": True,
        }


def _seg_to_dict(seg: SegmentView) -> Dict[str, Any]:
    return {
        "segment_id": seg.segment_id,
        "summary": seg.summary,
        "supercontext": seg.supercontext,
        "context": seg.context,
        "anchor": seg.anchor,
        "window_name": seg.window_name,
        "platform": seg.platform,
        "length_secs": seg.length_secs,
        "timestamp_start": seg.timestamp_start,
        "timestamp_end": seg.timestamp_end,
    }
