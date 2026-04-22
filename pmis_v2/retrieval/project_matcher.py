"""
Project Matcher — route work anchors to active deliverables.

Each day's activity-derived ANC memory nodes are scored against every anchor
under every active deliverable (falling back to the deliverable itself when no
anchors are attached).

Scoring is triangulated:
    combined = semantic^alpha * hyperbolic^beta * temporal^gamma

Product (not sum) means any near-zero component kills the match — a
semantically-close item in a closed deliverable still gets 0, and a subtree
neighbor with unrelated content also gets 0. Only triple-agreement survives.

Matches above matcher_tau_high are written to project_work_match_log with
is_correct=-1 (pending user confirm via thumbs up/down on the Goals page).
The top matcher_top_n_log candidates per work anchor are always written as a
diagnostic trace regardless of threshold, so false matches are debuggable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.poincare import poincare_distance

logger = logging.getLogger("pmis.project_matcher")


@dataclass
class MatchScore:
    candidate: Dict[str, Any]
    semantic: float
    hyperbolic: float
    temporal: float
    combined: float


class ProjectMatcher:
    """Scores today's activity anchors against active deliverables."""

    def __init__(self, db, hyperparams: Dict[str, Any]):
        self.db = db
        self.hp = hyperparams

        self.alpha = float(self.hp.get("matcher_alpha", 1.0))
        self.beta = float(self.hp.get("matcher_beta", 1.0))
        self.gamma = float(self.hp.get("matcher_gamma", 1.0))
        self.tau_high = float(self.hp.get("matcher_tau_high", 0.75))
        self.tau_low = float(self.hp.get("matcher_tau_low", 0.45))
        self.top_n_log = int(self.hp.get("matcher_top_n_log", 5))
        self.decay_days = float(self.hp.get("matcher_temporal_decay_days", 14))
        self.poincare_max = float(self.hp.get("matcher_poincare_max_dist", 4.0))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, target_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Match every activity anchor from target_date against active
        deliverables; write results to project_work_match_log.

        Two-stage strategy (2026-04-20, Phase 2):
          1. **Session-tagged inherit** — anchors whose source segments fell
             inside a user-tagged work_session get the project attached
             directly at combined=1.0, is_correct=1, source='session_tag'.
             No semantic matching, no review queue entry.
          2. **Semantic match for the rest** — the legacy scoring path, but
             now with explicit auto-confirm vs pending-review buckets:
                combined >= tau_high → is_correct=1  (auto-confirmed)
                tau_low <= combined  < tau_high → is_correct=-1 (review)
                combined <  tau_low                → no surfacing
             Source is set to 'semantic'.

        Manual consolidation (Phase 4) writes its own rows with
        source='manual_consolidation' and never passes through this code path.
        """
        if target_date is None:
            target_date = datetime.now().date().isoformat()

        work_anchors = self.db.get_activity_anchors_for_date(target_date)
        if not work_anchors:
            logger.info("No activity anchors to match for %s", target_date)
            return []

        # Stage 1: anchors whose segments have project_id set in
        # activity_time_log are user-tagged — inherit without scoring.
        tagged_map = self._tagged_project_for_anchors(
            [a["id"] for a in work_anchors]
        )
        actions: List[Dict[str, Any]] = []
        untagged_anchors: List[Dict[str, Any]] = []
        for anchor in work_anchors:
            pid = tagged_map.get(anchor["id"])
            if pid:
                actions.extend(self._log_session_tag_match(anchor, pid))
            else:
                untagged_anchors.append(anchor)

        # Stage 2: semantic matching for the untagged remainder.
        if untagged_anchors:
            candidates = self.db.get_active_deliverable_candidates()
            if candidates:
                cand_embeds = self._prefetch_candidate_embeddings(candidates)
                candidates = [c for c, e in zip(candidates, cand_embeds) if e is not None]
                cand_embeds = [e for e in cand_embeds if e is not None]
                if candidates:
                    for anchor in untagged_anchors:
                        action_rows = self._match_one(anchor, candidates, cand_embeds)
                        actions.extend(action_rows)
                else:
                    logger.warning("All deliverable candidates missing embeddings — skipping")
            else:
                logger.info("No active deliverable candidates to match against")

        logger.info(
            "Matcher: %d work anchors (%d tagged / %d semantic) → %d log rows",
            len(work_anchors), len(work_anchors) - len(untagged_anchors),
            len(untagged_anchors), len(actions),
        )
        return actions

    def _tagged_project_for_anchors(
        self, anchor_ids: List[str]
    ) -> Dict[str, str]:
        """Return {anchor_id: project_id} for anchors whose activity_time_log
        rows carry a project_id (populated by daily_activity_merge when a
        segment fell in a user-tagged work_session).

        If an anchor has multiple distinct project_ids across its segments
        (shouldn't happen given the cluster-level tagging, but defensive),
        we pick the most common.
        """
        if not anchor_ids:
            return {}
        import sqlite3
        from collections import Counter
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        placeholders = ",".join(["?"] * len(anchor_ids))
        rows = conn.execute(
            f"""
            SELECT memory_node_id, project_id
            FROM activity_time_log
            WHERE memory_node_id IN ({placeholders})
              AND project_id != '' AND project_id IS NOT NULL
            """,
            anchor_ids,
        ).fetchall()
        conn.close()

        votes: Dict[str, Counter] = {}
        for r in rows:
            votes.setdefault(r["memory_node_id"], Counter())[r["project_id"]] += 1
        return {nid: c.most_common(1)[0][0] for nid, c in votes.items()}

    def _log_session_tag_match(
        self, work_anchor: Dict[str, Any], project_id: str
    ) -> List[Dict[str, Any]]:
        """Write a session-tag match row with combined=1.0, is_correct=1,
        source='session_tag'. Zero semantic scoring.

        We still look up the deliverable (if any single deliverable is most
        appropriate) by scoring semantically within this project's
        deliverables only — this keeps time-assignment granularity.
        """
        candidates = self.db.get_active_deliverable_candidates()
        project_candidates = [c for c in candidates if c.get("project_id") == project_id]
        best_deliverable_id = ""
        best_sc = ""
        best_ctx = ""
        best_anchor = ""
        if project_candidates:
            # Pick highest semantic candidate within the project; if
            # embeddings missing just take the first.
            embeds = self._prefetch_candidate_embeddings(project_candidates)
            work_embeds = self.db.get_embeddings(work_anchor["id"])
            work_eu = work_embeds.get("euclidean")
            best_score = -1.0
            for cand, cand_emb in zip(project_candidates, embeds):
                if cand_emb is None:
                    continue
                sem = self._semantic_score(work_eu, cand_emb["euclidean"])
                if sem > best_score:
                    best_score = sem
                    best_deliverable_id = cand.get("deliverable_id", "")
                    best_sc = cand.get("sc_node_id", "")
                    best_ctx = cand.get("context_node_id", "")
                    best_anchor = cand.get("anchor_node_id", "")
            if best_deliverable_id == "" and project_candidates:
                c = project_candidates[0]
                best_deliverable_id = c.get("deliverable_id", "")
                best_sc = c.get("sc_node_id", "")
                best_ctx = c.get("context_node_id", "")
                best_anchor = c.get("anchor_node_id", "")

        mid = self.db.log_match({
            "segment_id": work_anchor["id"],
            "project_id": project_id,
            "deliverable_id": best_deliverable_id,
            "sc_node_id": best_sc,
            "context_node_id": best_ctx,
            "anchor_node_id": best_anchor,
            "semantic_score": 1.0,
            "hyperbolic_score": 1.0,
            "combined_match_pct": 1.0,
            "match_method": "session_tag",
            "work_description": (work_anchor.get("content") or "")[:500],
            "worker_type": "activity_merge",
            "time_mins": 0,
            "is_correct": 1,
            "source": "session_tag",
        })
        return [{
            "action": "project_match",
            "match_id": mid,
            "rank": 0,
            "work_node_id": work_anchor["id"],
            "deliverable_id": best_deliverable_id,
            "combined": 1.0,
            "source": "session_tag",
            "surfaced": True,
        }]

    # ------------------------------------------------------------------
    # Per-anchor matching
    # ------------------------------------------------------------------

    def _match_one(
        self,
        work_anchor: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        cand_embeds: List[Dict[str, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """Score one work anchor against all candidates; log the top-N and
        return action dicts for caller."""

        embeds = self.db.get_embeddings(work_anchor["id"])
        work_eu = embeds.get("euclidean")
        work_hyp = embeds.get("hyperbolic")
        if work_eu is None:
            return []

        scores: List[MatchScore] = []
        for cand, cand_emb in zip(candidates, cand_embeds):
            sem = self._semantic_score(work_eu, cand_emb["euclidean"])
            hyp = self._hyperbolic_score(work_hyp, cand_emb.get("hyperbolic"))
            tmp = self._temporal_score(cand.get("deadline", ""))
            combined = (
                (sem ** self.alpha) * (hyp ** self.beta) * (tmp ** self.gamma)
            )
            scores.append(
                MatchScore(cand, sem, hyp, tmp, float(combined))
            )

        scores.sort(key=lambda s: s.combined, reverse=True)
        top = scores[: self.top_n_log]

        actions: List[Dict[str, Any]] = []
        for rank, s in enumerate(top):
            if s.combined < self.tau_low and rank > 0:
                break  # don't even log rank>0 below tau_low
            mid = self._log_match(work_anchor, s, rank)
            actions.append({
                "action": "project_match",
                "match_id": mid,
                "rank": rank,
                "work_node_id": work_anchor["id"],
                "deliverable_id": s.candidate["deliverable_id"],
                "anchor_node_id": s.candidate.get("anchor_node_id", ""),
                "combined": s.combined,
                "semantic": s.semantic,
                "hyperbolic": s.hyperbolic,
                "temporal": s.temporal,
                "surfaced": s.combined >= self.tau_high and rank == 0,
            })

        return actions

    # ------------------------------------------------------------------
    # Component scores
    # ------------------------------------------------------------------

    @staticmethod
    def _semantic_score(a: np.ndarray, b: Optional[np.ndarray]) -> float:
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 0.0
        if a.shape != b.shape:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        # cosine → [-1, 1] → clip to [0, 1] (negatives are uninformative here)
        cos = float(np.dot(a, b) / (na * nb))
        return max(0.0, min(1.0, cos))

    def _hyperbolic_score(
        self, a: Optional[np.ndarray], b: Optional[np.ndarray]
    ) -> float:
        # Missing hyperbolic embeddings shouldn't zero the whole product —
        # return neutral 1.0 so the decision falls to semantic + temporal.
        # This also covers brand-new work anchors not yet seen by HGCN.
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 1.0
        if a.shape != b.shape:
            return 1.0
        try:
            d = poincare_distance(a.astype(np.float64), b.astype(np.float64))
        except Exception:
            return 1.0
        if not np.isfinite(d):
            return 0.0
        return max(0.0, 1.0 - (d / self.poincare_max))

    def _temporal_score(self, deadline: str) -> float:
        """1.0 within deadline; exponential decay afterwards. Empty deadline
        treated as open-ended → 1.0."""
        if not deadline:
            return 1.0
        try:
            # Accept YYYY-MM-DD or full ISO.
            due = datetime.fromisoformat(deadline)
        except ValueError:
            try:
                due = datetime.strptime(deadline, "%Y-%m-%d")
            except ValueError:
                return 1.0
        now = datetime.now()
        if now <= due:
            return 1.0
        overdue_days = (now - due).total_seconds() / 86400.0
        return float(np.exp(-overdue_days / max(self.decay_days, 1.0)))

    # ------------------------------------------------------------------
    # Candidate embedding prefetch
    # ------------------------------------------------------------------

    def _prefetch_candidate_embeddings(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Optional[Dict[str, np.ndarray]]]:
        """Load embeddings for each candidate. Prefer anchor_node_id; fall
        back to context_node_id, then sc_node_id."""
        out: List[Optional[Dict[str, np.ndarray]]] = []
        for c in candidates:
            node_id = (
                c.get("anchor_node_id")
                or c.get("context_node_id")
                or c.get("sc_node_id")
            )
            if not node_id:
                out.append(None)
                continue
            emb = self.db.get_embeddings(node_id)
            if emb.get("euclidean") is None:
                out.append(None)
                continue
            out.append(emb)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _log_match(
        self, work_anchor: Dict[str, Any], s: MatchScore, rank: int
    ) -> str:
        """Log a semantic match row.

        is_correct:
          1  → auto-confirmed (combined >= tau_high, rank 0)
          -1 → pending review  (tau_low <= combined < tau_high, rank 0)
          -1 → not surfaced    (rank > 0; kept for diagnostics only)

        source: 'semantic' for all rows written here. The review queue UI
        filters by `source='semantic' AND is_correct=-1 AND rank==0 via
        match_method` to surface only top candidates per anchor.
        """
        is_rank_zero = rank == 0
        auto_confirmed = is_rank_zero and s.combined >= self.tau_high
        method = "triangulated_v1"
        if not is_rank_zero:
            method = f"{method}_runnerup{rank}"
        elif not auto_confirmed:
            method = f"{method}_gated"
        is_correct = 1 if auto_confirmed else -1
        return self.db.log_match({
            "segment_id": work_anchor["id"],
            "project_id": s.candidate["project_id"],
            "deliverable_id": s.candidate["deliverable_id"],
            "sc_node_id": s.candidate.get("sc_node_id", ""),
            "context_node_id": s.candidate.get("context_node_id", ""),
            "anchor_node_id": s.candidate.get("anchor_node_id", ""),
            "semantic_score": s.semantic,
            "hyperbolic_score": s.hyperbolic,
            "combined_match_pct": s.combined,
            "match_method": method,
            "work_description": (work_anchor.get("content") or "")[:500],
            "worker_type": "activity_merge",
            "time_mins": 0,
            "is_correct": is_correct,
            "source": "semantic",
        })
