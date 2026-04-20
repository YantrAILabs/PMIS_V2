"""
Phase 3 — unified value_score on every memory_node.

  value_score(n) = w_g·G(n) + w_f·F(n) + w_u·U(n) + w_r·R(n)

Components (all bounded roughly in [0,1] except F which is in [-1,1] internally):

  G(n) — goal achievement ratio
         sum(goal.weight * link.weight for linked achieved goals)
         / sum(goal.weight * link.weight for all linked goals)
         Propagates UP the tree with 0.5 attenuation per hop, so an anchor
         contributing to an achieved goal raises its parent context's G too.

  F(n) — feedback, tanh-bounded:
         F = tanh(sum over f in feedback(n) of sign(polarity)*strength*exp(-dt/τ_f))
         positive = +1, negative = -1, correction = 0 (treated as neutral).
         τ_f = value_feedback_halflife_days (default 30). Floored at
         value_feedback_floor (default 0.2) so ancient endorsements still count.

  U(n) — log-normalized retrieval count in trailing window:
         U = log(1 + retrievals_30d(n)) / log(1 + max_j retrievals_30d(j))

  R(n) — recency exponential decay:
         R = exp(-age_days / τ_r); τ_r = value_recency_tau_days (default 60)

Writes results into memory_nodes.value_score + component columns. Meant to be
called nightly (Pass 9) or via POST /api/value/recompute. Display in wiki +
brief retrieval should read the materialized columns, not recompute.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("pmis.value_score")


@dataclass
class ValueComponents:
    node_id: str
    G: float
    F: float  # internal [-1,1]
    U: float
    R: float
    score: float  # composite, clipped at value_display_clip for display
    redflag: bool
    feedback_raw_sum: float = 0.0

    def to_row(self) -> Tuple:
        return (
            float(self.score),
            float(self.G),
            float(self.F),
            float(self.U),
            float(self.R),
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            self.node_id,
        )


class ValueScoreCalculator:
    def __init__(self, db, hyperparams: Optional[Dict[str, Any]] = None):
        self.db = db
        self.hp = hyperparams or {}
        self.w_g = float(self.hp.get("value_weight_goal", 0.4))
        self.w_f = float(self.hp.get("value_weight_feedback", 0.3))
        self.w_u = float(self.hp.get("value_weight_usage", 0.2))
        self.w_r = float(self.hp.get("value_weight_recency", 0.1))
        self.fb_halflife = float(self.hp.get("value_feedback_halflife_days", 30))
        self.fb_floor = float(self.hp.get("value_feedback_floor", 0.2))
        self.recency_tau = float(self.hp.get("value_recency_tau_days", 60))
        self.usage_window = float(self.hp.get("value_usage_window_days", 30))
        self.goal_propagation = float(self.hp.get("value_goal_propagation", 0.5))
        self.display_clip = float(self.hp.get("value_display_clip", 0.0))
        self.redflag_threshold = float(self.hp.get("value_feedback_redflag", -0.3))

    # ------------------------------------------------------------------
    # Batch recompute — cheap enough for <10k nodes with pure SQL+numpy
    # ------------------------------------------------------------------

    def recompute_all(self) -> Dict[str, int]:
        """Compute value_score for every non-deleted node, write into the
        materialized columns. Returns counts."""
        started = datetime.utcnow()

        with self.db._connect() as conn:
            node_rows = conn.execute(
                """SELECT id, level, created_at, last_modified FROM memory_nodes
                   WHERE is_deleted = 0"""
            ).fetchall()
            if not node_rows:
                return {"nodes": 0, "redflags": 0, "wall_time_ms": 0}

            node_ids = [r["id"] for r in node_rows]

            # Prefetch all signals in 4 grouped queries instead of N+1
            G_map = self._compute_goal_achievement_map(conn, node_ids)
            F_raw_map = self._compute_feedback_raw_map(conn, node_ids)
            U_map = self._compute_usage_map(conn, node_ids)
            R_map = self._compute_recency_map(node_rows)

            # Propagate G up the tree before composing
            G_propagated = self._propagate_goal_up(conn, G_map, node_ids)

            updates: List[Tuple] = []
            redflag_count = 0
            for r in node_rows:
                nid = r["id"]
                G = G_propagated.get(nid, 0.0)
                F_raw = F_raw_map.get(nid, 0.0)
                F = math.tanh(F_raw)
                U = U_map.get(nid, 0.0)
                R = R_map.get(nid, 0.0)

                composite = self.w_g * G + self.w_f * F + self.w_u * U + self.w_r * R
                display = max(self.display_clip, composite)
                redflag = F <= self.redflag_threshold
                if redflag:
                    redflag_count += 1

                comp = ValueComponents(
                    node_id=nid, G=G, F=F, U=U, R=R, score=display,
                    redflag=redflag, feedback_raw_sum=F_raw,
                )
                updates.append(comp.to_row())

            # Bulk update
            conn.executemany(
                """UPDATE memory_nodes
                   SET value_score = ?, value_goal = ?, value_feedback = ?,
                       value_usage = ?, value_recency = ?, value_computed_at = ?
                   WHERE id = ?""",
                updates,
            )
            conn.commit()

        wall = int((datetime.utcnow() - started).total_seconds() * 1000)
        return {"nodes": len(updates), "redflags": redflag_count, "wall_time_ms": wall}

    # ------------------------------------------------------------------
    # Single-node compute (for /api/node/{id}/value)
    # ------------------------------------------------------------------

    def compute_one(self, node_id: str) -> Optional[ValueComponents]:
        with self.db._connect() as conn:
            row = conn.execute(
                """SELECT id, level, created_at, last_modified, value_goal,
                          value_feedback, value_usage, value_recency, value_score,
                          value_computed_at
                   FROM memory_nodes WHERE id = ? AND is_deleted = 0""",
                (node_id,),
            ).fetchone()
            if not row:
                return None
            # Read materialized if present and fresh, else recompute this one only
            G_map = self._compute_goal_achievement_map(conn, [node_id])
            # Propagate just for this node
            G_propagated = self._propagate_goal_up(conn, G_map, [node_id])
            F_raw = self._compute_feedback_raw_map(conn, [node_id]).get(node_id, 0.0)
            U = self._compute_usage_map(conn, [node_id]).get(node_id, 0.0)
            R = self._compute_recency_map([row]).get(node_id, 0.0)

        G = G_propagated.get(node_id, 0.0)
        F = math.tanh(F_raw)
        composite = self.w_g * G + self.w_f * F + self.w_u * U + self.w_r * R
        display = max(self.display_clip, composite)
        return ValueComponents(
            node_id=node_id, G=G, F=F, U=U, R=R, score=display,
            redflag=F <= self.redflag_threshold, feedback_raw_sum=F_raw,
        )

    # ------------------------------------------------------------------
    # Component computations (grouped queries, O(nodes+feedback+goals))
    # ------------------------------------------------------------------

    def _compute_goal_achievement_map(
        self, conn: sqlite3.Connection, _node_ids: Iterable[str]
    ) -> Dict[str, float]:
        """Compute G(n) = sum(achieved weights) / sum(all weights) per node.
        Nodes with no linked goals get 0 (not penalized). One query covers all."""
        result: Dict[str, float] = {}
        rows = conn.execute(
            """SELECT gl.node_id, g.status, g.id AS goal_id, gl.weight AS link_weight
               FROM goal_links gl
               JOIN goals g ON g.id = gl.goal_id
               WHERE gl.link_type = 'supports'"""
        ).fetchall()
        # Use goal weight = 1.0 since schema doesn't have a per-goal weight column
        # (goal_links.weight is our priority signal). If a per-goal weight is ever
        # added, multiply here.
        numer: Dict[str, float] = {}
        denom: Dict[str, float] = {}
        for r in rows:
            nid = r["node_id"]
            w = float(r["link_weight"] or 0.5)
            denom[nid] = denom.get(nid, 0.0) + w
            if r["status"] == "achieved":
                numer[nid] = numer.get(nid, 0.0) + w
        for nid, d in denom.items():
            if d > 0:
                result[nid] = min(1.0, numer.get(nid, 0.0) / d)
        return result

    def _propagate_goal_up(
        self,
        conn: sqlite3.Connection,
        g_map: Dict[str, float],
        node_ids: Iterable[str],
    ) -> Dict[str, float]:
        """For each node, walk up child_of edges; accumulate ancestor G values
        attenuated by goal_propagation^depth. Take max with own G so that a node
        with direct linkage isn't dragged down by a weaker ancestor."""
        if not g_map:
            return {nid: 0.0 for nid in node_ids}

        # Build parent map once
        parent_map: Dict[str, str] = {}
        for crow in conn.execute(
            "SELECT source_id, target_id FROM relations WHERE relation_type = 'child_of'"
        ):
            parent_map[crow["source_id"]] = crow["target_id"]

        # Also propagate DOWN: if a child has G, its parent inherits attenuated.
        # Build children map from parent_map
        children_map: Dict[str, List[str]] = {}
        for child, parent in parent_map.items():
            children_map.setdefault(parent, []).append(child)

        # Breadth-first propagation DOWNward from nodes with G into their ancestors
        # (i.e. an anchor with G raises its CTX and SC). Classic inheritance up
        # from the leaf. We invert: for every node with direct G, walk its
        # ancestors and boost them by G * α^depth.
        propagated: Dict[str, float] = dict(g_map)  # start with direct G
        alpha = self.goal_propagation
        for leaf_id, g_val in g_map.items():
            cur = parent_map.get(leaf_id)
            depth = 1
            while cur and depth < 10:  # safety cap
                contrib = g_val * (alpha ** depth)
                if contrib < 0.01:
                    break
                propagated[cur] = max(propagated.get(cur, 0.0), contrib)
                cur = parent_map.get(cur)
                depth += 1

        return {nid: propagated.get(nid, 0.0) for nid in node_ids}

    def _compute_feedback_raw_map(
        self, conn: sqlite3.Connection, _node_ids: Iterable[str]
    ) -> Dict[str, float]:
        """Raw (pre-tanh) signed feedback sum with time decay."""
        now = datetime.utcnow()
        rows = conn.execute(
            """SELECT node_id, polarity, strength, timestamp FROM feedback"""
        ).fetchall()
        result: Dict[str, float] = {}
        halflife = self.fb_halflife
        for r in rows:
            pol = r["polarity"]
            sign = {"positive": 1.0, "negative": -1.0, "correction": 0.0}.get(pol, 0.0)
            if sign == 0.0:
                continue
            strength = float(r["strength"] or 1.0)
            ts = _parse_ts(r["timestamp"])
            if ts is None:
                decay = 1.0
            else:
                age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
                decay = max(self.fb_floor, math.exp(-age_days / halflife) if halflife > 0 else 1.0)
            result[r["node_id"]] = result.get(r["node_id"], 0.0) + sign * strength * decay
        return result

    def _compute_usage_map(
        self, conn: sqlite3.Connection, _node_ids: Iterable[str]
    ) -> Dict[str, float]:
        """Log-normalized retrieval count in the usage window. Uses access_log."""
        window = self.usage_window
        # SQLite friendly date math
        counts = conn.execute(
            """SELECT node_id, COUNT(*) AS c
               FROM access_log
               WHERE accessed_at >= datetime('now', ?)
               GROUP BY node_id""",
            (f"-{int(window)} days",),
        ).fetchall()
        raw = {r["node_id"]: int(r["c"]) for r in counts}
        if not raw:
            return {}
        max_c = max(raw.values())
        if max_c <= 0:
            return {}
        log_max = math.log1p(max_c)
        return {nid: math.log1p(c) / log_max for nid, c in raw.items()}

    def _compute_recency_map(self, node_rows: Iterable[Any]) -> Dict[str, float]:
        now = datetime.utcnow()
        tau = self.recency_tau
        result: Dict[str, float] = {}
        for r in node_rows:
            age_ts = _parse_ts(r["last_modified"]) or _parse_ts(r["created_at"])
            if age_ts is None:
                result[r["id"]] = 0.5  # unknown age — neutral
                continue
            age_days = max(0.0, (now - age_ts).total_seconds() / 86400.0)
            result[r["id"]] = math.exp(-age_days / tau) if tau > 0 else 1.0
        return result


# ── helpers ──────────────────────────────────────────────────────────

def _parse_ts(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)
    s = str(ts).strip()
    if not s:
        return None
    # Try common formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(s.replace("Z", ""), fmt.replace("Z", ""))
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None
