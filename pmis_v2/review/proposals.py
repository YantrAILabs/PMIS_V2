"""Review Proposals (Phase 3).

Two-stage flow powering /wiki/review:

  1. List unconsolidated segments — raw context_1 rows that haven't been
     claimed by a nightly cluster, a manual consolidation, or an earlier
     review confirmation.
  2. Consolidate on demand — cluster the unconsolidated segments, extract a
     one-line pattern per cluster via LLM, and score each cluster against
     active projects. Persisted as `review_proposals` rows in status='draft'.
  3. Confirm — create an ANC memory_node, write activity_time_log rows with
     match_source='user_review' (higher authority than nightly), log a
     project_work_match_log row marked is_correct=1, and flip the proposal
     to status='confirmed'. Runs under a date-scoped consolidation_lock
     so it can't race a nightly pass.
  4. Reject — mark the proposal status='rejected'; the segments fall back
     into the unconsolidated pool on the next refresh.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from consolidation.lock import consolidation_lock, LockBusy

logger = logging.getLogger("pmis.review")

TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")


# Legacy segments (captured before Phase 2 shipped short_title) all have the
# same LLM boilerplate opener — "During the segment, ...". Strip those so the
# derived title leads with real content.
_BOILERPLATE_OPENERS = (
    "during the segment,",
    "during this segment,",
    "in this segment,",
    "in the segment,",
    "the user is ",
    "the user was ",
    "a conversation was initiated ",
)


def _derive_short_title(short_title: Optional[str], detailed_summary: Optional[str],
                        window_name: Optional[str]) -> str:
    """Produce a readable, one-line title even for legacy rows that never got
    a short_title. Trims LLM boilerplate and caps at the first sentence or 80
    chars — whichever comes first."""
    if short_title:
        return short_title[:200]
    s = (detailed_summary or "").strip()
    if not s:
        return (window_name or "Activity")[:200]
    # Peel off stacked openers — LLM output often chains multiple
    # ("During the segment, a conversation was initiated where...").
    stripped = True
    while stripped:
        stripped = False
        low = s.lower()
        for opener in _BOILERPLATE_OPENERS:
            if low.startswith(opener):
                s = s[len(opener):].lstrip()
                stripped = True
                break
    for end in (". ", "? ", "! "):
        idx = s.find(end)
        if 0 < idx < 80:
            return s[:idx].rstrip()
    if len(s) <= 80:
        return s
    return s[:79].rstrip() + "…"


class ReviewProposals:
    """All state transitions for the review page."""

    def __init__(self, db, hyperparams: Dict[str, Any]):
        self.db = db
        self.hp = hyperparams

    # ------------------------------------------------------------------
    # Stage 1: list raw unconsolidated segments
    # ------------------------------------------------------------------

    def list_unconsolidated(
        self,
        target_date: Optional[str] = None,
        days: Optional[int] = 2,
    ) -> List[Dict]:
        """Return context_1 segments NOT claimed by activity_time_log AND
        not in a draft proposal.

        Windowing:
          target_date='YYYY-MM-DD' → only segments from that exact day.
          days=N (default 2)       → today + (N-1) prior days.
          days=None                → no window; returns the full backlog.

        The default is deliberately narrow. Nightly daily_activity_merge
        skips singleton clusters (size < 2), so over many days the
        un-clusterable residue piles up. Those segments are nightly's
        decision to NOT group, not a pending user task — showing the full
        historical residue on Review floods the UI with noise.
        """
        if not os.path.exists(TRACKER_DB):
            return []

        claimed = self._claimed_segment_ids()

        tconn = sqlite3.connect(TRACKER_DB)
        tconn.row_factory = sqlite3.Row
        base_cols = ("id, target_segment_id, detailed_summary, short_title, "
                     "window_name, platform, target_segment_length_secs, "
                     "worker, timestamp_start")
        if target_date:
            rows = tconn.execute(
                f"SELECT {base_cols} FROM context_1 "
                f"WHERE DATE(timestamp_start) = ? ORDER BY timestamp_start DESC",
                (target_date,),
            ).fetchall()
        elif days is not None and days > 0:
            rows = tconn.execute(
                f"SELECT {base_cols} FROM context_1 "
                f"WHERE DATE(timestamp_start) >= DATE('now', ?) "
                f"ORDER BY timestamp_start DESC",
                (f"-{days - 1} days",),
            ).fetchall()
        else:
            rows = tconn.execute(
                f"SELECT {base_cols} FROM context_1 ORDER BY timestamp_start DESC"
            ).fetchall()
        tconn.close()

        out: List[Dict] = []
        for r in rows:
            if r["id"] in claimed:
                continue
            if not (r["short_title"] or r["detailed_summary"]):
                continue
            out.append({
                "id": r["id"],
                "target_segment_id": r["target_segment_id"],
                "short_title": _derive_short_title(
                    r["short_title"], r["detailed_summary"], r["window_name"]
                ),
                "detailed_summary": r["detailed_summary"] or "",
                "window": r["window_name"] or "",
                "platform": r["platform"] or "",
                "duration_secs": r["target_segment_length_secs"] or 10,
                "worker": r["worker"] or "human",
                "timestamp_start": r["timestamp_start"] or "",
                "date": (r["timestamp_start"] or "")[:10],
            })
        return out

    def _claimed_segment_ids(self) -> set:
        """segment_ids that are already spoken for. Delegates to
        consolidation.claims so Review, Goals Unassigned, and Recent-days
        all agree on the definition of 'claimed' — including segments
        bound to confirmed or auto_attached proposals (the F1 leak fix)."""
        from consolidation.claims import all_claimed_segment_ids
        pconn = sqlite3.connect(self.db.db_path)
        try:
            return set(all_claimed_segment_ids(pconn).keys())
        finally:
            pconn.close()

    # ------------------------------------------------------------------
    # Stage 2: consolidate
    # ------------------------------------------------------------------

    def consolidate(
        self,
        target_date: Optional[str] = None,
        days: Optional[int] = 2,
    ) -> List[Dict]:
        """Cluster unconsolidated segments + score each cluster against active
        projects + persist as draft review_proposals. Uses the same windowing
        rules as list_unconsolidated (default: today + yesterday). Any
        previously-draft proposal for this scope is marked 'superseded' first
        so we don't accumulate stale groups on repeated clicks.
        """
        segments = self.list_unconsolidated(target_date, days=days)
        if not segments:
            return []

        # Reuse the nightly merger's clustering + pattern extraction so the
        # review proposals are semantically consistent with what nightly
        # would have produced.
        from daily_activity_merge import DailyActivityMerger
        merger = DailyActivityMerger(self.db, self.hp)

        # DailyActivityMerger._cluster_segments expects seg["summary"].
        # We populate it from short_title with detailed_summary as fallback.
        cluster_input = [
            {**s, "summary": (s["short_title"] or s["detailed_summary"] or "")[:500]}
            for s in segments
        ]
        clusters = merger._cluster_segments(cluster_input)

        # Supersede stale drafts for this target_date scope.
        self._supersede_drafts(target_date)

        proposals: List[Dict] = []
        for cluster in clusters:
            if not cluster:
                continue
            pattern = merger._extract_pattern(cluster) or cluster[0].get("summary") or "Activity"
            probs = self._score_against_projects(pattern, cluster)
            prop_id = f"rp-{uuid.uuid4().hex[:12]}"
            segment_ids = [s.get("id", "") for s in cluster]
            total_duration = sum(int(s.get("duration_secs", 10)) for s in cluster)
            windows = list({s.get("window", "") for s in cluster if s.get("window")})[:5]

            pconn = sqlite3.connect(self.db.db_path)
            pconn.execute(
                """INSERT INTO review_proposals
                   (id, target_date, author, status, proposed_content,
                    segment_ids_json, project_probs_json)
                   VALUES (?, ?, 'user', 'draft', ?, ?, ?)""",
                (
                    prop_id,
                    target_date or "",
                    pattern[:2000],
                    json.dumps(segment_ids),
                    json.dumps(probs),
                ),
            )
            pconn.commit()
            pconn.close()

            proposals.append({
                "id": prop_id,
                "target_date": target_date or "",
                "status": "draft",
                "proposed_content": pattern,
                "segment_ids": segment_ids,
                "segment_count": len(cluster),
                "duration_mins": round(total_duration / 60.0, 1),
                "windows": windows,
                "project_probs": probs,
            })
        return proposals

    def _supersede_drafts(self, target_date: Optional[str]) -> None:
        pconn = sqlite3.connect(self.db.db_path)
        if target_date:
            pconn.execute(
                "UPDATE review_proposals SET status='superseded' "
                "WHERE status='draft' AND target_date = ?",
                (target_date,),
            )
        else:
            pconn.execute(
                "UPDATE review_proposals SET status='superseded' WHERE status='draft'"
            )
        pconn.commit()
        pconn.close()

    def list_drafts(self, target_date: Optional[str] = None) -> List[Dict]:
        """Return active draft proposals (re-hydrated with duration/window info)."""
        pconn = sqlite3.connect(self.db.db_path)
        pconn.row_factory = sqlite3.Row
        if target_date:
            rows = pconn.execute(
                """SELECT id, target_date, status, proposed_content,
                          segment_ids_json, project_probs_json,
                          user_assigned_project_id, created_at
                   FROM review_proposals
                   WHERE status='draft' AND target_date = ?
                   ORDER BY created_at DESC""",
                (target_date,),
            ).fetchall()
        else:
            rows = pconn.execute(
                """SELECT id, target_date, status, proposed_content,
                          segment_ids_json, project_probs_json,
                          user_assigned_project_id, created_at
                   FROM review_proposals
                   WHERE status='draft'
                   ORDER BY created_at DESC"""
            ).fetchall()
        pconn.close()

        out: List[Dict] = []
        for r in rows:
            segment_ids = json.loads(r["segment_ids_json"] or "[]")
            probs = json.loads(r["project_probs_json"] or "[]")
            duration_mins, windows = self._segment_meta(segment_ids)
            out.append({
                "id": r["id"],
                "target_date": r["target_date"],
                "status": r["status"],
                "proposed_content": r["proposed_content"] or "",
                "segment_ids": segment_ids,
                "segment_count": len(segment_ids),
                "duration_mins": duration_mins,
                "windows": windows,
                "project_probs": probs,
                "user_assigned_project_id": r["user_assigned_project_id"] or "",
            })
        return out

    def _segment_meta(self, segment_ids: List[str]) -> Tuple[float, List[str]]:
        if not segment_ids or not os.path.exists(TRACKER_DB):
            return 0.0, []
        placeholders = ",".join("?" for _ in segment_ids)
        tconn = sqlite3.connect(TRACKER_DB)
        rows = tconn.execute(
            f"""SELECT target_segment_length_secs, window_name
                FROM context_1 WHERE id IN ({placeholders})""",
            segment_ids,
        ).fetchall()
        tconn.close()
        total = sum((r[0] or 10) for r in rows)
        windows = list({r[1] for r in rows if r[1]})[:5]
        return round(total / 60.0, 1), windows

    # ------------------------------------------------------------------
    # Stage 3: project probability scoring
    # ------------------------------------------------------------------

    def _score_against_projects(self, pattern: str, cluster: List[Dict]) -> List[Dict]:
        """Score a cluster's pattern against active projects. Returns list of
        {project_id, project_name, score}. score in [0,1], sums not required.

        Semantic path: embed the pattern, compare cosine against each active
        project's sc_node_id embedding.
        Keyword path: match goals.yaml match_patterns against window/summary.
        We combine both signals per project so a project with no SC node but
        with explicit keyword patterns still surfaces.
        """
        active_projects = self._list_active_projects()
        if not active_projects:
            return []

        # Semantic embedding of the pattern (single embed; cheap)
        query_emb = None
        try:
            from ingestion.embedder import Embedder
            emb = Embedder(hyperparams=self.hp)
            query_emb = emb.embed_text(pattern)
        except Exception:
            query_emb = None

        cluster_text = " | ".join(
            (s.get("window", "") + " " + s.get("short_title", "") + " " + s.get("detailed_summary", ""))
            for s in cluster
        ).lower()

        scored: List[Dict] = []
        for p in active_projects:
            sem = self._semantic_project_score(query_emb, p.get("sc_node_id"))
            kw = self._keyword_project_score(cluster_text, p.get("match_patterns") or [])
            # Prefer the stronger signal; nudge up when both fire.
            score = max(sem, kw)
            if sem > 0 and kw > 0:
                score = min(1.0, score + 0.1)
            if score > 0.0:
                scored.append({
                    "project_id": p["id"],
                    "project_name": p.get("name") or p["id"],
                    "score": round(float(score), 3),
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:5]

    def _semantic_project_score(self, query_emb, sc_node_id: Optional[str]) -> float:
        if query_emb is None or not sc_node_id:
            return 0.0
        try:
            embs = self.db.get_embeddings(sc_node_id)
            sc_emb = embs.get("euclidean") if embs else None
        except Exception:
            sc_emb = None
        if sc_emb is None:
            return 0.0
        dot = float(np.dot(query_emb, sc_emb))
        nq = float(np.linalg.norm(query_emb))
        ns = float(np.linalg.norm(sc_emb))
        if nq == 0 or ns == 0:
            return 0.0
        sim = dot / (nq * ns)
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))  # rescale [-1,1] -> [0,1]

    def _keyword_project_score(self, cluster_text: str, patterns: List[str]) -> float:
        if not patterns or not cluster_text:
            return 0.0
        hits = sum(1 for p in patterns if p and p.lower() in cluster_text)
        if not hits:
            return 0.0
        # Saturate at 3+ keyword hits = 0.9
        return min(0.9, 0.5 + 0.15 * hits)

    def _list_active_projects(self) -> List[Dict]:
        """Goals-page projects only (goals.yaml). The PMIS `projects` table
        holds historical rows that shouldn't surface in the Review picker —
        only the current Goals projects are valid tagging targets.

        We still enrich each project with its sc_node_id from the PMIS table
        when the id matches; that's a lookup, not a widening of the list.
        """
        catalog: Dict[str, Dict] = {}
        try:
            import yaml
            gy = Path.home() / "Desktop" / "memory" / "productivity-tracker" / "config" / "goals.yaml"
            if not gy.exists():
                return []
            with open(gy, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            for g in raw.get("goals") or []:
                for p in g.get("projects") or []:
                    pid = p.get("id")
                    if not pid:
                        continue
                    catalog[pid] = {
                        "id": pid,
                        "name": p.get("title") or pid,
                        "sc_node_id": "",
                        "match_patterns": list(p.get("match_patterns") or []),
                    }
        except Exception as e:
            logger.debug(f"goals.yaml scan failed: {e}")
            return []

        if not catalog:
            return []

        # Enrich with sc_node_id from PMIS table where the id matches.
        try:
            pconn = sqlite3.connect(self.db.db_path)
            pconn.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in catalog)
            rows = pconn.execute(
                f"SELECT id, sc_node_id FROM projects WHERE id IN ({placeholders})",
                list(catalog.keys()),
            ).fetchall()
            for r in rows:
                catalog[r["id"]]["sc_node_id"] = r["sc_node_id"] or ""
            pconn.close()
        except Exception:
            pass

        return list(catalog.values())

    # ------------------------------------------------------------------
    # Stage 4: confirm / reject
    # ------------------------------------------------------------------

    def confirm(
        self,
        proposal_id: str,
        project_id: str,
        deliverable_id: str = "",
    ) -> Dict[str, Any]:
        """Commit a draft proposal to memory as a user_review anchor."""
        pconn = sqlite3.connect(self.db.db_path)
        pconn.row_factory = sqlite3.Row
        row = pconn.execute(
            "SELECT * FROM review_proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
        pconn.close()
        if not row:
            return {"ok": False, "error": "not_found"}
        if row["status"] != "draft":
            return {"ok": False, "error": f"not_draft (status={row['status']})"}

        segment_ids: List[str] = json.loads(row["segment_ids_json"] or "[]")
        if not segment_ids:
            return {"ok": False, "error": "empty_cluster"}

        target_date = row["target_date"] or datetime.now().date().isoformat()
        lock_scope = f"date:{target_date}"

        try:
            with consolidation_lock(lock_scope, kind="manual"):
                return self._confirm_locked(
                    row, segment_ids, project_id, deliverable_id, target_date
                )
        except LockBusy as e:
            return {
                "ok": False,
                "error": "consolidation_locked",
                "detail": str(e),
                "retry_after_secs": e.retry_after_secs,
            }

    def _confirm_locked(
        self,
        row: sqlite3.Row,
        segment_ids: List[str],
        project_id: str,
        deliverable_id: str,
        target_date: str,
    ) -> Dict[str, Any]:
        from core.memory_node import MemoryNode, MemoryLevel
        from core.temporal import temporal_encode, compute_era
        from ingestion.embedder import Embedder

        content = (row["proposed_content"] or "").strip() or "User-reviewed activity"

        embedder = Embedder(hyperparams=self.hp)
        try:
            euclidean = embedder.embed_text(content)
        except Exception:
            euclidean = np.zeros(self.hp.get("local_embedding_dimensions", 768))

        hyp_dim = self.hp.get("poincare_dimensions", 16)
        temporal = temporal_encode(datetime.now(), self.hp.get("temporal_embedding_dim", 16))
        era = compute_era(datetime.now(), self.hp.get("era_boundaries", {}))

        # Project SC (for attachment)
        project_sc = ""
        project_name = project_id
        pconn = sqlite3.connect(self.db.db_path)
        try:
            r = pconn.execute(
                "SELECT sc_node_id, name FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if r:
                project_sc = r[0] or ""
                project_name = r[1] or project_id
        finally:
            pconn.close()

        node = MemoryNode.create(
            content=f"[User Review · {target_date} · {project_name}]\n\n{content}",
            level=MemoryLevel.ANCHOR,
            euclidean_embedding=euclidean,
            hyperbolic_coords=np.zeros(hyp_dim, dtype=np.float32),
            temporal_embedding=temporal,
            source_conversation_id=f"user_review_{row['id']}",
            surprise=0.0,
            precision=0.7,
            era=era,
        )
        node.is_orphan = False
        node.is_tentative = False
        self.db.create_node(node)

        if project_sc:
            tree_id = self._tree_id_for(project_sc) or "default"
            try:
                self.db.attach_to_parent(node.id, project_sc, tree_id)
            except Exception:
                pass

        total_duration = 0
        pconn = sqlite3.connect(self.db.db_path)
        try:
            # Per-segment durations come from the tracker DB.
            durations = self._segment_durations(segment_ids)
            for sid in segment_ids:
                dur = durations.get(sid, 10)
                total_duration += dur
                pconn.execute(
                    """INSERT OR IGNORE INTO activity_time_log
                       (segment_id, memory_node_id, matched_ctx_id, matched_sc_id,
                        duration_seconds, date, project_id, match_source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'user_review')""",
                    (sid, node.id, "", project_sc, dur, target_date, project_id),
                )
            pconn.execute(
                """UPDATE review_proposals
                   SET status='confirmed', user_assigned_project_id=?,
                       user_assigned_deliverable_id=?, anchor_node_id=?,
                       confirmed_at=datetime('now')
                   WHERE id = ?""",
                (project_id, deliverable_id, node.id, row["id"]),
            )
            pconn.commit()
        finally:
            pconn.close()

        self.db.log_match({
            "segment_id": node.id,
            "project_id": project_id,
            "deliverable_id": deliverable_id,
            "sc_node_id": project_sc,
            "context_node_id": "",
            "anchor_node_id": node.id,
            "semantic_score": 1.0,
            "hyperbolic_score": 1.0,
            "combined_match_pct": 1.0,
            "match_method": "user_review",
            "work_description": content[:500],
            "worker_type": "user_review",
            "time_mins": total_duration / 60.0,
            "is_correct": 1,
            "source": "user_review",
        })

        return {
            "ok": True,
            "proposal_id": row["id"],
            "anchor_id": node.id,
            "project_id": project_id,
            "segments_tagged": len(segment_ids),
            "duration_mins": round(total_duration / 60.0, 1),
        }

    def reject(self, proposal_id: str) -> Dict[str, Any]:
        pconn = sqlite3.connect(self.db.db_path)
        cur = pconn.execute(
            "UPDATE review_proposals SET status='rejected' WHERE id = ? AND status='draft'",
            (proposal_id,),
        )
        pconn.commit()
        changed = cur.rowcount
        pconn.close()
        return {"ok": bool(changed), "proposal_id": proposal_id}

    def set_assignment(self, proposal_id: str, project_id: str, deliverable_id: str = "") -> Dict[str, Any]:
        pconn = sqlite3.connect(self.db.db_path)
        cur = pconn.execute(
            """UPDATE review_proposals
               SET user_assigned_project_id = ?, user_assigned_deliverable_id = ?
               WHERE id = ? AND status='draft'""",
            (project_id, deliverable_id, proposal_id),
        )
        pconn.commit()
        changed = cur.rowcount
        pconn.close()
        return {"ok": bool(changed), "proposal_id": proposal_id}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _segment_durations(self, segment_ids: List[str]) -> Dict[str, int]:
        if not segment_ids or not os.path.exists(TRACKER_DB):
            return {}
        placeholders = ",".join("?" for _ in segment_ids)
        tconn = sqlite3.connect(TRACKER_DB)
        rows = tconn.execute(
            f"SELECT id, target_segment_length_secs FROM context_1 "
            f"WHERE id IN ({placeholders})",
            segment_ids,
        ).fetchall()
        tconn.close()
        return {r[0]: (r[1] or 10) for r in rows}

    def _tree_id_for(self, node_id: str) -> Optional[str]:
        conn = sqlite3.connect(self.db.db_path)
        try:
            row = conn.execute(
                "SELECT tree_id FROM relations "
                "WHERE (source_id = ? OR target_id = ?) AND tree_id != 'default' LIMIT 1",
                (node_id, node_id),
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else None
