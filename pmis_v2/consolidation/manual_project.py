"""
Manual Per-Project Daily Consolidation (Phase 4, 2026-04-20).

Triggered from the Goals/project UI via /api/project/{id}/consolidate-day.

For a given (project_id, date):
  1. Pull all activity_segments from tracker DB that fell inside any
     work_session tagged to this project on this date — these are the
     user-tagged segments.
  2. LLM generates a structured markdown summary with Accomplishments /
     Decisions / Open-items sections.
  3. The caller can edit the draft.
  4. On commit, we:
       - Create ONE Anchor node in PMIS with the final markdown.
       - Attach it to the project's SC/CTX via normal semantic match
         against the knowledge tree (memory tree attachment).
       - Write a row to project_work_match_log with source='manual_consolidation',
         is_correct=1, combined=1.0 (explicit user attachment).
       - Write activity_time_log rows for each included segment so the
         nightly consolidation skips them (see daily_activity_merge._read_segments).

Nightly never re-processes segments that appear in activity_time_log for the
same date — mutual-exclusion is enforced at read time.
"""

from __future__ import annotations

import os
import sqlite3
import logging
import numpy as np
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from consolidation.lock import consolidation_lock, LockBusy

logger = logging.getLogger("pmis.manual_project")

TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")


class ManualProjectConsolidator:
    """Produces a project-scoped daily markdown summary + commits it to memory."""

    def __init__(self, db, hyperparams: Dict[str, Any]):
        self.db = db
        self.hp = hyperparams

    # ------------------------------------------------------------------
    # Step 1: collect tagged segments
    # ------------------------------------------------------------------

    def collect_segments(
        self, project_id: str, target_date: str
    ) -> List[Dict[str, Any]]:
        """Return segments whose timestamp falls inside a work_session
        tagged to `project_id` on `target_date`. Segments already
        consolidated (present in activity_time_log for the date) are
        excluded — we never double-count.
        """
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        sessions = conn.execute(
            """SELECT started_at, ended_at FROM work_sessions
               WHERE project_id = ? AND DATE(started_at) <= ?
               AND (ended_at IS NULL OR DATE(ended_at) >= ?)""",
            (project_id, target_date, target_date),
        ).fetchall()
        already = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT segment_id FROM activity_time_log WHERE date = ?",
                (target_date,),
            ).fetchall()
        }
        conn.close()

        if not sessions or not os.path.exists(TRACKER_DB):
            return []

        tconn = sqlite3.connect(TRACKER_DB)
        tconn.row_factory = sqlite3.Row
        rows = tconn.execute(
            """SELECT id, detailed_summary, window_name, platform,
                      target_segment_length_secs, worker, timestamp_start
               FROM context_1 WHERE DATE(timestamp_start) = ?""",
            (target_date,),
        ).fetchall()
        tconn.close()

        session_windows = [
            (s["started_at"] or "", s["ended_at"] or "9999-12-31")
            for s in sessions
        ]

        out: List[Dict[str, Any]] = []
        for r in rows:
            if r["id"] in already:
                continue
            ts = r["timestamp_start"] or ""
            if not ts or not r["detailed_summary"]:
                continue
            for start, end in session_windows:
                if start <= ts <= end:
                    out.append({
                        "id": r["id"],
                        "summary": r["detailed_summary"] or "",
                        "window": r["window_name"] or "",
                        "platform": r["platform"] or "",
                        "duration_secs": r["target_segment_length_secs"] or 10,
                        "worker": r["worker"] or "human",
                        "timestamp_start": ts,
                    })
                    break
        return out

    # ------------------------------------------------------------------
    # Step 2: LLM draft
    # ------------------------------------------------------------------

    def draft_summary(
        self,
        project_id: str,
        target_date: str,
        segments: List[Dict[str, Any]],
    ) -> str:
        """LLM writes structured markdown. Returns empty string if no segments."""
        if not segments:
            return ""

        # Project name for the title
        conn = sqlite3.connect(self.db.db_path)
        row = conn.execute(
            "SELECT name FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        conn.close()
        project_name = row[0] if row else project_id

        descriptions: List[str] = []
        for s in segments[:60]:  # cap to keep prompt in bounds
            ts = (s.get("timestamp_start") or "")[11:16]  # HH:MM
            descriptions.append(
                f"- [{ts}] ({s.get('window','')}) {s['summary'][:240]}"
            )
        minutes = sum(s.get("duration_secs", 10) for s in segments) / 60.0

        prompt = (
            f"You are writing a concise daily work report for the project "
            f"'{project_name}' on {target_date}.\n\n"
            f"Total tracked time: {minutes:.0f} minutes across "
            f"{len(segments)} activity segments.\n\n"
            "Activity log:\n"
            + "\n".join(descriptions)
            + "\n\nProduce a structured markdown report with these sections "
              "(keep each section concise — 2-5 bullets):\n"
              "### Accomplishments\n### Decisions\n### Open items\n"
              "Rules:\n"
              "- Use plain English, no fabricated reference codes.\n"
              "- Focus on OUTCOMES and DECISIONS, not minute-by-minute actions.\n"
              "- If a section has no content, write '_None_' under it.\n"
        )

        try:
            if self.hp.get("use_local", True):
                return self._call_ollama(prompt)
            return self._call_anthropic(prompt)
        except Exception as e:
            logger.warning("Manual consolidation LLM failed: %s", e)
            # Fallback: mechanical summary
            return (
                f"### Accomplishments\n"
                + "\n".join(f"- {s['summary'][:120]}" for s in segments[:5])
                + "\n### Decisions\n_None_\n### Open items\n_None_\n"
            )

    # ------------------------------------------------------------------
    # Step 3: commit
    # ------------------------------------------------------------------

    def commit(
        self,
        project_id: str,
        target_date: str,
        final_markdown: str,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Persist the consolidation:
        1. Create an ANC memory_node under the project's SC/CTX.
        2. Write activity_time_log rows for every segment (so nightly skips).
        3. Write one project_work_match_log row with source='manual_consolidation'.

        Holds consolidation_lock scope='date:<target_date>' so a concurrent
        nightly run (global) or same-date manual run cannot race this write.
        """
        from core.memory_node import MemoryNode, MemoryLevel
        from core.temporal import temporal_encode, compute_era
        from ingestion.embedder import Embedder

        if not final_markdown.strip():
            return {"ok": False, "error": "empty markdown"}

        try:
            with consolidation_lock(f"date:{target_date}", kind="manual"):
                return self._commit_locked(
                    project_id, target_date, final_markdown, segments
                )
        except LockBusy as e:
            return {
                "ok": False,
                "error": "consolidation_locked",
                "detail": str(e),
                "retry_after_secs": e.retry_after_secs,
            }

    def _commit_locked(
        self,
        project_id: str,
        target_date: str,
        final_markdown: str,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Inner commit — runs under the date-scoped consolidation lock."""
        from core.memory_node import MemoryNode, MemoryLevel
        from core.temporal import temporal_encode, compute_era
        from ingestion.embedder import Embedder

        hp = self.hp
        embedder = Embedder(hyperparams=hp)
        try:
            euclidean = embedder.embed_text(final_markdown)
        except Exception:
            euclidean = np.zeros(hp.get("local_embedding_dimensions", 768))

        hyp_dim = hp.get("poincare_dimensions", 16)
        temporal = temporal_encode(
            datetime.now(), hp.get("temporal_embedding_dim", 16)
        )
        era = compute_era(datetime.now(), hp.get("era_boundaries", {}))

        # Attach to project's SC (if projects.sc_node_id set) — fallback to
        # semantic match against knowledge tree.
        conn = sqlite3.connect(self.db.db_path)
        row = conn.execute(
            "SELECT sc_node_id, name FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        conn.close()
        project_sc = row[0] if row else None
        project_name = row[1] if row else project_id

        content = (
            f"[Manual Consolidation · {target_date} · {project_name}]\n\n"
            + final_markdown.strip()
        )
        node = MemoryNode.create(
            content=content,
            level=MemoryLevel.ANCHOR,
            euclidean_embedding=euclidean,
            hyperbolic_coords=np.zeros(hyp_dim, dtype=np.float32),
            temporal_embedding=temporal,
            source_conversation_id=f"manual_consol_{project_id}_{target_date}",
            surprise=0.0,
            precision=0.6,
            era=era,
        )
        node.is_orphan = False
        node.is_tentative = False
        self.db.create_node(node)

        if project_sc:
            tree_id = self._tree_id_for(project_sc) or "default"
            self.db.attach_to_parent(node.id, project_sc, tree_id)

        # Write activity_time_log + mark consolidated.
        # INSERT OR IGNORE respects the UNIQUE(segment_id, date) guard so a
        # concurrent/earlier writer — typically nightly — keeps its row rather
        # than being clobbered. Manual tries to claim its segments first under
        # the date-scoped lock; any IGNOREd rows mean nightly already had them.
        pconn = sqlite3.connect(self.db.db_path)
        total_duration = 0
        for seg in segments:
            dur = seg.get("duration_secs", 10)
            total_duration += dur
            pconn.execute(
                """INSERT OR IGNORE INTO activity_time_log
                   (segment_id, memory_node_id, matched_ctx_id, matched_sc_id,
                    duration_seconds, date, project_id, match_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'manual_consolidation')""",
                (
                    seg.get("id", ""), node.id, "", project_sc or "",
                    dur, target_date, project_id,
                ),
            )
            pconn.execute(
                "UPDATE activity_segments SET consolidated_into_node_id = ? WHERE id = ?",
                (node.id, seg.get("id", "")),
            )
        pconn.commit()
        pconn.close()

        # Log match
        self.db.log_match({
            "segment_id": node.id,
            "project_id": project_id,
            "deliverable_id": "",
            "sc_node_id": project_sc or "",
            "context_node_id": "",
            "anchor_node_id": "",
            "semantic_score": 1.0,
            "hyperbolic_score": 1.0,
            "combined_match_pct": 1.0,
            "match_method": "manual_consolidation",
            "work_description": final_markdown[:500],
            "worker_type": "manual_consolidation",
            "time_mins": total_duration / 60.0,
            "is_correct": 1,
            "source": "manual_consolidation",
        })

        return {
            "ok": True,
            "anchor_id": node.id,
            "project_id": project_id,
            "date": target_date,
            "segments_consolidated": len(segments),
            "duration_mins": round(total_duration / 60.0, 1),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tree_id_for(self, node_id: str) -> Optional[str]:
        conn = sqlite3.connect(self.db.db_path)
        row = conn.execute(
            """SELECT tree_id FROM trees WHERE root_node_id = ?""",
            (node_id,),
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _call_ollama(self, prompt: str) -> str:
        model = self.hp.get("consolidation_model_local", "qwen2.5:14b")
        max_tokens = self.hp.get("consolidation_max_tokens", 2048)
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _call_anthropic(self, prompt: str) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = self.hp.get("consolidation_model", "claude-sonnet-4-20250514")
        max_tokens = self.hp.get("consolidation_max_tokens", 2048)
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()
