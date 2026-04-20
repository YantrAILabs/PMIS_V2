"""
Restructure pass — LLM regen of red-flagged nodes (audit-fix Item 7).

Integrated with production's existing Phase 3 value_score infrastructure:
  - Reads production's `feedback` table for polarity/strength history
  - Uses the materialized `value_feedback` column as threshold signal
  - Respects `is_user_edited` strictly (user authorship beats LLM regen)
  - Honors the existing `value_feedback_redflag` hyperparameter (default -0.3)

Does NOT replace `_pass_wiki_regen` (cache invalidation) — they are
orthogonal. This pass rewrites node *content* when feedback has turned
negative enough that the materialized value_feedback dropped below the
red-flag threshold.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("pmis.restructure")


class Restructurer:
    def __init__(self, db, hyperparams: Dict[str, Any], embedder: Any = None):
        self.db = db
        self.hp = hyperparams
        self._embedder = embedder

    # -----------------------------------------------------------------
    # Public entry points
    # -----------------------------------------------------------------

    def enqueue_red_flags(self) -> Dict[str, int]:
        """
        Scan memory_nodes for red-flagged nodes (value_feedback below threshold)
        and enqueue them for regen. Used by the nightly pass to turn passive
        feedback into active rewriting.
        """
        redflag_threshold = float(self.hp.get("value_feedback_redflag", -0.3))
        enqueued = 0
        with self.db._connect() as conn:
            rows = conn.execute("""
                SELECT id, level FROM memory_nodes
                WHERE is_deleted = 0
                  AND COALESCE(is_user_edited, 0) = 0
                  AND COALESCE(value_feedback, 0) < ?
            """, (redflag_threshold,)).fetchall()
            for r in rows:
                scope = "context" if r["level"] == "CTX" else "anchor"
                # Dedup: skip if already queued/processing
                existing = conn.execute("""
                    SELECT id FROM restructure_queue
                    WHERE node_id = ? AND status IN ('queued', 'processing')
                """, (r["id"],)).fetchone()
                if existing:
                    continue
                conn.execute("""
                    INSERT INTO restructure_queue (node_id, scope, reason)
                    VALUES (?, ?, ?)
                """, (r["id"], scope, f"value_feedback<{redflag_threshold}"))
                enqueued += 1
        return {"enqueued": enqueued, "redflag_threshold": redflag_threshold}

    def run(self, max_jobs: int = 50) -> List[Dict[str, Any]]:
        """Drain the queue. Returns list of action dicts."""
        actions: List[Dict[str, Any]] = []
        with self.db._connect() as conn:
            jobs = conn.execute("""
                SELECT id, node_id, scope, reason, queued_at
                FROM restructure_queue
                WHERE status = 'queued'
                ORDER BY queued_at ASC LIMIT ?
            """, (max_jobs,)).fetchall()
            jobs = [dict(j) for j in jobs]

        for job in jobs:
            actions.append(self._process_job(job))
        return actions

    def regen_now(self, node_id: str, scope: Optional[str] = None,
                  reason: str = "manual_override", force: bool = False) -> Dict[str, Any]:
        """Manual trigger: regen a single node immediately, bypass queue."""
        node = self.db.get_node(node_id)
        if not node:
            return {"action": "restructure_failed", "node_id": node_id,
                    "error": "node not found"}
        scope = scope or ("context" if node.get("level") == "CTX" else "anchor")
        synthetic_job = {
            "id": None, "node_id": node_id, "scope": scope,
            "reason": reason, "queued_at": datetime.now().isoformat(),
        }
        return self._process_job(synthetic_job, force=force)

    # -----------------------------------------------------------------
    # Per-job processing
    # -----------------------------------------------------------------

    def _process_job(self, job: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        node_id = job["node_id"]
        scope = job["scope"]
        queue_id = job.get("id")

        node = self.db.get_node(node_id)
        if not node:
            self._mark_processed(queue_id, "skipped")
            return {"action": "restructure_skipped", "node_id": node_id,
                    "reason": "node_missing"}

        if node.get("is_user_edited") and not force:
            self._mark_processed(queue_id, "skipped")
            return {"action": "restructure_skipped", "node_id": node_id,
                    "reason": "is_user_edited", "scope": scope}
        if node.get("is_deleted"):
            self._mark_processed(queue_id, "skipped")
            return {"action": "restructure_skipped", "node_id": node_id,
                    "reason": "node_deleted"}

        before = node.get("content") or ""
        prompt_ctx = self._build_context(node, scope)
        prompt = self._anchor_prompt(prompt_ctx) if scope == "anchor" \
                 else self._context_prompt(prompt_ctx)

        try:
            new_content = self._call_llm(prompt)
        except Exception as e:
            self._mark_processed(queue_id, "skipped")
            return {"action": "restructure_failed", "node_id": node_id,
                    "scope": scope, "error": str(e)[:200]}

        new_content = self._sanitize(new_content)
        if not new_content or new_content == before:
            self._mark_processed(queue_id, "skipped")
            return {"action": "restructure_skipped", "node_id": node_id,
                    "reason": "empty_or_unchanged"}

        self._apply_regen(node_id, before, new_content, scope, job.get("reason", ""))
        self._mark_processed(queue_id, "done")
        return {
            "action": "restructure",
            "node_id": node_id,
            "scope": scope,
            "trigger_reason": job.get("reason", ""),
            "before_chars": len(before),
            "after_chars": len(new_content),
            "applied_by": self._llm_label(),
        }

    # -----------------------------------------------------------------
    # Prompt context + assembly
    # -----------------------------------------------------------------

    def _build_context(self, node: Dict[str, Any], scope: str) -> Dict[str, Any]:
        """Gather parent + siblings / children + recent feedback for prompt."""
        if scope == "anchor":
            parents = self.db.get_parents(node["id"])
            parent = parents[0] if parents else None
            siblings: List[Dict[str, Any]] = []
            if parent:
                for c in self.db.get_children(parent["id"]):
                    if c["id"] != node["id"]:
                        siblings.append(c)
            return {
                "node": node, "parent": parent,
                "siblings": siblings[:6],
                "recent_feedback": self._recent_feedback(node["id"], 5),
            }
        # context
        children = self.db.get_children(node["id"])
        return {
            "node": node, "children": children[:12],
            "recent_feedback": self._recent_feedback(node["id"], 5),
        }

    def _recent_feedback(self, node_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self.db._connect() as conn:
            rows = conn.execute("""
                SELECT polarity, content, source, strength, timestamp
                FROM feedback
                WHERE node_id = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (node_id, limit)).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def _format_feedback(fb: List[Dict[str, Any]]) -> str:
        if not fb:
            return "  (no recent feedback)"
        lines = []
        for i, f in enumerate(fb, 1):
            content = (f.get("content") or "").replace("\n", " ")[:160]
            pol = f.get("polarity", "")
            ts = (f.get("timestamp") or "")[:16]
            lines.append(f"  {i}. [{ts}] {pol}: {content}")
        return "\n".join(lines)

    def _anchor_prompt(self, ctx: Dict[str, Any]) -> str:
        anchor = ctx["node"]
        parent = ctx["parent"]
        siblings = ctx["siblings"]
        before = anchor.get("content") or ""

        sib_block = "  (none)" if not siblings else "\n".join(
            f"  {i}. {(s.get('content') or '').strip()[:200]}"
            for i, s in enumerate(siblings, 1)
        )
        parent_line = (parent.get("content", "")[:300]
                       if parent else "(orphan — no parent context)")

        return (
            "You are rewriting a single memory anchor that has received sustained negative feedback.\n\n"
            "PARENT CONTEXT:\n"
            f"  {parent_line}\n\n"
            "SIBLING ANCHORS (for tone + scope reference):\n"
            f"{sib_block}\n\n"
            "CURRENT ANCHOR CONTENT (the one being replaced):\n"
            f"  {before}\n\n"
            "RECENT FEEDBACK ON THIS ANCHOR (most recent first):\n"
            f"{self._format_feedback(ctx['recent_feedback'])}\n\n"
            "RULES:\n"
            "- Output ONLY the new anchor text. No commentary, no headers, no quotes.\n"
            "- 1-3 sentences. Atomic, reusable insight.\n"
            "- Plain English only. Do NOT invent reference codes "
            "(e.g. no 'PM-25', 'MEM-3', 'CTX-7', 'ANC-12').\n"
            "- Address what the negative feedback suggests is wrong.\n"
            "- Stay in the topic scope of the parent context.\n\n"
            "NEW ANCHOR CONTENT:"
        )

    def _context_prompt(self, ctx: Dict[str, Any]) -> str:
        node = ctx["node"]
        children = ctx["children"]
        before = node.get("content") or ""

        ch_block = "  (no child anchors)" if not children else "\n".join(
            f"  {i}. {(c.get('content') or '').strip()[:200]}"
            for i, c in enumerate(children, 1)
        )

        return (
            "You are rewriting the SUMMARY TEXT of a memory context that has received "
            "sustained negative feedback.\n\n"
            "CURRENT CONTEXT CONTENT (the one being replaced):\n"
            f"  {before}\n\n"
            "CHILD ANCHORS (ground truth — summary must reflect them):\n"
            f"{ch_block}\n\n"
            "RECENT FEEDBACK ON THIS CONTEXT (most recent first):\n"
            f"{self._format_feedback(ctx['recent_feedback'])}\n\n"
            "RULES:\n"
            "- Output ONLY the new context summary. No commentary, no headers.\n"
            "- 1-2 sentences. Should describe the common theme of the child anchors.\n"
            "- Plain English only. Do NOT invent reference codes "
            "(e.g. no 'PM-25', 'CTX-7').\n"
            "- DO NOT change which child anchors belong here. Only rewrite text.\n\n"
            "NEW CONTEXT SUMMARY:"
        )

    # -----------------------------------------------------------------
    # Apply + audit
    # -----------------------------------------------------------------

    def _apply_regen(self, node_id: str, before: str, after: str,
                     scope: str, trigger_reason: str):
        applied_by = self._llm_label()
        with self.db._connect() as conn:
            conn.execute("""
                UPDATE memory_nodes
                SET content = ?, last_modified = datetime('now')
                WHERE id = ?
            """, (after, node_id))
            conn.execute("""
                INSERT INTO restructure_log
                (node_id, scope, trigger_reason, before_content,
                 after_content, applied_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (node_id, scope, trigger_reason, before, after, applied_by))

        # Re-embed so semantic ranking + ChromaDB match the new content.
        try:
            embedder = self._get_embedder()
            if embedder is not None:
                new_euc = embedder.embed_text(after)
                self.db.refresh_node_embedding(node_id, new_euc)
        except Exception as e:
            logger.warning(f"re-embed failed for {node_id}: {e}")

    def _mark_processed(self, queue_id: Optional[int], status: str):
        if queue_id is None:
            return
        with self.db._connect() as conn:
            conn.execute("""
                UPDATE restructure_queue
                SET status = ?, processed_at = datetime('now')
                WHERE id = ?
            """, (status, queue_id))

    # -----------------------------------------------------------------
    # LLM dispatch (mirrors nightly.py patterns)
    # -----------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from ingestion.embedder import Embedder
            self._embedder = Embedder(hyperparams=self.hp)
        except Exception as e:
            logger.warning(f"embedder unavailable: {e}")
            self._embedder = None
        return self._embedder

    def _call_llm(self, prompt: str) -> str:
        if self.hp.get("use_local", True):
            return self._call_ollama(prompt)
        return self._call_anthropic(prompt)

    def _llm_label(self) -> str:
        if self.hp.get("use_local", True):
            return f"llm_regen_{self.hp.get('consolidation_model_local', 'ollama')}"
        return f"llm_regen_{self.hp.get('consolidation_model', 'claude')}"

    def _call_ollama(self, prompt: str) -> str:
        model = self.hp.get("consolidation_model_local", "qwen2.5:14b")
        max_tokens = self.hp.get("consolidation_max_tokens", 2048)
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"num_predict": max_tokens}},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def _call_anthropic(self, prompt: str) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = self.hp.get("consolidation_model", "claude-sonnet-4-20250514")
        max_tokens = self.hp.get("consolidation_max_tokens", 2048)
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"].strip()

    @staticmethod
    def _sanitize(text: str) -> str:
        import re
        cleaned = re.sub(r'\b[A-Z]{1,5}-\d{1,4}\b', '', text or '')
        cleaned = re.sub(r'[\[\(\{][A-Z]{1,5}\d{1,4}[\]\)\}]', '', cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) >= 2 and cleaned[0] in '"\u201c\'`' and cleaned[-1] in '"\u201d\'`':
            cleaned = cleaned[1:-1].strip()
        cleaned = re.sub(r'  +', ' ', cleaned)
        return cleaned.strip()
