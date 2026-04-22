"""
Phase 3.5 — Meta-LLM Problem Statement Composer.

Given an active deliverable, compose a crisp `problem_statement.md` that a
separate Claude agent (Phase 4 harness) will consume to draft an
implementation plan.

Division of labor (locked decision from earlier rounds):
  • PMIS  — owns the problem statement (this module). Meta-LLM has its own
             system prompt role: synthesize memory + deliverable into a brief.
  • Claude — owns the plan and execution. Reads the brief, drafts plan.md,
             shows to user for approval, then runs [CLAUDE]/[USER]/[REVIEW]
             steps.

This module is deterministic in its inputs (deliverable + ranked anchors +
activity signals) so the output is reproducible for a given DB snapshot.

Two modes:
  • `render_prompt(...)` — pure template assembly, no LLM call. Deterministic.
    Useful for previewing what the meta-LLM will see, and as a fallback when
    no LLM is reachable.
  • `compose(...)` — calls the meta-LLM with a dedicated system prompt and
    returns a polished problem_statement.md.

Uses the same ollama/Anthropic dispatch as `consolidation/nightly.py` via a
local helper (kept inline rather than refactored into a shared module so
Phase 3.5 stays contained).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("pmis.meta_composer")


# ─── Dedicated system prompt for the Meta-LLM role ───────────────────
META_SYSTEM_PROMPT = """\
You are the Problem Statement Composer for PMIS — a productivity memory
system. Your job is NOT to solve the user's task. Your job is to produce a
precise, self-contained problem_statement.md that a separate Claude agent
will use to draft an implementation plan.

Output MUST follow this structure exactly:

# <Deliverable name>

## Goal
One or two sentences. What does "done" look like.

## Why it matters
One paragraph. Tie to the user's project goal or current constraint.

## What we know (from memory)
Bulleted synthesis of the top value_score anchors. Each bullet cites the
relevant memory in-line (first 8 words, no IDs). Prefer patterns ("we've
used X before") over raw facts.

## What we've tried / what to avoid
Only include if redflagged or negative-feedback anchors were provided.
Otherwise write "No known pitfalls logged."

## Signals from current activity
Summarize the active segment + recent window in 2-3 bullets: what apps,
what files, what the tracker observed.

## Out of scope
List 2-4 things the agent should explicitly NOT do. This keeps the plan
focused.

Rules:
- Be specific. Numbers and file paths where known.
- Don't speculate beyond the input. If memory is thin, say so.
- Don't write the plan. Don't propose steps. Don't decide [CLAUDE]/[USER]/
  [REVIEW] tagging — that's the executor agent's job.
- No reference codes, no IDs in prose, no "PM-25" style labels.
- Return ONLY the markdown, no preamble, no apology, no commentary.
"""


@dataclass
class ComposerBundle:
    """What PMIS hands to Claude. The Markdown is authoritative; the dict
    metadata exists so the Phase 4 harness runner can cross-reference anchors
    without re-parsing prose."""

    deliverable_id: str
    deliverable_name: str
    project_name: str
    problem_statement_md: str
    anchors_used: List[Dict[str, Any]] = field(default_factory=list)
    latest_segment: Optional[Dict[str, Any]] = None
    generated_at: str = ""
    model_used: str = ""
    mode: str = "llm"            # "llm" or "template" (if LLM unavailable)
    composer_version: str = "phase-3.5"

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class ProblemStatementComposer:
    def __init__(self, db, embedder, hyperparams: Optional[Dict[str, Any]] = None,
                 brief_composer=None):
        self.db = db
        self.embedder = embedder
        self.hp = hyperparams or {}
        self._brief_composer = brief_composer  # optional — lazy construct if None
        self.max_anchors = int(self.hp.get("meta_composer_max_anchors", 6))
        self.max_negative = int(self.hp.get("meta_composer_max_negative", 3))
        self.max_tokens = int(self.hp.get("meta_composer_max_tokens", 1400))

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def compose(
        self,
        deliverable_id: str,
        use_llm: bool = True,
        include_activity: bool = True,
    ) -> ComposerBundle:
        deliverable = self.db.get_deliverable(deliverable_id)
        if not deliverable:
            raise ValueError(f"deliverable_not_found: {deliverable_id}")

        brief = self._get_brief(deliverable_id)
        latest_segment = self._get_latest_segment() if include_activity else None

        # Assemble ranked anchors (use value-blended rankings Phase 3 gave us)
        positives = (brief.get("claude_can_do", []) + brief.get("you_did_before", []))[
            : self.max_anchors
        ]
        negatives = self._fetch_redflag_anchors(deliverable_id)[: self.max_negative]

        prompt = self._render_user_prompt(
            deliverable=deliverable,
            positives=positives,
            negatives=negatives,
            latest_segment=latest_segment,
        )

        md, model_used, mode = "", "", "template"
        if use_llm:
            try:
                md = self._call_meta_llm(prompt)
                model_used = self._llm_model_name()
                mode = "llm"
            except Exception as e:
                logger.warning("Meta-LLM call failed, falling back to template: %s", e)
                md = self._template_fallback(deliverable, positives, negatives, latest_segment)
                mode = "template"
        else:
            md = self._template_fallback(deliverable, positives, negatives, latest_segment)

        return ComposerBundle(
            deliverable_id=deliverable_id,
            deliverable_name=deliverable.get("name", ""),
            project_name=deliverable.get("project_name", ""),
            problem_statement_md=md.strip(),
            anchors_used=[{
                "node_id": a.get("node_id"),
                "preview": (a.get("preview") or a.get("content", ""))[:200],
                "similarity": a.get("similarity"),
                "automatability": a.get("automatability"),
                "value_score": a.get("value_score", 0.0),
                "is_negative": a.get("is_negative", False),
            } for a in positives + [dict(n, is_negative=True) for n in negatives]],
            latest_segment=latest_segment,
            generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            model_used=model_used,
            mode=mode,
        )

    def render_prompt(self, deliverable_id: str) -> str:
        """Return the user-prompt Markdown that would be handed to the meta-LLM.
        Useful for /api/work/compose-problem?preview=1."""
        deliverable = self.db.get_deliverable(deliverable_id)
        if not deliverable:
            raise ValueError(f"deliverable_not_found: {deliverable_id}")
        brief = self._get_brief(deliverable_id)
        positives = (brief.get("claude_can_do", []) + brief.get("you_did_before", []))[
            : self.max_anchors
        ]
        negatives = self._fetch_redflag_anchors(deliverable_id)[: self.max_negative]
        return self._render_user_prompt(
            deliverable=deliverable,
            positives=positives,
            negatives=negatives,
            latest_segment=self._get_latest_segment(),
        )

    # ------------------------------------------------------------------
    # Input assembly
    # ------------------------------------------------------------------

    def _get_brief(self, deliverable_id: str) -> Dict[str, Any]:
        if self._brief_composer is None:
            from retrieval.brief_composer import BriefComposer
            self._brief_composer = BriefComposer(self.db, self.embedder, hyperparams=self.hp)
        return self._brief_composer.compose(deliverable_id)

    def _get_latest_segment(self) -> Optional[Dict[str, Any]]:
        try:
            from retrieval.live_matcher import LiveMatcher
            lm = LiveMatcher(self.db, self.embedder, hyperparams=self.hp)
            seg = lm.get_latest_segment()
            if not seg:
                return None
            return {
                "segment_id": seg.segment_id,
                "summary": seg.summary,
                "window_name": seg.window_name,
                "platform": seg.platform,
                "length_secs": seg.length_secs,
                "timestamp_start": seg.timestamp_start,
            }
        except Exception as e:
            logger.debug("Latest segment unavailable: %s", e)
            return None

    def _fetch_redflag_anchors(self, deliverable_id: str) -> List[Dict[str, Any]]:
        """Pull redflagged (value_feedback <= -0.3) anchors scoped to this
        deliverable's project. Phase 2 BriefComposer filters these out of
        positive buckets — but the composer surfaces them as pitfalls."""
        threshold = float(self.hp.get("value_feedback_redflag", -0.3))
        with self.db._connect() as conn:
            rows = conn.execute(
                """SELECT id, content, value_feedback, value_score
                   FROM memory_nodes
                   WHERE is_deleted = 0 AND level = 'ANC'
                     AND value_feedback <= ?
                   ORDER BY value_feedback ASC LIMIT 20""",
                (threshold,),
            ).fetchall()
        return [
            {
                "node_id": r["id"],
                "preview": (r["content"] or "")[:200],
                "content": r["content"] or "",
                "value_feedback": float(r["value_feedback"] or 0.0),
                "value_score": float(r["value_score"] or 0.0),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_user_prompt(
        self,
        deliverable: Dict[str, Any],
        positives: List[Dict[str, Any]],
        negatives: List[Dict[str, Any]],
        latest_segment: Optional[Dict[str, Any]],
    ) -> str:
        parts: List[str] = []
        parts.append("## Deliverable")
        parts.append(f"- Name: {deliverable.get('name')}")
        parts.append(f"- Project: {deliverable.get('project_name') or deliverable.get('project_id')}")
        if deliverable.get("description"):
            parts.append(f"- Description: {deliverable['description']}")
        if deliverable.get("deadline"):
            parts.append(f"- Deadline: {deliverable['deadline']}")

        parts.append("")
        parts.append("## Ranked memory anchors (value_score-weighted)")
        if not positives:
            parts.append("_No anchors matched._")
        else:
            for i, a in enumerate(positives, 1):
                sim = a.get("similarity", 0.0) or 0.0
                val = a.get("value_score", 0.0) or 0.0
                auto = a.get("automatability", 0.0) or 0.0
                preview = (a.get("preview") or a.get("content") or "").replace("\n", " ")[:260]
                parts.append(
                    f"{i}. [sim={sim:.2f} value={val:.2f} auto={auto:.2f}] {preview}"
                )

        parts.append("")
        parts.append("## Redflagged pitfalls (negative feedback)")
        if not negatives:
            parts.append("_None logged._")
        else:
            for i, n in enumerate(negatives, 1):
                preview = (n.get("preview") or n.get("content") or "").replace("\n", " ")[:240]
                parts.append(f"{i}. [feedback={n.get('value_feedback'):+.2f}] {preview}")

        parts.append("")
        parts.append("## Current activity signal")
        if latest_segment:
            mins = int((latest_segment.get("length_secs") or 0) / 60)
            parts.append(
                f"- Latest segment: {mins} min on {latest_segment.get('platform') or 'unknown'} "
                f"(window: {latest_segment.get('window_name') or '—'})"
            )
            if latest_segment.get("summary"):
                parts.append(f"- Summary: {(latest_segment['summary'] or '')[:300]}")
        else:
            parts.append("_No active segment._")

        parts.append("")
        parts.append("Now produce the problem_statement.md per the system prompt spec.")
        return "\n".join(parts)

    def _template_fallback(
        self,
        deliverable: Dict[str, Any],
        positives: List[Dict[str, Any]],
        negatives: List[Dict[str, Any]],
        latest_segment: Optional[Dict[str, Any]],
    ) -> str:
        """Deterministic no-LLM rendering. Used when the LLM is unreachable
        (no ANTHROPIC_API_KEY + ollama down). Maintains the same section
        structure so Phase 4 can consume either mode."""
        lines: List[str] = []
        lines.append(f"# {deliverable.get('name','Untitled deliverable')}")
        lines.append("")
        lines.append("## Goal")
        desc = deliverable.get("description") or deliverable.get("name") or ""
        lines.append(desc or "No description provided.")
        lines.append("")
        lines.append("## Why it matters")
        lines.append(
            f"Tracked under project **{deliverable.get('project_name','')}**. "
            f"Deadline: {deliverable.get('deadline') or 'unset'}."
        )
        lines.append("")
        lines.append("## What we know (from memory)")
        if not positives:
            lines.append("- Memory is thin — no anchors scored above threshold for this deliverable.")
        else:
            for a in positives:
                preview = (a.get("preview") or a.get("content") or "").replace("\n", " ")[:180]
                lines.append(f"- {preview}")
        lines.append("")
        lines.append("## What we've tried / what to avoid")
        if not negatives:
            lines.append("No known pitfalls logged.")
        else:
            for n in negatives:
                preview = (n.get("preview") or n.get("content") or "").replace("\n", " ")[:180]
                lines.append(f"- ⚠ {preview}")
        lines.append("")
        lines.append("## Signals from current activity")
        if latest_segment:
            mins = int((latest_segment.get("length_secs") or 0) / 60)
            lines.append(
                f"- Latest segment: **{mins} min** on {latest_segment.get('platform') or '—'} "
                f"(`{latest_segment.get('window_name') or '—'}`)"
            )
            if latest_segment.get("summary"):
                lines.append(f"- Summary: {(latest_segment['summary'] or '')[:240]}")
        else:
            lines.append("- No active segment.")
        lines.append("")
        lines.append("## Out of scope")
        lines.append("- Anything unrelated to the deliverable's description above.")
        lines.append("- Refactoring adjacent modules unless required by the goal.")
        lines.append("- Changing hyperparameters, schemas, or infra beyond this deliverable.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Meta-LLM dispatch (ollama or Anthropic)
    # ------------------------------------------------------------------

    def _llm_model_name(self) -> str:
        if self.hp.get("use_local", True):
            return self.hp.get("consolidation_model_local", "qwen2.5:14b")
        return self.hp.get("consolidation_model", "claude-sonnet-4-20250514")

    def _call_meta_llm(self, user_prompt: str) -> str:
        if self.hp.get("use_local", True):
            return self._call_ollama(user_prompt)
        return self._call_anthropic(user_prompt)

    def _call_ollama(self, user_prompt: str) -> str:
        model = self.hp.get("consolidation_model_local", "qwen2.5:14b")
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": META_SYSTEM_PROMPT + "\n\n---\n\n" + user_prompt,
                "stream": False,
                "options": {"num_predict": self.max_tokens, "temperature": 0.3},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def _call_anthropic(self, user_prompt: str) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        model = self.hp.get("consolidation_model", "claude-sonnet-4-20250514")
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": self.max_tokens,
                "system": META_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"].strip()
