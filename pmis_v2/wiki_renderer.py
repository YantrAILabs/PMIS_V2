"""
Wiki Renderer — Queries DB and assembles context for wiki page templates.

Two layers:
  - Front (prose): LLM-generated markdown rendered as HTML. Cached.
  - Backend (data): Live DB queries for PageValue, scores, diagnostics.

Pages: index, SC, CTX, ANC, goals, feedback, health, log, diagnostics.
"""

import json
import hashlib
import numpy as np
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class WikiRenderer:
    """Assembles data for wiki page templates from the database."""

    def __init__(self, db):
        self.db = db

    # ─── INDEX ─────────────────────────────────────────

    def render_index(self) -> Dict:
        """All SCs grouped by category with stats."""
        scs = self.db.get_nodes_by_level("SC")
        sc_list = []
        for sc in scs:
            children = self.db.get_children(sc["id"])
            # Count total descendants
            total_anc = 0
            for child in children:
                total_anc += len(self.db.get_children(child["id"]))

            # Get tree info
            tree_id = self._get_tree_id(sc["id"])

            # Feedback summary
            fb = self.db.get_feedback_summary(node_id=sc["id"])

            # Goals
            goals = self.db.get_goals_for_node(sc["id"])

            sc_list.append({
                **sc,
                "ctx_count": len(children),
                "anc_count": total_anc,
                "total_nodes": len(children) + total_anc + 1,
                "tree_id": tree_id,
                "feedback": fb,
                "goals": goals,
                "last_accessed": sc.get("last_accessed", ""),
                "is_stale": self._is_stale(sc.get("last_accessed")),
            })

        # Sort: active first, then by total nodes
        sc_list.sort(key=lambda s: (s["is_stale"], -s["total_nodes"]))

        # Global stats
        orphan_count = len(self.db.get_orphan_nodes()) if hasattr(self.db, 'get_orphan_nodes') else 0

        return {
            "super_contexts": sc_list,
            "total_scs": len(sc_list),
            "total_nodes": sum(s["total_nodes"] for s in sc_list),
            "orphan_count": orphan_count,
        }

    # ─── NODE DETAIL ──────────────────────────────────

    def render_node(self, node_id: str) -> Optional[Dict]:
        """Full detail for any node (SC, CTX, ANC)."""
        node = self.db.get_node(node_id)
        if not node:
            return None

        children = self.db.get_children(node_id)
        feedback = self.db.get_feedback_for_node(node_id)
        goals = self.db.get_goals_for_node(node_id)
        fb_score = self.db.get_feedback_score(node_id)

        # Embeddings
        embs = self.db.get_embeddings(node_id)
        hyp_norm = 0.0
        if embs.get("hyperbolic") is not None:
            hyp_norm = float(np.linalg.norm(embs["hyperbolic"]))

        # Access history
        access_history = self._get_access_history(node_id, limit=10)

        # Co-retrieval neighbors
        co_retrieved = self._get_co_retrieved(node_id, limit=5)

        # Parent chain
        parents = self._get_parent_chain(node_id)

        # Page value components
        page_value = self._compute_page_value(node, feedback, goals, co_retrieved)

        # Children with their feedback scores
        enriched_children = []
        for child in children:
            child_fb = self.db.get_feedback_score(child["id"])
            enriched_children.append({
                **child,
                "feedback_score": child_fb,
                "content_preview": child.get("content", "")[:100],
            })
        enriched_children.sort(key=lambda c: c["feedback_score"], reverse=True)

        return {
            "node": node,
            "level": node.get("level", "ANC"),
            "children": enriched_children,
            "parents": parents,
            "feedback": feedback,
            "feedback_score": fb_score,
            "goals": goals,
            "hyperbolic_norm": hyp_norm,
            "access_history": access_history,
            "co_retrieved": co_retrieved,
            "page_value": page_value,
            "tree_id": self._get_tree_id(node_id),
        }

    # ─── GOALS PAGE ───────────────────────────────────

    def render_goals(self) -> Dict:
        """All goals with linked nodes and progress."""
        goals = self.db.list_goals()
        enriched = []
        for g in goals:
            nodes = self.db.get_nodes_for_goal(g["id"])
            fb = self.db.get_feedback_summary(node_id=None)
            enriched.append({
                **g,
                "linked_nodes": nodes,
                "node_count": len(nodes),
                "feedback": fb,
            })
        return {"goals": enriched, "total": len(enriched)}

    # ─── FEEDBACK PAGE ────────────────────────────────

    def render_feedback_log(self, limit: int = 50) -> Dict:
        """Recent feedback entries across all nodes."""
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT f.*, mn.content as node_content, mn.level as node_level,
                   g.title as goal_title
            FROM feedback f
            JOIN memory_nodes mn ON mn.id = f.node_id
            LEFT JOIN goals g ON g.id = f.goal_id
            ORDER BY f.timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()
        return {"entries": [dict(r) for r in rows], "total": len(rows)}

    # ─── HEALTH PAGE ──────────────────────────────────

    def render_health(self) -> Dict:
        """Lint report: orphans, stale, oversized, unvalidated."""
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row

        # Orphans
        orphans = conn.execute("""
            SELECT id, content, created_at FROM memory_nodes
            WHERE is_orphan = 1 AND is_deleted = 0
            ORDER BY created_at DESC LIMIT 20
        """).fetchall()

        # Stale contexts (>14 days no access)
        stale = conn.execute("""
            SELECT id, content, last_accessed FROM memory_nodes
            WHERE level IN ('SC', 'CTX') AND is_deleted = 0
            AND (last_accessed IS NULL OR last_accessed < datetime('now', '-14 days'))
            ORDER BY last_accessed ASC LIMIT 20
        """).fetchall()

        # Oversized (>25 children)
        oversized = conn.execute("""
            SELECT r.target_id, mn.content, COUNT(r.source_id) as children
            FROM relations r
            JOIN memory_nodes mn ON mn.id = r.target_id
            WHERE r.relation_type = 'child_of' AND mn.is_deleted = 0
            GROUP BY r.target_id HAVING children > 25
            ORDER BY children DESC LIMIT 10
        """).fetchall()

        # Unvalidated (no feedback, accessed 3+ times)
        unvalidated = conn.execute("""
            SELECT mn.id, mn.content, mn.access_count FROM memory_nodes mn
            WHERE mn.is_deleted = 0 AND mn.access_count >= 3
            AND mn.id NOT IN (SELECT DISTINCT node_id FROM feedback)
            ORDER BY mn.access_count DESC LIMIT 20
        """).fetchall()

        conn.close()

        return {
            "orphans": [dict(r) for r in orphans],
            "stale": [dict(r) for r in stale],
            "oversized": [dict(r) for r in oversized],
            "unvalidated": [dict(r) for r in unvalidated],
            "orphan_count": len(orphans),
            "stale_count": len(stale),
            "oversized_count": len(oversized),
            "unvalidated_count": len(unvalidated),
        }

    # ─── DIAGNOSTICS PAGE ─────────────────────────────

    def render_diagnostics(self, limit: int = 50) -> Dict:
        """Recent turn diagnostics for trending."""
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM turn_diagnostics
            ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
        conn.close()

        entries = [dict(r) for r in rows]

        # Compute summary stats
        if entries:
            gammas = [e["gamma_final"] for e in entries if e.get("gamma_final")]
            hier_discs = [e["hierarchy_score_discriminative"] for e in entries if e.get("hierarchy_score_discriminative")]
            modes = {}
            for e in entries:
                m = e.get("gamma_mode", "UNKNOWN")
                modes[m] = modes.get(m, 0) + 1

            summary = {
                "gamma_mean": np.mean(gammas) if gammas else 0,
                "gamma_std": np.std(gammas) if gammas else 0,
                "hier_disc_mean": np.mean(hier_discs) if hier_discs else 0,
                "mode_distribution": modes,
                "total_turns": len(entries),
            }
        else:
            summary = {}

        return {"entries": entries, "summary": summary}

    # ─── BACKEND DATA (per node, on click) ────────────

    def render_backend(self, node_id: str) -> Optional[Dict]:
        """Backend data panel: PageValue decomposition, stats, diagnostics."""
        node_data = self.render_node(node_id)
        if not node_data:
            return None

        # Diagnostics for this node (when it was top-1)
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        diag_rows = conn.execute("""
            SELECT turn_number, gamma_final, gamma_mode,
                   top1_final_score, top1_semantic_score, top1_hierarchy_score,
                   timestamp
            FROM turn_diagnostics
            WHERE top1_node_id = ?
            ORDER BY id DESC LIMIT 10
        """, (node_id,)).fetchall()
        conn.close()

        return {
            **node_data,
            "diagnostics": [dict(r) for r in diag_rows],
        }

    # ─── PRODUCTIVITY DASHBOARD ─────────────────────

    def render_productivity(self, target_date: str = None) -> Dict:
        """Rich productivity dashboard from tracker.db + node_time_log."""
        import os
        from datetime import date as dt_date, timedelta

        if not target_date:
            target_date = dt_date.today().isoformat()

        tracker_db = os.path.expanduser("~/.productivity-tracker/tracker.db")
        if not os.path.exists(tracker_db):
            return {"has_data": False}

        conn = sqlite3.connect(tracker_db)
        conn.row_factory = sqlite3.Row

        # Total time
        totals = conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(target_segment_length_secs)/60.0, 0) as mins "
            "FROM context_1 WHERE DATE(timestamp_start) = ?", (target_date,)
        ).fetchone()

        if not totals or totals["cnt"] == 0:
            conn.close()
            return {"has_data": False}

        # Productive vs not
        productive = conn.execute(
            "SELECT COALESCE(SUM(target_segment_length_secs)/60.0, 0) as mins "
            "FROM context_1 WHERE DATE(timestamp_start) = ? AND is_productive = 1",
            (target_date,)
        ).fetchone()
        productive_mins = productive["mins"] if productive else 0

        # Worker split
        worker_split = conn.execute(
            "SELECT worker, COALESCE(SUM(target_segment_length_secs)/60.0, 0) as mins "
            "FROM context_1 WHERE DATE(timestamp_start) = ? GROUP BY worker",
            (target_date,)
        ).fetchall()
        human_mins = sum(r["mins"] for r in worker_split if r["worker"] == "human")
        ai_mins = sum(r["mins"] for r in worker_split if r["worker"] in ("ai", "agent"))

        # By window/app — NON-ZERO TIME ONLY, min 0.1 min
        by_window = conn.execute(
            "SELECT window_name as window, COUNT(*) as count, "
            "SUM(target_segment_length_secs)/60.0 as mins "
            "FROM context_1 WHERE DATE(timestamp_start) = ? "
            "AND target_segment_length_secs > 0 AND window_name IS NOT NULL "
            "GROUP BY window_name HAVING mins >= 0.1 "
            "ORDER BY mins DESC LIMIT 10",
            (target_date,)
        ).fetchall()
        by_window = [dict(r) for r in by_window]
        max_window_mins = max((r["mins"] or 0.1) for r in by_window) if by_window else 1

        # Weekly trend (last 7 days)
        today = dt_date.fromisoformat(target_date)
        weekly_trend = []
        max_weekly_mins = 1
        for i in range(6, -1, -1):
            day = today - timedelta(days=i)
            day_str = day.isoformat()
            row = conn.execute(
                "SELECT COALESCE(SUM(target_segment_length_secs)/60.0, 0) as mins "
                "FROM context_1 WHERE DATE(timestamp_start) = ?", (day_str,)
            ).fetchone()
            mins = row["mins"] if row else 0
            max_weekly_mins = max(max_weekly_mins, mins)
            weekly_trend.append({
                "date": day_str,
                "label": day.strftime("%a"),
                "mins": mins,
                "is_today": day_str == target_date,
            })

        # Hourly distribution
        hourly = []
        max_hourly_mins = 1
        for h in range(24):
            row = conn.execute(
                "SELECT COALESCE(SUM(target_segment_length_secs)/60.0, 0) as mins "
                "FROM context_1 WHERE DATE(timestamp_start) = ? "
                "AND CAST(STRFTIME('%%H', timestamp_start) AS INTEGER) = ?",
                (target_date, h)
            ).fetchone()
            mins = row["mins"] if row else 0
            max_hourly_mins = max(max_hourly_mins, mins)
            hourly.append({"hour": h, "mins": mins})

        # Input stats from context_2
        input_row = conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(has_keyboard_activity) as kb, SUM(has_mouse_activity) as mouse "
            "FROM context_2 WHERE DATE(frame_timestamp) = ?",
            (target_date,)
        ).fetchone()
        input_stats = {
            "total_frames": input_row["total"] or 0,
            "keyboard_frames": input_row["kb"] or 0,
            "mouse_frames": input_row["mouse"] or 0,
        } if input_row else None

        # Recent frames
        recent_frames = conn.execute(
            "SELECT frame_timestamp as timestamp, detailed_summary as summary, "
            "worker_type as worker "
            "FROM context_2 WHERE DATE(frame_timestamp) = ? "
            "ORDER BY frame_timestamp DESC LIMIT 15",
            (target_date,)
        ).fetchall()
        recent_frames = [dict(r) for r in recent_frames]

        conn.close()

        # Branch time from node_time_log
        branch_time = []
        try:
            pmis_conn = sqlite3.connect(self.db.db_path)
            pmis_conn.row_factory = sqlite3.Row
            rows = pmis_conn.execute("""
                SELECT ntl.node_id, ntl.total_duration_mins, ntl.segment_count,
                       ntl.project_id, mn.content as node_content
                FROM node_time_log ntl
                JOIN memory_nodes mn ON mn.id = ntl.node_id
                WHERE ntl.date = ? AND ntl.total_duration_mins > 0
                ORDER BY ntl.total_duration_mins DESC
            """, (target_date,)).fetchall()
            branch_time = [dict(r) for r in rows]
            pmis_conn.close()
        except Exception:
            pass

        return {
            "has_data": True,
            "total_segments": totals["cnt"],
            "total_duration_mins": totals["mins"],
            "productive_mins": productive_mins,
            "human_mins": human_mins,
            "ai_mins": ai_mins,
            "by_window": by_window,
            "max_window_mins": max_window_mins,
            "weekly_trend": weekly_trend,
            "max_weekly_mins": max_weekly_mins,
            "hourly": hourly,
            "max_hourly_mins": max_hourly_mins,
            "input_stats": input_stats,
            "recent_frames": recent_frames,
            "branch_time": branch_time,
        }

    # ─── LLM PROSE GENERATION ─────────────────────────

    def generate_wiki_prose(self, node_id: str) -> Optional[str]:
        """Generate LLM-written wiki prose for a node. Caches result."""
        import hashlib

        # Check cache first
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        cached = conn.execute(
            "SELECT prose_markdown, context_hash FROM wiki_page_cache WHERE node_id=?",
            (node_id,)
        ).fetchone()

        # Build context for generation
        node_data = self.render_node(node_id)
        if not node_data:
            conn.close()
            return None

        # Create content hash to detect if data changed
        content_parts = [node_data["node"].get("content", "")]
        for c in node_data.get("children", [])[:30]:
            content_parts.append(c.get("content_preview", ""))
        context_hash = hashlib.md5("|".join(content_parts).encode()).hexdigest()[:16]

        # Return cache if still valid
        if cached and cached["context_hash"] == context_hash and cached["prose_markdown"]:
            conn.close()
            return cached["prose_markdown"]

        conn.close()

        # Generate prose via LLM
        prose = self._call_llm_for_prose(node_data)
        if not prose:
            return None

        # Cache it
        conn2 = sqlite3.connect(self.db.db_path)
        conn2.execute("""
            INSERT OR REPLACE INTO wiki_page_cache
            (node_id, prose_markdown, context_hash, generated_at, llm_model, word_count)
            VALUES (?, ?, ?, datetime('now'), ?, ?)
        """, (node_id, prose, context_hash, "qwen2.5:3b", len(prose.split())))
        conn2.commit()
        conn2.close()

        return prose

    def _call_llm_for_prose(self, node_data: Dict) -> Optional[str]:
        """Call local Ollama to generate wiki prose from node data."""
        node = node_data["node"]
        level = node_data["level"]
        children = node_data.get("children", [])

        # Build prompt based on level
        if level == "SC":
            prompt = self._build_sc_prompt(node, children)
        elif level in ("CTX", "CTX-1", "CTX-2"):
            prompt = self._build_ctx_prompt(node, children, node_data)
        else:
            prompt = self._build_anc_prompt(node, node_data)

        # Call Ollama
        try:
            import httpx
            resp = httpx.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 2000},
                },
                timeout=180,
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip()
                text = text.replace("```markdown", "").replace("```", "")
                return text
        except Exception as e:
            import logging
            logging.getLogger("pmis.wiki").warning(f"LLM prose generation failed: {e}")

        # Fallback: template-based prose from tree data
        return self._generate_template_prose(node_data)

    def _build_sc_prompt(self, node: Dict, children: List[Dict]) -> str:
        """Build prompt for SC-level wiki page."""
        sc_name = node.get("content", "")[:100]

        # Get anchors for each CTX child
        sections = []
        # Limit to top 10 CTXs by access count, 3 anchors each (keep prompt manageable)
        sorted_ctx = sorted(children, key=lambda c: c.get("access_count", 0), reverse=True)
        for ctx in sorted_ctx[:10]:
            ctx_id = ctx.get("id", "")
            ctx_name = ctx.get("content_preview", ctx.get("content", ""))[:60]

            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            anchors = conn.execute("""
                SELECT mn.content FROM relations r
                JOIN memory_nodes mn ON mn.id = r.source_id
                WHERE r.target_id=? AND r.relation_type='child_of' AND mn.is_deleted=0
                ORDER BY mn.access_count DESC LIMIT 3
            """, (ctx_id,)).fetchall()
            conn.close()

            anchor_texts = [a["content"][:120] for a in anchors]
            if anchor_texts:
                sections.append(f"TOPIC: {ctx_name}\nKEY KNOWLEDGE:\n" + "\n".join(f"- {a}" for a in anchor_texts))

        return f"""You are writing a wiki page for a personal knowledge base.

DOMAIN: {sc_name}

This domain has {len(children)} sub-topics:

{chr(10).join(sections)}

Write a comprehensive wiki article following these rules:
- Start with a 2-3 sentence overview of the domain
- Create one H2 section per major topic (group related topics together, skip trivial ones)
- Within each section, weave the knowledge into readable prose paragraphs
- Use specific numbers, facts, and details from the knowledge points
- Write as if explaining to a colleague who just joined the team
- Keep it under 1000 words
- NO reference codes, IDs, node numbers, or technical metadata
- NO bullet lists — write in prose paragraphs
- Use markdown: # for title, ## for sections, **bold** for key terms
"""

    def _build_ctx_prompt(self, node: Dict, children: List[Dict], node_data: Dict) -> str:
        """Build prompt for CTX-level wiki page."""
        ctx_name = node.get("content", "")[:100]
        parents = node_data.get("parents", [])
        parent_name = parents[0].get("content", "")[:60] if parents else "Unknown"

        anchor_texts = [c.get("content_preview", c.get("content", ""))[:150] for c in children[:10]]

        return f"""You are writing a wiki page for a personal knowledge base.

TOPIC: {ctx_name}
PARENT DOMAIN: {parent_name}

KEY KNOWLEDGE POINTS:
{chr(10).join(f"- {a}" for a in anchor_texts)}

Write a focused wiki article (400-600 words):
- Start with 1-2 sentence overview of this topic
- Weave ALL the knowledge points into readable prose paragraphs
- Use specific numbers and facts
- Write as if explaining to a colleague
- NO bullet lists, NO IDs, NO metadata
- Use markdown: # for title, ## for sub-sections if needed
"""

    def _build_anc_prompt(self, node: Dict, node_data: Dict) -> str:
        """Build prompt for ANC-level knowledge card."""
        content = node.get("content", "")[:300]
        parents = node_data.get("parents", [])
        parent_name = parents[0].get("content", "")[:60] if parents else ""

        return f"""You are writing a knowledge card for a personal wiki.

INSIGHT: {content}
CONTEXT: Part of "{parent_name}"

Write a concise knowledge card (100-200 words):
- State the core insight clearly in the first sentence
- Add context about why this matters
- Include any specific numbers or evidence
- NO metadata, NO IDs
- Use markdown
"""

    def _generate_template_prose(self, node_data: Dict) -> str:
        """Template-based prose generation when LLM is unavailable.
        Reads tree data and writes a structured wiki page from it."""
        node = node_data["node"]
        level = node_data.get("level", "ANC")
        children = node_data.get("children", [])
        content = node.get("content", "")

        if level == "SC":
            return self._template_sc_prose(content, children)
        elif level in ("CTX", "CTX-1", "CTX-2"):
            return self._template_ctx_prose(content, children, node_data)
        else:
            return self._template_anc_prose(content, node_data)

    def _template_sc_prose(self, sc_content: str, children: List[Dict]) -> str:
        """Generate readable SC page using tree structure for headings."""
        sc_name = sc_content.split(".")[0].strip()
        desc = sc_content[len(sc_name):].strip(". ")

        lines = [f"# {sc_name}\n"]
        if desc:
            lines.append(f"{desc}\n")

        # Group related CTXs into logical sections
        sorted_children = sorted(children, key=lambda c: c.get("access_count", 0), reverse=True)

        # Only include CTXs with anchors
        active_ctxs = []
        for ctx in sorted_children:
            ctx_id = ctx.get("id", "")
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            anchors = conn.execute("""
                SELECT mn.id, mn.content, mn.access_count FROM relations r
                JOIN memory_nodes mn ON mn.id = r.source_id
                WHERE r.target_id=? AND r.relation_type='child_of' AND mn.is_deleted=0
                ORDER BY mn.access_count DESC
            """, (ctx_id,)).fetchall()
            conn.close()
            if anchors:
                active_ctxs.append({
                    "name": ctx.get("content_preview", ctx.get("content", "")).split(".")[0].strip(),
                    "id": ctx_id,
                    "access": ctx.get("access_count", 0),
                    "anchors": anchors,
                })

        lines.append(f"This knowledge base covers **{len(active_ctxs)} key areas** across the domain.\n")

        # Table of contents
        lines.append("---\n")
        for ctx_data in active_ctxs:
            slug = ctx_data["name"].lower().replace(" ", "-").replace("&", "and").replace("/", "-")
            slug = ''.join(c for c in slug if c.isalnum() or c == '-')
            lines.append(f'[{ctx_data["name"]}](#{slug}) &middot; ')
        lines.append("\n---\n")

        for ctx_data in active_ctxs:
            ctx_name = ctx_data["name"]
            anchors = ctx_data["anchors"]

            slug = ctx_name.lower().replace(" ", "-").replace("&", "and").replace("/", "-")
            slug = ''.join(c for c in slug if c.isalnum() or c == '-')
            lines.append(f'\n## <span id="{slug}">{ctx_name}</span>\n')

            # Write each anchor as a paragraph with bold lead-in
            for anc in anchors[:5]:
                text = anc["content"]
                # Split into headline + detail if there's a natural break
                if ". " in text:
                    parts = text.split(". ", 1)
                    headline = parts[0].strip()
                    detail = parts[1].strip()
                    if len(headline) < 80 and detail:
                        lines.append(f"**{headline}.** {detail}\n")
                    else:
                        lines.append(f"{text}\n")
                else:
                    lines.append(f"{text}\n")

        # Related section
        lines.append("\n---\n")
        lines.append(f"*This page is part of the PMIS knowledge wiki. "
                     f"It covers {sum(len(c['anchors']) for c in active_ctxs)} knowledge anchors "
                     f"across {len(active_ctxs)} contexts.*")

        return "\n".join(lines)

    def _template_ctx_prose(self, content: str, children: List[Dict], node_data: Dict) -> str:
        """Generate readable CTX page."""
        ctx_name = content.split(".")[0].strip()
        parents = node_data.get("parents", [])
        parent_name = parents[0].get("content", "")[:40] if parents else ""

        lines = [f"# {ctx_name}\n"]
        if parent_name:
            lines.append(f"Part of **{parent_name.split('.')[0]}**.\n")

        for child in children[:10]:
            text = child.get("content_preview", child.get("content", ""))
            if ". " in text:
                parts = text.split(". ", 1)
                if len(parts[0]) < 60:
                    text = f"**{parts[0]}.** {parts[1]}"
            lines.append(f"{text}\n")

        return "\n".join(lines)

    def _template_anc_prose(self, content: str, node_data: Dict) -> str:
        """Generate readable ANC card."""
        parents = node_data.get("parents", [])
        parent_name = parents[0].get("content", "")[:40] if parents else ""

        lines = []
        if ". " in content:
            parts = content.split(". ", 1)
            lines.append(f"# {parts[0]}\n")
            lines.append(f"{parts[1]}\n")
        else:
            lines.append(f"# Knowledge\n\n{content}\n")

        if parent_name:
            lines.append(f"\nPart of **{parent_name.split('.')[0]}**.")

        return "\n".join(lines)

    # ─── HELPERS ──────────────────────────────────────

    def _compute_page_value(self, node, feedback, goals, co_retrieved) -> Dict:
        """Compute PageValue decomposition."""
        components = {}
        active_weights = {}

        # Goal
        if goals:
            STATUS_MULT = {"active": 1.0, "achieved": 0.3, "paused": 0.1, "abandoned": 0.0}
            goal_scores = [
                g.get("weight", 0.5) * STATUS_MULT.get(g.get("status", "active"), 0.5)
                for g in goals
            ]
            components["goal"] = min(np.mean(goal_scores), 1.0) if goal_scores else 0
            active_weights["goal"] = 0.30

        # Feedback
        if feedback:
            pos = sum(1 for f in feedback if f.get("polarity") == "positive")
            neg = sum(1 for f in feedback if f.get("polarity") == "negative")
            raw = pos * 0.10 - neg * 0.15
            components["feedback"] = 1.0 / (1.0 + np.exp(-raw * 5.0))
            active_weights["feedback"] = 0.25

        # Usage
        access_count = node.get("access_count", 0)
        if access_count > 0:
            components["usage"] = min(np.log1p(access_count) / np.log1p(50), 1.0)
            active_weights["usage"] = 0.20

        # Recency (always active)
        last_acc = node.get("last_accessed")
        if last_acc:
            try:
                hours = (datetime.now() - datetime.fromisoformat(last_acc)).total_seconds() / 3600
            except (ValueError, TypeError):
                hours = 720
        else:
            hours = 720
        components["recency"] = float(np.exp(-hours / 720))
        active_weights["recency"] = 0.10

        # Linkage
        if co_retrieved:
            components["linkage"] = min(np.log1p(len(co_retrieved)) / np.log1p(20), 1.0)
            active_weights["linkage"] = 0.10

        # Structural (always)
        components["structural"] = 0.5
        active_weights["structural"] = 0.05

        # Normalize
        total_w = sum(active_weights.values()) or 1.0
        normalized = {k: v / total_w for k, v in active_weights.items()}

        page_value = sum(normalized.get(k, 0) * v for k, v in components.items())

        return {
            "value": round(page_value, 3),
            "components": {k: round(v, 3) for k, v in components.items()},
            "weights": {k: round(v, 3) for k, v in normalized.items()},
            "active_count": len(active_weights),
        }

    def _get_tree_id(self, node_id: str) -> Optional[str]:
        conn = sqlite3.connect(self.db.db_path)
        row = conn.execute("""
            SELECT tree_id FROM relations
            WHERE (source_id = ? OR target_id = ?) AND tree_id != 'default'
            LIMIT 1
        """, (node_id, node_id)).fetchone()
        conn.close()
        return row[0] if row else None

    def _is_stale(self, last_accessed) -> bool:
        if not last_accessed:
            return True
        try:
            dt = datetime.fromisoformat(last_accessed)
            return (datetime.now() - dt).days > 14
        except (ValueError, TypeError):
            return True

    def _get_access_history(self, node_id: str, limit: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT accessed_at, query_text, gamma_at_access, surprise_at_access
            FROM access_log WHERE node_id = ?
            ORDER BY accessed_at DESC LIMIT ?
        """, (node_id, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _get_co_retrieved(self, node_id: str, limit: int = 5) -> List[Dict]:
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT t2.memory_node_id as co_node_id,
                   mn.content as co_content, mn.level as co_level,
                   COUNT(*) as co_count
            FROM turn_retrieved_memories t1
            JOIN turn_retrieved_memories t2
                ON t1.turn_id = t2.turn_id AND t1.memory_node_id != t2.memory_node_id
            JOIN memory_nodes mn ON mn.id = t2.memory_node_id
            WHERE t1.memory_node_id = ? AND mn.is_deleted = 0
            GROUP BY t2.memory_node_id
            ORDER BY co_count DESC
            LIMIT ?
        """, (node_id, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _get_parent_chain(self, node_id: str) -> List[Dict]:
        """Walk up the hierarchy from node to SC root."""
        chain = []
        current = node_id
        visited = set()
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row

        while current and current not in visited:
            visited.add(current)
            row = conn.execute("""
                SELECT r.target_id, mn.content, mn.level
                FROM relations r
                JOIN memory_nodes mn ON mn.id = r.target_id
                WHERE r.source_id = ? AND r.relation_type = 'child_of'
                LIMIT 1
            """, (current,)).fetchone()
            if row:
                chain.append(dict(row))
                current = row["target_id"]
            else:
                break

        conn.close()
        return chain
