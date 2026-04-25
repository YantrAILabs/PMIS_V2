"""
Wiki Renderer — Queries DB and assembles context for wiki page templates.

Two layers:
  - Front (prose): LLM-generated markdown rendered as HTML. Cached.
  - Backend (data): Live DB queries for PageValue, scores, diagnostics.

Pages: index, SC, CTX, ANC, goals, feedback, health, log, diagnostics.
"""

import json
import hashlib
import logging
import numpy as np
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger("pmis.wiki_renderer")


def _section_md_to_html(body_md: str) -> str:
    """Phase C2: render deliverable section markdown to HTML by reusing
    server._markdown_to_html so all wiki surfaces share the same
    converter. Lazy-imported to avoid the wiki_renderer ↔ server import
    cycle (server imports wiki_renderer at module load)."""
    if not body_md:
        return ""
    try:
        from server import _markdown_to_html as _to_html
        return _to_html(body_md)
    except Exception:
        from html import escape
        return f"<pre>{escape(body_md)}</pre>"
from pathlib import Path

# Repo root (parent of pmis_v2/) — productivity-tracker/ lives as a sibling.
# Resolved relative to this file so Windows / Linux / macOS all work regardless
# of where the user cloned the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent


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

        # Shell sidebar / rail data for /wiki/ index.
        ws_payload = self.render_workspaces()
        total_nodes = sum(s["total_nodes"] for s in sc_list)
        return {
            "super_contexts": sc_list,
            "total_scs": len(sc_list),
            "total_nodes": total_nodes,
            "orphan_count": orphan_count,
            "workspaces": ws_payload["workspaces"],
            "workspace_active_id": "",
            "review_counts": ws_payload["review"],
            "index_stats": {
                "visible_scs": len([w for w in ws_payload["workspaces"]
                                    if w["status"] != "stale"]),
                "total_scs": len(sc_list),
                "total_nodes": total_nodes,
                "orphan_count": orphan_count,
            },
        }

    # ─── WORKSPACE TREE (Phase D1 — sidebar + cmd-K data source) ───

    def render_workspaces(self) -> Dict:
        """SC tree with freshness signals for the unified sidebar and
        cmd-K palette. Slimmer than render_index — no feedback/goals joins,
        context list trimmed to what a sidebar needs."""
        from datetime import date as _date, timedelta as _timedelta

        today = _date.today()
        scs = self.db.get_nodes_by_level("SC")

        out: List[Dict] = []
        unclassified_count = 0
        stale_count = 0

        for sc in scs:
            children = self.db.get_children(sc["id"])
            anchor_count = 0
            for child in children:
                anchor_count += len(self.db.get_children(child["id"]))
            node_count = len(children) + anchor_count + 1

            last_accessed = sc.get("last_accessed") or ""
            days_dormant: Optional[int] = None
            last_active_date: Optional[str] = None
            if last_accessed:
                try:
                    dt = datetime.fromisoformat(last_accessed)
                    last_active_date = dt.date().isoformat()
                    days_dormant = (today - dt.date()).days
                except (ValueError, TypeError):
                    pass

            # Status buckets
            title_lower = (sc.get("content") or "").strip().lower()
            if title_lower.startswith("unclassified"):
                status = "stale"
            elif days_dormant is None:
                status = "stale"
            elif days_dormant <= 7:
                status = "active"
            elif days_dormant <= 30:
                status = "recent"
            else:
                status = "stale"
            if node_count <= 1:  # SCs with no contexts/anchors
                status = "stale"

            if status == "stale":
                stale_count += 1
            if title_lower.startswith("unclassified"):
                unclassified_count += 1

            # Trim contexts to sidebar-friendly shape
            ctx_slim: List[Dict] = []
            for ctx in children:
                anchors = self.db.get_children(ctx["id"])
                ctx_slim.append({
                    "id": ctx["id"],
                    "title": (ctx.get("content") or "").split("\n", 1)[0][:120],
                    "anchor_count": len(anchors),
                    "precision": round(float(ctx.get("precision") or 0.5), 2),
                })
            ctx_slim.sort(key=lambda c: -c["anchor_count"])

            title_full = (sc.get("content") or "").split(".", 1)
            title = title_full[0].strip() if title_full else sc["id"]
            description = title_full[1].strip() if len(title_full) > 1 else ""

            out.append({
                "id": sc["id"],
                "title": title,
                "description": description,
                "node_count": node_count,
                "context_count": len(children),
                "anchor_count": anchor_count,
                "last_active": last_active_date,
                "days_dormant": days_dormant,
                "status": status,
                "contexts": ctx_slim,
            })

        # Sort: biggest workspaces first (matches /wiki/ index behaviour).
        out.sort(key=lambda w: -w["node_count"])

        orphan_count = (
            len(self.db.get_orphan_nodes())
            if hasattr(self.db, "get_orphan_nodes") else 0
        )

        return {
            "count": len(out),
            "current_date": today.isoformat(),
            "workspaces": out,
            "review": {
                "unclassified_count": unclassified_count,
                "stale_count": stale_count,
                "orphan_count": orphan_count,
            },
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

        # Phase 3 — unified value_score from materialized columns
        value_v3 = {
            "score": float(node.get("value_score") or 0.0),
            "G": float(node.get("value_goal") or 0.0),
            "F": float(node.get("value_feedback") or 0.0),
            "U": float(node.get("value_usage") or 0.0),
            "R": float(node.get("value_recency") or 0.0),
            "computed_at": node.get("value_computed_at") or "",
            "redflag": float(node.get("value_feedback") or 0.0) <= -0.3,
        }

        # Children with their feedback scores + Phase 3 value_score
        enriched_children = []
        for child in children:
            child_fb = self.db.get_feedback_score(child["id"])
            enriched_children.append({
                **child,
                "feedback_score": child_fb,
                "content_preview": child.get("content", "")[:100],
                "value_score": float(child.get("value_score") or 0.0),
                "value_feedback": float(child.get("value_feedback") or 0.0),
                "is_pitfall": float(child.get("value_feedback") or 0.0) <= -0.3,
            })
        # Phase 5 — rank children by value_score (was feedback_score)
        enriched_children.sort(key=lambda c: c["value_score"], reverse=True)

        # Phase 5 — pitfalls in this node's subtree (any ANC descendant with F <= -0.3)
        pitfalls = self._fetch_pitfalls_for_subtree(node_id)

        # Phase D2 — data for the unified 3-column shell.
        shell = self._build_node_shell(node, parents, enriched_children, goals)

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
            "value_v3": value_v3,
            "pitfalls": pitfalls,
            "tree_id": self._get_tree_id(node_id),
            # Shell (sidebar / breadcrumb / right rail)
            "workspaces": shell["workspaces"],
            "workspace_active_id": shell["workspace_active_id"],
            "breadcrumb_trail": shell["breadcrumb"],
            "stats_rail": shell["stats"],
            "backlinks": shell["backlinks"],
            "review_counts": shell["review"],
        }

    def _build_node_shell(
        self,
        node: Dict,
        parents: List[Dict],
        children: List[Dict],
        goals_for_node: List[Dict],
    ) -> Dict:
        """Assemble workspace tree, breadcrumb, stats, backlinks for the
        unified shell. Kept cheap: reuses render_workspaces plus a handful
        of small queries."""
        from datetime import date as _date, timedelta as _timedelta

        ws_payload = self.render_workspaces()

        # Resolve active workspace: the first SC seen in the parent chain,
        # or the node itself if it's an SC.
        node_id = node["id"]
        node_level = node.get("level", "ANC")
        active_sc_id = node_id if node_level == "SC" else ""
        if not active_sc_id:
            for p in parents:
                if p.get("level") == "SC":
                    active_sc_id = p["target_id"]
                    break

        # Breadcrumb: Home → SC → (CTX) → current
        trail: List[Dict] = [{"title": "Home", "href": "/wiki/"}]
        for p in reversed(parents):
            if not p.get("target_id"):
                continue
            title = (p.get("content") or p["target_id"]).split(".", 1)[0].strip()[:60]
            trail.append({
                "title": title,
                "href": f"/wiki/node/{p['target_id']}",
            })
        current_title = (node.get("content") or "").split(".", 1)[0].strip()[:60] or node["id"]
        trail.append({"title": current_title, "href": None})

        # Stats: counts + last_active + precision + 14-day access spark.
        today = _date.today()
        node_count = len(children) + 1
        anchor_count = 0
        if node_level == "SC":
            for ctx in children:
                anchor_count += len(self.db.get_children(ctx["id"]))
            node_count += anchor_count

        last_accessed = node.get("last_accessed") or ""
        last_active_str: Optional[str] = None
        days_dormant: Optional[int] = None
        if last_accessed:
            try:
                dt = datetime.fromisoformat(last_accessed)
                last_active_str = dt.date().isoformat()
                days_dormant = (today - dt.date()).days
            except (ValueError, TypeError):
                pass

        # 14-day access spark — count access_log rows per day for the last 14.
        spark = [0] * 14
        try:
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            start_iso = (today - _timedelta(days=13)).isoformat()
            rows = conn.execute(
                "SELECT DATE(accessed_at) AS d, COUNT(*) AS c "
                "FROM access_log "
                "WHERE node_id = ? AND DATE(accessed_at) >= ? "
                "GROUP BY DATE(accessed_at)",
                (node_id, start_iso),
            ).fetchall()
            conn.close()
            by_day = {r["d"]: r["c"] for r in rows}
            for i in range(14):
                d = (today - _timedelta(days=13 - i)).isoformat()
                spark[i] = int(by_day.get(d, 0) or 0)
        except Exception:
            pass

        stats = {
            "context_count": len(children) if node_level == "SC" else 0,
            "anchor_count": anchor_count if node_level == "SC" else len(children),
            "node_count": node_count,
            "last_active": last_active_str,
            "days_dormant": days_dormant,
            "access_count": int(node.get("access_count") or 0),
            "precision": round(float(node.get("precision") or 0.5), 2),
            "pulse_spark_14d": spark,
            "pulse_max": max(spark) if spark else 0,
        }

        # Backlinks — up to 3 goals + up to 3 narratives citing this node
        # or (for SCs) any child node via source_page_ids_json substring.
        backlinks: List[Dict] = []
        for g in goals_for_node[:3]:
            backlinks.append({
                "kind": "goal",
                "title": g.get("title") or g.get("id", ""),
                "href": "/wiki/goals",
            })
        try:
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT heading, date_local, source_page_ids_json "
                "FROM work_narratives "
                "WHERE source_page_ids_json LIKE ? "
                "ORDER BY date_local DESC, ordinal ASC LIMIT 3",
                (f'%{node_id}%',),
            ).fetchall()
            conn.close()
            for r in rows:
                backlinks.append({
                    "kind": "narrative",
                    "title": f"{r['heading']} · {r['date_local']}",
                    "href": "/wiki/goals",
                })
        except Exception:
            pass

        return {
            "workspaces": ws_payload["workspaces"],
            "workspace_active_id": active_sc_id,
            "breadcrumb": trail,
            "stats": stats,
            "backlinks": backlinks,
            "review": ws_payload["review"],
        }

    # ─── GOALS PAGE ───────────────────────────────────

    def render_goals(self) -> Dict:
        """All goals with linked nodes, recent surfaced matches, and progress.

        Also includes the PM board (Goal → Project → Deliverable) read from
        productivity-tracker/config/goals.yaml, enriched with 7-day tracked
        minutes per deliverable from tracker.db.
        """
        goals = self.db.list_goals()
        enriched = []
        for g in goals:
            nodes = self.db.get_nodes_for_goal(g["id"])
            fb = self.db.get_feedback_summary(node_id=None)
            matches = self.db.get_matches_for_goal(g["id"], min_score=0.75, limit=10)
            enriched.append({
                **g,
                "linked_nodes": nodes,
                "node_count": len(nodes),
                "feedback": fb,
                "matches": matches,
                "match_count": len(matches),
            })

        pm = self._render_pm_projects()

        # Unassigned lane — state=open work_pages in the recent window.
        # Window widens beyond today so a backfill's historical pages show up
        # instead of being invisible; knob is goals_recent_window_days.
        from datetime import date as _date, timedelta as _timedelta
        today = _date.today().isoformat()
        try:
            from core import config as _cfg
            _window_days = int(_cfg.get("goals_recent_window_days", 2) or 0)
        except Exception:
            _window_days = 2
        cutoff = (_date.today() - _timedelta(days=_window_days)).isoformat()
        unassigned_pages: List[Dict] = []
        leak_filtered_count = 0
        try:
            project_titles: Dict[str, str] = {}
            deliverable_names: Dict[str, str] = {}
            for _g in pm.get("goals", []):
                for _p in _g.get("projects", []):
                    if _p.get("id"):
                        project_titles[_p["id"]] = _p.get("title", _p["id"])
                    for _d in _p.get("deliverables", []):
                        if _d.get("id"):
                            deliverable_names[_d["id"]] = _d.get("name", _d["id"])

            # F1 leak-fix: precompute the set of work_pages whose segments
            # are already fully claimed by nightly, so we can skip them in
            # the loop below.
            from consolidation.claims import (
                fully_consolidated_page_ids, is_workpage_claimed,
            )
            import sqlite3 as _sqlite3
            claim_conn = _sqlite3.connect(self.db.db_path)
            try:
                consolidated_set = fully_consolidated_page_ids(claim_conn)
                raw_pages = self.db.list_work_pages_by_state("open")
                for p in raw_pages:
                    if (p.get("date_local") or "") < cutoff:
                        continue
                    claimed, _reason = is_workpage_claimed(
                        claim_conn, p["id"],
                        fully_consolidated=consolidated_set,
                    )
                    if claimed:
                        leak_filtered_count += 1
                        continue
                    p.pop("embedding_blob", None)
                    p["segment_count"] = len(self.db.get_page_segments(p["id"]))
                    p["is_today"] = (p.get("date_local") == today)
                    if p.get("tag_state") == "proposed":
                        p["proposed_project_title"] = project_titles.get(
                            p.get("project_id") or "", p.get("project_id") or ""
                        )
                        did = p.get("deliverable_id") or ""
                        p["proposed_deliverable_name"] = deliverable_names.get(did, did)
                    unassigned_pages.append(p)
            finally:
                claim_conn.close()
            unassigned_pages.sort(
                key=lambda p: (p.get("date_local", ""), p.get("created_at", "")),
                reverse=True,
            )
        except Exception:
            unassigned_pages = []

        # Phase A — split into salient (main lane) + kachra (folded footer).
        # Pages with salience='pending' (e.g. predates Phase A) are treated
        # as salient so nothing is silently hidden.
        salient_pages: List[Dict] = []
        kachra_pages: List[Dict] = []
        kachra_total_minutes = 0.0
        kachra_reason_counts: Dict[str, int] = {}
        for p in unassigned_pages:
            if (p.get("salience") or "pending") == "kachra":
                kachra_pages.append(p)
                kachra_total_minutes += (p.get("segment_count", 0) * 10) / 60.0
                reason = p.get("kachra_reason") or "other"
                kachra_reason_counts[reason] = kachra_reason_counts.get(reason, 0) + 1
            else:
                salient_pages.append(p)

        # Phase C2 — goal hero cards (prose + 6-week pulse + status pill).
        goals_with_narratives, current_iso_week, hero_counts = \
            self._build_goals_hero(pm["goals"])

        try:
            review_pending_count = self.db._conn.execute(
                "SELECT COUNT(*) FROM work_pages "
                "WHERE state='open' AND (salience='salient' OR salience='pending')"
            ).fetchone()[0] or 0
        except Exception:
            review_pending_count = 0

        # F1 leak-fix: narratives for projects already shown in the PM
        # board above would be a double-render — collect active project ids
        # and let _render_narratives_for_today roll them up into a count.
        active_project_ids: Set[str] = {
            (p.get("id") or "")
            for g in pm.get("goals", []) or []
            for p in (g.get("projects") or [])
            if p.get("id")
        }
        narratives, narratives_rolled_up_count = \
            self._render_narratives_for_today(today, active_project_ids)

        return {
            "goals": enriched,
            "total": len(enriched),
            "pm_goals": pm["goals"],
            "pm_total_projects": pm["total_projects"],
            "pm_alive_projects": pm["alive_projects"],
            "pm_column_counts": pm["column_counts"],
            "pm_config_path": pm["config_path"],
            "unassigned_pages": salient_pages,
            "unassigned_count": len(salient_pages),
            "unassigned_leak_filtered_count": leak_filtered_count,
            "kachra_pages": kachra_pages,
            "kachra_count": len(kachra_pages),
            "kachra_total_minutes": round(kachra_total_minutes, 1),
            "kachra_reason_counts": kachra_reason_counts,
            "narratives": narratives,
            "narratives_rolled_up_count": narratives_rolled_up_count,
            "goals_with_narratives": goals_with_narratives,
            "current_iso_week": current_iso_week,
            "hero_counts": hero_counts,
            "review_pending_count": review_pending_count,
        }

    def _build_goals_hero(self, pm_goals: List[Dict]) -> tuple:
        """Attach weekly prose narrative + 6-week pulse strip to each goal.
        Returns (list_of_goals, current_iso_week, counts_by_status)."""
        from datetime import date as _date, timedelta as _timedelta

        today = _date.today()
        y, w, _ = today.isocalendar()
        current_iso_week = f"{y}-W{w:02d}"

        # Pull all narratives for the current ISO week in one query.
        try:
            gw_rows = self.db.list_goal_narratives(iso_week=current_iso_week)
        except Exception:
            gw_rows = []
        gw_by_id: Dict[str, Dict] = {r["goal_id"]: r for r in gw_rows}

        # Legacy (YAML) → new vocab fallback for goals not yet composed.
        legacy_map = {
            "active": "active", "alive": "active",
            "cooling": "slowing_down",
            "cold": "stalled", "paused": "stalled", "abandoned": "stalled",
            "shipped": "shipped", "achieved": "shipped", "done": "shipped",
        }

        def _iso_week_str(d: _date) -> str:
            yy, ww, _ = d.isocalendar()
            return f"{yy}-W{ww:02d}"

        def _week_bounds(iso_wk: str) -> tuple:
            year_s, week_s = iso_wk.split("-W")
            mon = _date.fromisocalendar(int(year_s), int(week_s), 1)
            return mon, mon + _timedelta(days=6)

        # Each goal gets 6 weekly buckets ending on the current ISO week.
        last_6_weeks: List[str] = []
        for i in range(5, -1, -1):
            monday_i = today - _timedelta(days=today.weekday()) - _timedelta(weeks=i)
            last_6_weeks.append(_iso_week_str(monday_i))

        import os
        tracker_db = Path(os.path.expanduser("~/.productivity-tracker/tracker.db"))
        tracker_conn = None
        if tracker_db.is_file():
            try:
                tracker_conn = sqlite3.connect(
                    f"file:{tracker_db}?mode=ro", uri=True, timeout=3.0
                )
                tracker_conn.row_factory = sqlite3.Row
            except Exception:
                tracker_conn = None

        def _mins_for(patterns: List[str], iso_wk: str) -> float:
            if not patterns or tracker_conn is None:
                return 0.0
            mon, sun = _week_bounds(iso_wk)
            clauses, params = [], []
            for p in patterns:
                like = f"%{p}%"
                clauses.append("(window_name LIKE ? OR detailed_summary LIKE ?)")
                params.extend([like, like])
            sql = (
                "SELECT COALESCE(SUM(target_segment_length_secs)/60.0, 0) AS m "
                "FROM context_1 WHERE DATE(timestamp_start) BETWEEN ? AND ? "
                f"AND ({' OR '.join(clauses)})"
            )
            try:
                row = tracker_conn.execute(
                    sql, [mon.isoformat(), sun.isoformat(), *params]
                ).fetchone()
                return float(row["m"] or 0.0) if row else 0.0
            except Exception:
                return 0.0

        goals_out: List[Dict] = []
        counts = {"active": 0, "slowing_down": 0, "stalled": 0,
                  "shipped": 0, "not_started": 0}
        for g in pm_goals:
            gid = g.get("id") or ""
            patterns: List[str] = []
            for p in g.get("projects", []) or []:
                patterns.extend(p.get("match_patterns", []) or [])

            weekly_pulse: List[Dict] = []
            for wk in last_6_weeks:
                m = _mins_for(patterns, wk)
                if m >= 180:
                    bucket = "hot"
                elif m >= 60:
                    bucket = "warm"
                elif m > 0:
                    bucket = "light"
                else:
                    bucket = "cold"
                weekly_pulse.append({
                    "iso_week": wk,
                    "mins": round(m),
                    "bucket": bucket,
                    "is_current": wk == current_iso_week,
                })

            gw = gw_by_id.get(gid, {})
            status = (gw.get("status_this_week") or
                      legacy_map.get((g.get("status") or "").lower(), ""))
            if not status:
                status = "not_started" if not (g.get("projects") or []) else "stalled"
            mins_this_week = float(gw.get("minutes_this_week") or
                                   (weekly_pulse[-1]["mins"] if weekly_pulse else 0))
            prose = (gw.get("narrative_prose") or "").strip()
            if not prose:
                if status == "not_started":
                    prose = ("Not started — no linked projects yet." if not (g.get("projects") or [])
                             else "No work logged yet. Run Compose to generate this week's summary.")
                else:
                    prose = "No prose composed yet. Run Compose to generate this week's summary."

            # Projects slimmed for the evidence drawer (pulse + deliverables
            # already enriched in _render_pm_projects output).
            projects_trimmed = []
            for p in g.get("projects", []) or []:
                projects_trimmed.append({
                    "id": p.get("id", ""),
                    "title": p.get("title", ""),
                    "status": p.get("status", ""),
                    "target_week": p.get("target_week", ""),
                    "pulse_mins_7d": p.get("pulse_mins_7d", 0) or 0,
                    "deliverables": p.get("deliverables", []) or [],
                })

            # Up to 3 short story snippets from this ISO week whose text
            # mentions any of the goal's project titles or match patterns.
            haystack_needles: List[str] = [g.get("title", "") or ""]
            for p in g.get("projects", []) or []:
                if p.get("title"):
                    haystack_needles.append(p["title"])
                haystack_needles.extend(p.get("match_patterns", []) or [])
            haystack_needles = [n for n in haystack_needles if n and len(n) >= 3]

            story_snippets: List[str] = []
            try:
                mon, sun = (_date.fromisocalendar(
                    *tuple(int(s) for s in current_iso_week.split("-W"))
                ), None)
                # fromisocalendar wants (year, week, day); rebuild monday → sun
                year_s, week_s = current_iso_week.split("-W")
                mon = _date.fromisocalendar(int(year_s), int(week_s), 1)
                sun = mon + _timedelta(days=6)
                narr_rows = self.db._conn.execute(
                    """SELECT heading, body_prose, body_markdown
                       FROM work_narratives
                       WHERE date_local BETWEEN ? AND ?
                       ORDER BY date_local DESC, ordinal ASC""",
                    (mon.isoformat(), sun.isoformat()),
                ).fetchall()
                for r in narr_rows:
                    hay = " ".join(filter(None, (
                        r["heading"] or "", r["body_prose"] or "",
                        r["body_markdown"] or "",
                    ))).lower()
                    if not any(n.lower() in hay for n in haystack_needles):
                        continue
                    snippet = (r["body_prose"] or r["body_markdown"] or r["heading"] or "").strip()
                    snippet = " ".join(snippet.split())
                    if snippet:
                        story_snippets.append(snippet[:220])
                    if len(story_snippets) >= 3:
                        break
            except Exception:
                pass

            counts[status] = counts.get(status, 0) + 1
            goals_out.append({
                "id": gid,
                "title": g.get("title", ""),
                "outcome": g.get("why", "") or "",
                "status": status,
                "minutes_this_week": mins_this_week,
                "n_projects": len(g.get("projects", []) or []),
                "narrative_prose": prose,
                "weekly_pulse": weekly_pulse,
                "projects": projects_trimmed,
                "story_snippets": story_snippets,
            })

        if tracker_conn is not None:
            tracker_conn.close()

        return goals_out, current_iso_week, counts

    def _render_narratives_for_today(
        self, today: str,
        active_project_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[Dict], int]:
        """Returns narratives in the recent window (today + previous N days
        where N = goals_recent_window_days). Today first, then older by date
        desc.

        F1 leak-fix: narratives whose project_id is already represented by
        an active project card in the PM board are excluded from the list
        and counted in the second return value (rolled_up_count) so the UI
        can surface "N narratives rolled into project cards" without
        showing them twice.
        """
        from datetime import date as _date, timedelta as _timedelta
        try:
            from core import config as _cfg
            window_days = int(_cfg.get("goals_recent_window_days", 2) or 0)
        except Exception:
            window_days = 2
        try:
            all_rows: List[Dict] = []
            rolled_up = 0
            for i in range(window_days + 1):
                d = (_date.today() - _timedelta(days=i)).isoformat()
                day_rows = self.db.list_narratives(date_local=d) or []
                for r in day_rows:
                    pid = (r.get("project_id") or "").strip()
                    if active_project_ids and pid and pid in active_project_ids:
                        rolled_up += 1
                        continue
                    r["is_today"] = (d == today)
                    # Prefer prose when the narrator filled it; fall back to
                    # the bullet list otherwise (old rows, LLM-unavailable run).
                    prose = (r.get("body_prose") or "").strip()
                    r["render_mode"] = "prose" if prose else "bullets"
                    all_rows.append(r)
            return all_rows, rolled_up
        except Exception:
            return [], 0

    # ─── PM BOARD (Goal → Project → Deliverable) ─────

    def _render_pm_projects(self) -> Dict:
        """Read productivity-tracker/config/goals.yaml and enrich each project
        with 7-day pulse sourced from tracker.db.

        Pulse is computed via per-project match_patterns (case-insensitive
        substring match against context_1.window_name and detailed_summary).
        The tracker's deliverable_id/sc_node_id columns are empty in practice,
        so we don't join through them — patterns keep this declarative and
        visible in one file.
        """
        import os, yaml
        from datetime import date as dt_date, timedelta

        pt_root = _REPO_ROOT / "productivity-tracker"
        goals_path = pt_root / "config" / "goals.yaml"
        deliv_path = pt_root / "config" / "deliverables.yaml"
        tracker_db = Path(os.path.expanduser("~/.productivity-tracker/tracker.db"))

        if not goals_path.exists():
            return {
                "goals": [],
                "total_projects": 0,
                "alive_projects": 0,
                "column_counts": {"alive": 0, "cooling": 0, "cold": 0, "shipped": 0},
                "config_path": str(goals_path),
            }

        try:
            with open(goals_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception:
            raw = {}
        goals = raw.get("goals", []) or []

        # Deliverable catalog: id → {name, supercontext}
        deliv_ix: Dict[str, Dict] = {}
        if deliv_path.exists():
            try:
                with open(deliv_path, encoding="utf-8") as f:
                    d_raw = yaml.safe_load(f) or {}
                for d in d_raw.get("deliverables", []) or []:
                    deliv_ix[d["id"]] = {
                        "name": d.get("name", d["id"]),
                        "supercontext": d.get("supercontext"),
                    }
            except Exception:
                pass

        tracker_conn = None
        if tracker_db.exists():
            try:
                tracker_conn = sqlite3.connect(str(tracker_db))
                tracker_conn.row_factory = sqlite3.Row
            except Exception:
                tracker_conn = None

        today = dt_date.today()
        start_date = (today - timedelta(days=6)).isoformat()
        seven_days = [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]

        def _where_clause(patterns: List[str]):
            """Build an OR-clause + params tuple for a pattern list."""
            clauses = []
            params: List[Any] = []
            for p in patterns:
                like = f"%{p}%"
                clauses.append("(window_name LIKE ? OR detailed_summary LIKE ?)")
                params.extend([like, like])
            return " OR ".join(clauses), params

        def pulse_for_patterns(patterns: List[str]) -> float:
            """Sum minutes over the last 7 days where any pattern matches."""
            if not patterns or tracker_conn is None:
                return 0.0
            where, pp = _where_clause(patterns)
            sql = (
                "SELECT COALESCE(SUM(target_segment_length_secs)/60.0, 0) AS mins "
                "FROM context_1 WHERE DATE(timestamp_start) >= ? "
                f"AND ({where})"
            )
            try:
                row = tracker_conn.execute(sql, [start_date, *pp]).fetchone()
                return float(row["mins"] or 0) if row else 0.0
            except Exception:
                return 0.0

        def rich_pulse(patterns: List[str]) -> Dict:
            """Full pulse for a pattern set:
                {mins_7d, spark[7], last_active, days_dormant}.
            spark is a 7-element list of minutes (oldest → newest).
            last_active is the most recent YYYY-MM-DD with any match (or None).
            days_dormant is today - last_active (None if never active)."""
            if not patterns or tracker_conn is None:
                return {"mins_7d": 0.0, "spark": [0]*7, "last_active": None, "days_dormant": None}
            where, pp = _where_clause(patterns)
            # Per-day sparkline
            spark_sql = (
                "SELECT DATE(timestamp_start) AS d, "
                "COALESCE(SUM(target_segment_length_secs)/60.0, 0) AS mins "
                "FROM context_1 WHERE DATE(timestamp_start) >= ? "
                f"AND ({where}) GROUP BY d"
            )
            by_day: Dict[str, float] = {}
            try:
                for r in tracker_conn.execute(spark_sql, [start_date, *pp]).fetchall():
                    by_day[r["d"]] = float(r["mins"] or 0)
            except Exception:
                pass
            spark = [int(round(by_day.get(d, 0))) for d in seven_days]
            mins_7d = sum(by_day.values())
            # Last-active date (all-time, not just 7d)
            last_sql = (
                "SELECT MAX(DATE(timestamp_start)) AS last_d "
                f"FROM context_1 WHERE {where}"
            )
            last_active = None
            try:
                row = tracker_conn.execute(last_sql, pp).fetchone()
                last_active = row["last_d"] if row and row["last_d"] else None
            except Exception:
                pass
            days_dormant = None
            if last_active:
                try:
                    la = dt_date.fromisoformat(last_active)
                    days_dormant = (today - la).days
                except Exception:
                    pass
            return {
                "mins_7d": round(mins_7d, 1),
                "spark": spark,
                "last_active": last_active,
                "days_dormant": days_dormant,
            }

        # Pulse Kanban thresholds (see feedback_pulse_threshold.md)
        ALIVE_MIN_7D = 5.0          # min of pulse in last 7d to count as Alive
        COOLING_WINDOW_DAYS = 30    # within this many days of last pulse = Cooling
        SHIPPED_KEYWORDS = ("ship", "complete", "done", "launch")

        def _column_for(project_status: str, pulse_7d: float,
                        days_since: Optional[int], lifecycle: List[Dict]) -> str:
            """Derive Pulse Kanban column from project state + pulse history.

            Priority:
              shipped  — status flag OR any lifecycle event matches SHIPPED_KEYWORDS
              alive    — >= ALIVE_MIN_7D minutes in last 7d
              cooling  — last pulse within (7, COOLING_WINDOW_DAYS] days
              cold     — last pulse > COOLING_WINDOW_DAYS days ago
              spark    — never had pulse
            """
            if project_status == "shipped":
                return "shipped"
            for ev in lifecycle:
                ev_text = (ev.get("event") or "").lower()
                if any(k in ev_text for k in SHIPPED_KEYWORDS):
                    return "shipped"
            if pulse_7d >= ALIVE_MIN_7D:
                return "alive"
            if days_since is None:
                return "spark"
            if days_since <= COOLING_WINDOW_DAYS:
                return "cooling"
            return "cold"

        enriched_goals = []
        total_projects = 0
        alive_projects = 0
        column_counts = {"spark": 0, "alive": 0, "cooling": 0, "cold": 0, "shipped": 0}
        for goal in goals:
            projects = goal.get("projects", []) or []
            enriched_projects = []
            for p in projects:
                total_projects += 1
                patterns = p.get("match_patterns", []) or []
                per_deliv_patterns = p.get("deliverable_patterns", {}) or {}
                project_rp = rich_pulse(patterns)
                project_pulse = project_rp["mins_7d"]
                project_spark = project_rp["spark"]
                project_spark_peak = max(project_spark) if project_spark else 0
                days_since_pulse = project_rp["days_dormant"]

                delivs = []
                # goals.yaml stores deliverable IDs as keys of
                # deliverable_patterns; fall back to those keys when an
                # explicit deliverable_ids list isn't present.
                deliv_ids = (
                    p.get("deliverable_ids")
                    or list((p.get("deliverable_patterns") or {}).keys())
                )
                for did in deliv_ids:
                    d_meta = deliv_ix.get(did, {})
                    # Per-deliverable patterns fall back to project-level
                    d_patterns = per_deliv_patterns.get(did) or patterns
                    rp = rich_pulse(d_patterns)
                    spark_peak = max(rp["spark"]) if rp["spark"] else 0
                    # Phase 5 polish — count pitfalls in the deliverable's
                    # linked-anchor subtree so the card can warn the user.
                    pitfall_count = self._count_pitfalls_for_deliverable(did)
                    delivs.append({
                        "id": did,
                        "name": d_meta.get("name", did),
                        "supercontext": d_meta.get("supercontext"),
                        "known": did in deliv_ix,
                        "mins_7d": rp["mins_7d"],
                        "spark": rp["spark"],
                        "spark_peak": spark_peak,
                        "last_active": rp["last_active"],
                        "days_dormant": rp["days_dormant"],
                        "alive": rp["mins_7d"] >= ALIVE_MIN_7D,
                        "inherited_patterns": did not in per_deliv_patterns,
                        "pitfall_count": pitfall_count,
                    })

                alive = project_pulse >= ALIVE_MIN_7D
                if alive:
                    alive_projects += 1

                # Inception → live breadcrumb: first lifecycle date → current state
                lc = p.get("lifecycle", []) or []
                inception_date = lc[0]["date"] if lc else None
                latest_event = lc[-1] if lc else None

                status = p.get("status", "active")
                column = _column_for(status, project_pulse, days_since_pulse, lc)
                column_counts[column] = column_counts.get(column, 0) + 1

                enriched_projects.append({
                    "id": p.get("id"),
                    "title": p.get("title", "Untitled project"),
                    "status": status,
                    "target_week": p.get("target_week", ""),
                    "match_patterns": patterns,
                    "deliverable_patterns": per_deliv_patterns,
                    "lifecycle": lc,
                    "latest_event": latest_event,
                    "deliverables": delivs,
                    "pulse_mins_7d": round(project_pulse, 1),
                    "pulse_spark": project_spark,
                    "pulse_spark_peak": project_spark_peak,
                    "alive": alive,
                    "days_since_pulse": days_since_pulse,
                    "inception_date": inception_date,
                    "column": column,
                })
            enriched_goals.append({
                "id": goal.get("id"),
                "title": goal.get("title", "Untitled goal"),
                "why": goal.get("why", ""),
                "status": goal.get("status", "active"),
                "projects": enriched_projects,
            })

        if tracker_conn is not None:
            try:
                tracker_conn.close()
            except Exception:
                pass

        return {
            "goals": enriched_goals,
            "total_projects": total_projects,
            "alive_projects": alive_projects,
            "column_counts": column_counts,
            "config_path": str(goals_path),
        }

    # Public wrapper (API surface)
    def render_pm_projects(self) -> Dict:
        return self._render_pm_projects()

    # Phase C2: fixed scaffold for deliverable pages. Order matters —
    # the template renders slots in this sequence.
    DELIVERABLE_SLOT_ORDER: Tuple[str, ...] = (
        "overview", "progress", "decisions", "questions", "risks", "links",
    )

    DELIVERABLE_SLOT_LABELS: Dict[str, str] = {
        "overview": "Overview",
        "progress": "Progress",
        "decisions": "Decisions",
        "questions": "Open questions",
        "risks": "Risks",
        "links": "Contributing links",
    }

    def load_deliverable_sections(
        self, deliverable_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Return the 6-slot scaffold for `deliverable_id`. Always
        returns every slot — missing rows surface as empty bodies so
        the template doesn't have to handle absence per-slot.

        C3: an eligible slot (no row OR source='auto' with empty body)
        is auto-filled at read time. Currently only the `progress`
        slot uses auto-fill; other slots remain empty until the user
        edits them. The returned dict carries `auto_filled` so the
        template can label it accordingly."""
        rows: Dict[str, Dict[str, Any]] = {}
        try:
            cursor = self.db._conn.execute(
                "SELECT slot, body_md, source, updated_at "
                "FROM deliverable_sections WHERE deliverable_id = ?",
                (deliverable_id,),
            )
            for r in cursor.fetchall():
                slot = r[0]
                if slot in self.DELIVERABLE_SLOT_LABELS:
                    rows[slot] = {
                        "body_md": r[1] or "",
                        "source": r[2] or "auto",
                        "updated_at": r[3] or "",
                    }
        except Exception:
            logger.exception("load_deliverable_sections failed")

        out: Dict[str, Dict[str, Any]] = {}
        for slot in self.DELIVERABLE_SLOT_ORDER:
            saved = rows.get(slot, {})
            body_md = saved.get("body_md", "")
            source = saved.get("source", "auto")

            # Eligible: no saved content AND the slot's source isn't 'user'.
            # 'user' source means the human has explicitly saved (even
            # empty) — respect that and don't auto-fill.
            auto_filled = False
            tagged_count = 0
            link_rows: List[Dict[str, Any]] = []
            if (not body_md.strip()) and source != "user":
                if slot == "progress":
                    body_md, tagged_count = (
                        self._progress_from_tagged_proposals(deliverable_id)
                    )
                    if body_md:
                        auto_filled = True
                elif slot == "links":
                    # D4: structured payload, not markdown — the template
                    # renders a chip list with toggle buttons, so we leave
                    # body_md/body_html empty and stash the rows in
                    # `link_rows` for the template.
                    link_rows = self.load_deliverable_links(deliverable_id)
                    if link_rows:
                        auto_filled = True
                        tagged_count = len(link_rows)

            out[slot] = {
                "slot": slot,
                "label": self.DELIVERABLE_SLOT_LABELS[slot],
                "body_md": body_md,
                "body_html": _section_md_to_html(body_md) if body_md else "",
                "source": source,
                "updated_at": saved.get("updated_at", ""),
                "is_empty": not bool(body_md.strip()) and not link_rows,
                "auto_filled": auto_filled,
                "tagged_count": tagged_count,
                "link_rows": link_rows,
            }
        return out

    def load_daily_view(
        self, deliverable_id: str, date: str,
    ) -> Dict[str, Any]:
        """Phase E — read surface for the Daily pane on a deliverable.

        Returns:
            {
                tagged_items: [{match_id, segment_id, work_description,
                                 time_mins, matched_at, is_correct,
                                 project_id, deliverable_id}, ...],
                daily_summary: {id, body_md, body_html, status,
                                composed_at, project_id, deliverable_id,
                                date} or None,
                total_minutes: float,
                contributing_links: [{link_id, url, kind, dwell_frames,
                                      contributed}, ...],
                can_feedback: bool   # True for past dates only.
                date: YYYY-MM-DD
                deliverable_id: str
            }
        """
        from datetime import date as _date
        out: Dict[str, Any] = {
            "tagged_items": [],
            "daily_summary": None,
            "total_minutes": 0.0,
            "contributing_links": [],
            "can_feedback": False,
            "date": date,
            "deliverable_id": deliverable_id,
        }
        if not deliverable_id or not date:
            return out

        try:
            today_iso = _date.today().isoformat()
            out["can_feedback"] = date < today_iso
        except Exception:
            out["can_feedback"] = False

        try:
            cursor = self.db._conn.execute(
                """SELECT id, segment_id, project_id, deliverable_id,
                          work_description, time_mins, matched_at,
                          is_correct
                   FROM project_work_match_log
                   WHERE deliverable_id = ?
                     AND DATE(matched_at) = ?
                   ORDER BY matched_at DESC""",
                (deliverable_id, date),
            )
            rows = cursor.fetchall()
            total = 0.0
            for r in rows:
                mins = float(r[5] or 0.0)
                total += mins
                out["tagged_items"].append({
                    "match_id": r[0],
                    "segment_id": r[1] or "",
                    "project_id": r[2] or "",
                    "deliverable_id": r[3] or "",
                    "work_description": r[4] or "",
                    "time_mins": round(mins, 2),
                    "matched_at": r[6] or "",
                    "is_correct": int(r[7]) if r[7] is not None else -1,
                })
            out["total_minutes"] = round(total, 1)
        except Exception:
            logger.exception("load_daily_view tagged_items failed")

        # Daily summary row (if any).
        try:
            row = self.db._conn.execute(
                """SELECT id, project_id, deliverable_id, date, body_md,
                          status, composed_at
                   FROM daily_summaries
                   WHERE deliverable_id = ? AND date = ?""",
                (deliverable_id, date),
            ).fetchone()
            if row:
                body_md = row[4] or ""
                out["daily_summary"] = {
                    "id": row[0],
                    "project_id": row[1] or "",
                    "deliverable_id": row[2] or "",
                    "date": row[3] or date,
                    "body_md": body_md,
                    "body_html": _section_md_to_html(body_md) if body_md else "",
                    "status": row[5] or "auto",
                    "composed_at": row[6] or "",
                }
        except Exception:
            logger.exception("load_daily_view daily_summary failed")

        # Reuse load_deliverable_links for the contributing strip — same
        # bindings drive both Overall and Daily views, so the user sees
        # consistent attribution.
        out["contributing_links"] = [
            r for r in self.load_deliverable_links(deliverable_id)
            if r["contributed"] == 1
        ]
        return out

    def load_deliverable_links(
        self, deliverable_id: str,
    ) -> List[Dict[str, Any]]:
        """Phase D4 — return all link_bindings for this deliverable as
        a sorted list. Order: contributed=1 first, then dwell desc."""
        if not deliverable_id:
            return []
        try:
            cursor = self.db._conn.execute(
                """SELECT lb.link_id, l.url, l.kind, lb.contributed,
                          lb.dwell_frames
                   FROM link_bindings lb
                   JOIN links l ON l.id = lb.link_id
                   WHERE lb.scope = 'deliverable' AND lb.scope_id = ?
                   ORDER BY lb.contributed DESC, lb.dwell_frames DESC,
                            l.url ASC""",
                (deliverable_id,),
            )
            return [
                {
                    "link_id": r[0],
                    "url": r[1],
                    "kind": r[2] or "other",
                    "contributed": int(r[3] or 0),
                    "dwell_frames": int(r[4] or 0),
                }
                for r in cursor.fetchall()
            ]
        except Exception:
            logger.exception("load_deliverable_links failed")
            return []

    def project_link_rollup(
        self, project_id: str, limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Phase D4 — aggregate contributed=1 link_bindings across every
        deliverable belonging to `project_id`. URL-deduped; total dwell
        summed across deliverables. Top `limit` by summed dwell."""
        if not project_id:
            return []

        # Resolve the project's deliverable IDs from goals.yaml — the same
        # source render_project_detail uses, so the rollup matches what
        # the user sees in the sidebar.
        deliverable_ids: List[str] = []
        try:
            pm = self._render_pm_projects()
            for g in pm.get("goals", []) or []:
                for p in g.get("projects", []) or []:
                    if p.get("id") == project_id:
                        for d in p.get("deliverables") or []:
                            did = d.get("id")
                            if did:
                                deliverable_ids.append(did)
                        break
        except Exception:
            return []

        if not deliverable_ids:
            return []

        placeholders = ",".join("?" * len(deliverable_ids))
        try:
            cursor = self.db._conn.execute(
                f"""SELECT l.url, l.kind, SUM(lb.dwell_frames) AS dwell_sum
                    FROM link_bindings lb
                    JOIN links l ON l.id = lb.link_id
                    WHERE lb.scope = 'deliverable'
                      AND lb.contributed = 1
                      AND lb.scope_id IN ({placeholders})
                    GROUP BY l.url, l.kind
                    ORDER BY dwell_sum DESC, l.url ASC
                    LIMIT ?""",
                deliverable_ids + [int(limit)],
            )
            return [
                {
                    "url": r[0],
                    "kind": r[1] or "other",
                    "dwell_total": int(r[2] or 0),
                }
                for r in cursor.fetchall()
            ]
        except Exception:
            logger.exception("project_link_rollup failed")
            return []

    def _progress_from_tagged_proposals(
        self, deliverable_id: str, limit: int = 10,
    ) -> Tuple[str, int]:
        """Render the deliverable's recent confirmed / auto_attached
        proposals as wiki prose for the Progress slot. Returns
        (markdown, count). Empty markdown when no eligible proposals."""
        if not deliverable_id:
            return "", 0
        try:
            from wiki_tree_prose import render_tree_as_prose
        except Exception:
            return "", 0

        try:
            cursor = self.db._conn.execute(
                """SELECT tree_json, confirmed_at, status
                   FROM review_proposals
                   WHERE (user_assigned_deliverable_id = ?
                          OR auto_attached_to_deliverable_id = ?)
                     AND status IN ('confirmed', 'auto_attached')
                     AND tree_json IS NOT NULL
                     AND tree_json != ''
                   ORDER BY confirmed_at DESC
                   LIMIT ?""",
                (deliverable_id, deliverable_id, int(limit)),
            )
            rows = cursor.fetchall()
        except Exception:
            logger.exception("_progress_from_tagged_proposals query failed")
            return "", 0

        chunks: List[str] = []
        for tree_json, confirmed_at, status in rows:
            try:
                tree = json.loads(tree_json)
            except Exception:
                continue
            prose = render_tree_as_prose(tree)
            if not prose:
                continue
            date_str = (confirmed_at or "")[:10]
            badge = (
                "*auto-attached*" if status == "auto_attached"
                else "*confirmed*"
            )
            header = f"*on {date_str}* · {badge}" if date_str else badge
            chunks.append(f"{header}\n\n{prose}")

        if not chunks:
            return "", 0
        return "\n\n---\n\n".join(chunks), len(chunks)

    def render_project_detail(
        self, project_id: str, deliverable_id: Optional[str] = None,
        view: str = "overall", date: Optional[str] = None,
    ) -> Optional[Dict]:
        """Project shell-view data for /wiki/goals/p/{pid}[/d/{did}].

        Returns None when project_id is not present in goals.yaml — the
        caller should 404 in that case. When deliverable_id is given but
        not under this project, `selected_deliverable` is None and the
        caller should 404 as well; this method does not differentiate
        between "no selection requested" and "bad selection requested".

        Reuses _render_pm_projects so deliverable pulse stats stay in
        sync with the Goals landing — no duplicated enrichment logic.
        """
        pm = self._render_pm_projects()

        project: Optional[Dict] = None
        parent_goal: Optional[Dict] = None
        for g in pm.get("goals", []) or []:
            for p in g.get("projects", []) or []:
                if p.get("id") == project_id:
                    project = p
                    parent_goal = g
                    break
            if project is not None:
                break

        if project is None:
            return None

        deliverables: List[Dict] = list(project.get("deliverables") or [])
        selected: Optional[Dict] = None
        if deliverable_id:
            selected = next(
                (d for d in deliverables if d.get("id") == deliverable_id),
                None,
            )

        # Phase C2: load the 6-slot scaffold for the selected deliverable.
        # Only when one is actually selected — the project Overview state
        # doesn't have its own sections (yet).
        sections: Dict[str, Dict[str, Any]] = {}
        section_order: Tuple[str, ...] = ()
        if selected and selected.get("id"):
            sections = self.load_deliverable_sections(selected["id"])
            section_order = self.DELIVERABLE_SLOT_ORDER

        # Phase D4: top contributed links across the whole project.
        # Renders in the right rail; informational only.
        project_links_rollup = self.project_link_rollup(project_id)

        # Phase E: daily view payload when ?view=daily is requested AND a
        # deliverable is selected. Resolves date to today when not given.
        daily_view: Optional[Dict[str, Any]] = None
        if view == "daily" and selected and selected.get("id"):
            from datetime import date as _date
            target_date = (date or _date.today().isoformat())[:10]
            daily_view = self.load_daily_view(selected["id"], target_date)

        return {
            "project": project,
            "deliverables": deliverables,
            "selected_deliverable": selected,
            "selected_deliverable_id": deliverable_id or "",
            "sections": sections,
            "section_order": list(section_order),
            "project_links_rollup": project_links_rollup,
            "view": view,
            "daily_view": daily_view,
            "parent_goal": ({
                "id": parent_goal.get("id"),
                "title": parent_goal.get("title", ""),
            } if parent_goal else None),
            "breadcrumb": [
                {"label": "Goals", "href": "/wiki/goals"},
                {"label": (parent_goal or {}).get("title", "Goal"),
                 "href": "/wiki/goals"},
                {"label": project.get("title", project_id), "href": ""},
            ],
        }

    def save_pm_goals(self, goals_payload: List[Dict]) -> Dict:
        """Persist goals.yaml from an API payload, stripping runtime-only
        enrichment fields so the file stays a clean spec. Returns the
        re-enriched view (same shape as render_pm_projects)."""
        import yaml
        pt_root = _REPO_ROOT / "productivity-tracker"
        goals_path = pt_root / "config" / "goals.yaml"
        goals_path.parent.mkdir(parents=True, exist_ok=True)

        computed = {
            "deliverables", "pulse_mins_7d", "alive",
            "pulse_spark", "pulse_spark_peak", "days_since_pulse",
            "inception_date", "latest_event", "column",
        }
        clean_goals = []
        for g in goals_payload or []:
            cg = {
                "id": g.get("id"),
                "title": g.get("title", "Untitled goal"),
                "why": g.get("why", "") or "",
                "status": g.get("status", "active"),
            }
            projects = []
            for p in g.get("projects", []) or []:
                cp = {k: v for k, v in p.items() if k not in computed}
                # Preserve stable ordering: id, title, target_week, match_patterns,
                # deliverable_ids, lifecycle.
                ordered = {}
                for key in ("id", "title", "status", "target_week", "match_patterns",
                            "deliverable_ids", "deliverable_patterns", "lifecycle"):
                    if key in cp:
                        ordered[key] = cp[key]
                # carry any other custom fields
                for k, v in cp.items():
                    if k not in ordered:
                        ordered[k] = v
                projects.append(ordered)
            cg["projects"] = projects
            clean_goals.append(cg)

        with open(goals_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"goals": clean_goals}, f,
                           sort_keys=False, default_flow_style=False)
        return self._render_pm_projects()

    def list_pm_deliverables(self) -> List[Dict]:
        """Read productivity-tracker/config/deliverables.yaml for the picker."""
        import yaml
        pt_root = _REPO_ROOT / "productivity-tracker"
        deliv_path = pt_root / "config" / "deliverables.yaml"
        out: List[Dict] = []
        if not deliv_path.exists():
            return out
        try:
            with open(deliv_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            for d in raw.get("deliverables", []) or []:
                out.append({
                    "id": d.get("id"),
                    "name": d.get("name", d.get("id")),
                    "supercontext": d.get("supercontext"),
                })
        except Exception:
            pass
        return out

    def create_pm_deliverable(self, name: str,
                              supercontext: Optional[str] = None) -> Dict:
        """Append a new deliverable to deliverables.yaml with an auto-assigned
        D-00X id (next available). Returns the new row.
        """
        import re, yaml
        name = (name or "").strip()
        if not name:
            raise ValueError("name required")

        pt_root = _REPO_ROOT / "productivity-tracker"
        deliv_path = pt_root / "config" / "deliverables.yaml"
        deliv_path.parent.mkdir(parents=True, exist_ok=True)

        raw: Dict[str, Any] = {}
        if deliv_path.exists():
            try:
                with open(deliv_path, encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
            except Exception:
                raw = {}
        items = raw.get("deliverables", []) or []

        # Next available D-00X (based on existing numeric suffixes)
        used = set()
        for d in items:
            m = re.match(r"^D-(\d+)$", str(d.get("id") or ""))
            if m:
                used.add(int(m.group(1)))
        nxt = 1
        while nxt in used:
            nxt += 1
        new_id = f"D-{nxt:03d}"

        # Guard against duplicate names (case-insensitive)
        for d in items:
            if (d.get("name") or "").strip().lower() == name.lower():
                return {
                    "id": d.get("id"),
                    "name": d.get("name"),
                    "supercontext": d.get("supercontext"),
                    "already_existed": True,
                }

        row: Dict[str, Any] = {"id": new_id, "name": name}
        sc = (supercontext or "").strip()
        if sc:
            row["supercontext"] = sc

        items.append(row)
        raw["deliverables"] = items
        with open(deliv_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False, default_flow_style=False)

        return {
            "id": new_id,
            "name": name,
            "supercontext": sc or None,
            "already_existed": False,
        }

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
        """Rich productivity dashboard from tracker.db + node_time_log.

        If target_date has no data, fall back to the most recent day that
        does. Run the tracker-health probe on every call so the red banner
        on the page auto-lights when the tracker's gone silent.
        """
        import os
        from datetime import date as dt_date, timedelta

        requested_date = target_date or dt_date.today().isoformat()

        # Tracker freshness → productivity_degradation_log.
        try:
            from tracker_health import probe_and_log
            degradation = probe_and_log(self.db._conn)
        except Exception:
            degradation = {"degraded": False, "reason": "", "detected_at": "",
                           "last_segment_ts": ""}

        tracker_db = os.path.expanduser("~/.productivity-tracker/tracker.db")
        empty_shell = {
            "has_data": False,
            "requested_date": requested_date,
            "effective_date": requested_date,
            "fallback_date_used": False,
            "tracker_degraded": degradation.get("degraded", False),
            "degraded_reason": degradation.get("reason", ""),
            "degraded_detected_at": degradation.get("detected_at", ""),
            "tracker_last_ts": degradation.get("last_segment_ts", ""),
        }
        if not os.path.exists(tracker_db):
            return empty_shell

        conn = sqlite3.connect(tracker_db)
        conn.row_factory = sqlite3.Row

        def _totals_for(d: str):
            return conn.execute(
                "SELECT COUNT(*) as cnt, "
                "COALESCE(SUM(target_segment_length_secs)/60.0, 0) as mins "
                "FROM context_1 WHERE DATE(timestamp_start) = ?", (d,)
            ).fetchone()

        # Resolve effective date: requested → most-recent-with-data fallback.
        effective_date = requested_date
        totals = _totals_for(effective_date)
        fallback_used = False
        if not totals or totals["cnt"] == 0:
            fallback_row = conn.execute(
                "SELECT DATE(timestamp_start) as d "
                "FROM context_1 "
                "WHERE DATE(timestamp_start) < ? AND target_segment_length_secs > 0 "
                "ORDER BY timestamp_start DESC LIMIT 1",
                (requested_date,)
            ).fetchone()
            if fallback_row and fallback_row["d"]:
                effective_date = fallback_row["d"]
                totals = _totals_for(effective_date)
                fallback_used = True

        if not totals or totals["cnt"] == 0:
            conn.close()
            return empty_shell

        # From here on, queries read from effective_date (NOT requested_date).
        target_date = effective_date

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

        # 30-day coverage — uses the tracker.db connection before it closes.
        cov_rows = conn.execute(
            "SELECT DATE(timestamp_start) AS d, "
            "COALESCE(SUM(target_segment_length_secs)/60.0, 0) AS mins, "
            "COUNT(*) AS segments "
            "FROM context_1 "
            "WHERE DATE(timestamp_start) BETWEEN DATE(?, '-29 days') AND ? "
            "AND target_segment_length_secs > 0 "
            "GROUP BY DATE(timestamp_start)",
            (effective_date, effective_date)
        ).fetchall()
        by_day = {r["d"]: {"mins": r["mins"], "segments": r["segments"]} for r in cov_rows}

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

        # 30-day coverage heatmap — by_day already built above before conn
        # closed. Pull degradation markers and bucket into cells.
        try:
            deg_rows = self.db._conn.execute(
                "SELECT DISTINCT date_local FROM productivity_degradation_log "
                "WHERE date_local BETWEEN DATE(?, '-29 days') AND ?",
                (effective_date, effective_date)
            ).fetchall()
            degraded_days = {r["date_local"] for r in deg_rows}
        except sqlite3.OperationalError:
            degraded_days = set()
        coverage_30d = []
        effective_dt = dt_date.fromisoformat(effective_date)
        for i in range(29, -1, -1):
            d = (effective_dt - timedelta(days=i)).isoformat()
            cell = by_day.get(d, {"mins": 0.0, "segments": 0})
            mins = cell["mins"]
            if mins <= 0:
                level = 0
            elif mins <= 30:
                level = 1
            elif mins <= 90:
                level = 2
            elif mins <= 180:
                level = 3
            else:
                level = 4
            coverage_30d.append({
                "date": d,
                "mins": mins,
                "segments": cell["segments"],
                "degraded": d in degraded_days,
                "intensity_level": level,
                "is_effective": d == effective_date,
            })
        coverage_max_mins = max((c["mins"] for c in coverage_30d), default=0.0)

        return {
            "has_data": True,
            "requested_date": requested_date,
            "effective_date": effective_date,
            "fallback_date_used": fallback_used,
            "tracker_degraded": degradation.get("degraded", False),
            "degraded_reason": degradation.get("reason", ""),
            "degraded_detected_at": degradation.get("detected_at", ""),
            "tracker_last_ts": degradation.get("last_segment_ts", ""),
            "coverage_30d": coverage_30d,
            "coverage_max_mins": coverage_max_mins,
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
            """SELECT prose_markdown, context_hash, value_score_avg_at_render,
                      pitfall_count_at_render
               FROM wiki_page_cache WHERE node_id=?""",
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

        # Phase 5 — staleness: recompute when value_score distribution drifts,
        # even if content_hash is identical. Two triggers:
        #   1. avg child value_score drift > drift_threshold (default 0.2)
        #   2. pitfall count changed (new warning or resolved warning)
        children = node_data.get("children", [])
        cur_avg = (sum((c.get("value_score") or 0.0) for c in children) / len(children)
                   if children else 0.0)
        cur_pitfall_count = len(node_data.get("pitfalls", []))
        drift_threshold = 0.2
        is_stale_by_value = False
        if cached:
            prev_avg = float(cached["value_score_avg_at_render"] or 0.0)
            prev_pitfalls = int(cached["pitfall_count_at_render"] or 0)
            if abs(cur_avg - prev_avg) > drift_threshold:
                is_stale_by_value = True
            if cur_pitfall_count != prev_pitfalls:
                is_stale_by_value = True

        # Return cache if content is unchanged AND value distribution hasn't drifted
        if (cached and cached["context_hash"] == context_hash
                and cached["prose_markdown"] and not is_stale_by_value):
            conn.close()
            return cached["prose_markdown"]

        conn.close()

        # Generate prose via LLM
        prose = self._call_llm_for_prose(node_data)
        if not prose:
            return None

        # Cache it — include value_score snapshot so next call can detect drift
        conn2 = sqlite3.connect(self.db.db_path)
        conn2.execute("""
            INSERT OR REPLACE INTO wiki_page_cache
            (node_id, prose_markdown, context_hash, generated_at, llm_model,
             word_count, value_score_avg_at_render, pitfall_count_at_render)
            VALUES (?, ?, ?, datetime('now'), ?, ?, ?, ?)
        """, (node_id, prose, context_hash, "qwen2.5:3b", len(prose.split()),
              cur_avg, cur_pitfall_count))
        conn2.commit()
        conn2.close()

        return prose

    def list_stale_pages(self, drift_threshold: float = 0.2, limit: int = 50) -> List[Dict]:
        """Phase 5 — pages whose cached prose is out of sync with current
        value_score distribution. Consolidation can prioritize these."""
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("""
                SELECT wpc.node_id, mn.content, mn.level,
                       wpc.generated_at, wpc.value_score_avg_at_render,
                       wpc.pitfall_count_at_render,
                       (SELECT AVG(cmn.value_score)
                        FROM relations r JOIN memory_nodes cmn ON cmn.id = r.source_id
                        WHERE r.target_id = wpc.node_id AND r.relation_type = 'child_of'
                          AND cmn.is_deleted = 0) AS cur_avg
                FROM wiki_page_cache wpc
                JOIN memory_nodes mn ON mn.id = wpc.node_id
                WHERE mn.is_deleted = 0
            """).fetchall()
        finally:
            conn.close()

        stale: List[Dict] = []
        for r in rows:
            cur = float(r["cur_avg"] or 0.0)
            prev = float(r["value_score_avg_at_render"] or 0.0)
            drift = abs(cur - prev)
            if drift > drift_threshold:
                stale.append({
                    "node_id": r["node_id"],
                    "level": r["level"],
                    "content": (r["content"] or "")[:120],
                    "generated_at": r["generated_at"],
                    "prev_avg": prev,
                    "cur_avg": cur,
                    "drift": round(drift, 3),
                    "pitfall_count": r["pitfall_count_at_render"],
                })
        stale.sort(key=lambda s: s["drift"], reverse=True)
        return stale[:limit]

    def _call_llm_for_prose(self, node_data: Dict) -> Optional[str]:
        """Call local Ollama to generate wiki prose from node data."""
        node = node_data["node"]
        level = node_data["level"]
        children = node_data.get("children", [])
        pitfalls = node_data.get("pitfalls", [])

        # Build prompt based on level
        if level == "SC":
            prompt = self._build_sc_prompt(node, children, pitfalls)
        elif level in ("CTX", "CTX-1", "CTX-2"):
            prompt = self._build_ctx_prompt(node, children, node_data, pitfalls)
        else:
            prompt = self._build_anc_prompt(node, node_data, pitfalls)

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

    def _build_sc_prompt(self, node: Dict, children: List[Dict],
                          pitfalls: Optional[List[Dict]] = None) -> str:
        """Build prompt for SC-level wiki page."""
        sc_name = node.get("content", "")[:100]
        pitfalls_block = _format_pitfalls_block(pitfalls)

        # Get anchors for each CTX child — Phase 5: rank by value_score
        sections = []
        # Top 10 CTXs by value_score, top 3 anchors each (keep prompt manageable)
        sorted_ctx = sorted(children, key=lambda c: c.get("value_score", 0) or c.get("access_count", 0), reverse=True)
        for ctx in sorted_ctx[:10]:
            ctx_id = ctx.get("id", "")
            ctx_name = ctx.get("content_preview", ctx.get("content", ""))[:60]

            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            # Exclude redflagged anchors from positive knowledge feed — they
            # surface separately as Known Pitfalls so the LLM treats them as
            # warnings, not facts to weave in.
            anchors = conn.execute("""
                SELECT mn.content FROM relations r
                JOIN memory_nodes mn ON mn.id = r.source_id
                WHERE r.target_id=? AND r.relation_type='child_of' AND mn.is_deleted=0
                  AND (mn.value_feedback IS NULL OR mn.value_feedback > -0.3)
                ORDER BY mn.value_score DESC, mn.access_count DESC LIMIT 3
            """, (ctx_id,)).fetchall()
            conn.close()

            anchor_texts = [a["content"][:120] for a in anchors]
            if anchor_texts:
                sections.append(f"TOPIC: {ctx_name}\nKEY KNOWLEDGE:\n" + "\n".join(f"- {a}" for a in anchor_texts))

        return f"""You are writing a wiki page for a personal knowledge base.

DOMAIN: {sc_name}

This domain has {len(children)} sub-topics:

{chr(10).join(sections)}

{pitfalls_block}

Write a comprehensive wiki article following these rules:
- Start with a 2-3 sentence overview of the domain
- Create one H2 section per major topic (group related topics together, skip trivial ones)
- Within each section, weave the knowledge into readable prose paragraphs
- Use specific numbers, facts, and details from the knowledge points
- If KNOWN PITFALLS are provided above, add a final `## Known Pitfalls` section
  that warns about them in prose — DO NOT treat them as facts to incorporate
- Write as if explaining to a colleague who just joined the team
- Keep it under 1000 words
- NO reference codes, IDs, node numbers, or technical metadata
- NO bullet lists in normal sections — write in prose paragraphs
- Use markdown: # for title, ## for sections, **bold** for key terms
"""

    def _build_ctx_prompt(self, node: Dict, children: List[Dict], node_data: Dict,
                           pitfalls: Optional[List[Dict]] = None) -> str:
        """Build prompt for CTX-level wiki page."""
        ctx_name = node.get("content", "")[:100]
        parents = node_data.get("parents", [])
        parent_name = parents[0].get("content", "")[:60] if parents else "Unknown"

        # Phase 5: exclude redflagged from positives; they surface as pitfalls
        positive_children = [c for c in children if not c.get("is_pitfall")]
        anchor_texts = [c.get("content_preview", c.get("content", ""))[:150]
                        for c in positive_children[:10]]
        pitfalls_block = _format_pitfalls_block(pitfalls)

        return f"""You are writing a wiki page for a personal knowledge base.

TOPIC: {ctx_name}
PARENT DOMAIN: {parent_name}

KEY KNOWLEDGE POINTS (ranked by value_score):
{chr(10).join(f"- {a}" for a in anchor_texts)}

{pitfalls_block}

Write a focused wiki article (400-600 words):
- Start with 1-2 sentence overview of this topic
- Weave ALL the knowledge points into readable prose paragraphs
- If KNOWN PITFALLS are provided above, close the article with a short
  `## Known Pitfalls` paragraph treating them as warnings, NOT facts
- Use specific numbers and facts
- Write as if explaining to a colleague
- NO bullet lists, NO IDs, NO metadata
- Use markdown: # for title, ## for sub-sections if needed
"""

    def _build_anc_prompt(self, node: Dict, node_data: Dict,
                           pitfalls: Optional[List[Dict]] = None) -> str:
        """Build prompt for ANC-level knowledge card."""
        content = node.get("content", "")[:300]
        parents = node_data.get("parents", [])
        parent_name = parents[0].get("content", "")[:60] if parents else ""

        # If this ANC itself is redflagged, frame the whole card as a warning
        v3 = node_data.get("value_v3") or {}
        is_redflag = bool(v3.get("redflag"))
        frame = ("## Warning — Known Pitfall\nThis insight is logged as a "
                 "pitfall. Treat as cautionary, not prescriptive.\n\n"
                 if is_redflag else "")

        return f"""You are writing a knowledge card for a personal wiki.

{frame}INSIGHT: {content}
CONTEXT: Part of "{parent_name}"

Write a concise knowledge card (100-200 words):
- State the core insight clearly in the first sentence
- Add context about why this matters
- Include any specific numbers or evidence
{"- Frame this as a WARNING or pitfall, not a recommendation" if is_redflag else ""}
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

        # Group related CTXs into logical sections — Phase 5: rank by value_score
        sorted_children = sorted(
            children,
            key=lambda c: (c.get("value_score", 0) or 0, c.get("access_count", 0)),
            reverse=True,
        )

        # Only include CTXs with anchors
        active_ctxs = []
        for ctx in sorted_children:
            ctx_id = ctx.get("id", "")
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            anchors = conn.execute("""
                SELECT mn.id, mn.content, mn.access_count, mn.value_score, mn.value_feedback
                FROM relations r
                JOIN memory_nodes mn ON mn.id = r.source_id
                WHERE r.target_id=? AND r.relation_type='child_of' AND mn.is_deleted=0
                  AND (mn.value_feedback IS NULL OR mn.value_feedback > -0.3)
                ORDER BY mn.value_score DESC, mn.access_count DESC
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
        lines.append(f"*This page is part of the ProMe memory wiki. "
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

    def _count_pitfalls_for_deliverable(self, deliverable_id: str,
                                          redflag_threshold: float = -0.3) -> int:
        """Phase 1 polish — count redflagged ANC nodes reachable from this
        deliverable's context_node + explicit anchor list. Cheap — one query."""
        if not deliverable_id:
            return 0
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT context_node_id, anchor_node_ids FROM deliverables WHERE id = ?",
                (deliverable_id,),
            ).fetchone()
            if not row:
                return 0
            ctx_id = row["context_node_id"] or ""
            try:
                anchor_ids = json.loads(row["anchor_node_ids"] or "[]")
            except Exception:
                anchor_ids = []

            if not ctx_id and not anchor_ids:
                return 0

            count = 0
            if ctx_id:
                result = conn.execute(
                    """WITH RECURSIVE sub(id) AS (
                         SELECT ?
                         UNION ALL
                         SELECT r.source_id FROM relations r
                         JOIN sub ON r.target_id = sub.id
                         WHERE r.relation_type = 'child_of'
                       )
                       SELECT COUNT(*) FROM memory_nodes mn
                       JOIN sub ON mn.id = sub.id
                       WHERE mn.is_deleted = 0 AND mn.level = 'ANC'
                         AND mn.value_feedback <= ?""",
                    (ctx_id, redflag_threshold),
                ).fetchone()
                count += int(result[0] or 0) if result else 0

            if anchor_ids:
                ph = ",".join("?" * len(anchor_ids))
                result = conn.execute(
                    f"""SELECT COUNT(*) FROM memory_nodes
                        WHERE id IN ({ph}) AND is_deleted = 0
                          AND value_feedback <= ?""",
                    (*anchor_ids, redflag_threshold),
                ).fetchone()
                count += int(result[0] or 0) if result else 0

            return count
        except Exception:
            return 0
        finally:
            conn.close()

    def _fetch_pitfalls_for_subtree(self, root_id: str, redflag_threshold: float = -0.3,
                                     limit: int = 10) -> List[Dict]:
        """Phase 5 — return redflagged (value_feedback <= threshold) ANC nodes
        reachable from root via child_of relations. Surfaced as Known Pitfalls
        in the prose and on the page."""
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Direct descendants (1-hop for speed; tree is shallow SC→CTX→ANC)
            rows = conn.execute("""
                WITH RECURSIVE sub(id) AS (
                    SELECT ?
                    UNION ALL
                    SELECT r.source_id FROM relations r
                    JOIN sub ON r.target_id = sub.id
                    WHERE r.relation_type = 'child_of'
                )
                SELECT mn.id, mn.content, mn.level, mn.value_feedback, mn.value_score
                FROM memory_nodes mn
                JOIN sub ON mn.id = sub.id
                WHERE mn.is_deleted = 0
                  AND mn.level = 'ANC'
                  AND mn.value_feedback <= ?
                ORDER BY mn.value_feedback ASC
                LIMIT ?
            """, (root_id, redflag_threshold, limit)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            conn.close()

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


def _format_pitfalls_block(pitfalls: Optional[List[Dict]]) -> str:
    """Phase 5 — render pitfalls as a clearly-labeled warning block for LLM
    prompts. The LLM is instructed to treat these as warnings, not facts."""
    if not pitfalls:
        return ""
    lines = ["KNOWN PITFALLS (negative feedback — treat as warnings, do NOT "
             "weave into main narrative):"]
    for p in pitfalls[:6]:
        preview = (p.get("content") or "").replace("\n", " ")[:180]
        fb = float(p.get("value_feedback") or 0.0)
        lines.append(f"- [{fb:+.2f}] {preview}")
    return "\n".join(lines)
