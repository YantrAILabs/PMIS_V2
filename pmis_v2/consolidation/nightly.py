"""
Nightly Consolidation Engine for PMIS v2.

Runs four passes:
  1. COMPRESS: Merge redundant Anchors into parent Context
  2. PROMOTE: Orphan Anchors with 3+ hits → attach to nearest Context
  3. BIRTH:   Cluster of 3+ orphans → create new Context node (LLM summarized)
  4. PRUNE:   Low-precision, low-access, low-surprise Anchors → soft-delete

Uses LLM (Ollama/Claude) for generating Context summaries during
COMPRESS (updating parent) and BIRTH (naming new Context).
"""

import json
import httpx
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from core.memory_node import MemoryNode, MemoryLevel
from core.surprise import compute_raw_surprise
from core.temporal import temporal_encode, compute_era
from core.poincare import assign_hyperbolic_coords, ProjectionManager
from core import config
from db.manager import DBManager


class NightlyConsolidation:
    def __init__(self, db: DBManager, hyperparams: Optional[Dict[str, Any]] = None):
        self.db = db
        self.hp = hyperparams or config.get_all()
        self.projection = ProjectionManager(
            input_dim=self.hp.get("local_embedding_dimensions", 768)
                if self.hp.get("use_local", True)
                else self.hp.get("embedding_dimensions", 1536),
            output_dim=self.hp.get("poincare_dimensions", 32),
        )
        self.actions_log: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute consolidation passes in order.

        The order was reshaped on 2026-04-20: activity-derived ANCs are now
        created *before* graph maintenance so they participate in COMPRESS /
        PROMOTE / BIRTH / CELL-DIVISION / PRUNE just like any other anchor.
        Previously they were injected after maintenance and so never had a
        chance to form new contexts or be merged with redundant siblings.

        Pass order:
          1.  ACTIVITY MERGE       — create today's activity-derived anchors
          2.  BACKFILL EMBEDDINGS  — ensure every node has a row in embeddings
          3.  COMPRESS             — merge redundant anchors (incl. today's)
          4.  PROMOTE              — adopt accessed orphans
          5.  BIRTH                — create contexts from orphan clusters
          6.  CELL DIVISION        — split oversized contexts
          7.  PRUNE                — soft-delete low-value anchors
          8.  HGCN TRAIN           — learn Poincare embeddings on final graph
          9.  PROJECT MATCHING     — tagged-session inherit + semantic match
          10. TIME ASSIGNMENT      — assign activity time to tree + projects
          11. REFRESH CTX STATS    — materialize cached context stats
          12. REINDEX              — rebuild ChromaDB ANN index
          13. WIKI REGEN           — invalidate changed wiki pages
        """
        self.actions_log = []
        results: Dict[str, List[Dict[str, Any]]] = {}

        # 1. Activity merge — today's raw activity segments become ANCs
        results["activity_merged"] = self._pass_activity_merge()

        # 2. Backfill embeddings — safety net so new ANCs (and any other
        #    batch-inserted node) have embeddings before maintenance runs.
        results["embeddings_backfilled"] = self._pass_backfill_embeddings()

        # 3-7. Graph maintenance — now operates on the full node set including
        #      today's activity anchors.
        results["compressed"] = self._pass_compress()
        results["promoted"] = self._pass_promote()
        results["birthed"] = self._pass_birth()
        results["cell_divided"] = self._pass_cell_division()
        results["pruned"] = self._pass_prune()

        # 8. HGCN train on the now-stable graph.
        hgcn_result = self._pass_hgcn_train()
        results["hgcn"] = [hgcn_result] if hgcn_result else []

        # 9. Project matching (tagged-session inherit + semantic for untagged).
        results["project_matches"] = self._pass_project_matching()

        # 9b. Work-page auto-match — bootstrap-gated, writes tag_state=proposed
        #     on state=open pages. User confirms via the Unassigned lane
        #     morning review; only confirmed tags feed next nightly's HGCN.
        try:
            from consolidation.work_page_matcher import run_work_page_matching
            wp_match = run_work_page_matching(self.db, self.hp)
            results["work_page_matches"] = [wp_match]
        except Exception as e:
            results["work_page_matches"] = [{"status": "error", "error": str(e)}]

        # 10. Time assignment.
        results["time_assigned"] = self._pass_time_assignment()

        # Persist all collected actions.
        for action in self.actions_log:
            self.db.log_consolidation(action)

        # 11. Refresh materialized context stats after bulk changes.
        total_actions = sum(len(v) for v in results.values())
        if total_actions > 0:
            self._refresh_all_context_stats()

        # 12. Reindex ANN.
        if self.db.has_ann_index and self.db._chroma:
            self.db._chroma.rebuild_from_db(self.db)

        # 12.5. Restructure red-flagged content — LLM regen of anchors/contexts
        #       whose materialized value_feedback dropped below the red-flag
        #       threshold. Honors is_user_edited. Runs AFTER reindex so ANN
        #       returns the old embeddings while deciding what to rewrite, and
        #       AFTER value_score was recomputed in its own Phase 3 pass. The
        #       regen itself re-embeds each rewritten node + syncs ChromaDB.
        results["restructured"] = self._pass_restructure()

        # 13. Invalidate changed wiki pages.
        self._pass_wiki_regen()

        return results

    # -------------------------------------------------------------------
    # PASS 12.5: RESTRUCTURE (audit-fix Item 7)
    # -------------------------------------------------------------------

    def _pass_restructure(self) -> List[Dict[str, Any]]:
        """
        Enqueue red-flagged nodes (value_feedback < redflag_threshold) and
        drain the queue with LLM regen. Skips is_user_edited=1 nodes.
        """
        try:
            from consolidation.restructure import Restructurer
        except Exception as e:
            logger = __import__("logging").getLogger("pmis.nightly")
            logger.warning(f"Restructure pass unavailable: {e}")
            return []

        rs = Restructurer(self.db, self.hp)
        enq = rs.enqueue_red_flags()
        actions = rs.run(max_jobs=self.hp.get("restructure_max_jobs_per_run", 50))
        for a in actions:
            self.actions_log.append(a)
        if enq.get("enqueued"):
            self.actions_log.append({
                "action": "restructure_enqueue",
                "reason": f"value_feedback<{enq.get('redflag_threshold')}",
                "details": enq,
            })
        return actions

    # NOTE: RSGD was removed on 2026-04-20. HGCN (pass 8) is its successor and
    # handles hyperbolic refinement with richer signals (co-retrieval + feedback
    # edges, not just parent-child). The rsgd_runs table is retained as a
    # historical log; no new rows are written.

    def _refresh_all_context_stats(self):
        """P2a: Recompute cached stats for all Context nodes after consolidation."""
        contexts = self.db.get_nodes_by_level("CTX")
        for ctx in contexts:
            self.db._refresh_context_stats(ctx["id"])

    # -------------------------------------------------------------------
    # PASS 1: COMPRESS
    # -------------------------------------------------------------------

    def _pass_compress(self) -> List[Dict[str, Any]]:
        """
        Merge Anchors that are perfectly predicted by their parent Context.
        Low surprise relative to parent + low individual access = absorb.
        """
        actions = []
        contexts = self.db.get_nodes_by_level("CTX")

        for ctx in contexts:
            children = self.db.get_children(ctx["id"])
            ctx_embs = self.db.get_embeddings(ctx["id"])
            ctx_emb = ctx_embs.get("euclidean")

            for child in children:
                child_embs = self.db.get_embeddings(child["id"])
                child_emb = child_embs.get("euclidean")

                if ctx_emb is None or child_emb is None:
                    continue

                distance = compute_raw_surprise(ctx_emb, child_emb)

                if (distance < self.hp.get("compress_distance_threshold", 0.12) and
                    child.get("access_count", 0) < self.hp.get("compress_access_threshold", 3) and
                    child.get("surprise_at_creation", 0) < self.hp.get("compress_surprise_threshold", 0.20)):

                    # Absorb: merge child content into parent
                    self.db.merge_into_parent(child["id"], ctx["id"])

                    action = {
                        "action": "compress",
                        "child_id": child["id"],
                        "parent_id": ctx["id"],
                        "distance": distance,
                    }
                    actions.append(action)
                    self.actions_log.append(action)

        return actions

    # -------------------------------------------------------------------
    # PASS 2: PROMOTE
    # -------------------------------------------------------------------

    def _pass_promote(self) -> List[Dict[str, Any]]:
        """
        Orphan Anchors accessed 3+ times → attach to nearest Context.
        """
        actions = []
        orphans = self.db.get_orphan_nodes()
        min_accesses = self.hp.get("promote_min_accesses", 3)

        for orphan in orphans:
            if orphan.get("access_count", 0) >= min_accesses:
                # Find nearest Context by embedding distance
                nearest_ctx = self._find_nearest_context_for_node(orphan["id"])
                if nearest_ctx:
                    self.db.attach_to_parent(orphan["id"], nearest_ctx["id"])

                    action = {
                        "action": "promote",
                        "orphan_id": orphan["id"],
                        "parent_id": nearest_ctx["id"],
                        "access_count": orphan.get("access_count", 0),
                    }
                    actions.append(action)
                    self.actions_log.append(action)

        return actions

    # -------------------------------------------------------------------
    # PASS 3: BIRTH
    # -------------------------------------------------------------------

    def _pass_birth(self) -> List[Dict[str, Any]]:
        """
        When 3+ orphans cluster together, create a new Context node.
        Uses LLM to generate the Context summary/name.
        """
        actions = []
        orphans = self.db.get_orphan_nodes()
        min_orphans = self.hp.get("birth_min_orphans", 3)

        if len(orphans) < min_orphans:
            return actions

        # Simple clustering: group by pairwise similarity
        clusters = self._cluster_orphans(orphans)

        for cluster in clusters:
            if len(cluster) < min_orphans:
                continue

            # Generate Context summary via LLM
            contents = [o.get("content", "")[:200] for o in cluster]
            summary = self._generate_context_summary(contents)

            # Create the new Context node
            # Use centroid of cluster embeddings
            cluster_embeddings = []
            for o in cluster:
                embs = self.db.get_embeddings(o["id"])
                if embs.get("euclidean") is not None:
                    cluster_embeddings.append(embs["euclidean"])

            if not cluster_embeddings:
                continue

            centroid = np.mean(cluster_embeddings, axis=0)
            hyp_coords = assign_hyperbolic_coords(
                euclidean_embedding=centroid,
                level="CTX",
                projection_manager=self.projection,
                hyperparams=self.hp,
            )
            temporal = temporal_encode(datetime.now(), self.hp.get("temporal_embedding_dim", 16))
            era = compute_era(datetime.now(), self.hp.get("era_boundaries", {}))

            new_ctx = MemoryNode.create(
                content=summary,
                level=MemoryLevel.CONTEXT,
                euclidean_embedding=centroid,
                hyperbolic_coords=hyp_coords,
                temporal_embedding=temporal,
                surprise=0.0,
                precision=0.3,
                era=era,
            )
            new_ctx.is_orphan = False
            new_ctx.is_tentative = False
            self.db.create_node(new_ctx)

            # Attach all orphans to new Context
            for o in cluster:
                self.db.attach_to_parent(o["id"], new_ctx.id, tree_id="auto_" + new_ctx.id[:8])

            action = {
                "action": "birth",
                "orphan_ids": [o["id"] for o in cluster],
                "new_context_id": new_ctx.id,
                "summary": summary[:200],
            }
            actions.append(action)
            self.actions_log.append(action)

        return actions

    # -------------------------------------------------------------------
    # PASS 6: DAILY ACTIVITY MERGE
    # -------------------------------------------------------------------

    def _pass_activity_merge(self) -> List[Dict[str, Any]]:
        """Create memories from today's activity tracker data."""
        try:
            from daily_activity_merge import DailyActivityMerger
            merger = DailyActivityMerger(self.db, self.hp)
            actions = merger.run()
            self.actions_log.extend(actions)
            return actions
        except Exception as e:
            import logging
            logging.getLogger("pmis.consolidation").warning(f"Activity merge failed: {e}")
            return []

    # -------------------------------------------------------------------
    # PASS 8: PROJECT MATCHING
    # -------------------------------------------------------------------

    def _pass_project_matching(self) -> List[Dict[str, Any]]:
        """Score today's activity anchors against active deliverables and
        write rows to project_work_match_log for surfacing on the Goals page."""
        try:
            from retrieval.project_matcher import ProjectMatcher
            matcher = ProjectMatcher(self.db, self.hp)
            actions = matcher.run()
            self.actions_log.extend(actions)
            return actions
        except Exception as e:
            import logging
            logging.getLogger("pmis.consolidation").warning(
                f"Project matching failed: {e}"
            )
            return []

    # -------------------------------------------------------------------
    # PASS 9: TIME ASSIGNMENT
    # -------------------------------------------------------------------

    def _pass_time_assignment(self) -> List[Dict[str, Any]]:
        """Assign activity time to knowledge tree branches and projects."""
        try:
            from time_assignment import TimeAssignment
            assigner = TimeAssignment(self.db, self.hp)
            actions = assigner.run()
            self.actions_log.extend(actions)
            return actions
        except Exception as e:
            import logging
            logging.getLogger("pmis.consolidation").warning(f"Time assignment failed: {e}")
            return []

    # -------------------------------------------------------------------
    # PASS 10: WIKI REGEN
    # -------------------------------------------------------------------

    def _pass_wiki_regen(self):
        """Invalidate wiki page cache for nodes changed in this consolidation."""
        try:
            import sqlite3
            changed_ids = set()
            for action in self.actions_log:
                for key in ["new_context_id", "new_node_id", "node_id", "memory_node_id",
                             "matched_ctx_id", "matched_sc_id"]:
                    if action.get(key):
                        changed_ids.add(action[key])
                for key in ["orphan_ids", "source_node_ids"]:
                    if action.get(key):
                        changed_ids.update(action[key])

            if not changed_ids:
                return

            conn = sqlite3.connect(self.db.db_path)
            for node_id in changed_ids:
                conn.execute("DELETE FROM wiki_page_cache WHERE node_id = ?", (node_id,))
                # Also invalidate parents
                parents = conn.execute(
                    "SELECT target_id FROM relations WHERE source_id=? AND relation_type='child_of'",
                    (node_id,)
                ).fetchall()
                for p in parents:
                    conn.execute("DELETE FROM wiki_page_cache WHERE node_id = ?", (p[0],))
            conn.commit()
            conn.close()
        except Exception as e:
            import logging
            logging.getLogger("pmis.consolidation").warning(f"Wiki regen failed: {e}")

    # -------------------------------------------------------------------
    # PASS 6b: BACKFILL MISSING EMBEDDINGS
    # -------------------------------------------------------------------

    def _pass_backfill_embeddings(self) -> List[Dict[str, Any]]:
        """
        Find any non-deleted memory_node that lacks an embeddings row, generate
        its euclidean embedding from content, and insert. Guards against batch
        insert scripts that bypass the normal ingestion pipeline.
        """
        import sqlite3
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT m.id, m.content, m.level
            FROM memory_nodes m
            LEFT JOIN embeddings e ON e.node_id = m.id
            WHERE m.is_deleted = 0
              AND e.node_id IS NULL
              AND m.content IS NOT NULL
              AND length(m.content) > 0
        """).fetchall()
        if not rows:
            conn.close()
            return []

        try:
            from ingestion.embedder import Embedder
            embedder = Embedder(self.hp)
            texts = [r["content"] for r in rows]
            vectors = embedder.batch_embed_texts(texts)
        except Exception as e:
            import logging
            logging.getLogger("pmis.consolidation").warning(
                f"Embedding backfill failed: {e}"
            )
            conn.close()
            return []

        actions = []
        for r, vec in zip(rows, vectors):
            blob = vec.astype(np.float32).tobytes()
            conn.execute(
                "INSERT INTO embeddings (node_id, euclidean, is_learned) VALUES (?, ?, 0)",
                (r["id"], blob),
            )
            actions.append({
                "action": "embedding_backfill",
                "node_id": r["id"],
                "level": r["level"],
            })
        conn.commit()
        conn.close()
        for a in actions:
            self.actions_log.append(a)
        return actions

    # -------------------------------------------------------------------
    # PASS 8: HGCN TRAIN
    # -------------------------------------------------------------------

    def _pass_hgcn_train(self) -> Optional[Dict[str, Any]]:
        """
        Train Hyperbolic GCN on graph structure + co-retrieval + feedback.
        This is the sole hyperbolic-training pass (RSGD was removed
        2026-04-20). Always runs — co-retrieval data grows daily.
        """
        try:
            from core.hgcn import HGCNTrainer
            trainer = HGCNTrainer(self.db, self.hp)
            result = trainer.train()

            action = {
                "action": "hgcn_train",
                "nodes_trained": result.get("nodes_trained", 0),
                "structural_edges": result.get("structural_edges", 0),
                "co_retrieval_edges": result.get("co_retrieval_edges", 0),
                "feedback_edges": result.get("feedback_pos_edges", 0) + result.get("feedback_neg_edges", 0),
                "epochs": result.get("epochs_run", 0),
                "final_loss": result.get("final_loss"),
                "wall_time": result.get("wall_time_seconds"),
            }
            self.actions_log.append(action)
            return result
        except Exception as e:
            import logging
            logging.getLogger("pmis.consolidation").error(f"HGCN training failed: {e}")
            return {"error": str(e)}

    # -------------------------------------------------------------------
    # PASS 3b: CELL DIVISION
    # -------------------------------------------------------------------

    def _pass_cell_division(self) -> List[Dict[str, Any]]:
        """
        Split oversized CTX/SC nodes recursively.
        Runs after BIRTH (new contexts exist) and before PRUNE.
        """
        from consolidation.cell_division import CellDivision

        divider = CellDivision(
            db=self.db,
            hyperparams=self.hp,
            projection=self.projection,
            generate_summary_fn=self._generate_context_summary,
        )
        actions = divider.run()
        self.actions_log.extend(actions)
        return actions

    # -------------------------------------------------------------------
    # PASS 4: PRUNE
    # -------------------------------------------------------------------

    def _pass_prune(self) -> List[Dict[str, Any]]:
        """
        Soft-delete Anchors that are low-value:
        low precision + low access + low surprise + old enough.
        """
        actions = []
        candidates = self.db.get_prune_candidates(
            max_precision=self.hp.get("prune_max_precision", 0.15),
            max_access_count=self.hp.get("prune_max_access_count", 2),
            max_surprise=self.hp.get("prune_max_surprise", 0.15),
            min_age_days=self.hp.get("prune_min_age_days", 14),
        )

        for c in candidates:
            self.db.soft_delete(c["id"])
            action = {
                "action": "prune",
                "node_id": c["id"],
                "reason": "low_value",
                "precision": c.get("precision"),
                "access_count": c.get("access_count"),
                "surprise": c.get("surprise_at_creation"),
            }
            actions.append(action)
            self.actions_log.append(action)

        return actions

    # -------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------

    def _find_nearest_context_for_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Find nearest Context to a given node by embedding distance."""
        node_embs = self.db.get_embeddings(node_id)
        node_emb = node_embs.get("euclidean")
        if node_emb is None:
            return None

        contexts = self.db.get_nodes_by_level("CTX")
        best = None
        best_dist = float("inf")

        for ctx in contexts:
            ctx_embs = self.db.get_embeddings(ctx["id"])
            ctx_emb = ctx_embs.get("euclidean")
            if ctx_emb is None:
                continue
            dist = compute_raw_surprise(node_emb, ctx_emb)
            if dist < best_dist:
                best_dist = dist
                best = ctx

        return best

    def _cluster_orphans(self, orphans: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Simple greedy clustering of orphans by embedding similarity.
        Returns list of clusters (each cluster is a list of orphan dicts).
        """
        threshold = self.hp.get("birth_cluster_threshold", 0.35)

        # Load embeddings
        orphan_embs = {}
        for o in orphans:
            embs = self.db.get_embeddings(o["id"])
            if embs.get("euclidean") is not None:
                orphan_embs[o["id"]] = embs["euclidean"]

        if not orphan_embs:
            return []

        # Greedy clustering
        assigned = set()
        clusters = []

        orphan_ids = list(orphan_embs.keys())
        orphan_map = {o["id"]: o for o in orphans}

        for seed_id in orphan_ids:
            if seed_id in assigned:
                continue

            cluster = [orphan_map[seed_id]]
            assigned.add(seed_id)

            for other_id in orphan_ids:
                if other_id in assigned:
                    continue
                dist = compute_raw_surprise(orphan_embs[seed_id], orphan_embs[other_id])
                if dist < threshold:
                    cluster.append(orphan_map[other_id])
                    assigned.add(other_id)

            clusters.append(cluster)

        return clusters

    def _generate_context_summary(self, child_contents: List[str]) -> str:
        """
        Use LLM to generate a concise Context summary from child Anchors.
        Falls back to simple concatenation if LLM is unavailable.
        """
        prompt = (
            "You are a memory summarization system. Given these memory fragments, "
            "generate a single concise topic label and 1-sentence summary that captures "
            "the common theme. Respond with ONLY the label and summary, nothing else.\n\n"
            "RULES:\n"
            "- Use plain descriptive English only.\n"
            "- Do NOT invent reference codes, IDs, shorthand labels, or alphanumeric "
            "identifiers (e.g. no 'PM-25', 'MEM-3', 'CTX-7', etc.).\n"
            "- Do NOT quote system internals, table names, or index names.\n"
            "- The label should read like a natural topic name a human would use.\n\n"
            "Memory fragments:\n"
        )
        for i, content in enumerate(child_contents[:10], 1):
            prompt += f"  {i}. {content}\n"
        prompt += "\nTopic label and summary:"

        try:
            if self.hp.get("use_local", True):
                raw = self._call_ollama(prompt)
            else:
                raw = self._call_anthropic(prompt)
            return self._sanitize_summary(raw)
        except Exception as e:
            # Fallback: just concatenate first 3
            combined = " | ".join(c[:80] for c in child_contents[:3])
            return f"[Auto-Context] {combined}"

    @staticmethod
    def _sanitize_summary(text: str) -> str:
        """
        Strip fabricated reference codes from LLM-generated summaries.
        Prevents hallucinated identifiers like 'PM-25', 'MEM-3', 'CTX-7'
        from being stored as node content.
        """
        import re
        # Remove patterns like PM-25, MEM-3, CTX-07, ANC-12, REF-001, etc.
        cleaned = re.sub(
            r'\b[A-Z]{1,5}-\d{1,4}\b',
            '',
            text,
        )
        # Remove patterns like [PM25], (MEM3), {CTX7}
        cleaned = re.sub(
            r'[\[\(\{][A-Z]{1,5}\d{1,4}[\]\)\}]',
            '',
            cleaned,
        )
        # Collapse multiple spaces / leading-trailing whitespace
        cleaned = re.sub(r'  +', ' ', cleaned).strip()
        return cleaned

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
        import os
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
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"].strip()
