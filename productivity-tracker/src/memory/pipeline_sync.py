"""
Full ProMe memory pipeline sync for productivity segments.

Runs the complete pipeline: SQL → Vector → Hyperbolic → Project Matching → RSGD.
Called every 30 minutes by the tracker daemon when new frames exist.
"""

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("tracker.pipeline_sync")

# Add ProMe (pmis_v2) to path
PMIS_V2_DIR = Path(__file__).resolve().parents[3] / "pmis_v2"
if str(PMIS_V2_DIR) not in sys.path:
    sys.path.insert(0, str(PMIS_V2_DIR))


class ProductivityPipelineSync:
    """Runs full ProMe pipeline on unsynced productivity segments."""

    def __init__(self, db_manager, chroma_store, embedder, poincare_pm,
                 rsgd_trainer, nightly_consolidation, tracker_db,
                 hyperparams: Optional[Dict] = None):
        self.db = db_manager
        self.chroma = chroma_store
        self.embedder = embedder
        self.poincare_pm = poincare_pm
        self.rsgd = rsgd_trainer
        self.consolidation = nightly_consolidation
        self.tracker_db = tracker_db
        self.hyperparams = hyperparams or {}

    @classmethod
    def from_config(cls, config: dict) -> "ProductivityPipelineSync":
        """Factory method — creates all ProMe components from tracker config."""
        import yaml

        hyper_path = PMIS_V2_DIR / "hyperparameters.yaml"
        hyperparams = {}
        if hyper_path.exists():
            with open(hyper_path, encoding="utf-8") as f:
                hyperparams = yaml.safe_load(f) or {}

        from db.manager import DBManager
        from db.chroma_store import ChromaStore
        from ingestion.embedder import Embedder
        from core.poincare import ProjectionManager
        from core.rsgd import RSGDTrainer
        from consolidation.nightly import NightlyConsolidation

        pmis_db_path = config.get("pmis", {}).get(
            "db_path",
            str(Path.home() / "Desktop" / "memory" / "pmis_v2" / "data" / "memory.db")
        )
        chroma_path = config.get("pmis", {}).get(
            "chromadb_path",
            str(Path.home() / "Desktop" / "memory" / "pmis_v2" / "data" / "chroma")
        )

        chroma = ChromaStore(persist_dir=chroma_path)
        db_mgr = DBManager(db_path=pmis_db_path, chroma_store=chroma)

        # Match embedding model + dimensions to existing DB to avoid mismatch
        import numpy as np
        existing_dim = 768  # default
        try:
            sample = db_mgr._conn.execute(
                "SELECT euclidean FROM embeddings WHERE euclidean IS NOT NULL LIMIT 1"
            ).fetchone()
            if sample and sample[0]:
                existing_dim = len(np.frombuffer(sample[0], dtype=np.float32))
        except Exception:
            pass

        if existing_dim == 1536:
            hyperparams["use_local"] = False
            hyperparams["embedding_dimensions"] = 1536
            hyperparams["local_embedding_dimensions"] = 1536
        else:
            hyperparams["use_local"] = True
            hyperparams["local_embedding_dimensions"] = existing_dim
            hyperparams["embedding_dimensions"] = existing_dim

        logger.info(f"Embedder: dim={existing_dim}, use_local={hyperparams.get('use_local')}")

        embedder = Embedder(hyperparams=hyperparams)
        pm = ProjectionManager(input_dim=existing_dim)
        rsgd = RSGDTrainer(hyperparams=hyperparams)
        consolidation = NightlyConsolidation(db_mgr, hyperparams=hyperparams)

        from src.storage.db import Database
        tracker_db = Database()

        return cls(db_mgr, chroma, embedder, pm, rsgd, consolidation,
                   tracker_db, hyperparams)

    def run(self, unsynced_segments: List[Dict]) -> Dict[str, Any]:
        """Execute full pipeline: SQL → Vector → Hyperbolic → Match → RSGD."""
        sync_id = str(uuid.uuid4())[:10]
        started_at = datetime.now().isoformat()

        self.db.log_sync({
            "id": sync_id,
            "sync_type": "30min",
            "triggered_at": started_at,
            "segments_processed": len(unsynced_segments),
            "status": "running",
        })

        nodes_created = 0
        nodes_updated = 0
        all_touched_ids = []
        match_scores = []

        try:
            for segment in unsynced_segments:
                try:
                    result = self._process_segment(segment)
                    nodes_created += result.get("nodes_created", 0)
                    nodes_updated += result.get("nodes_updated", 0)
                    all_touched_ids.extend(result.get("node_ids", []))
                    if result.get("match_score", 0) > 0:
                        match_scores.append(result["match_score"])
                except Exception as seg_err:
                    logger.warning(f"Segment {segment.get('target_segment_id','?')} failed: {seg_err}")
                    continue

            # Step 5: Incremental RSGD on touched nodes
            rsgd_epochs = 0
            if all_touched_ids:
                rsgd_epochs = self._run_incremental_rsgd(all_touched_ids)

            avg_match = sum(match_scores) / len(match_scores) if match_scores else 0

            self.db.update_sync_status(sync_id, "completed", datetime.now().isoformat())

            return {
                "sync_id": sync_id,
                "segments_processed": len(unsynced_segments),
                "nodes_created": nodes_created,
                "nodes_updated": nodes_updated,
                "matches_found": len(match_scores),
                "avg_match_pct": round(avg_match * 100, 1),
                "rsgd_epochs_run": rsgd_epochs,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Pipeline sync failed: {e}")
            self.db.update_sync_status(sync_id, "failed", datetime.now().isoformat())
            return {
                "sync_id": sync_id,
                "segments_processed": len(unsynced_segments),
                "nodes_created": nodes_created,
                "nodes_updated": nodes_updated,
                "matches_found": 0,
                "avg_match_pct": 0,
                "rsgd_epochs_run": 0,
                "status": "failed",
                "error": str(e),
            }

    def _process_segment(self, segment: Dict) -> Dict[str, Any]:
        """Process a single segment through the full pipeline."""
        sc_title = segment.get("supercontext", "Unclassified")
        ctx_title = segment.get("context", "Unclassified")
        anc_title = segment.get("anchor", "Unclassified")
        summary = segment.get("detailed_summary", "")
        segment_id = segment.get("target_segment_id", "")
        duration_secs = segment.get("target_segment_length_secs") or 0
        duration_mins = max(duration_secs / 60.0, 0.01)  # minimum 0.01 min
        worker = segment.get("worker", "human")
        human_mins = duration_mins if worker == "human" else 0
        ai_mins = duration_mins if worker == "ai" else 0

        node_ids = []
        created = 0

        # Step 1-3: Find or create SC → CTX → ANC with triple embeddings
        sc_id, sc_new = self._find_or_create_node(content=sc_title, level="SC", parent_id=None)
        node_ids.append(sc_id)
        created += sc_new

        ctx_id, ctx_new = self._find_or_create_node(content=ctx_title, level="CTX", parent_id=sc_id)
        node_ids.append(ctx_id)
        created += ctx_new

        anc_content = f"{anc_title}: {summary}" if summary else anc_title
        anc_id, anc_new = self._find_or_create_node(content=anc_content, level="ANC", parent_id=ctx_id)
        node_ids.append(anc_id)
        created += anc_new

        # Update productivity time on all three nodes
        for nid in node_ids:
            self.db.update_node_productivity_time(nid, duration_mins, human_mins, ai_mins)

        # Step 4: Project matching
        # First check segment_override_bindings — Phase 1 hard-bind from
        # a user-confirmed work_session short-circuits the semantic matcher.
        match_result = {"combined_match_pct": 0, "project_id": "", "deliverable_id": ""}
        override = None
        try:
            override = self.db.get_segment_override(segment_id) if segment_id else None
        except Exception:
            override = None

        if override and override.get("deliverable_id"):
            match_result = {
                "combined_match_pct": 1.0,
                "project_id": override.get("project_id", ""),
                "deliverable_id": override.get("deliverable_id", ""),
                "match_method": f"session_override:{override.get('source','session')}",
            }
        else:
            try:
                from src.memory.project_matcher import ProjectMatcher
                matcher = ProjectMatcher(self.db, self.chroma, self.embedder)
                match_result = matcher.match_segment(segment, sc_id, ctx_id, anc_id)
            except Exception as match_err:
                logger.debug(f"Project matching skipped: {match_err}")

        match_score = match_result.get("combined_match_pct", 0)
        is_productive = 1 if match_score >= 0.5 else (0 if match_score == 0 else -1)

        # Mark segment as synced in tracker DB
        self.tracker_db.mark_segment_synced(
            segment_id=segment_id,
            sc_node_id=sc_id,
            context_node_id=ctx_id,
            anchor_node_id=anc_id,
            match_score=match_score,
            is_productive=is_productive,
            project_id=match_result.get("project_id", ""),
            deliverable_id=match_result.get("deliverable_id", ""),
        )

        return {
            "action": "created" if created > 0 else "updated",
            "nodes_created": created,
            "nodes_updated": 3 - created,
            "node_ids": node_ids,
            "match_score": match_score,
        }

    def _find_or_create_node(self, content: str, level: str,
                              parent_id: Optional[str] = None) -> tuple:
        """
        Find an existing node by semantic similarity or create a new one.
        Returns (node_id, was_created: 0 or 1).
        Runs Steps 1-3: SQL → Vector → Hyperbolic.
        """
        from core.memory_node import MemoryNode, MemoryLevel
        from core.poincare import assign_hyperbolic_coords, place_near_parent
        from core.temporal import temporal_encode

        level_enum = {
            "SC": MemoryLevel.SUPER_CONTEXT,
            "CTX": MemoryLevel.CONTEXT,
            "ANC": MemoryLevel.ANCHOR,
        }[level]

        # Step 1a: Check EXACT content match in SQL (must also match parent for CTX/ANC)
        with self.db._connect() as conn:
            if parent_id:
                # For CTX/ANC: must be exact content AND child of same parent
                exact = conn.execute("""
                    SELECT mn.id FROM memory_nodes mn
                    JOIN relations r ON mn.id = r.source_id AND r.relation_type = 'child_of'
                    WHERE mn.content = ? AND mn.level = ? AND mn.is_deleted = 0
                      AND r.target_id = ?
                """, (content, level, parent_id)).fetchone()
            else:
                # For SC: just exact content match
                exact = conn.execute(
                    "SELECT id FROM memory_nodes WHERE content = ? AND level = ? AND is_deleted = 0",
                    (content, level)
                ).fetchone()
            if exact:
                self.db.update_node_access(exact["id"], query_text=content)
                return exact["id"], 0

        # Step 2: Generate Euclidean embedding (WHAT?)
        euclidean = self.embedder.embed_text(content)

        # Step 2a: Check SEMANTIC match via ChromaDB ANN
        # For SC: use 0.85 threshold (broad matching across domains)
        # For CTX/ANC: use 0.95 threshold AND verify same parent (prevent cross-branch merging)
        threshold = 0.85 if level == "SC" else 0.95
        candidates = self.db.ann_query(euclidean, n_results=5, level_filter=level)
        for candidate in candidates:
            if candidate.get("chroma_similarity", 0) > threshold:
                existing_id = candidate["id"]
                node = self.db.get_node(existing_id)
                if node and node.get("level") == level:
                    # For CTX/ANC: verify same parent
                    if parent_id:
                        is_child = self.db._conn.execute(
                            "SELECT 1 FROM relations WHERE source_id = ? AND target_id = ? AND relation_type = 'child_of'",
                            (existing_id, parent_id)
                        ).fetchone()
                        if not is_child:
                            continue  # Wrong parent — skip this candidate
                    self.db.update_node_access(existing_id, query_text=content)
                    return existing_id, 0  # Found existing under same parent

        # No match — create new node with triple embeddings

        # Step 3a: Hyperbolic coordinates (WHERE?)
        parent_coords = None
        if parent_id:
            parent_emb = self.db.get_embeddings(parent_id)
            parent_coords = parent_emb.get("hyperbolic")

        if parent_coords is not None:
            hyp_coords = place_near_parent(
                euclidean, parent_coords, level, self.poincare_pm, self.hyperparams
            )
        else:
            hyp_coords = assign_hyperbolic_coords(
                euclidean, level, self.poincare_pm, hyperparams=self.hyperparams
            )

        # Step 3b: Temporal embedding (WHEN?)
        temporal = temporal_encode(datetime.now())

        # Create the node with all three embeddings
        node = MemoryNode.create(
            content=content,
            level=level_enum,
            euclidean_embedding=euclidean,
            hyperbolic_coords=hyp_coords,
            temporal_embedding=temporal,
            source_conversation_id="productivity_tracker",
            surprise=0.3,
            precision=0.5,
        )

        # Persist to SQL + ChromaDB (auto-synced by DBManager)
        node_id = self.db.create_node(node)

        # Create parent-child relation
        if parent_id:
            self.db.create_relation(node_id, parent_id, "child_of")

        # Mark as productivity-sourced node
        with self.db._connect() as conn:
            conn.execute(
                "UPDATE memory_nodes SET is_project_node = 1 WHERE id = ?",
                (node_id,)
            )

        return node_id, 1  # Created new

    def _run_incremental_rsgd(self, touched_ids: List[str]) -> int:
        """Run RSGD on touched nodes + neighbors to refine hyperbolic positions."""
        try:
            all_hyp = self.db.get_all_hyperbolic()
            edges = self.db.get_child_of_edges()
            levels = self.db.get_node_levels()

            if not all_hyp or not edges:
                return 0

            # Expand to 1-hop neighbors
            touched_set = set(touched_ids)
            for child_id, parent_id in edges:
                if child_id in touched_set or parent_id in touched_set:
                    touched_set.add(child_id)
                    touched_set.add(parent_id)

            epochs = self.hyperparams.get("rsgd_epochs_incremental", 8)

            # RSGDTrainer.train() expects embeddings dict, edges list, levels dict
            result = self.rsgd.train(
                embeddings=all_hyp,
                edges=edges,
                node_levels=levels,
                epochs=epochs,
            )

            # RSGDResult has: updated_embeddings, final_loss, epochs_run, wall_time_seconds
            if result.updated_embeddings:
                self.db.batch_update_hyperbolic(result.updated_embeddings)

            self.db.log_rsgd_run({
                "run_type": "incremental_productivity",
                "epochs": result.epochs_run,
                "final_loss": result.final_loss,
                "nodes_updated": result.nodes_updated,
                "edges_used": result.edges_used,
                "learning_rate": self.rsgd.lr,
                "wall_time_seconds": result.wall_time_seconds,
            })

            logger.info(f"Incremental RSGD: {epochs} epochs, {len(touched_set)} nodes")
            return epochs

        except Exception as e:
            logger.warning(f"Incremental RSGD failed: {e}")
            return 0
