"""
Project-Deliverable matcher — maps work segments to projects using
semantic search (ChromaDB cosine) + hyperbolic navigation (Poincaré distance).

Projects (company-level like "YantrAI") map to SC nodes.
Deliverables map to Context/Anchor nodes.
The matcher traverses the Poincaré ball from SC → CTX → ANC to find
the best matching branch and leaf.
"""

import logging
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger("tracker.project_matcher")

PMIS_V2_DIR = Path(__file__).resolve().parents[3] / "pmis_v2"
if str(PMIS_V2_DIR) not in sys.path:
    sys.path.insert(0, str(PMIS_V2_DIR))


class ProjectMatcher:
    """Two-stage matching: semantic search → hyperbolic navigation."""

    def __init__(self, db_manager, chroma_store, embedder):
        self.db = db_manager
        self.chroma = chroma_store
        self.embedder = embedder

    def match_segment(self, segment: Dict, sc_node_id: str,
                      ctx_node_id: str, anc_node_id: str) -> Dict[str, Any]:
        """
        Match a work segment to a project/deliverable.

        Returns:
            {
                "project_id": str,
                "deliverable_id": str,
                "sc_node_id": str,
                "context_node_id": str,
                "anchor_node_id": str,
                "semantic_score": float,
                "hyperbolic_score": float,
                "combined_match_pct": float,  # 0.6×semantic + 0.4×hyperbolic
                "match_method": str,
            }
        """
        # Get all active projects
        projects = self.db.list_projects(status="active")
        if not projects:
            return self._no_match(segment)

        work_text = self._build_work_text(segment)

        # Stage 1: Semantic search against project SC nodes
        semantic_results = self._semantic_match(work_text, projects)
        if not semantic_results:
            return self._no_match(segment)

        # Stage 2: Hyperbolic navigation for best candidates
        best_match = self._hyperbolic_navigate(
            segment, sc_node_id, ctx_node_id, anc_node_id,
            semantic_results
        )

        # Log the match
        if best_match["combined_match_pct"] > 0:
            self._log_match(segment, best_match)

        return best_match

    def _build_work_text(self, segment: Dict) -> str:
        """Build search text from segment data."""
        parts = [
            segment.get("supercontext", ""),
            segment.get("context", ""),
            segment.get("anchor", ""),
            segment.get("detailed_summary", ""),
        ]
        return " > ".join(p for p in parts if p and p != "Unclassified")

    def _semantic_match(self, work_text: str, projects: List[Dict]) -> List[Dict]:
        """Stage 1: Embed work description and search against project nodes."""
        try:
            query_emb = self.embedder.embed_text(work_text)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return []

        # Try ChromaDB ANN first
        candidates = self.db.ann_query(query_emb, n_results=10, level_filter="SC")

        results = []
        for candidate in candidates:
            node_id = candidate["id"]
            similarity = candidate.get("chroma_similarity", 0)

            # Check if this SC node belongs to a project
            for project in projects:
                if project.get("sc_node_id") == node_id:
                    results.append({
                        "project_id": project["id"],
                        "project_name": project["name"],
                        "sc_node_id": node_id,
                        "semantic_score": similarity,
                    })
                    break

        # Also try exact SC title matching as fallback
        if not results:
            for project in projects:
                sc_node_id = project.get("sc_node_id", "")
                if sc_node_id:
                    node = self.db.get_node(sc_node_id)
                    if node:
                        # Simple title overlap
                        node_words = set(node.get("content", "").lower().split())
                        work_words = set(work_text.lower().split())
                        overlap = len(node_words & work_words)
                        if overlap > 0:
                            score = overlap / max(len(node_words), 1)
                            results.append({
                                "project_id": project["id"],
                                "project_name": project["name"],
                                "sc_node_id": sc_node_id,
                                "semantic_score": min(score, 1.0),
                            })

        # Sort by semantic score
        results.sort(key=lambda r: r["semantic_score"], reverse=True)
        return results[:5]

    def _hyperbolic_navigate(self, segment: Dict, seg_sc_id: str,
                             seg_ctx_id: str, seg_anc_id: str,
                             semantic_results: List[Dict]) -> Dict[str, Any]:
        """Stage 2: Use Poincaré distance to find the best branch and leaf."""
        from core.poincare import poincare_distance

        # Get segment node embeddings
        seg_anc_emb = self.db.get_embeddings(seg_anc_id)
        seg_hyp = seg_anc_emb.get("hyperbolic")

        best = None
        best_combined = 0

        for candidate in semantic_results:
            project_id = candidate["project_id"]
            semantic_score = candidate["semantic_score"]
            sc_node_id = candidate["sc_node_id"]

            # Navigate: SC → best CTX → best ANC
            contexts = self.db.get_children(sc_node_id)
            best_ctx = None
            best_ctx_dist = float("inf")
            best_anc = None
            best_anc_dist = float("inf")

            if seg_hyp is not None and contexts:
                for ctx in contexts:
                    ctx_emb = self.db.get_embeddings(ctx["id"])
                    ctx_hyp = ctx_emb.get("hyperbolic")
                    if ctx_hyp is not None:
                        dist = poincare_distance(seg_hyp, ctx_hyp)
                        if dist < best_ctx_dist:
                            best_ctx_dist = dist
                            best_ctx = ctx

                # Navigate to anchor level
                if best_ctx:
                    anchors = self.db.get_children(best_ctx["id"])
                    for anc in anchors:
                        anc_emb = self.db.get_embeddings(anc["id"])
                        anc_hyp = anc_emb.get("hyperbolic")
                        if anc_hyp is not None:
                            dist = poincare_distance(seg_hyp, anc_hyp)
                            if dist < best_anc_dist:
                                best_anc_dist = dist
                                best_anc = anc

            # Compute hyperbolic score (inverse distance, normalized to 0-1)
            if best_ctx_dist < float("inf"):
                h_score = 1.0 / (1.0 + best_ctx_dist)
            else:
                h_score = 0.0

            # Combined: 60% semantic + 40% hyperbolic
            combined = 0.6 * semantic_score + 0.4 * h_score

            # Check deliverable match
            deliverable_id = ""
            if best_ctx:
                deliverables = self.db.get_deliverables(project_id)
                for d in deliverables:
                    if d.get("context_node_id") == best_ctx["id"]:
                        deliverable_id = d["id"]
                        break

            if combined > best_combined:
                best_combined = combined
                best = {
                    "project_id": project_id,
                    "deliverable_id": deliverable_id,
                    "sc_node_id": sc_node_id,
                    "context_node_id": best_ctx["id"] if best_ctx else "",
                    "anchor_node_id": best_anc["id"] if best_anc else "",
                    "semantic_score": semantic_score,
                    "hyperbolic_score": h_score,
                    "combined_match_pct": combined,
                    "match_method": "semantic+hyperbolic" if h_score > 0 else "semantic_only",
                }

        return best if best else self._no_match(segment)

    def _no_match(self, segment: Dict) -> Dict[str, Any]:
        """Return empty match result."""
        return {
            "project_id": "",
            "deliverable_id": "",
            "sc_node_id": "",
            "context_node_id": "",
            "anchor_node_id": "",
            "semantic_score": 0,
            "hyperbolic_score": 0,
            "combined_match_pct": 0,
            "match_method": "none",
        }

    def _log_match(self, segment: Dict, match: Dict):
        """Log match to project_work_match_log table."""
        try:
            self.db.log_match({
                "segment_id": segment.get("target_segment_id", ""),
                "project_id": match["project_id"],
                "deliverable_id": match.get("deliverable_id", ""),
                "sc_node_id": match.get("sc_node_id", ""),
                "context_node_id": match.get("context_node_id", ""),
                "anchor_node_id": match.get("anchor_node_id", ""),
                "semantic_score": match["semantic_score"],
                "hyperbolic_score": match["hyperbolic_score"],
                "combined_match_pct": match["combined_match_pct"],
                "match_method": match["match_method"],
                "work_description": segment.get("detailed_summary", "")[:500],
                "worker_type": segment.get("worker", "human"),
                "time_mins": (segment.get("target_segment_length_secs", 0) or 0) / 60.0,
            })
        except Exception as e:
            logger.warning(f"Failed to log match: {e}")
