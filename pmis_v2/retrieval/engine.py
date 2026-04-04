"""
Retrieval Engine for PMIS v2.

Performs γ-weighted blended retrieval:
  - Narrow search (high threshold, small k) weighted by γ
  - Broad search (low threshold, large k) weighted by (1-γ)

Results scored by: semantic(0.4) + hierarchy(0.3) + temporal(0.15) + precision(0.15)
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.poincare import poincare_distance, hierarchy_level, RelationTransform
from core.temporal import temporal_similarity, temporal_encode
from core.surprise import compute_raw_surprise
from db.manager import DBManager
from core import config


class RetrievalEngine:
    def __init__(self, db: DBManager, hyperparams: Optional[Dict[str, Any]] = None):
        self.db = db
        self.hp = hyperparams or config.get_all()
        self.relation_transform = RelationTransform(
            dim=self.hp.get("poincare_dimensions", 32)
        )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        gamma: float,
        effective_surprise: float = 0.5,
        tree_context: Optional[str] = None,
        target_level: Optional[str] = None,
        top_k: int = 10,
        query_timestamp: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval entry point.
        Returns ranked list of memory nodes with scores.
        """
        top_k = min(top_k, self.hp.get("retrieval_max_results", 10))
        ts = query_timestamp or datetime.now()
        query_temporal = temporal_encode(ts, dim=self.hp.get("temporal_embedding_dim", 16))

        # Determine how many to fetch from each strategy
        narrow_k = max(3, int(top_k * 1.5 * gamma))
        broad_k = max(3, int(top_k * 1.5 * (1.0 - gamma)))

        # --- NARROW RETRIEVAL: high threshold, from best Context subtree ---
        narrow_results = self._retrieve_narrow(
            query_embedding, narrow_k, tree_context, target_level
        )

        # --- BROAD RETRIEVAL: low threshold, cross-tree ---
        broad_results = self._retrieve_broad(
            query_embedding, broad_k, target_level
        )

        # --- MERGE & DEDUPLICATE ---
        all_candidates = {}
        for r in narrow_results:
            all_candidates[r["id"]] = {**r, "_source": "narrow"}
        for r in broad_results:
            if r["id"] not in all_candidates:
                all_candidates[r["id"]] = {**r, "_source": "broad"}

        # Compute query hyperbolic coords for hierarchy scoring
        # Use nearest context centroid if available, else place from projection
        query_hyperbolic = None
        if tree_context:
            query_hyperbolic = self.db.get_centroid(tree_context)
        if query_hyperbolic is None:
            # Fallback: use projection for approximate position
            from core.poincare import assign_hyperbolic_coords, ProjectionManager
            pm = ProjectionManager(input_dim=len(query_embedding), output_dim=self.hp.get("poincare_dimensions", 32))
            query_hyperbolic = assign_hyperbolic_coords(
                query_embedding, "ANC", pm, hyperparams=self.hp
            )

        # --- SCORE EACH CANDIDATE ---
        scored = []
        for node_id, candidate in all_candidates.items():
            scores = self._compute_score(
                query_embedding=query_embedding,
                query_temporal=query_temporal,
                candidate=candidate,
                gamma=gamma,
                tree_context=tree_context,
                query_hyperbolic=query_hyperbolic,
            )
            candidate["final_score"] = scores["final_score"]
            candidate["semantic_score"] = scores["semantic_score"]
            candidate["hierarchy_score"] = scores["hierarchy_score"]
            candidate["temporal_score"] = scores["temporal_score"]
            candidate["precision_score"] = scores["precision_score"]
            scored.append(candidate)

        # --- SORT & FILTER ---
        scored.sort(key=lambda x: x["final_score"], reverse=True)

        if target_level:
            scored = [s for s in scored if s.get("level") == target_level]

        # Record access on returned results
        for item in scored[:top_k]:
            self.db.update_node_access(
                item["id"],
                gamma=gamma,
                surprise=effective_surprise,
                semantic_distance=item.get("semantic_distance"),
            )

        return scored[:top_k]

    def _retrieve_narrow(
        self,
        query_embedding: np.ndarray,
        k: int,
        tree_context: Optional[str],
        target_level: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Narrow retrieval: high similarity threshold.
        P1a: Uses ChromaDB ANN if available, falls back to linear scan.
        """
        threshold = self.hp.get("retrieval_narrow_threshold", 0.82)

        # P1a: Try ANN index first (O(log N) instead of O(N))
        if self.db.has_ann_index:
            candidates = self.db.ann_query(
                query_embedding, n_results=k * 3,
                level_filter=target_level, tree_filter=tree_context,
            )
            # Hydrate candidates with full data from SQLite
            return self._hydrate_ann_candidates(candidates, threshold)

        # Fallback: linear scan
        if target_level:
            candidates = self.db.get_nodes_by_level(target_level)
        else:
            candidates = (
                self.db.get_nodes_by_level("ANC") +
                self.db.get_nodes_by_level("CTX")
            )

        if tree_context:
            candidates = [
                c for c in candidates
                if tree_context in c.get("tree_ids", []) or not c.get("tree_ids")
            ]

        return self._rank_by_similarity(query_embedding, candidates, k, threshold)

    def _retrieve_broad(
        self,
        query_embedding: np.ndarray,
        k: int,
        target_level: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Broad retrieval: low threshold, cross-tree.
        P1a: Uses ChromaDB ANN if available, falls back to linear scan.
        """
        threshold = self.hp.get("retrieval_broad_threshold", 0.45)

        # P1a: Try ANN index first
        if self.db.has_ann_index:
            candidates = self.db.ann_query(
                query_embedding, n_results=k * 3,
                level_filter=target_level, tree_filter=None,
            )
            return self._hydrate_ann_candidates(candidates, threshold)

        # Fallback: linear scan
        if target_level:
            candidates = self.db.get_nodes_by_level(target_level)
        else:
            candidates = (
                self.db.get_nodes_by_level("SC") +
                self.db.get_nodes_by_level("CTX") +
                self.db.get_nodes_by_level("ANC")
            )

        return self._rank_by_similarity(query_embedding, candidates, k, threshold)

    def _hydrate_ann_candidates(
        self,
        ann_results: List[Dict[str, Any]],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        P1a: Take ChromaDB ANN results (IDs + distances) and hydrate with
        full metadata + embeddings from SQLite.
        """
        hydrated = []
        for candidate in ann_results:
            sim = candidate.get("chroma_similarity", 0)
            if sim < threshold:
                continue

            node = self.db.get_node(candidate["id"])
            if not node:
                continue

            embs = self.db.get_embeddings(candidate["id"])
            hydrated.append({
                **node,
                "euclidean_embedding": embs.get("euclidean"),
                "hyperbolic_coords": embs.get("hyperbolic"),
                "temporal_embedding": embs.get("temporal"),
                "semantic_distance": 1.0 - sim,
                "semantic_similarity": sim,
            })

        return hydrated

    def _rank_by_similarity(
        self,
        query_embedding: np.ndarray,
        candidates: List[Dict[str, Any]],
        k: int,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Rank candidates by cosine similarity, filter by threshold."""
        results = []

        for c in candidates:
            embs = self.db.get_embeddings(c["id"])
            c_emb = embs.get("euclidean")
            if c_emb is None:
                continue

            distance = compute_raw_surprise(query_embedding, c_emb)
            similarity = 1.0 - distance

            if similarity >= threshold:
                results.append({
                    **c,
                    "euclidean_embedding": c_emb,
                    "hyperbolic_coords": embs.get("hyperbolic"),
                    "temporal_embedding": embs.get("temporal"),
                    "semantic_distance": distance,
                    "semantic_similarity": similarity,
                })

        results.sort(key=lambda x: x["semantic_similarity"], reverse=True)
        return results[:k]

    def _compute_score(
        self,
        query_embedding: np.ndarray,
        query_temporal: np.ndarray,
        candidate: Dict[str, Any],
        gamma: float,
        tree_context: Optional[str],
        query_hyperbolic: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Combined scoring. Returns dict with final_score + all component scores.
          score = w_sem × semantic + w_hier × hierarchy +
                  w_temp × temporal + w_prec × precision + source_bonus
        """
        w_sem = self.hp.get("score_weight_semantic", 0.40)
        w_hier = self.hp.get("score_weight_hierarchy", 0.30)
        w_temp = self.hp.get("score_weight_temporal", 0.15)
        w_prec = self.hp.get("score_weight_precision", 0.15)

        # 1. Semantic similarity (already computed)
        semantic_sim = candidate.get("semantic_similarity", 0.5)

        # 2. Hierarchical proximity (Poincaré distance + norm preference)
        hier_score = self._compute_hierarchy_score(candidate, gamma, tree_context, query_hyperbolic)

        # 3. Temporal relevance
        temp_score = self._compute_temporal_score(candidate, query_temporal)

        # 4. Precision weight
        precision_score = candidate.get("precision", 0.5)

        # 5. Source bonus: narrow results get small γ-based boost
        source_bonus = 0.05 if candidate.get("_source") == "narrow" else 0.0

        final = (
            w_sem * semantic_sim +
            w_hier * hier_score +
            w_temp * temp_score +
            w_prec * precision_score +
            source_bonus
        )

        return {
            "final_score": final,
            "semantic_score": semantic_sim,
            "hierarchy_score": hier_score,
            "temporal_score": temp_score,
            "precision_score": precision_score,
        }

    def _compute_hierarchy_score(
        self,
        candidate: Dict[str, Any],
        gamma: float,
        tree_context: Optional[str],
        query_hyperbolic: Optional[np.ndarray] = None,
    ) -> float:
        """
        Score based on Poincaré distance (uses all 32 dims) + norm preference.

        If query_hyperbolic is available (from nearest context centroid),
        uses actual poincare_distance for geometric scoring.
        Falls back to norm-based scoring otherwise.
        """
        hyp_coords = candidate.get("hyperbolic_coords")
        if hyp_coords is None:
            return 0.5

        # Primary: actual Poincaré distance (if query coords available)
        if query_hyperbolic is not None:
            d = poincare_distance(query_hyperbolic, hyp_coords)
            # Convert distance to score: closer = higher (temperature=2.0)
            distance_score = float(np.exp(-d / 2.0))
        else:
            distance_score = 0.5

        # Secondary: norm preference (gamma-dependent level bias)
        cand_norm = hierarchy_level(hyp_coords)
        target_norm = gamma * 0.85 + (1.0 - gamma) * 0.35
        norm_score = 1.0 - abs(cand_norm - target_norm)

        # Blend: 70% distance (if available), 30% norm preference
        if query_hyperbolic is not None:
            score = 0.7 * distance_score + 0.3 * norm_score
        else:
            score = norm_score

        # Tree bonus
        if tree_context:
            tree_ids = candidate.get("tree_ids", [])
            if isinstance(tree_ids, str):
                import json as _json
                try:
                    tree_ids = _json.loads(tree_ids)
                except Exception:
                    tree_ids = []
            if tree_context in tree_ids:
                score = min(score + 0.1, 1.0)

        return float(np.clip(score, 0.0, 1.0))

    def _compute_temporal_score(
        self,
        candidate: Dict[str, Any],
        query_temporal: np.ndarray,
    ) -> float:
        """
        Score based on temporal proximity + recency decay.
        Combines embedding similarity with access-based temporal weight.
        """
        cand_temporal = candidate.get("temporal_embedding")

        # Temporal embedding similarity
        if cand_temporal is not None:
            embed_sim = temporal_similarity(query_temporal, cand_temporal)
        else:
            embed_sim = 0.5

        # Recency: how recently was this node accessed or modified?
        last_modified = candidate.get("last_modified")
        if last_modified and isinstance(last_modified, str):
            try:
                lm = datetime.fromisoformat(last_modified)
                age_hours = (datetime.now() - lm).total_seconds() / 3600
                recency = np.exp(-age_hours / 720)  # 30-day halflife
            except ValueError:
                recency = 0.3
        else:
            recency = 0.3

        # Blend: 60% embedding similarity + 40% recency
        return 0.6 * embed_sim + 0.4 * recency
