"""
Ingestion Pipeline for PMIS v2.

Full per-turn flow:
  raw text → embed → check dedup → compute surprise → compute gamma →
  storage decision → create/update/skip → link sequence → return result
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

from core.memory_node import MemoryNode, MemoryLevel
from core.surprise import compute_full_surprise, detect_staleness
from core.gamma import compute_gamma, GammaResult
from core.temporal import compute_era
from core.session_state import SessionState
from core import config

from ingestion.embedder import Embedder
from ingestion.surprise_gate import decide_storage, StorageDecision
from ingestion.dedup import find_near_duplicates
from ingestion.sequence_linker import link_sequence

from db.manager import DBManager


class IngestionResult:
    def __init__(self):
        self.node_id: Optional[str] = None
        self.action: str = "skip"
        self.gamma_result: Optional[GammaResult] = None
        self.surprise_result = None
        self.storage_decision: Optional[Dict[str, Any]] = None
        self.is_stale: bool = False
        self.reason: str = ""


class IngestionPipeline:
    def __init__(self, db: DBManager, embedder: Embedder,
                 hyperparams: Optional[Dict[str, Any]] = None):
        self.db = db
        self.embedder = embedder
        self.hp = hyperparams or config.get_all()

    def process_turn(
        self,
        content: str,
        session: SessionState,
        role: str = "user",
        conversation_id: str = "",
    ) -> IngestionResult:
        """
        Process a single conversation turn through the full pipeline.
        """
        result = IngestionResult()
        now = datetime.now()

        # --- 1. EMBED ---
        embeddings = self.embedder.generate_triple_embedding(
            text=content,
            level="ANC",  # All new turns start as Anchor candidates
            timestamp=now,
        )
        query_embedding = embeddings["euclidean"]

        # --- 2. FIND NEAREST CONTEXT (+ all context embeddings for K-nearest) ---
        nearest_context, all_ctx_embeddings = self._find_nearest_context_k(query_embedding)

        # --- 3. COMPUTE SURPRISE (K-nearest) ---
        surprise_result = compute_full_surprise(
            query_embedding=query_embedding,
            nearest_context=nearest_context,
            hyperparams=self.hp,
            all_context_embeddings=all_ctx_embeddings,
        )
        result.surprise_result = surprise_result

        # --- 4. DETECT STALENESS ---
        is_stale = detect_staleness(
            recent_surprises=session.recent_surprises,
            hyperparams=self.hp,
        )
        result.is_stale = is_stale

        # --- 5. COMPUTE GAMMA (with session boost) ---
        # Session precision accumulator for multi-turn gamma boost
        boost_alpha = self.hp.get("gamma_session_boost_alpha", 0.3)
        boost_threshold = self.hp.get("gamma_session_boost_threshold", 0.6)
        boost_factor = self.hp.get("gamma_session_boost_factor", 0.15)
        boost_min_turns = self.hp.get("gamma_session_boost_min_turns", 3)

        if not hasattr(session, "precision_accumulator"):
            session.precision_accumulator = 0.0
        session.precision_accumulator = (
            (1 - boost_alpha) * session.precision_accumulator +
            boost_alpha * surprise_result.cluster_precision
        )

        session_boost = 0.0
        if (session.turn_counter >= boost_min_turns and
                session.precision_accumulator > boost_threshold):
            session_boost = boost_factor * session.precision_accumulator

        gamma_result = compute_gamma(
            effective_surprise=surprise_result.effective_surprise,
            staleness=is_stale,
            hyperparams=self.hp,
            session_boost=session_boost,
        )

        # Apply gamma override if set by /memory explore or /memory exploit
        if session.gamma_override is not None and session.gamma_override_turns_remaining > 0:
            override_gamma = session.gamma_override
            if override_gamma > 0.7:
                mode = "ASSOCIATIVE"
            elif override_gamma < 0.4:
                mode = "PREDICTIVE"
            else:
                mode = "BALANCED"
            gamma_result = GammaResult(
                gamma=override_gamma,
                mode_label=mode,
                retrieval_narrow_weight=override_gamma,
                retrieval_broad_weight=1.0 - override_gamma,
                confidence_instruction=gamma_result.confidence_instruction,
                storage_instruction=gamma_result.storage_instruction,
            )
            session.gamma_override_turns_remaining -= 1
            if session.gamma_override_turns_remaining <= 0:
                session.gamma_override = None

        result.gamma_result = gamma_result

        # --- 6. STORAGE DECISION ---
        storage = decide_storage(
            gamma_result=gamma_result,
            surprise_result=surprise_result,
            is_stale=is_stale,
            content_length=len(content),
            hyperparams=self.hp,
        )
        result.storage_decision = storage
        result.action = storage["action"]
        result.reason = storage.get("reason", "")

        # --- 7. EXECUTE STORAGE ---
        node_id = None

        if storage["action"] == StorageDecision.CREATE:
            # Check dedup first
            if not self._is_duplicate(query_embedding, content):
                # Determine parent coords for hyperbolic placement
                parent_coords = None
                parent_id = storage.get("target_context_id")
                if parent_id:
                    parent_embs = self.db.get_embeddings(parent_id)
                    if parent_embs.get("hyperbolic") is not None:
                        parent_coords = parent_embs["hyperbolic"]

                # Zero vector placeholder — HGCN assigns real Poincare coords nightly
                hyp_dim = self.hp.get("poincare_dimensions", 16)
                embeddings["hyperbolic"] = np.zeros(hyp_dim, dtype=np.float32)

                # Compute era
                era = compute_era(now, self.hp.get("era_boundaries", {}))

                # Create node
                node = MemoryNode.create(
                    content=content,
                    level=MemoryLevel.ANCHOR,
                    euclidean_embedding=embeddings["euclidean"],
                    hyperbolic_coords=embeddings["hyperbolic"],
                    temporal_embedding=embeddings["temporal"],
                    source_conversation_id=conversation_id,
                    surprise=surprise_result.effective_surprise,
                    precision=surprise_result.cluster_precision,
                    era=era,
                )
                node.is_orphan = storage.get("is_orphan", False)
                node.is_tentative = storage.get("is_tentative", False)

                # Persist
                self.db.create_node(node)
                node_id = node.id

                # Attach to parent if specified
                if parent_id and not node.is_orphan:
                    tree_id = session.active_tree_id or "default"
                    self.db.attach_to_parent(node.id, parent_id, tree_id)

                # Link sequence
                prev_id = session.last_stored_node_id
                link_sequence(self.db, node.id, prev_id,
                              tree_id=session.active_tree_id or "default")

        elif storage["action"] == StorageDecision.UPDATE:
            target_id = storage.get("target_context_id")
            if target_id:
                self.db.update_node_access(
                    target_id,
                    query_text=content[:500],
                    gamma=gamma_result.gamma,
                    surprise=surprise_result.effective_surprise,
                )

        # --- 8. RECORD IN SESSION ---
        prev_node = session.record_turn(
            role=role,
            content=content,
            embedding=query_embedding,
            node_id=node_id,
            gamma=gamma_result.gamma,
            effective_surprise=surprise_result.effective_surprise,
            mode=gamma_result.mode_label,
        )

        # Update active context in session
        if surprise_result.nearest_context_id:
            session.set_active_context(surprise_result.nearest_context_id)

        result.node_id = node_id
        return result

    def _find_nearest_context(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Find the nearest Context node to the query (legacy, single-point)."""
        nearest, _ = self._find_nearest_context_k(query_embedding)
        return nearest

    def _find_nearest_context_k(
        self, query_embedding: np.ndarray
    ) -> tuple:
        """
        Find nearest Context + collect top-K context embeddings for K-nearest
        surprise. Returns (nearest_context_dict, list_of_top_k_embeddings).

        Fast path: ChromaDB ANN with level='CTX' filter — pulls only the top-K
        contexts (K = surprise_k_nearest) instead of every CTX node's BLOB.
        Fallback: linear scan over all CTX nodes.
        """
        from core.surprise import compute_raw_surprise

        k_nearest = int(self.hp.get("surprise_k_nearest", 5))

        # Fast path: ANN with level filter
        if self.db.has_ann_index:
            ann_results = self.db.ann_query(
                query_embedding,
                n_results=max(k_nearest, 1),
                level_filter="CTX",
            )
            if ann_results:
                top_embeddings: List[np.ndarray] = []
                best_ctx = None
                for i, r in enumerate(ann_results):
                    embs = self.db.get_embeddings(r["id"])
                    ctx_emb = embs.get("euclidean")
                    if ctx_emb is None:
                        continue
                    top_embeddings.append(ctx_emb)
                    if i == 0:  # ANN is sorted; first is nearest
                        ctx_node = self.db.get_node(r["id"])
                        if ctx_node:
                            stats = self.db.get_context_stats(r["id"])
                            best_ctx = {
                                **ctx_node,
                                **stats,
                                "embedding": ctx_emb,
                                "name": ctx_node.get("content", "")[:100],
                            }
                if best_ctx is not None:
                    return best_ctx, top_embeddings

        # Fallback: linear scan
        contexts = self.db.get_nodes_by_level("CTX")
        if not contexts:
            return None, []

        best_ctx = None
        best_dist = float("inf")
        all_embeddings = []

        for ctx in contexts:
            embs = self.db.get_embeddings(ctx["id"])
            ctx_emb = embs.get("euclidean")
            if ctx_emb is None:
                continue

            all_embeddings.append(ctx_emb)
            dist = compute_raw_surprise(query_embedding, ctx_emb)

            if dist < best_dist:
                best_dist = dist
                stats = self.db.get_context_stats(ctx["id"])
                best_ctx = {
                    **ctx,
                    **stats,
                    "embedding": ctx_emb,
                    "name": ctx.get("content", "")[:100],
                }

        return best_ctx, all_embeddings

    def _is_duplicate(self, embedding: np.ndarray, content: str) -> bool:
        """Quick dedup check against recent nodes."""
        node_id = MemoryNode.generate_id(content)
        existing = self.db.get_node(node_id)
        return existing is not None
