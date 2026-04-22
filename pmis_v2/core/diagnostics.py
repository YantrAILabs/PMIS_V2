"""
Turn Diagnostics — End-to-end instrumentation for every PMIS pipeline touchpoint.

Captures a complete diagnostic row per turn, covering:
  1. Embedding stage (model, dim, latency)
  2. Nearest context stage (distances, gap analysis)
  3. Surprise stage (raw, precision components, effective)
  4. Gamma stage (inputs, sigmoid params, output, mode)
  5. Tree resolution stage (method, match quality)
  6. Retrieval stage (narrow/broad k, candidates, latency)
  7. Scoring breakdown (top-1 detail)
  8. Scoring spread (within-turn variance)
  9. Storage decision (action, reason, dedup)
  10. Epistemic & predictive counts
  11. Session state (turn count, accumulators)
  12. Poincaré health (norm distribution, discriminative power)

Usage:
    diag = DiagnosticCapture()
    diag.mark_embedding(model="nomic-embed-text", dim=768)
    # ... pipeline runs ...
    diag.mark_retrieval_results(retrieved_memories)
    row = diag.finalize(conversation_id, turn_number)
    db.log_diagnostics(row)
"""

import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class DiagnosticRow:
    """Complete diagnostic snapshot for one turn."""
    conversation_id: str = ""
    turn_number: int = 0
    timestamp: str = ""

    # 1. Embedding stage
    embed_model: str = ""
    embed_dim: int = 0
    embed_latency_ms: float = 0.0

    # 2. Nearest context stage
    nearest_ctx_id: str = ""
    nearest_ctx_name: str = ""
    nearest_ctx_distance: float = 0.0
    second_ctx_distance: float = 0.0
    contexts_searched: int = 0

    # 3. Surprise stage
    raw_surprise: float = 0.0
    cluster_precision: float = 0.0
    precision_anchor_factor: float = 0.0
    precision_recency_factor: float = 0.0
    precision_consistency_factor: float = 0.0
    effective_surprise: float = 0.0
    is_orphan_territory: int = 0

    # 4. Gamma stage
    gamma_input_surprise: float = 0.0
    gamma_temperature: float = 0.0
    gamma_bias: float = 0.0
    gamma_staleness_penalty: float = 0.0
    gamma_session_boost: float = 0.0
    gamma_raw: float = 0.0
    gamma_final: float = 0.0
    gamma_mode: str = ""
    gamma_override_active: int = 0

    # 5. Tree resolution stage
    tree_resolution_method: str = ""
    tree_id: str = ""
    tree_name: str = ""
    tree_root_distance: float = 0.0

    # 6. Retrieval stage
    narrow_k: int = 0
    narrow_threshold: float = 0.0
    narrow_candidates_found: int = 0
    broad_k: int = 0
    broad_threshold: float = 0.0
    broad_candidates_found: int = 0
    total_candidates_scored: int = 0
    retrieval_latency_ms: float = 0.0

    # 7. Scoring breakdown (top-1)
    top1_node_id: str = ""
    top1_final_score: float = 0.0
    top1_semantic_score: float = 0.0
    top1_hierarchy_score: float = 0.0
    top1_temporal_score: float = 0.0
    top1_precision_score: float = 0.0
    top1_node_level: str = ""
    top1_source: str = ""

    # 8. Scoring spread
    avg_semantic: float = 0.0
    avg_hierarchy: float = 0.0
    avg_temporal: float = 0.0
    avg_precision: float = 0.0
    std_semantic: float = 0.0
    std_hierarchy: float = 0.0
    std_temporal: float = 0.0
    std_precision: float = 0.0
    score_range: float = 0.0

    # 9. Storage decision
    storage_action: str = ""
    storage_reason: str = ""
    stored_node_id: str = ""
    stored_as_orphan: int = 0
    stored_as_tentative: int = 0
    dedup_blocked: int = 0

    # 10. Epistemic & predictive
    epistemic_questions_count: int = 0
    top_epistemic_info_gain: float = 0.0
    predictive_memories_count: int = 0

    # 11. Session state
    session_turn_count: int = 0
    session_avg_gamma: float = 0.0
    session_precision_accumulator: float = 0.0
    session_is_stale: int = 0

    # 12. Poincaré health
    avg_poincare_norm: float = 0.0
    poincare_norm_spread: float = 0.0
    hierarchy_score_discriminative: float = 0.0


class DiagnosticCapture:
    """
    Builder pattern for capturing diagnostics incrementally during pipeline execution.

    Usage:
        cap = DiagnosticCapture(conversation_id, turn_number)
        cap.mark_embedding_start()
        # ... embedding happens ...
        cap.mark_embedding_done(model, dim)
        cap.mark_surprise(surprise_result)
        cap.mark_gamma(gamma_result, hp)
        cap.mark_retrieval_results(retrieved)
        cap.mark_storage(action, reason, node_id)
        row = cap.finalize(session)
    """

    def __init__(self, conversation_id: str = "", turn_number: int = 0):
        self.row = DiagnosticRow(
            conversation_id=conversation_id,
            turn_number=turn_number,
            timestamp=datetime.now().isoformat(),
        )
        self._embed_start: float = 0.0
        self._retrieval_start: float = 0.0

    def mark_embedding_start(self):
        self._embed_start = time.time()

    def mark_embedding_done(self, model: str, dim: int):
        self.row.embed_model = model
        self.row.embed_dim = dim
        if self._embed_start > 0:
            self.row.embed_latency_ms = (time.time() - self._embed_start) * 1000

    def mark_nearest_context(
        self,
        ctx_id: str,
        ctx_name: str,
        distance: float,
        second_distance: float = 0.0,
        contexts_searched: int = 0,
    ):
        self.row.nearest_ctx_id = ctx_id or ""
        self.row.nearest_ctx_name = ctx_name or ""
        self.row.nearest_ctx_distance = distance
        self.row.second_ctx_distance = second_distance
        self.row.contexts_searched = contexts_searched

    def mark_surprise(
        self,
        raw_surprise: float,
        cluster_precision: float,
        effective_surprise: float,
        is_orphan_territory: bool = False,
        anchor_factor: float = 0.0,
        recency_factor: float = 0.0,
        consistency_factor: float = 0.0,
    ):
        self.row.raw_surprise = raw_surprise
        self.row.cluster_precision = cluster_precision
        self.row.effective_surprise = effective_surprise
        self.row.is_orphan_territory = int(is_orphan_territory)
        self.row.precision_anchor_factor = anchor_factor
        self.row.precision_recency_factor = recency_factor
        self.row.precision_consistency_factor = consistency_factor

    def mark_gamma(
        self,
        gamma_final: float,
        mode: str,
        input_surprise: float = 0.0,
        temperature: float = 0.0,
        bias: float = 0.0,
        staleness_penalty: float = 0.0,
        session_boost: float = 0.0,
        gamma_raw: float = 0.0,
        override_active: bool = False,
    ):
        self.row.gamma_final = gamma_final
        self.row.gamma_mode = mode
        self.row.gamma_input_surprise = input_surprise
        self.row.gamma_temperature = temperature
        self.row.gamma_bias = bias
        self.row.gamma_staleness_penalty = staleness_penalty
        self.row.gamma_session_boost = session_boost
        self.row.gamma_raw = gamma_raw
        self.row.gamma_override_active = int(override_active)

    def mark_tree_resolution(
        self,
        method: str = "",
        tree_id: str = "",
        tree_name: str = "",
        root_distance: float = 0.0,
    ):
        self.row.tree_resolution_method = method or ""
        self.row.tree_id = tree_id or ""
        self.row.tree_name = tree_name or ""
        self.row.tree_root_distance = root_distance

    def mark_retrieval_start(self):
        self._retrieval_start = time.time()

    def mark_retrieval_params(
        self,
        narrow_k: int,
        narrow_threshold: float,
        broad_k: int,
        broad_threshold: float,
    ):
        self.row.narrow_k = narrow_k
        self.row.narrow_threshold = narrow_threshold
        self.row.broad_k = broad_k
        self.row.broad_threshold = broad_threshold

    def mark_retrieval_results(self, retrieved: List[Dict[str, Any]]):
        """Compute all retrieval diagnostics from the scored result list."""
        if self._retrieval_start > 0:
            self.row.retrieval_latency_ms = (time.time() - self._retrieval_start) * 1000

        if not retrieved:
            return

        # Count narrow vs broad
        self.row.narrow_candidates_found = sum(
            1 for r in retrieved if r.get("_source") == "narrow"
        )
        self.row.broad_candidates_found = sum(
            1 for r in retrieved if r.get("_source") == "broad"
        )
        self.row.total_candidates_scored = len(retrieved)

        # Top-1 detail
        top = retrieved[0]
        self.row.top1_node_id = top.get("id", "")
        self.row.top1_final_score = top.get("final_score", 0.0)
        self.row.top1_semantic_score = top.get("semantic_score", 0.0)
        self.row.top1_hierarchy_score = top.get("hierarchy_score", 0.0)
        self.row.top1_temporal_score = top.get("temporal_score", 0.0)
        self.row.top1_precision_score = top.get("precision_score", 0.0)
        self.row.top1_node_level = top.get("level", "")
        self.row.top1_source = top.get("_source", "")

        # Scoring spread across all results
        sem = [r.get("semantic_score", 0) for r in retrieved]
        hier = [r.get("hierarchy_score", 0) for r in retrieved]
        temp = [r.get("temporal_score", 0) for r in retrieved]
        prec = [r.get("precision_score", 0) for r in retrieved]
        final = [r.get("final_score", 0) for r in retrieved]

        self.row.avg_semantic = float(np.mean(sem)) if sem else 0.0
        self.row.avg_hierarchy = float(np.mean(hier)) if hier else 0.0
        self.row.avg_temporal = float(np.mean(temp)) if temp else 0.0
        self.row.avg_precision = float(np.mean(prec)) if prec else 0.0
        self.row.std_semantic = float(np.std(sem)) if len(sem) > 1 else 0.0
        self.row.std_hierarchy = float(np.std(hier)) if len(hier) > 1 else 0.0
        self.row.std_temporal = float(np.std(temp)) if len(temp) > 1 else 0.0
        self.row.std_precision = float(np.std(prec)) if len(prec) > 1 else 0.0
        self.row.score_range = (max(final) - min(final)) if final else 0.0

        # Hierarchy discriminative power: std/mean
        mean_hier = self.row.avg_hierarchy
        self.row.hierarchy_score_discriminative = (
            self.row.std_hierarchy / mean_hier if mean_hier > 0.01 else 0.0
        )

        # Poincaré health — check norm distribution of hierarchy scores
        # Higher norms = closer to boundary = more specific
        # Spread in norms = hierarchy is differentiating levels
        self.row.avg_poincare_norm = mean_hier  # proxy: hierarchy score correlates with norm
        self.row.poincare_norm_spread = self.row.std_hierarchy

    def mark_storage(
        self,
        action: str,
        reason: str = "",
        node_id: str = "",
        is_orphan: bool = False,
        is_tentative: bool = False,
        dedup_blocked: bool = False,
    ):
        self.row.storage_action = action
        self.row.storage_reason = reason
        self.row.stored_node_id = node_id or ""
        self.row.stored_as_orphan = int(is_orphan)
        self.row.stored_as_tentative = int(is_tentative)
        self.row.dedup_blocked = int(dedup_blocked)

    def mark_epistemic(self, questions: List[Dict]):
        self.row.epistemic_questions_count = len(questions) if questions else 0
        if questions:
            gains = [q.get("information_gain", 0) for q in questions if q.get("information_gain")]
            self.row.top_epistemic_info_gain = max(gains) if gains else 0.0

    def mark_predictive(self, memories: List[Dict]):
        self.row.predictive_memories_count = len(memories) if memories else 0

    def finalize(self, session=None) -> DiagnosticRow:
        """Capture session state and return the complete row."""
        if session:
            self.row.session_turn_count = getattr(session, "turn_counter", 0)
            self.row.session_avg_gamma = getattr(session, "avg_gamma", 0.0)
            self.row.session_precision_accumulator = getattr(
                session, "precision_accumulator", 0.0
            )
            self.row.session_is_stale = int(getattr(session, "is_stale", False))

        return self.row


# SQL column list for INSERT (matches DiagnosticRow field order)
DIAGNOSTIC_COLUMNS = [
    "conversation_id", "turn_number", "timestamp",
    "embed_model", "embed_dim", "embed_latency_ms",
    "nearest_ctx_id", "nearest_ctx_name", "nearest_ctx_distance",
    "second_ctx_distance", "contexts_searched",
    "raw_surprise", "cluster_precision",
    "precision_anchor_factor", "precision_recency_factor", "precision_consistency_factor",
    "effective_surprise", "is_orphan_territory",
    "gamma_input_surprise", "gamma_temperature", "gamma_bias",
    "gamma_staleness_penalty", "gamma_session_boost",
    "gamma_raw", "gamma_final", "gamma_mode", "gamma_override_active",
    "tree_resolution_method", "tree_id", "tree_name", "tree_root_distance",
    "narrow_k", "narrow_threshold", "narrow_candidates_found",
    "broad_k", "broad_threshold", "broad_candidates_found",
    "total_candidates_scored", "retrieval_latency_ms",
    "top1_node_id", "top1_final_score",
    "top1_semantic_score", "top1_hierarchy_score",
    "top1_temporal_score", "top1_precision_score",
    "top1_node_level", "top1_source",
    "avg_semantic", "avg_hierarchy", "avg_temporal", "avg_precision",
    "std_semantic", "std_hierarchy", "std_temporal", "std_precision",
    "score_range",
    "storage_action", "storage_reason", "stored_node_id",
    "stored_as_orphan", "stored_as_tentative", "dedup_blocked",
    "epistemic_questions_count", "top_epistemic_info_gain",
    "predictive_memories_count",
    "session_turn_count", "session_avg_gamma",
    "session_precision_accumulator", "session_is_stale",
    "avg_poincare_norm", "poincare_norm_spread", "hierarchy_score_discriminative",
]


def diagnostic_row_to_tuple(row: DiagnosticRow) -> tuple:
    """Convert DiagnosticRow to tuple matching DIAGNOSTIC_COLUMNS order."""
    d = asdict(row)
    return tuple(d[col] for col in DIAGNOSTIC_COLUMNS)
