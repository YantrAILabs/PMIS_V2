"""
Surprise computation engine for PMIS v2.

Computes per-turn surprise using:
  effective_surprise = raw_surprise × cluster_precision

Phase 1 changes:
  - K=5 weighted nearest contexts (replaces single nearest)
  - Precision floor for established contexts
  - Precision component factors exposed for diagnostics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class SurpriseResult:
    raw_surprise: float
    cluster_precision: float
    effective_surprise: float
    nearest_context_id: Optional[str]
    nearest_context_name: str
    nearest_distance: float
    is_orphan_territory: bool
    # Phase 1: expose precision components for diagnostics
    precision_anchor_factor: float = 0.0
    precision_recency_factor: float = 0.0
    precision_consistency_factor: float = 0.0
    # Phase 1: K-nearest metadata
    second_nearest_distance: float = 0.0
    contexts_searched: int = 0


def compute_raw_surprise(
    query_embedding: np.ndarray,
    nearest_embedding: np.ndarray,
) -> float:
    """
    Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite.
    """
    dot = np.dot(query_embedding, nearest_embedding)
    norm_q = np.linalg.norm(query_embedding)
    norm_n = np.linalg.norm(nearest_embedding)
    if norm_q < 1e-8 or norm_n < 1e-8:
        return 1.0
    cosine_sim = dot / (norm_q * norm_n)
    return float(np.clip(1.0 - cosine_sim, 0.0, 2.0))


def compute_raw_surprise_k(
    query_embedding: np.ndarray,
    context_embeddings: List[np.ndarray],
    k: int = 5,
) -> Tuple[float, float]:
    """
    K-nearest weighted surprise. More robust than single-point.

    Returns (raw_surprise, confidence).
    Confidence replaces cluster_precision's role as noise filter:
      - Tight cluster of K neighbors = high confidence (real signal)
      - Spread out = low confidence (ambiguous territory)
    """
    if not context_embeddings:
        return 1.0, 0.1

    # Compute distances to all contexts
    distances = []
    for ctx_emb in context_embeddings:
        distances.append(compute_raw_surprise(query_embedding, ctx_emb))

    distances.sort()
    top_k = distances[:k]

    # Weighted average: nearest matters most
    weights = [1.0 / (2 ** i) for i in range(len(top_k))]
    raw_surprise = sum(w * d for w, d in zip(weights, top_k)) / sum(weights)

    # Spread measure: tight cluster = high confidence
    if len(top_k) > 1:
        spread = max(top_k) - min(top_k)
        confidence = 1.0 - min(spread / 0.5, 1.0)
    else:
        confidence = 0.5

    return float(raw_surprise), float(confidence)


def compute_cluster_precision(
    num_anchors: int,
    avg_recency_hours: float,
    internal_consistency: float,
    hyperparams: Dict[str, Any],
    access_count: int = 0,
) -> Tuple[float, float, float, float]:
    """
    Precision of a Context cluster.
    High precision = system is confident about this territory.

    Returns (precision, anchor_factor, recency_factor, consistency_factor)
    for diagnostics visibility.
    """
    # Anchor count (log scale, saturates at threshold)
    saturation = hyperparams.get("precision_anchor_saturation", 50)
    anchor_factor = min(np.log1p(num_anchors) / np.log1p(saturation), 1.0)

    # Recency (exponential decay)
    halflife = hyperparams.get("precision_recency_halflife", 720)
    recency_factor = np.exp(-avg_recency_hours / halflife)

    # Consistency (direct, already 0-1)
    consistency_factor = np.clip(internal_consistency, 0.0, 1.0)

    precision = (
        hyperparams.get("precision_weight_anchors", 0.4) * anchor_factor +
        hyperparams.get("precision_weight_recency", 0.35) * recency_factor +
        hyperparams.get("precision_weight_consistency", 0.25) * consistency_factor
    )

    # Phase 1: Precision floor for established contexts
    floor = hyperparams.get("precision_established_floor", 0.7)
    min_anchors = hyperparams.get("precision_established_min_anchors", 15)
    min_access = hyperparams.get("precision_established_min_access", 10)
    if num_anchors >= min_anchors and access_count >= min_access:
        precision = max(precision, floor)

    precision = float(np.clip(precision, 0.05, 1.0))
    return precision, float(anchor_factor), float(recency_factor), float(consistency_factor)


def compute_effective_surprise(raw_surprise: float, cluster_precision: float) -> float:
    """
    effective = raw x precision

    High surprise + high precision = genuinely novel -> EXPLORE
    High surprise + low precision  = noise           -> IGNORE
    Low surprise  + high precision = confirmation    -> EXPLOIT
    Low surprise  + low precision  = weak signal     -> GATHER DATA
    """
    return raw_surprise * cluster_precision


def compute_full_surprise(
    query_embedding: np.ndarray,
    nearest_context: Optional[Dict[str, Any]],
    hyperparams: Dict[str, Any],
    all_context_embeddings: Optional[List[np.ndarray]] = None,
) -> SurpriseResult:
    """
    Full surprise computation for a single turn.

    Phase 1: If all_context_embeddings is provided, uses K-nearest
    weighted surprise instead of single nearest.
    """
    orphan_threshold = hyperparams.get("orphan_distance_threshold", 0.80)
    k = hyperparams.get("surprise_k_nearest", 5)

    if nearest_context is None:
        return SurpriseResult(
            raw_surprise=1.0,
            cluster_precision=0.1,
            effective_surprise=0.1,
            nearest_context_id=None,
            nearest_context_name="NONE",
            nearest_distance=1.0,
            is_orphan_territory=True,
        )

    # Phase 1: K-nearest surprise if context embeddings provided
    if all_context_embeddings and len(all_context_embeddings) >= 2:
        raw, k_confidence = compute_raw_surprise_k(
            query_embedding, all_context_embeddings, k=k
        )
        contexts_searched = len(all_context_embeddings)

        # Compute distances for second-nearest metadata
        nearest_embedding = nearest_context.get("embedding")
        if nearest_embedding is not None:
            all_dists = sorted([
                compute_raw_surprise(query_embedding, e)
                for e in all_context_embeddings
            ])
            second_dist = all_dists[1] if len(all_dists) > 1 else 0.0
        else:
            second_dist = 0.0
    else:
        # Fallback: single nearest (original behavior)
        nearest_embedding = nearest_context.get("embedding")
        if nearest_embedding is None:
            raw = 1.0
        else:
            raw = compute_raw_surprise(query_embedding, np.array(nearest_embedding))
        k_confidence = None
        contexts_searched = 1
        second_dist = 0.0

    # Compute precision from context stats
    precision, anchor_f, recency_f, consistency_f = compute_cluster_precision(
        num_anchors=nearest_context.get("num_anchors", 1),
        avg_recency_hours=nearest_context.get("avg_recency_hours", 720),
        internal_consistency=nearest_context.get("internal_consistency", 0.5),
        hyperparams=hyperparams,
        access_count=nearest_context.get("access_count", 0),
    )

    # If K-nearest confidence is available, blend with cluster precision
    if k_confidence is not None:
        # K-nearest confidence modulates precision:
        # tight K-nearest cluster + high precision = very confident
        # spread K-nearest + high precision = precision might be misleading
        precision = precision * (0.5 + 0.5 * k_confidence)
        precision = float(np.clip(precision, 0.05, 1.0))

    effective = compute_effective_surprise(raw, precision)
    is_orphan = raw > orphan_threshold

    return SurpriseResult(
        raw_surprise=raw,
        cluster_precision=precision,
        effective_surprise=effective,
        nearest_context_id=nearest_context.get("id"),
        nearest_context_name=nearest_context.get("name", "unknown"),
        nearest_distance=raw,
        is_orphan_territory=is_orphan,
        precision_anchor_factor=anchor_f,
        precision_recency_factor=recency_f,
        precision_consistency_factor=consistency_f,
        second_nearest_distance=second_dist,
        contexts_searched=contexts_searched,
    )


def detect_staleness(
    recent_surprises: List[float],
    hyperparams: Dict[str, Any],
) -> bool:
    """
    Boredom detection.
    Returns True if surprise has been consistently low for too long.
    """
    window = hyperparams.get("staleness_window", 10)
    threshold = hyperparams.get("staleness_threshold", 0.15)

    if len(recent_surprises) < window:
        return False

    avg = np.mean(recent_surprises[-window:])
    return float(avg) < threshold
