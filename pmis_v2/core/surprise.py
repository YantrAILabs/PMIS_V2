"""
Surprise computation engine for PMIS v2.

Computes per-turn surprise using:
  effective_surprise = raw_surprise × cluster_precision

Where raw_surprise is cosine distance from nearest memory cluster,
and cluster_precision reflects how confident the system is about that territory.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class SurpriseResult:
    raw_surprise: float
    cluster_precision: float
    effective_surprise: float
    nearest_context_id: Optional[str]
    nearest_context_name: str
    nearest_distance: float
    is_orphan_territory: bool


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


def compute_cluster_precision(
    num_anchors: int,
    avg_recency_hours: float,
    internal_consistency: float,
    hyperparams: Dict[str, Any],
) -> float:
    """
    Precision of a Context cluster.
    High precision = system is confident about this territory.

    precision = w1 * anchor_factor + w2 * recency_factor + w3 * consistency
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
    return float(np.clip(precision, 0.05, 1.0))


def compute_effective_surprise(raw_surprise: float, cluster_precision: float) -> float:
    """
    effective = raw × precision

    High surprise + high precision = genuinely novel → EXPLORE
    High surprise + low precision  = noise           → IGNORE
    Low surprise  + high precision = confirmation    → EXPLOIT
    Low surprise  + low precision  = weak signal     → GATHER DATA
    """
    return raw_surprise * cluster_precision


def compute_full_surprise(
    query_embedding: np.ndarray,
    nearest_context: Optional[Dict[str, Any]],
    hyperparams: Dict[str, Any],
) -> SurpriseResult:
    """
    Full surprise computation for a single turn.
    Takes query embedding + nearest context info, returns SurpriseResult.
    """
    orphan_threshold = hyperparams.get("orphan_distance_threshold", 0.80)

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

    nearest_embedding = nearest_context.get("embedding")
    if nearest_embedding is None:
        raw = 1.0
    else:
        raw = compute_raw_surprise(query_embedding, np.array(nearest_embedding))

    precision = compute_cluster_precision(
        num_anchors=nearest_context.get("num_anchors", 1),
        avg_recency_hours=nearest_context.get("avg_recency_hours", 720),
        internal_consistency=nearest_context.get("internal_consistency", 0.5),
        hyperparams=hyperparams,
    )

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
