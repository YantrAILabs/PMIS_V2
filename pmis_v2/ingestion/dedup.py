"""
Duplicate Detection for PMIS v2.

Distinguishes genuine duplicates (same memory, merge) from
similar-but-distinct memories (keep separate) using three signals:
  1. Semantic distance (cosine)
  2. Temporal proximity (created within N hours)
  3. Source overlap (Jaccard of source conversation IDs)
"""

import numpy as np
from typing import Dict, Any, Optional, List
from core.surprise import compute_raw_surprise


def is_same_memory(
    new_embedding: np.ndarray,
    candidate: Dict[str, Any],
    new_created_at_hours: float,
    candidate_created_hours: float,
    new_source_id: str,
    candidate_source_id: str,
    hyperparams: Dict[str, Any],
) -> str:
    """
    Determine if a new memory is a duplicate of an existing one.

    Returns:
        "duplicate"  — same memory, should merge
        "maybe"      — might be same, flag for review
        "distinct"   — different memory, keep separate
    """
    sem_thresh = hyperparams.get("dedup_semantic_threshold", 0.05)
    temp_thresh = hyperparams.get("dedup_temporal_threshold_hours", 2)

    # Semantic distance
    candidate_emb = candidate.get("euclidean_embedding")
    if candidate_emb is None:
        return "distinct"
    if isinstance(candidate_emb, list):
        candidate_emb = np.array(candidate_emb, dtype=np.float32)

    semantic_dist = compute_raw_surprise(new_embedding, candidate_emb)

    # Temporal distance
    temporal_dist = abs(new_created_at_hours - candidate_created_hours)

    # Source overlap
    source_match = (new_source_id == candidate_source_id)

    # Decision logic
    if semantic_dist < sem_thresh and source_match:
        return "duplicate"
    elif semantic_dist < sem_thresh * 3 and temporal_dist < temp_thresh:
        return "maybe"
    else:
        return "distinct"


def find_near_duplicates(
    new_embedding: np.ndarray,
    candidates: List[Dict[str, Any]],
    hyperparams: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Find potential duplicates from a list of candidate memories.
    Returns candidates tagged with duplicate status.
    """
    results = []
    for c in candidates:
        c_emb = c.get("euclidean_embedding")
        if c_emb is None:
            continue
        if isinstance(c_emb, list):
            c_emb = np.array(c_emb, dtype=np.float32)

        dist = compute_raw_surprise(new_embedding, c_emb)
        if dist < hyperparams.get("dedup_semantic_threshold", 0.05) * 3:
            results.append({**c, "dedup_distance": dist})

    return sorted(results, key=lambda x: x["dedup_distance"])
