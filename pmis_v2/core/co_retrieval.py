"""
Co-Retrieval Graph Builder for PMIS V2.

Builds edge lists from turn_retrieved_memories:
- Co-retrieval pairs: nodes retrieved together in the same turn
- Feedback edges: from the feedback table (Phase 2)

These feed into HGCN training as additional loss terms beyond structural edges.

Co-retrieval signal: 3,048 cross-branch pairs, 9,000 observations.
88.8% cross-branch (novel), 9.6% same-parent (redundant).
Negatives: nodes never co-retrieved with either member.
"""

import sqlite3
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict


def build_co_retrieval_edges(
    db_path: str,
    id_to_idx: Dict[str, int],
    min_count: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build co-retrieval edge list from turn_retrieved_memories.

    Returns:
        edges: (E, 2) int array — [node_a_idx, node_b_idx]
        weights: (E,) float array — log(1 + co_count) as edge weight
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all pairs from same turn
    cursor.execute("""
        SELECT t1.memory_node_id as a, t2.memory_node_id as b, COUNT(*) as cnt
        FROM turn_retrieved_memories t1
        JOIN turn_retrieved_memories t2
            ON t1.turn_id = t2.turn_id AND t1.memory_node_id < t2.memory_node_id
        GROUP BY t1.memory_node_id, t2.memory_node_id
        HAVING cnt >= ?
    """, (min_count,))

    edges = []
    weights = []

    for row in cursor.fetchall():
        a_id, b_id, count = row
        if a_id in id_to_idx and b_id in id_to_idx:
            edges.append([id_to_idx[a_id], id_to_idx[b_id]])
            weights.append(np.log1p(count))

    conn.close()

    if not edges:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)

    return np.array(edges, dtype=np.int64), np.array(weights, dtype=np.float32)


def build_feedback_edges(
    db_path: str,
    id_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feedback edges from the feedback table.

    For each positive feedback on a node, create edges to that node's
    parent and siblings (strengthen neighborhood).
    For negative feedback, create edges that push the node away.

    Returns:
        positive_edges: (E_pos, 2) — pairs to pull closer
        negative_edges: (E_neg, 2) — pairs to push apart
        strengths: (E_pos + E_neg,) — edge strengths
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all feedback with node's parent info
    cursor.execute("""
        SELECT f.node_id, f.polarity, f.strength,
               r.target_id as parent_id
        FROM feedback f
        LEFT JOIN relations r ON r.source_id = f.node_id AND r.relation_type = 'child_of'
    """)

    pos_edges = []
    neg_edges = []
    pos_strengths = []
    neg_strengths = []

    for row in cursor.fetchall():
        node_id = row["node_id"]
        parent_id = row["parent_id"]
        polarity = row["polarity"]
        strength = row["strength"] or 1.0

        if node_id not in id_to_idx:
            continue

        node_idx = id_to_idx[node_id]

        if polarity == "positive" and parent_id and parent_id in id_to_idx:
            # Positive: pull node closer to parent (strengthen hierarchy)
            pos_edges.append([node_idx, id_to_idx[parent_id]])
            pos_strengths.append(strength)

            # Also get siblings and pull them closer
            cursor2 = conn.cursor()
            cursor2.execute("""
                SELECT source_id FROM relations
                WHERE target_id = ? AND relation_type = 'child_of'
                AND source_id != ?
                LIMIT 5
            """, (parent_id, node_id))
            for sib in cursor2.fetchall():
                if sib[0] in id_to_idx:
                    pos_edges.append([node_idx, id_to_idx[sib[0]]])
                    pos_strengths.append(strength * 0.5)  # weaker for siblings

        elif polarity == "negative" and parent_id and parent_id in id_to_idx:
            # Negative: push node away from parent region
            neg_edges.append([node_idx, id_to_idx[parent_id]])
            neg_strengths.append(strength)

    conn.close()

    pos = np.array(pos_edges, dtype=np.int64) if pos_edges else np.zeros((0, 2), dtype=np.int64)
    neg = np.array(neg_edges, dtype=np.int64) if neg_edges else np.zeros((0, 2), dtype=np.int64)
    strengths = np.array(pos_strengths + neg_strengths, dtype=np.float32)

    return pos, neg, strengths


def sample_co_retrieval_negatives(
    co_edges: np.ndarray,
    n_nodes: int,
    n_neg_per_edge: int = 5,
) -> np.ndarray:
    """
    Sample negatives for co-retrieval: nodes never co-retrieved with either member.

    Returns: (E_neg, 2) — [anchor_idx, negative_idx]
    """
    # Build co-retrieval adjacency set for fast lookup
    co_neighbors: Dict[int, Set[int]] = defaultdict(set)
    for a, b in co_edges:
        co_neighbors[a].add(b)
        co_neighbors[b].add(a)

    neg_edges = []
    for a, b in co_edges:
        # Negative for pair (a, b): node never co-retrieved with a OR b
        excluded = co_neighbors[a] | co_neighbors[b] | {a, b}
        candidates = [i for i in range(n_nodes) if i not in excluded]

        if not candidates:
            continue

        n_sample = min(n_neg_per_edge, len(candidates))
        sampled = np.random.choice(candidates, size=n_sample, replace=False)
        for neg_idx in sampled:
            neg_edges.append([a, neg_idx])

    if not neg_edges:
        return np.zeros((0, 2), dtype=np.int64)

    return np.array(neg_edges, dtype=np.int64)
