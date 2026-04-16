"""
Cell Division — Recursive splitting of oversized Context nodes.

When a CTX has more children than `division_threshold` (default 25),
it is split into 2-3 sub-contexts via k-means clustering in embedding
space. The process recurses until all leaf clusters are below threshold.

Level naming: CTX-1, CTX-2, CTX-3... (numeric depth, no limit).

Called during consolidation, after BIRTH and before PRUNE.
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from core.memory_node import MemoryNode, MemoryLevel
from core.surprise import compute_raw_surprise
from core.temporal import temporal_encode, compute_era
from core.poincare import assign_hyperbolic_coords, ProjectionManager
from db.manager import DBManager


def find_best_k(embeddings: np.ndarray, k_range: List[int] = [2, 3]) -> Tuple[int, np.ndarray]:
    """
    Find best k for k-means by silhouette-like heuristic.
    Returns (best_k, labels).
    """
    if len(embeddings) < 4:
        # Too few points, just split in 2
        labels = np.zeros(len(embeddings), dtype=int)
        labels[len(embeddings) // 2:] = 1
        return 2, labels

    best_k = k_range[0]
    best_score = -1
    best_labels = None

    for k in k_range:
        if k >= len(embeddings):
            continue
        labels, centroids = _simple_kmeans(embeddings, k, max_iter=50)

        # Silhouette-like score: avg(inter-cluster dist) / avg(intra-cluster dist)
        intra = 0
        inter = 0
        intra_count = 0
        inter_count = 0

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                if labels[i] == labels[j]:
                    intra += dist
                    intra_count += 1
                else:
                    inter += dist
                    inter_count += 1

        avg_intra = intra / max(intra_count, 1)
        avg_inter = inter / max(inter_count, 1)
        score = (avg_inter - avg_intra) / max(avg_inter, 1e-8)

        # Also penalize very unbalanced splits
        counts = [np.sum(labels == l) for l in range(k)]
        min_count = min(counts)
        balance_penalty = 0 if min_count >= 3 else 0.5

        score -= balance_penalty

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_labels is None:
        best_labels = np.zeros(len(embeddings), dtype=int)
        best_labels[len(embeddings) // 2:] = 1
        best_k = 2

    return best_k, best_labels


def _simple_kmeans(embeddings: np.ndarray, k: int, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means clustering. Returns (labels, centroids)."""
    n = len(embeddings)
    # k-means++ initialization
    indices = [np.random.randint(n)]
    for _ in range(1, k):
        dists = np.array([
            min(np.linalg.norm(embeddings[i] - embeddings[j]) for j in indices)
            for i in range(n)
        ])
        dists = dists / (dists.sum() + 1e-10)
        indices.append(np.random.choice(n, p=dists))

    centroids = embeddings[indices].copy()

    for _ in range(max_iter):
        # Assign
        dists = np.array([[np.linalg.norm(e - c) for c in centroids] for e in embeddings])
        labels = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for ki in range(k):
            mask = labels == ki
            if mask.sum() > 0:
                new_centroids[ki] = embeddings[mask].mean(axis=0)
            else:
                new_centroids[ki] = centroids[ki]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels, centroids


class CellDivision:
    """
    Recursive cell division for oversized Context nodes.

    Usage:
        divider = CellDivision(db, hp, projection, summary_fn)
        actions = divider.run()
    """

    def __init__(
        self,
        db: DBManager,
        hyperparams: Dict[str, Any],
        projection: ProjectionManager,
        generate_summary_fn=None,
    ):
        self.db = db
        self.hp = hyperparams
        self.projection = projection
        self.generate_summary = generate_summary_fn or self._default_summary
        self.actions: List[Dict[str, Any]] = []
        self.nodes_created = 0
        self.nodes_reparented = 0

    def run(self) -> List[Dict[str, Any]]:
        """Find and split all oversized nodes. Returns action log."""
        threshold = self.hp.get("division_threshold", 25)
        self.actions = []
        self.nodes_created = 0
        self.nodes_reparented = 0

        # Find all nodes (CTX and SC) with too many children
        candidates = self._find_oversized_nodes(threshold)

        for node_id, node_content, node_level, child_count in candidates:
            self._recursive_divide(
                node_id=node_id,
                node_content=node_content,
                parent_level=node_level,
                depth=0,
                threshold=threshold,
            )

        return self.actions

    def _find_oversized_nodes(self, threshold: int) -> List[Tuple[str, str, str, int]]:
        """Find all nodes with children > threshold."""
        with self.db._connect() as conn:
            rows = conn.execute("""
                SELECT r.target_id, mn.content, mn.level, COUNT(r.source_id) as children
                FROM relations r
                JOIN memory_nodes mn ON mn.id = r.target_id
                WHERE r.relation_type = 'child_of'
                AND mn.is_deleted = 0
                GROUP BY r.target_id
                HAVING children > ?
                ORDER BY children DESC
            """, (threshold,)).fetchall()
            return [(r[0], r[1], r[2], r[3]) for r in rows]

    def _recursive_divide(
        self,
        node_id: str,
        node_content: str,
        parent_level: str,
        depth: int,
        threshold: int,
    ):
        """Recursively split a node until all children clusters are <= threshold."""
        # Get children
        children = self.db.get_children(node_id)
        if len(children) <= threshold:
            return

        # Load embeddings for children
        child_data = []
        for child in children:
            embs = self.db.get_embeddings(child["id"])
            euc = embs.get("euclidean")
            if euc is not None:
                child_data.append({"node": child, "embedding": euc})

        if len(child_data) < 4:
            return  # Too few to split meaningfully

        embeddings = np.array([cd["embedding"] for cd in child_data])

        # Find best k and cluster
        k_range = self.hp.get("division_k_range", [2, 3])
        best_k, labels = find_best_k(embeddings, k_range)

        # Determine level name for new intermediate nodes
        if parent_level == "SC":
            new_level = "CTX"
        elif parent_level == "CTX":
            new_level = "CTX-1"
        elif parent_level.startswith("CTX-"):
            try:
                current_depth = int(parent_level.split("-")[1])
                new_level = f"CTX-{current_depth + 1}"
            except (ValueError, IndexError):
                new_level = f"CTX-{depth + 1}"
        else:
            new_level = f"CTX-{depth + 1}"

        # Create sub-nodes for each cluster
        for cluster_label in range(best_k):
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_label]
            cluster_children = [child_data[i] for i in cluster_indices]

            if len(cluster_children) < self.hp.get("division_min_cluster_size", 3):
                continue  # Skip tiny clusters, leave them under parent

            # Generate summary
            contents = [cd["node"].get("content", "")[:200] for cd in cluster_children]
            summary = self.generate_summary(contents)

            # Compute centroid embedding
            cluster_embs = np.array([cd["embedding"] for cd in cluster_children])
            centroid = np.mean(cluster_embs, axis=0)

            # Assign hyperbolic coords with depth-based norm
            hyp_coords = assign_hyperbolic_coords(
                euclidean_embedding=centroid,
                level="CTX",  # Use CTX norms as base — HGCN will refine
                projection_manager=self.projection,
                hyperparams=self.hp,
            )

            temporal = temporal_encode(
                datetime.now(),
                self.hp.get("temporal_embedding_dim", 16),
            )
            era = compute_era(datetime.now(), self.hp.get("era_boundaries", {}))

            # Create the new intermediate node
            new_node = MemoryNode.create(
                content=summary,
                level=MemoryLevel.CONTEXT,  # DB stores as CTX; we track depth separately
                euclidean_embedding=centroid,
                hyperbolic_coords=hyp_coords,
                temporal_embedding=temporal,
                surprise=0.0,
                precision=0.3,
                era=era,
            )
            new_node.is_orphan = False
            new_node.is_tentative = False
            self.db.create_node(new_node)
            self.nodes_created += 1

            # Attach new node to parent
            # Determine tree_id from parent's existing tree
            tree_id = self._get_tree_id(node_id) or f"auto_{new_node.id[:8]}"
            self.db.attach_to_parent(new_node.id, node_id, tree_id)

            # Reparent children from parent to new sub-node
            for cd in cluster_children:
                child_id = cd["node"]["id"]
                self._reparent(child_id, from_parent=node_id, to_parent=new_node.id, tree_id=tree_id)
                self.nodes_reparented += 1

            # Log action
            action = {
                "action": "cell_division",
                "parent_id": node_id,
                "parent_content": node_content[:100],
                "new_node_id": new_node.id,
                "new_level": new_level,
                "summary": summary[:200],
                "children_moved": len(cluster_children),
                "depth": depth,
            }
            self.actions.append(action)

            # Recurse if the new cluster is still too big
            if len(cluster_children) > threshold:
                self._recursive_divide(
                    node_id=new_node.id,
                    node_content=summary,
                    parent_level=new_level,
                    depth=depth + 1,
                    threshold=threshold,
                )

    def _reparent(self, child_id: str, from_parent: str, to_parent: str, tree_id: str):
        """Move a child from one parent to another."""
        with self.db._connect() as conn:
            # Remove old child_of relation
            conn.execute("""
                DELETE FROM relations
                WHERE source_id = ? AND target_id = ? AND relation_type = 'child_of'
            """, (child_id, from_parent))

            # Create new child_of relation
            conn.execute("""
                INSERT OR IGNORE INTO relations (source_id, target_id, relation_type, tree_id, weight)
                VALUES (?, ?, 'child_of', ?, 0.85)
            """, (child_id, to_parent, tree_id))

            # Update parent_ids in memory_nodes
            import json
            row = conn.execute(
                "SELECT parent_ids FROM memory_nodes WHERE id = ?", (child_id,)
            ).fetchone()
            if row:
                try:
                    parents = json.loads(row[0]) if row[0] else []
                except (json.JSONDecodeError, TypeError):
                    parents = [row[0]] if row[0] and row[0] != "[]" else []

                # Replace from_parent with to_parent
                parents = [p for p in parents if p != from_parent]
                if to_parent not in parents:
                    parents.append(to_parent)

                conn.execute(
                    "UPDATE memory_nodes SET parent_ids = ? WHERE id = ?",
                    (json.dumps(parents), child_id)
                )

    def _get_tree_id(self, node_id: str) -> Optional[str]:
        """Get the tree_id for a node from its relations."""
        with self.db._connect() as conn:
            row = conn.execute("""
                SELECT tree_id FROM relations
                WHERE (source_id = ? OR target_id = ?) AND relation_type = 'child_of'
                AND tree_id != 'default'
                LIMIT 1
            """, (node_id, node_id)).fetchone()
            return row[0] if row else None

    @staticmethod
    def _default_summary(contents: List[str]) -> str:
        """Fallback summary without LLM."""
        if not contents:
            return "Unnamed cluster"
        # Take first 3 content fragments, extract key phrases
        fragments = [c[:60].strip().rstrip(".") for c in contents[:3]]
        return ". ".join(fragments) + "."
