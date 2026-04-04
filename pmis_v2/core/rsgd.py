"""
Riemannian Stochastic Gradient Descent (RSGD) for PMIS V2.

Learns hyperbolic embeddings from the parent-child hierarchy
so that poincare_distance(parent, child) is small and
cross-branch distances are large.

The only difference from regular SGD:
  riemannian_grad = ((1 - ||x||^2)^2 / 4) * euclidean_grad
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from core.poincare import poincare_distance, project_to_ball, exp_map_origin, log_map_origin

EPS = 1e-7


@dataclass
class RSGDResult:
    updated_embeddings: Dict[str, np.ndarray]
    loss_history: List[float]
    final_loss: float
    epochs_run: int
    nodes_updated: int
    edges_used: int
    converged: bool
    wall_time_seconds: float


class RSGDTrainer:
    """
    Learns hyperbolic positions from parent-child edges via Riemannian SGD.

    Loss per positive edge (child, parent):
      L_pos = d(parent, child)^2

    Loss per negative sample (parent, random_node):
      L_neg = max(0, margin - d(parent, negative))^2

    Riemannian correction:
      grad_R = ((1 - ||x||^2)^2 / 4) * grad_E
    """

    def __init__(
        self,
        dim: int = 32,
        learning_rate: float = 0.01,
        neg_samples: int = 10,
        margin: float = 1.0,
        max_norm: float = 0.95,
        norm_reg_weight: float = 0.1,
        cross_branch_neg_ratio: float = 0.5,
        eps_fd: float = 1e-5,
        hyperparams: Optional[Dict] = None,
    ):
        self.dim = dim
        self.lr = learning_rate
        self.neg_samples = neg_samples
        self.margin = margin
        self.max_norm = max_norm
        self.norm_reg_weight = norm_reg_weight
        self.cross_branch_neg_ratio = cross_branch_neg_ratio
        self.eps_fd = eps_fd
        self.hp = hyperparams or {}

    def train(
        self,
        embeddings: Dict[str, np.ndarray],
        edges: List[Tuple[str, str]],
        node_levels: Dict[str, str],
        epochs: int = 50,
        burn_in_epochs: int = 10,
        burn_in_lr: float = 0.1,
        convergence_tolerance: float = 1e-4,
        convergence_patience: int = 5,
    ) -> RSGDResult:
        """
        Full RSGD training on all edges.

        Args:
            embeddings: {node_id: np.ndarray} — will be modified in-place
            edges: [(child_id, parent_id)] — child_of relationships
            node_levels: {node_id: "SC"|"CTX"|"ANC"}
            epochs: total training epochs
            burn_in_epochs: initial epochs with higher LR
            burn_in_lr: learning rate during burn-in
        """
        t_start = time.time()
        all_ids = list(embeddings.keys())
        n_nodes = len(all_ids)

        if not edges or n_nodes < 2:
            return RSGDResult(
                updated_embeddings=embeddings, loss_history=[], final_loss=0,
                epochs_run=0, nodes_updated=0, edges_used=0, converged=True,
                wall_time_seconds=0,
            )

        # Build SC subtree map for cross-branch negative sampling
        sc_subtrees = self._build_sc_subtrees(edges, node_levels)

        # Detect burn-in condition
        unlearned_count = sum(1 for nid in all_ids if np.linalg.norm(embeddings[nid]) < 0.01)
        needs_burn_in = unlearned_count > n_nodes * 0.5

        loss_history = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Learning rate schedule
            if needs_burn_in and epoch < burn_in_epochs:
                lr = burn_in_lr
            else:
                lr = self.lr

            epoch_loss = self._train_epoch(
                embeddings, edges, all_ids, node_levels, sc_subtrees, lr
            )
            loss_history.append(epoch_loss)

            # Convergence check
            if epoch_loss < best_loss - convergence_tolerance:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= convergence_patience and epoch > burn_in_epochs:
                break

        wall_time = time.time() - t_start

        return RSGDResult(
            updated_embeddings=embeddings,
            loss_history=loss_history,
            final_loss=loss_history[-1] if loss_history else 0,
            epochs_run=len(loss_history),
            nodes_updated=n_nodes,
            edges_used=len(edges),
            converged=patience_counter >= convergence_patience,
            wall_time_seconds=wall_time,
        )

    def train_incremental(
        self,
        embeddings: Dict[str, np.ndarray],
        edges: List[Tuple[str, str]],
        node_levels: Dict[str, str],
        touched_ids: set,
        epochs: int = 10,
    ) -> RSGDResult:
        """
        Incremental RSGD: only update touched nodes + 1-hop neighbors.
        """
        # Expand touched set to include neighbors
        expanded = set(touched_ids)
        for child_id, parent_id in edges:
            if child_id in touched_ids or parent_id in touched_ids:
                expanded.add(child_id)
                expanded.add(parent_id)

        # Filter edges to only those involving expanded set
        relevant_edges = [
            (c, p) for c, p in edges
            if c in expanded or p in expanded
        ]

        if not relevant_edges:
            return RSGDResult(
                updated_embeddings=embeddings, loss_history=[], final_loss=0,
                epochs_run=0, nodes_updated=0, edges_used=0, converged=True,
                wall_time_seconds=0,
            )

        return self.train(
            embeddings, relevant_edges, node_levels,
            epochs=epochs, burn_in_epochs=0, burn_in_lr=self.lr,
        )

    def frechet_mean(self, points: List[np.ndarray], iters: int = 50, lr: float = 0.1) -> np.ndarray:
        """
        Compute the Frechet mean (geometric centroid) in hyperbolic space.
        Iterative: log_map to tangent space, average, exp_map back.
        """
        if not points:
            return np.zeros(self.dim, dtype=np.float32)
        if len(points) == 1:
            return points[0].copy()

        # Start from the first point
        mu = points[0].copy()

        for _ in range(iters):
            tangent_sum = np.zeros_like(mu)
            for p in points:
                # Log map from mu to p (approximate: use origin-based)
                diff = p - mu
                tangent_sum += diff

            update = lr * tangent_sum / len(points)
            mu = mu + update
            mu = project_to_ball(mu, max_norm=self.max_norm)

        return mu

    # ================================================================
    # INTERNAL
    # ================================================================

    def _train_epoch(
        self,
        embeddings: Dict[str, np.ndarray],
        edges: List[Tuple[str, str]],
        all_ids: List[str],
        node_levels: Dict[str, str],
        sc_subtrees: Dict[str, set],
        lr: float,
    ) -> float:
        """One epoch of RSGD."""
        total_loss = 0.0
        indices = list(range(len(edges)))
        np.random.shuffle(indices)

        for idx in indices:
            child_id, parent_id = edges[idx]

            if child_id not in embeddings or parent_id not in embeddings:
                continue

            u = embeddings[parent_id]
            v = embeddings[child_id]

            # Positive loss: d(parent, child)^2
            d_pos = poincare_distance(u, v)
            pos_loss = d_pos ** 2

            # Negative sampling
            neg_loss = 0.0
            negatives = self._sample_negatives(
                parent_id, child_id, all_ids, sc_subtrees, node_levels
            )

            for neg_id in negatives:
                if neg_id not in embeddings:
                    continue
                d_neg = poincare_distance(u, embeddings[neg_id])
                if d_neg < self.margin:
                    neg_loss += (self.margin - d_neg) ** 2

            # Compute numerical gradient for parent and child
            grad_u = self._numerical_gradient(
                embeddings, parent_id, child_id, negatives, is_parent=True
            )
            grad_v = self._numerical_gradient(
                embeddings, parent_id, child_id, negatives, is_parent=False
            )

            # Add norm regularization gradient
            grad_u += self._norm_reg_gradient(u, node_levels.get(parent_id, "CTX"))
            grad_v += self._norm_reg_gradient(v, node_levels.get(child_id, "ANC"))

            # Riemannian correction
            scale_u = ((1 - np.dot(u, u)) ** 2) / 4
            scale_v = ((1 - np.dot(v, v)) ** 2) / 4

            # Update
            embeddings[parent_id] = project_to_ball(
                u - lr * scale_u * grad_u, max_norm=self.max_norm
            )
            embeddings[child_id] = project_to_ball(
                v - lr * scale_v * grad_v, max_norm=self.max_norm
            )

            total_loss += pos_loss + neg_loss

        return total_loss

    def _numerical_gradient(
        self,
        embeddings: Dict[str, np.ndarray],
        parent_id: str,
        child_id: str,
        negatives: List[str],
        is_parent: bool,
    ) -> np.ndarray:
        """Compute gradient via finite differences (sparse: only incident edges)."""
        target_id = parent_id if is_parent else child_id
        point = embeddings[target_id]
        grad = np.zeros(self.dim, dtype=np.float64)
        h = self.eps_fd

        for d in range(self.dim):
            # Perturb +h
            point_p = point.copy()
            point_p[d] += h

            # Perturb -h
            point_m = point.copy()
            point_m[d] -= h

            # Compute loss for both perturbations
            loss_p = self._local_loss(embeddings, parent_id, child_id, negatives, target_id, point_p)
            loss_m = self._local_loss(embeddings, parent_id, child_id, negatives, target_id, point_m)

            grad[d] = (loss_p - loss_m) / (2 * h)

        return grad

    def _local_loss(
        self,
        embeddings: Dict[str, np.ndarray],
        parent_id: str,
        child_id: str,
        negatives: List[str],
        perturbed_id: str,
        perturbed_point: np.ndarray,
    ) -> float:
        """Compute loss with one point perturbed."""
        u = perturbed_point if perturbed_id == parent_id else embeddings[parent_id]
        v = perturbed_point if perturbed_id == child_id else embeddings[child_id]

        # Positive loss
        d_pos = poincare_distance(u, v)
        loss = d_pos ** 2

        # Negative loss (only for parent)
        if perturbed_id == parent_id:
            for neg_id in negatives:
                if neg_id in embeddings:
                    d_neg = poincare_distance(perturbed_point, embeddings[neg_id])
                    if d_neg < self.margin:
                        loss += (self.margin - d_neg) ** 2

        return loss

    def _norm_reg_gradient(self, point: np.ndarray, level: str) -> np.ndarray:
        """Regularization: keep norm within level-appropriate band."""
        norm = np.linalg.norm(point)
        ranges = {"SC": (0.05, 0.20), "CTX": (0.35, 0.60), "ANC": (0.70, 0.95)}
        lo, hi = ranges.get(level, (0.35, 0.60))
        center = (lo + hi) / 2
        half_width = (hi - lo) / 2

        if abs(norm - center) <= half_width:
            return np.zeros_like(point)

        # Push back toward center
        direction = point / (norm + EPS)
        penalty = self.norm_reg_weight * (norm - center)
        return penalty * direction

    def _sample_negatives(
        self,
        parent_id: str,
        child_id: str,
        all_ids: List[str],
        sc_subtrees: Dict[str, set],
        node_levels: Dict[str, str],
    ) -> List[str]:
        """Sample negative nodes: 50% cross-branch, 50% random."""
        negatives = []
        n_cross = int(self.neg_samples * self.cross_branch_neg_ratio)
        n_random = self.neg_samples - n_cross

        # Find which SC subtree the parent belongs to
        parent_sc = None
        for sc_id, subtree in sc_subtrees.items():
            if parent_id in subtree:
                parent_sc = sc_id
                break

        # Cross-branch negatives (from different SC)
        if parent_sc and n_cross > 0:
            other_nodes = []
            for sc_id, subtree in sc_subtrees.items():
                if sc_id != parent_sc:
                    other_nodes.extend(subtree)
            if other_nodes:
                sampled = np.random.choice(
                    other_nodes, size=min(n_cross, len(other_nodes)), replace=False
                )
                negatives.extend(sampled.tolist())

        # Random negatives
        for _ in range(n_random):
            ni = np.random.choice(all_ids)
            while ni == parent_id or ni == child_id:
                ni = np.random.choice(all_ids)
            negatives.append(ni)

        return negatives

    def _build_sc_subtrees(
        self,
        edges: List[Tuple[str, str]],
        node_levels: Dict[str, str],
    ) -> Dict[str, set]:
        """Build map: SC_id → set of all descendant node IDs."""
        # Build adjacency: parent → [children]
        children_of = {}
        for child_id, parent_id in edges:
            if parent_id not in children_of:
                children_of[parent_id] = []
            children_of[parent_id].append(child_id)

        sc_ids = [nid for nid, level in node_levels.items() if level == "SC"]
        subtrees = {}

        for sc_id in sc_ids:
            subtree = {sc_id}
            frontier = [sc_id]
            while frontier:
                current = frontier.pop()
                for child in children_of.get(current, []):
                    if child not in subtree:
                        subtree.add(child)
                        frontier.append(child)
            subtrees[sc_id] = subtree

        return subtrees
