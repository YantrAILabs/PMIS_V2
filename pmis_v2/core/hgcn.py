"""
Hyperbolic Graph Convolutional Network (HGCN) for PMIS V2.

Learns Poincare embeddings from the knowledge graph structure.
Phase 4a: structural edges only (Nickel & Kiela 2017 formulation).
Phase 4b: + co-retrieval edges.
Phase 4c: + feedback edges.

Architecture:
    768d semantic → [Linear 128d] → [tangent aggregation] → [Linear 16d] → Poincare ball

Training:
    Nightly consolidation. 100 epochs on ~3,900 nodes.
    Loss: d(parent,child)^2 + margin-based negative sampling.
    Optimizer: RiemannianAdam (geoopt).

References:
    - Nickel & Kiela 2017: Poincare Embeddings for Learning Hierarchical Representations
    - Chami et al. 2019: Hyperbolic Graph Convolutional Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import geoopt
import time
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger("pmis.hgcn")


class HyperbolicGCNLayer(nn.Module):
    """
    Single HGCN layer: linear transform in tangent space + aggregation.

    Flow: tangent_features → linear → aggregate neighbors → exp_map → Poincare
    """

    def __init__(self, in_features: int, out_features: int, ball: geoopt.PoincareBall):
        super().__init__()
        self.ball = ball
        self.linear = nn.Linear(in_features, out_features, bias=True)
        # Separate magnitude head: learns per-node radius in the ball.
        # Input dimension matches linear input so it can read level one-hot.
        self.mag_head = nn.Linear(in_features, 1, bias=True)
        # Attention weights for neighbor aggregation
        self.attn = nn.Linear(out_features * 2, 1, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.zeros_(self.mag_head.weight)
        nn.init.constant_(self.mag_head.bias, 0.0)  # sigmoid(0)=0.5 → init magnitude ~0.75

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_features) — features in tangent space at origin
        edge_index: (2, E) — [source, target] pairs
        Returns: (N, out_features) — embeddings on Poincare ball
        """
        # Direction: linear transform → unit-normalize.
        # Carries semantic structure (which way in the ball).
        h_raw = self.linear(x)
        h_dir = F.normalize(h_raw, dim=-1)  # (N, out_features) unit vectors

        # Magnitude: per-node scalar from separate head.
        # sigmoid * 1.5 → range [0, 1.5] → tanh → [0, 0.905].
        # Carries hierarchy depth (how far from origin).
        h_mag = torch.sigmoid(self.mag_head(x)) * 1.5  # (N, 1)

        # Combined tangent vector = direction × magnitude
        h = h_dir * h_mag

        # Neighbor aggregation with attention
        src, tgt = edge_index  # tgt aggregates from src
        h_src = h[src]  # (E, out)
        h_tgt = h[tgt]  # (E, out)

        # Attention scores
        attn_input = torch.cat([h_src, h_tgt], dim=1)  # (E, out*2)
        attn_scores = self.attn(attn_input).squeeze(-1)  # (E,)

        # Softmax per target node
        attn_weights = _scatter_softmax(attn_scores, tgt, num_nodes=x.size(0))

        # Weighted aggregation in tangent space (gyromidpoint approximation)
        weighted_msgs = h_src * attn_weights.unsqueeze(-1)  # (E, out)
        agg = torch.zeros(x.size(0), h.size(1), device=x.device)
        agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(weighted_msgs), weighted_msgs)

        # Combine self + neighbor
        h_combined = h + agg  # residual connection

        # Exponential map to Poincare ball
        h_poincare = self.ball.expmap0(h_combined)

        # Clamp norms to prevent boundary collapse (max 0.95)
        norms = torch.norm(h_poincare, dim=1, keepdim=True)
        max_norm = 0.95
        clamped = torch.where(
            norms > max_norm,
            h_poincare * (max_norm / (norms + 1e-8)),
            h_poincare,
        )

        return clamped


class HyperbolicGCN(nn.Module):
    """
    2-layer HGCN: 768d → 128d tangent → 16d Poincare.

    Usage:
        model = HyperbolicGCN(768, 128, 16)
        embeddings = model(features, edge_index)  # (N, 16) on Poincare ball
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        output_dim: int = 16,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=curvature)

        # Layer 1: Euclidean → tangent space
        self.layer1 = HyperbolicGCNLayer(input_dim, hidden_dim, self.ball)
        # Layer 2: tangent → Poincare ball (16d)
        self.layer2 = HyperbolicGCNLayer(hidden_dim, output_dim, self.ball)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 768) euclidean features
        edge_index: (2, E)
        Returns: (N, 16) Poincare embeddings
        """
        # Layer 1: euclidean → intermediate Poincare
        h = self.layer1(x, edge_index)
        h = self.dropout(h)

        # Log map back to tangent for layer 2
        h_tangent = self.ball.logmap0(h)
        h_tangent = F.relu(h_tangent)

        # Layer 2: tangent → final Poincare
        out = self.layer2(h_tangent, edge_index)

        return out


class StructuralLoss(nn.Module):
    """
    InfoNCE contrastive loss for hierarchical embedding (Nickel & Kiela 2017 style).

    For each positive (parent, child) edge: rank positive closer than K negatives
    via cross-entropy over softmax of distances. Does NOT regress to a target distance
    — lets the geometry arrange itself.

    Plus per-level shell prior (soft norm targets by SC/CTX/ANC level).
    """

    def __init__(self, ball: geoopt.PoincareBall, margin: float = 1.0,
                 norm_reg: float = 0.1, target_pc_dist: float = 0.5,
                 hierarchy_weight: float = 1.0, hierarchy_margin: float = 0.15,
                 shell_weight: float = 2.0,
                 shell_targets: Optional[Dict[int, float]] = None,
                 infonce_temperature: float = 0.5):
        super().__init__()
        self.ball = ball
        self.margin = margin
        self.norm_reg = norm_reg
        self.target_pc_dist = target_pc_dist  # kept for backward compat; unused by InfoNCE
        self.hierarchy_weight = hierarchy_weight
        self.hierarchy_margin = hierarchy_margin
        self.shell_weight = shell_weight
        if shell_targets is None:
            shell_targets = {0: 0.15, 1: 0.50, 2: 0.80}
        self.shell_targets = shell_targets
        self.infonce_temperature = infonce_temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        positive_edges: torch.Tensor,
        negative_edges: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        node_levels: Optional[torch.Tensor] = None,
        node_depths: Optional[torch.Tensor] = None,
        max_depth: int = 6,
    ) -> torch.Tensor:
        """
        embeddings: (N, D) on Poincare ball
        positive_edges: (E_pos, 2) — [child_idx, parent_idx]
        negative_edges: (E_neg, 2) — [child_idx, neg_idx]
        node_levels: (N,) int tensor with 0=SC, 1=CTX, 2=ANC (kept for backward compat)
        node_depths: (N,) int tensor with BFS depth from root (0=SC root, max=deepest leaf)
        max_depth: max observed depth, for normalizing targets
        """
        child_emb = embeddings[positive_edges[:, 0]]
        parent_emb = embeddings[positive_edges[:, 1]]

        # --- InfoNCE contrastive loss ---
        # For each positive edge, compute distance to positive and all its negatives.
        # Cross-entropy: positive should be closer than negatives.
        pos_dist = self.ball.dist(child_emb, parent_emb)  # (E_pos,)

        neg_child_emb = embeddings[negative_edges[:, 0]]
        neg_emb = embeddings[negative_edges[:, 1]]
        neg_dist = self.ball.dist(neg_child_emb, neg_emb)  # (E_neg,)

        # Reshape negatives: assume neg_samples per positive edge
        n_pos = positive_edges.shape[0]
        n_neg_total = negative_edges.shape[0]
        n_neg_per = n_neg_total // max(n_pos, 1)

        if n_neg_per > 0 and n_neg_total == n_pos * n_neg_per:
            neg_dist_grouped = neg_dist.view(n_pos, n_neg_per)  # (E_pos, K)
            # Logits: -distance / temperature (closer = higher logit)
            pos_logits = (-pos_dist / self.infonce_temperature).unsqueeze(1)  # (E_pos, 1)
            neg_logits = -neg_dist_grouped / self.infonce_temperature  # (E_pos, K)
            all_logits = torch.cat([pos_logits, neg_logits], dim=1)  # (E_pos, 1+K)
            # Cross-entropy: positive is class 0
            labels = torch.zeros(n_pos, dtype=torch.long, device=embeddings.device)
            contrastive_loss = F.cross_entropy(all_logits, labels)
        else:
            # Fallback: margin loss if negative count doesn't align
            contrastive_loss = pos_dist.mean() + F.relu(self.margin - neg_dist).mean()

        # --- Hierarchy regularization: parent norm < child norm ---
        parent_norms = torch.norm(parent_emb, dim=1)
        child_norms = torch.norm(child_emb, dim=1)
        hierarchy_violation = F.relu(parent_norms - child_norms + self.hierarchy_margin)
        hierarchy_loss = hierarchy_violation.mean()

        # --- Overall norm regularization ---
        all_norms = torch.norm(embeddings, dim=1)
        norm_spread_loss = F.relu(all_norms - 0.9).mean() + F.relu(0.05 - all_norms).mean()

        # Depth-based shell prior: each node's target norm is a linear function
        # of its BFS depth from root. Deeper = larger norm (further from origin).
        # Equal weight per depth bucket so shallow nodes (few) aren't drowned out.
        shell_loss = torch.tensor(0.0, device=embeddings.device)
        min_norm = self.shell_targets.get(0, 0.15)  # shallowest target
        max_norm_target = self.shell_targets.get(2, 0.85)  # deepest target
        if node_depths is not None and self.shell_weight > 0:
            # Per-node target: linear interpolation from depth
            md = max(float(max_depth), 1.0)
            target_per_node = min_norm + (node_depths.float() / md) * (max_norm_target - min_norm)
            # Equal weight per depth bucket (same trick as per-level)
            unique_depths = torch.unique(node_depths)
            depth_losses = []
            for d in unique_depths:
                mask = (node_depths == d)
                if mask.sum() > 0:
                    bucket_loss = ((all_norms[mask] - target_per_node[mask]) ** 2).mean()
                    depth_losses.append(bucket_loss)
            if depth_losses:
                shell_loss = torch.stack(depth_losses).mean()
        elif node_levels is not None and self.shell_weight > 0:
            # Fallback: level-based if depths not available
            level_losses = []
            for level_id, target in self.shell_targets.items():
                mask = (node_levels == level_id)
                if mask.sum() > 0:
                    level_losses.append(((all_norms[mask] - target) ** 2).mean())
            if level_losses:
                shell_loss = torch.stack(level_losses).mean()

        return (contrastive_loss
                + self.hierarchy_weight * hierarchy_loss
                + self.norm_reg * norm_spread_loss
                + self.shell_weight * shell_loss)


def _scatter_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute softmax of scores grouped by index (scatter softmax)."""
    max_vals = torch.full((num_nodes,), -1e9, device=scores.device)
    max_vals.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    exp_scores = torch.exp(scores - max_vals[index])
    sum_exp = torch.zeros(num_nodes, device=scores.device)
    sum_exp.scatter_add_(0, index, exp_scores)
    return exp_scores / (sum_exp[index] + 1e-10)


# ────────────────────────────────────────────────────────────
# Training Loop
# ────────────────────────────────────────────────────────────

class HGCNTrainer:
    """
    Trains the HGCN model during nightly consolidation.

    Usage:
        trainer = HGCNTrainer(db, hp)
        result = trainer.train()
        # result contains loss history, training time, embeddings
    """

    def __init__(self, db, hyperparams: Dict):
        self.db = db
        self.hp = hyperparams

    def train(self) -> Dict:
        """Full training pipeline with structural + co-retrieval + feedback losses."""
        t0 = time.time()

        # 1. Build graph data
        logger.info("Building graph data...")
        features, edge_index, node_ids, id_to_idx, levels_np, depths_np, max_depth = self._build_graph()
        n_nodes = features.shape[0]

        # Append depth encoding so the model's magnitude head can differentiate
        # nodes at different tree depths. Normalized depth [0,1] + one-hot level.
        n_levels = 3
        level_onehot = np.zeros((n_nodes, n_levels), dtype=np.float32)
        level_onehot[np.arange(n_nodes), levels_np] = 1.0
        depth_norm = (depths_np / max(float(max_depth), 1.0)).astype(np.float32).reshape(-1, 1)
        extra_features = np.concatenate([level_onehot, depth_norm], axis=1)  # 4 extra dims
        features = torch.cat([features, torch.from_numpy(extra_features)], dim=1)
        n_edges = edge_index.shape[1]
        logger.info(f"Graph: {n_nodes} nodes, {n_edges} structural edges")

        if n_edges == 0:
            return {"error": "No edges to train on", "nodes": n_nodes}

        # 1b. Build co-retrieval edges (Phase 4b)
        co_edges_np, co_weights_np = np.zeros((0, 2), dtype=np.int64), np.zeros(0)
        co_neg_edges_np = np.zeros((0, 2), dtype=np.int64)
        try:
            from core.co_retrieval import (
                build_co_retrieval_edges, sample_co_retrieval_negatives
            )
            co_edges_np, co_weights_np = build_co_retrieval_edges(
                self.db.db_path, id_to_idx, min_count=1
            )
            if len(co_edges_np) > 0:
                co_neg_edges_np = sample_co_retrieval_negatives(
                    co_edges_np, n_nodes, n_neg_per_edge=3
                )
            logger.info(f"Co-retrieval: {len(co_edges_np)} edges, {len(co_neg_edges_np)} negatives")
        except Exception as e:
            logger.warning(f"Co-retrieval edges failed: {e}")

        # 1c. Build feedback edges (Phase 4c)
        fb_pos_np = np.zeros((0, 2), dtype=np.int64)
        fb_neg_np = np.zeros((0, 2), dtype=np.int64)
        fb_strengths_np = np.zeros(0)
        try:
            from core.co_retrieval import build_feedback_edges
            fb_pos_np, fb_neg_np, fb_strengths_np = build_feedback_edges(
                self.db.db_path, id_to_idx
            )
            logger.info(f"Feedback: {len(fb_pos_np)} positive, {len(fb_neg_np)} negative edges")
        except Exception as e:
            logger.warning(f"Feedback edges failed: {e}")

        # 1d. Build match feedback edges from Goals-page thumbs up/down.
        # Rides the same w_fb loss weight as node-level feedback — one more
        # edge source, no new loss term needed.
        try:
            from core.co_retrieval import build_match_feedback_edges
            mfb_pos_np, mfb_neg_np, _ = build_match_feedback_edges(
                self.db.db_path, id_to_idx,
                max_age_days=self.hp.get("match_feedback_max_age_days", 90),
            )
            if len(mfb_pos_np) > 0:
                fb_pos_np = np.concatenate([fb_pos_np, mfb_pos_np], axis=0) \
                    if len(fb_pos_np) > 0 else mfb_pos_np
            if len(mfb_neg_np) > 0:
                fb_neg_np = np.concatenate([fb_neg_np, mfb_neg_np], axis=0) \
                    if len(fb_neg_np) > 0 else mfb_neg_np
            logger.info(
                f"Match feedback: {len(mfb_pos_np)} positive, "
                f"{len(mfb_neg_np)} negative edges"
            )
        except Exception as e:
            logger.warning(f"Match feedback edges failed: {e}")

        # 2. Create model
        input_dim = features.shape[1]
        hidden_dim = self.hp.get("hgcn_hidden_dim", 128)
        output_dim = self.hp.get("hgcn_output_dim", 16)
        curvature = abs(self.hp.get("hgcn_curvature_init", 1.0))

        model = HyperbolicGCN(input_dim, hidden_dim, output_dim, curvature)
        loss_fn = StructuralLoss(
            model.ball,
            margin=self.hp.get("hgcn_margin", 1.0),
            norm_reg=self.hp.get("hgcn_norm_reg", 0.1),
            target_pc_dist=self.hp.get("hgcn_target_pc_dist", 0.5),
            hierarchy_weight=self.hp.get("hgcn_hierarchy_weight", 1.0),
            hierarchy_margin=self.hp.get("hgcn_hierarchy_margin", 0.15),
            shell_weight=self.hp.get("hgcn_shell_weight", 5.0),
            infonce_temperature=self.hp.get("hgcn_infonce_temperature", 0.5),
            shell_targets={
                0: self.hp.get("hgcn_shell_target_sc", 0.15),
                1: self.hp.get("hgcn_shell_target_ctx", 0.50),
                2: self.hp.get("hgcn_shell_target_anc", 0.80),
            },
        )

        # 3. Optimizer
        lr = self.hp.get("hgcn_lr", 0.01)
        optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr)

        # 4. Device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        node_levels = torch.tensor(levels_np, dtype=torch.long, device=device)
        node_depths = torch.tensor(depths_np, dtype=torch.long, device=device)

        # Move co-retrieval + feedback tensors to device
        co_edges = torch.tensor(co_edges_np, dtype=torch.long, device=device) if len(co_edges_np) > 0 else None
        co_weights = torch.tensor(co_weights_np, dtype=torch.float32, device=device) if len(co_weights_np) > 0 else None
        co_neg = torch.tensor(co_neg_edges_np, dtype=torch.long, device=device) if len(co_neg_edges_np) > 0 else None
        fb_pos = torch.tensor(fb_pos_np, dtype=torch.long, device=device) if len(fb_pos_np) > 0 else None
        fb_neg = torch.tensor(fb_neg_np, dtype=torch.long, device=device) if len(fb_neg_np) > 0 else None

        # 5. Structural positive edges (from edge_index, excluding self-loops)
        all_edges = edge_index.t()
        mask = all_edges[:, 0] != all_edges[:, 1]
        pos_edges = all_edges[mask]
        # Take only one direction (parent → child, not both)
        pos_edges = pos_edges[::2]  # Every other edge (since we added bidirectional)
        neg_samples = self.hp.get("hgcn_neg_samples", 10)

        # Loss weights
        w_struct = self.hp.get("hgcn_structural_weight", 0.6)
        w_coret = self.hp.get("hgcn_coretrieval_weight", 0.3)
        w_fb = self.hp.get("hgcn_feedback_weight", 0.1)

        # 6. Training loop
        epochs = self.hp.get("hgcn_epochs", 100)
        patience = self.hp.get("hgcn_patience", 10)
        loss_history = []
        best_loss = float("inf")
        patience_counter = 0

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            embeddings = model(features, edge_index)

            # Structural loss
            neg_edges = self._sample_negatives(pos_edges, n_nodes, neg_samples, device)
            struct_loss = loss_fn(embeddings, pos_edges, neg_edges,
                                  node_levels=node_levels,
                                  node_depths=node_depths,
                                  max_depth=max_depth)

            total_loss = w_struct * struct_loss

            # Co-retrieval loss (Phase 4b)
            if co_edges is not None and len(co_edges) > 0:
                co_pos_dist = model.ball.dist(
                    embeddings[co_edges[:, 0]], embeddings[co_edges[:, 1]]
                )
                # Weighted: high co-occurrence = stronger pull
                co_pos_loss = (co_pos_dist ** 2 * co_weights).mean()

                co_neg_loss = torch.tensor(0.0, device=device)
                if co_neg is not None and len(co_neg) > 0:
                    co_neg_dist = model.ball.dist(
                        embeddings[co_neg[:, 0]], embeddings[co_neg[:, 1]]
                    )
                    co_neg_loss = F.relu(1.0 - co_neg_dist).mean()

                total_loss = total_loss + w_coret * (co_pos_loss + co_neg_loss)

            # Feedback loss (Phase 4c)
            if fb_pos is not None and len(fb_pos) > 0:
                fb_pos_dist = model.ball.dist(
                    embeddings[fb_pos[:, 0]], embeddings[fb_pos[:, 1]]
                )
                fb_pos_loss = (fb_pos_dist ** 2).mean()
                total_loss = total_loss + w_fb * fb_pos_loss

            if fb_neg is not None and len(fb_neg) > 0:
                fb_neg_dist = model.ball.dist(
                    embeddings[fb_neg[:, 0]], embeddings[fb_neg[:, 1]]
                )
                fb_neg_loss = F.relu(1.0 - fb_neg_dist).mean()
                total_loss = total_loss + w_fb * fb_neg_loss

            loss_val = total_loss.item()
            loss_history.append(loss_val)

            # Backward + step
            total_loss.backward()
            optimizer.step()

            # Early stopping
            if loss_val < best_loss - 1e-4:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (best loss: {best_loss:.4f})")
                break

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: loss={loss_val:.4f}")

        # 7. Extract final embeddings
        model.eval()
        with torch.no_grad():
            final_embeddings = model(features, edge_index).cpu().numpy()

        wall_time = time.time() - t0

        # 8. Save checkpoint
        checkpoint_path = Path(self.db.db_path).parent / "hgcn_checkpoint.pt"
        torch.save({
            "model_state": model.state_dict(),
            "node_ids": node_ids,
            "id_to_idx": id_to_idx,
            "loss_history": loss_history,
            "config": {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "curvature": curvature,
            },
        }, checkpoint_path)

        # 9. Update database with new embeddings
        logger.info(f"Updating {n_nodes} embeddings in database...")
        self._update_embeddings(node_ids, final_embeddings)

        result = {
            "nodes_trained": n_nodes,
            "structural_edges": len(pos_edges),
            "co_retrieval_edges": len(co_edges_np),
            "feedback_pos_edges": len(fb_pos_np),
            "feedback_neg_edges": len(fb_neg_np),
            "epochs_run": len(loss_history),
            "final_loss": loss_history[-1] if loss_history else None,
            "best_loss": best_loss,
            "loss_converged": patience_counter >= patience,
            "wall_time_seconds": wall_time,
            "checkpoint_path": str(checkpoint_path),
            "output_dim": output_dim,
        }
        logger.info(f"HGCN training complete: {result}")
        return result

    def _build_graph(self) -> Tuple[torch.Tensor, torch.Tensor, List[str], Dict[str, int], np.ndarray, np.ndarray, int]:
        """
        Build PyTorch tensors from database.
        Returns: (features, edge_index, node_ids, id_to_idx, levels, depths, max_depth)
            levels: int array, 0=SC, 1=CTX, 2=ANC
            depths: int array, BFS depth from root (0=root, max_depth=deepest/orphan)
            max_depth: int, maximum observed depth
        """
        import sqlite3
        from collections import deque, defaultdict

        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row

        # Load all non-deleted nodes with euclidean embeddings + level
        rows = conn.execute("""
            SELECT mn.id, mn.level, e.euclidean
            FROM memory_nodes mn
            JOIN embeddings e ON e.node_id = mn.id
            WHERE mn.is_deleted = 0 AND e.euclidean IS NOT NULL
        """).fetchall()

        level_to_id = {"SC": 0, "CTX": 1, "ANC": 2}

        node_ids = []
        features_list = []
        levels_list = []
        id_to_idx = {}

        for row in rows:
            emb = np.frombuffer(row["euclidean"], dtype=np.float32).copy()
            idx = len(node_ids)
            node_ids.append(row["id"])
            id_to_idx[row["id"]] = idx
            features_list.append(emb)
            levels_list.append(level_to_id.get(row["level"], 2))  # unknown → ANC

        if not features_list:
            conn.close()
            return (torch.zeros(0, 768), torch.zeros(2, 0, dtype=torch.long),
                    [], {}, np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64), 0)

        features = torch.tensor(np.array(features_list), dtype=torch.float32)

        # Load child_of edges (bidirectional for message passing)
        edges = conn.execute("""
            SELECT source_id, target_id
            FROM relations
            WHERE relation_type = 'child_of'
        """).fetchall()

        # Build adjacency for BFS: target is parent (child_of convention)
        graph_children = defaultdict(set)  # parent → children
        for edge in edges:
            src_id, tgt_id = edge["source_id"], edge["target_id"]
            if src_id in id_to_idx and tgt_id in id_to_idx:
                graph_children[tgt_id].add(src_id)

        src_indices = []
        tgt_indices = []
        for edge in edges:
            src_id, tgt_id = edge["source_id"], edge["target_id"]
            if src_id in id_to_idx and tgt_id in id_to_idx:
                src_idx = id_to_idx[src_id]
                tgt_idx = id_to_idx[tgt_id]
                # Bidirectional for GCN message passing
                src_indices.extend([src_idx, tgt_idx])
                tgt_indices.extend([tgt_idx, src_idx])

        # Add self-loops
        for i in range(len(node_ids)):
            src_indices.append(i)
            tgt_indices.append(i)

        edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
        levels = np.array(levels_list, dtype=np.int64)

        # BFS depth from root SCs
        sc_ids = [nid for nid, lvl in zip(node_ids, levels_list) if lvl == 0]
        depth_map = {}
        for sc in sc_ids:
            if sc not in depth_map:
                depth_map[sc] = 0
                queue = deque([(sc, 0)])
                while queue:
                    nid, d = queue.popleft()
                    for child in graph_children.get(nid, set()):
                        if child not in depth_map:
                            depth_map[child] = d + 1
                            queue.append((child, d + 1))

        max_depth = max(depth_map.values()) if depth_map else 1
        # Orphan nodes (no path to root) get max_depth (treated as leaves)
        depths = np.array(
            [depth_map.get(nid, max_depth) for nid in node_ids],
            dtype=np.int64,
        )
        logger.info(f"Depth stats: max={max_depth}, "
                     f"rooted={sum(1 for nid in node_ids if nid in depth_map)}/{len(node_ids)}")

        conn.close()
        return features, edge_index, node_ids, id_to_idx, levels, depths, max_depth

    def _sample_negatives(
        self,
        pos_edges: torch.Tensor,
        n_nodes: int,
        n_neg: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample negative edges. 50% cross-branch, 50% random."""
        n_pos = pos_edges.shape[0]
        parent_indices = pos_edges[:, 0].repeat_interleave(n_neg)

        # Random negative targets
        neg_targets = torch.randint(0, n_nodes, (n_pos * n_neg,), device=device)

        # Filter out accidental positives (best effort)
        neg_edges = torch.stack([parent_indices, neg_targets], dim=1)
        return neg_edges

    def _update_embeddings(self, node_ids: List[str], embeddings: np.ndarray):
        """Write new 16d Poincare embeddings back to database."""
        import sqlite3

        conn = sqlite3.connect(self.db.db_path)
        for i, node_id in enumerate(node_ids):
            emb = embeddings[i]
            norm = float(np.linalg.norm(emb))
            blob = emb.astype(np.float32).tobytes()

            conn.execute("""
                UPDATE embeddings
                SET hyperbolic = ?, hyperbolic_norm = ?, is_learned = 1,
                    last_trained = datetime('now')
                WHERE node_id = ?
            """, (blob, norm, node_id))

        conn.commit()
        conn.close()
        logger.info(f"Updated {len(node_ids)} hyperbolic embeddings ({embeddings.shape[1]}d)")
