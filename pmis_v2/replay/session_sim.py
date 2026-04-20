"""
session_sim.py — Pure math simulation of the PMIS V2 Session Tree Engine.

Shadow-mode only: computes what the session engine WOULD do without
affecting actual retrieval. All functions are stateless pure math.
The SimulatedSessionEngine class accumulates state across turns.

Reference: PMIS V2 Session Tree Engine Architecture v1.0, Sections 5-8
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import uuid
import time


# ─── Primitives ──────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─── Hyperparameters (all from architecture doc Section 5) ───────────────

@dataclass
class SessionHyperparams:
    # Confidence update
    alpha: float = 0.35           # confidence learning rate
    beta: float = 4.0             # affinity temperature
    delta: float = 3.0            # support temperature
    epsilon: float = 2.5          # baseline bias

    # Embedding accumulation
    lambda_base: float = 0.3     # EMA rate
    decay_accel: float = 0.5     # later turns matter more

    # Convergence
    T_c: float = 8.0             # convergence temperature
    b_c: float = 0.25            # minimum gap for inflection
    theta_converged: float = 0.85

    # Divergence
    T_d: float = 6.0             # divergence temperature
    b_d: float = 0.45            # surprise inflection
    theta_diverge: float = 0.75
    diverge_smoothing: float = 0.3  # EMA for tree surprise

    # Schema boost weights
    sc_boost: float = 0.15       # SC match
    ctx_boost: float = 0.25      # CTX match
    anc_boost: float = 0.10      # ANC match

    # Pruning
    prune_min_confidence: float = 0.1
    prune_min_turns: int = 3

    # Storage
    store_min_score: float = 0.5
    store_min_turns: int = 3

    # Frustration
    frustration_repetition_thresh: float = 0.75
    frustration_specificity_thresh: float = 0.15

    # Dual gamma/surprise
    gamma_T: float = 3.0
    gamma_bias: float = 0.5

    # Session gamma multipliers for convergence
    gamma_high_thresh: float = 0.6
    gamma_low_thresh: float = 0.4
    gamma_high_mult: float = 1.1
    gamma_low_mult: float = 0.7

    # Trajectory
    trajectory_narrow_mult: float = 1.2
    trajectory_broad_mult: float = 0.8

    # Convergence/divergence boosts from surprise gap
    conv_boost_value: float = 0.10
    div_boost_value: float = 0.15

    # Limits
    max_candidates: int = 5
    max_turns_no_convergence: int = 15


# ─── Data Structures ────────────────────────────────────────────────────

@dataclass
class SimTreeNode:
    node_id: str
    node_type: str              # "SC", "CTX", "ANC"
    permanent_id: Optional[str] = None  # link to real memory node
    name: str = ""
    is_exploratory: bool = False  # x-node from frustration fork
    confidence: float = 0.3

@dataclass
class SimTree:
    tree_id: str
    root_sc_id: Optional[str]   # permanent SC id (None for cold start)
    root_sc_name: str
    confidence: float = 0.5
    confidence_history: List[float] = field(default_factory=list)
    accumulated_embedding: Optional[np.ndarray] = None
    nodes: List[SimTreeNode] = field(default_factory=list)
    matched_memory_ids: List[str] = field(default_factory=list)
    turns_alive: int = 0
    created_at_turn: int = 0

@dataclass
class FrustrationState:
    stage: int = 0              # 0=none, 1=reduce+fork, 2=cross-domain, 3=reset
    turns_in_stage: int = 0
    repetition_scores: List[float] = field(default_factory=list)
    blend_penalty: float = 0.0

@dataclass
class SimTurnResult:
    """Everything the harness logs per turn."""
    turn_number: int
    timestamp: float

    # Dual gamma/surprise
    gamma_global: float
    gamma_session: float
    surprise_global: float
    surprise_session: float
    surprise_gap: float

    # Trees
    num_candidates: int
    trees_summary: List[Dict]   # [{id, sc_name, confidence, turns_alive}]
    best_tree_id: Optional[str]
    best_tree_sc: Optional[str]
    second_tree_id: Optional[str]
    confidence_gap: float

    # Convergence & divergence
    convergence: float
    divergence: float
    is_converged: bool
    blend_weight: float

    # Trajectory
    depth: float
    trajectory_direction: float  # positive=narrowing, negative=broadening
    narrowing_trend: bool

    # Frustration
    frustration_stage: int
    frustration_repetition: float
    frustration_blend_penalty: float

    # Re-ranking effect
    rerank_changes: int          # how many memories changed position
    top_memory_boosted: bool     # did schema boost change the #1 result?
    max_schema_boost: float

    # Evidence
    supporting_memories: int     # memories matching active tree
    challenging_memories: int    # memories matching OTHER trees

    # Feedback (filled in NEXT turn)
    feedback_bit: Optional[int] = None       # 1=confirmed, 0=rejected
    feedback_continuous: Optional[float] = None  # continuous version

    # What WOULD have happened
    would_suppress_epistemic: bool = False
    would_hippocampus_fallback: bool = False


# ─── Simulated Session Engine ───────────────────────────────────────────

class SimulatedSessionEngine:
    """
    Shadow-mode session engine. Call observe_turn() on each conversation
    turn with the real data from the main orchestrator. Computes all
    session tree math without affecting actual retrieval.
    """

    def __init__(self, hp: Optional[SessionHyperparams] = None):
        self.hp = hp or SessionHyperparams()
        self.trees: List[SimTree] = []
        self.turn_count: int = 0
        self.is_converged: bool = False
        self.converged_tree_id: Optional[str] = None
        self.turns_since_convergence: int = 0
        self.smoothed_tree_surprise: float = 0.0
        self.frustration: FrustrationState = FrustrationState()

        # History for trajectory and feedback
        self.turn_embeddings: List[np.ndarray] = []
        self.depth_history: List[float] = []
        self.direction_history: List[float] = []
        self.convergence_history: List[float] = []
        self.divergence_history: List[float] = []
        self.results_history: List[SimTurnResult] = []

        # For feedback signal
        self._prev_accumulated: Optional[np.ndarray] = None
        self._prev_global_centroid: Optional[np.ndarray] = None
        self._prev_converged: bool = False

    def observe_turn(
        self,
        turn_embedding: np.ndarray,
        gamma_global: float,
        surprise_global: float,
        unbiased_retrieved: List[Dict],
        conversation_id: str = "",
    ) -> SimTurnResult:
        """
        Process one turn in shadow mode.

        Args:
            turn_embedding: 1536-d embedding of the user's message
            gamma_global: gamma from main orchestrator
            surprise_global: surprise from main orchestrator
            unbiased_retrieved: list of dicts with keys:
                {id, sc_id, ctx_id, anc_id, score, embedding}
        """
        self.turn_count += 1
        t = self.turn_count
        self.turn_embeddings.append(turn_embedding.copy())

        # ── 1. Compute feedback for PREVIOUS turn ──
        feedback_bit, feedback_continuous = self._compute_feedback(turn_embedding)
        if self.results_history:
            self.results_history[-1].feedback_bit = feedback_bit
            self.results_history[-1].feedback_continuous = feedback_continuous

        # ── 2. Compute session gamma & surprise ──
        gamma_session, surprise_session = self._compute_session_gamma_surprise(
            turn_embedding
        )
        surprise_gap = abs(surprise_global - surprise_session)

        # ── 3. Initialize or update trees ──
        if not self.trees:
            self._initialize_trees(turn_embedding, unbiased_retrieved)
        else:
            self._update_tree_embeddings(turn_embedding)

        # ── 4. Score evidence against each tree ──
        evidence = self._score_evidence(unbiased_retrieved)

        # ── 5. Update tree confidences ──
        self._update_confidences(turn_embedding, evidence)

        # ── 6. Grow matching trees ──
        self._grow_trees(unbiased_retrieved, turn_embedding)

        # ── 7. Compute convergence ──
        convergence, best_tree, second_tree, gap = self._compute_convergence(
            gamma_session, surprise_global, surprise_session
        )
        self.convergence_history.append(convergence)

        # ── 8. Compute trajectory ──
        depth, direction, narrowing = self._compute_trajectory(unbiased_retrieved)

        # ── 9. Simulate re-ranking ──
        rerank_changes, top_boosted, max_boost = self._simulate_rerank(
            unbiased_retrieved, best_tree, convergence
        )

        # ── 10. Compute blend weight ──
        best_conf = best_tree.confidence if best_tree else 0.0
        blend_weight = convergence * best_conf

        # ── 11. Detect frustration ──
        frust_rep = self._detect_frustration(turn_embedding)

        # ── 12. Check divergence ──
        div = self._compute_divergence(
            turn_embedding, surprise_global, surprise_session
        )
        self.divergence_history.append(div)

        # ── 13. Handle convergence state ──
        was_converged = self.is_converged
        if convergence >= self.hp.theta_converged and not self.is_converged:
            self.is_converged = True
            self.converged_tree_id = best_tree.tree_id if best_tree else None
            self.turns_since_convergence = 0
        elif self.is_converged:
            self.turns_since_convergence += 1

        # ── 14. Handle divergence ──
        diverged = False
        if (self.is_converged
            and div >= self.hp.theta_diverge
            and self.turns_since_convergence >= 3):
            diverged = True
            # In shadow mode: just record it, don't actually reset
            # (we track what WOULD happen)

        # ── 15. Apply frustration penalty to blend ──
        if self.frustration.stage > 0:
            blend_weight *= (1.0 - self.frustration.blend_penalty)

        # ── 16. Prune dead trees ──
        pruned = self._prune_trees()

        # ── 17. Count supporting/challenging evidence ──
        supporting = 0
        challenging = 0
        if best_tree:
            for mem in unbiased_retrieved:
                sc = mem.get("sc_id", "")
                if sc == best_tree.root_sc_id:
                    supporting += 1
                elif sc and sc != best_tree.root_sc_id:
                    challenging += 1

        # ── 18. Store state for next turn's feedback ──
        if best_tree and best_tree.accumulated_embedding is not None:
            self._prev_accumulated = best_tree.accumulated_embedding.copy()
        self._prev_global_centroid = self._compute_global_centroid()
        self._prev_converged = self.is_converged

        # ── 19. Would-have checks ──
        would_suppress_epistemic = convergence > 0.7
        would_hippocampus_fallback = (
            t >= self.hp.max_turns_no_convergence and not self.is_converged
        )

        # ── 20. Build result ──
        trees_summary = [
            {
                "id": tr.tree_id,
                "sc_name": tr.root_sc_name,
                "sc_id": tr.root_sc_id,
                "confidence": round(tr.confidence, 4),
                "turns_alive": tr.turns_alive,
                "num_nodes": len(tr.nodes),
                "matched_memories": len(tr.matched_memory_ids),
            }
            for tr in sorted(self.trees, key=lambda x: -x.confidence)
        ]

        result = SimTurnResult(
            turn_number=t,
            timestamp=time.time(),
            gamma_global=round(gamma_global, 4),
            gamma_session=round(gamma_session, 4),
            surprise_global=round(surprise_global, 4),
            surprise_session=round(surprise_session, 4),
            surprise_gap=round(surprise_gap, 4),
            num_candidates=len(self.trees),
            trees_summary=trees_summary,
            best_tree_id=best_tree.tree_id if best_tree else None,
            best_tree_sc=best_tree.root_sc_name if best_tree else None,
            second_tree_id=second_tree.tree_id if second_tree else None,
            confidence_gap=round(gap, 4),
            convergence=round(convergence, 4),
            divergence=round(div, 4),
            is_converged=self.is_converged,
            blend_weight=round(blend_weight, 4),
            depth=round(depth, 4),
            trajectory_direction=round(direction, 4),
            narrowing_trend=narrowing,
            frustration_stage=self.frustration.stage,
            frustration_repetition=round(frust_rep, 4),
            frustration_blend_penalty=round(self.frustration.blend_penalty, 4),
            rerank_changes=rerank_changes,
            top_memory_boosted=top_boosted,
            max_schema_boost=round(max_boost, 4),
            supporting_memories=supporting,
            challenging_memories=challenging,
            would_suppress_epistemic=would_suppress_epistemic,
            would_hippocampus_fallback=would_hippocampus_fallback,
        )

        self.results_history.append(result)
        return result

    # ─── Internal Methods ───────────────────────────────────────────────

    def _compute_session_gamma_surprise(
        self, turn_embedding: np.ndarray
    ) -> Tuple[float, float]:
        """Dual gamma/surprise independent of main orchestrator."""
        if len(self.turn_embeddings) < 2:
            return 0.5, 0.5  # neutral on first turn

        # Session surprise = distance from accumulated trajectory
        accumulated = np.mean(self.turn_embeddings[:-1], axis=0)
        s_session = 1.0 - cosine_sim(turn_embedding, accumulated)
        g_session = sigmoid(-s_session * self.hp.gamma_T + self.hp.gamma_bias)
        return float(g_session), float(s_session)

    def _initialize_trees(
        self, turn_embedding: np.ndarray, retrieved: List[Dict]
    ) -> None:
        """Spawn candidate trees from retrieved memories' SCs."""
        seen_scs = {}
        for mem in retrieved[:10]:  # top 10
            sc_id = mem.get("sc_id")
            sc_name = mem.get("sc_name", "unknown")
            if sc_id and sc_id not in seen_scs:
                seen_scs[sc_id] = sc_name

        if not seen_scs:
            # Cold start: exploratory tree with no SC
            tree = SimTree(
                tree_id=f"st_{uuid.uuid4().hex[:8]}",
                root_sc_id=None,
                root_sc_name="[exploratory]",
                confidence=0.5,
                accumulated_embedding=turn_embedding.copy(),
                created_at_turn=self.turn_count,
            )
            self.trees.append(tree)
            return

        for sc_id, sc_name in list(seen_scs.items())[:self.hp.max_candidates]:
            tree = SimTree(
                tree_id=f"st_{uuid.uuid4().hex[:8]}",
                root_sc_id=sc_id,
                root_sc_name=sc_name,
                confidence=0.5,
                confidence_history=[0.5],
                accumulated_embedding=turn_embedding.copy(),
                created_at_turn=self.turn_count,
            )
            self.trees.append(tree)

    def _update_tree_embeddings(self, turn_embedding: np.ndarray) -> None:
        """Turn-weighted EMA update for each tree's accumulated embedding."""
        t = self.turn_count
        lam = self.hp.lambda_base * (
            1 + self.hp.decay_accel * min(t, 10) / 10.0
        )
        for tree in self.trees:
            tree.turns_alive += 1
            if tree.accumulated_embedding is not None:
                tree.accumulated_embedding = (
                    (1 - lam) * tree.accumulated_embedding + lam * turn_embedding
                )
            else:
                tree.accumulated_embedding = turn_embedding.copy()

    def _score_evidence(self, retrieved: List[Dict]) -> Dict[str, Dict]:
        """Score how well retrieved memories support each candidate tree."""
        evidence = {}
        for tree in self.trees:
            matching = 0
            strongly_matching = 0
            total = len(retrieved)

            tree_ctx_ids = {
                n.permanent_id for n in tree.nodes if n.node_type == "CTX"
            }

            for mem in retrieved:
                if mem.get("sc_id") == tree.root_sc_id:
                    matching += 1
                if mem.get("ctx_id") in tree_ctx_ids:
                    strongly_matching += 1

            support = (matching + 2 * strongly_matching) / (total + 1)
            evidence[tree.tree_id] = {
                "matching": matching,
                "strongly_matching": strongly_matching,
                "support": support,
                "total": total,
            }
        return evidence

    def _update_confidences(
        self, turn_embedding: np.ndarray, evidence: Dict
    ) -> None:
        """Sigmoid-gated EMA confidence update per tree."""
        for tree in self.trees:
            if tree.accumulated_embedding is None:
                continue

            # Affinity: how well this turn matches tree trajectory
            # Compare to PREVIOUS accumulated (before this turn's update)
            # We approximate since we already updated
            affinity = cosine_sim(turn_embedding, tree.accumulated_embedding)

            # Support from evidence
            ev = evidence.get(tree.tree_id, {"support": 0.0})
            support = ev["support"]

            # Sigmoid-gated raw signal
            raw = sigmoid(
                self.hp.beta * affinity
                + self.hp.delta * support
                - self.hp.epsilon
            )

            # EMA update
            tree.confidence = (
                (1 - self.hp.alpha) * tree.confidence
                + self.hp.alpha * raw
            )
            tree.confidence_history.append(round(tree.confidence, 4))

    def _grow_trees(
        self, retrieved: List[Dict], turn_embedding: np.ndarray
    ) -> None:
        """Add CTX/ANC nodes to matching trees from evidence."""
        for tree in self.trees:
            if tree.confidence < 0.3:
                continue
            existing_ids = {n.permanent_id for n in tree.nodes}
            for mem in retrieved[:5]:
                if mem.get("sc_id") == tree.root_sc_id:
                    ctx_id = mem.get("ctx_id")
                    if ctx_id and ctx_id not in existing_ids:
                        tree.nodes.append(SimTreeNode(
                            node_id=f"n_{uuid.uuid4().hex[:6]}",
                            node_type="CTX",
                            permanent_id=ctx_id,
                            name=mem.get("ctx_name", ""),
                        ))
                        existing_ids.add(ctx_id)
                    anc_id = mem.get("anc_id")
                    if anc_id and anc_id not in existing_ids:
                        tree.nodes.append(SimTreeNode(
                            node_id=f"n_{uuid.uuid4().hex[:6]}",
                            node_type="ANC",
                            permanent_id=anc_id,
                            name=mem.get("anc_name", ""),
                        ))
                        existing_ids.add(anc_id)
                    mem_id = mem.get("id")
                    if mem_id and mem_id not in tree.matched_memory_ids:
                        tree.matched_memory_ids.append(mem_id)

    def _compute_convergence(
        self,
        gamma_session: float,
        surprise_global: float,
        surprise_session: float,
    ) -> Tuple[float, Optional[SimTree], Optional[SimTree], float]:
        """Sigmoid convergence with trajectory and gamma adjustments."""
        if len(self.trees) < 1:
            return 0.0, None, None, 0.0

        sorted_trees = sorted(self.trees, key=lambda t: -t.confidence)
        best = sorted_trees[0]
        second = sorted_trees[1] if len(sorted_trees) > 1 else None
        gap = best.confidence - (second.confidence if second else 0.0)

        # Trajectory multiplier
        if self.direction_history:
            last_dir = self.direction_history[-1]
            if last_dir > 0.05:
                traj_mult = self.hp.trajectory_narrow_mult
            elif last_dir < -0.05:
                traj_mult = self.hp.trajectory_broad_mult
            else:
                traj_mult = 1.0
        else:
            traj_mult = 1.0

        # Gamma multiplier
        if gamma_session > self.hp.gamma_high_thresh:
            gamma_mult = self.hp.gamma_high_mult
        elif gamma_session < self.hp.gamma_low_thresh:
            gamma_mult = self.hp.gamma_low_mult
        else:
            gamma_mult = 1.0

        # Convergence boost from surprise gap
        conv_boost = 0.0
        if surprise_global > 0.5 and surprise_session < 0.25:
            conv_boost = self.hp.conv_boost_value

        T_adj = self.hp.T_c * traj_mult * gamma_mult
        convergence = sigmoid(T_adj * (gap + conv_boost - self.hp.b_c))

        return convergence, best, second, gap

    def _compute_divergence(
        self,
        turn_embedding: np.ndarray,
        surprise_global: float,
        surprise_session: float,
    ) -> float:
        """Sigmoid divergence detection."""
        if not self.is_converged:
            return 0.0

        # Find converged tree
        conv_tree = None
        for t in self.trees:
            if t.tree_id == self.converged_tree_id:
                conv_tree = t
                break

        if conv_tree is None or conv_tree.accumulated_embedding is None:
            return 0.0

        tree_surprise = 1.0 - cosine_sim(
            turn_embedding, conv_tree.accumulated_embedding
        )
        self.smoothed_tree_surprise = (
            (1 - self.hp.diverge_smoothing) * self.smoothed_tree_surprise
            + self.hp.diverge_smoothing * tree_surprise
        )

        # Known topic shift boost
        div_boost = 0.0
        if surprise_global < 0.25 and surprise_session > 0.5:
            div_boost = self.hp.div_boost_value

        div = sigmoid(self.hp.T_d * (
            self.smoothed_tree_surprise + div_boost - self.hp.b_d
        ))
        return div

    def _compute_trajectory(
        self, retrieved: List[Dict]
    ) -> Tuple[float, float, bool]:
        """Track depth and direction within session."""
        # Depth = weighted mean of matched node levels
        weights = {"SC": 0.1, "CTX": 0.5, "ANC": 0.9}
        if not retrieved:
            depth = 0.0
        else:
            depths = []
            for mem in retrieved[:5]:
                if mem.get("anc_id"):
                    depths.append(weights["ANC"])
                elif mem.get("ctx_id"):
                    depths.append(weights["CTX"])
                elif mem.get("sc_id"):
                    depths.append(weights["SC"])
            depth = float(np.mean(depths)) if depths else 0.0

        self.depth_history.append(depth)

        if len(self.depth_history) >= 2:
            direction = depth - self.depth_history[-2]
        else:
            direction = 0.0

        self.direction_history.append(direction)

        narrowing = (
            len(self.direction_history) >= 3
            and all(d > 0 for d in self.direction_history[-3:])
        )

        return depth, direction, narrowing

    def _simulate_rerank(
        self,
        retrieved: List[Dict],
        best_tree: Optional[SimTree],
        convergence: float,
    ) -> Tuple[int, bool, float]:
        """Simulate re-ranking to measure its effect without applying it."""
        if not best_tree or convergence < 0.3 or not retrieved:
            return 0, False, 0.0

        tree_ctx_ids = {
            n.permanent_id for n in best_tree.nodes if n.node_type == "CTX"
        }
        tree_anc_ids = {
            n.permanent_id for n in best_tree.nodes if n.node_type == "ANC"
        }

        original_order = [m.get("id") for m in retrieved]
        boosted = []
        max_boost = 0.0

        for mem in retrieved:
            boost = 0.0
            if mem.get("sc_id") == best_tree.root_sc_id:
                boost += self.hp.sc_boost
            if mem.get("ctx_id") in tree_ctx_ids:
                boost += self.hp.ctx_boost
            if mem.get("anc_id") in tree_anc_ids:
                boost += self.hp.anc_boost

            w = convergence * best_tree.confidence
            effective_boost = boost * w
            max_boost = max(max_boost, effective_boost)

            new_score = mem.get("score", 0.0) + effective_boost
            boosted.append((mem.get("id"), new_score))

        boosted.sort(key=lambda x: -x[1])
        new_order = [b[0] for b in boosted]

        changes = sum(
            1 for i, mid in enumerate(original_order)
            if i < len(new_order) and new_order[i] != mid
        )
        top_changed = (
            len(new_order) > 0
            and len(original_order) > 0
            and new_order[0] != original_order[0]
        )

        return changes, top_changed, max_boost

    def _detect_frustration(self, turn_embedding: np.ndarray) -> float:
        """Detect repetition-without-deepening pattern."""
        if len(self.turn_embeddings) < 3:
            self.frustration.repetition_scores.append(0.0)
            return 0.0

        # Repetition score = mean similarity to last 2 turns
        sims = []
        for prev in self.turn_embeddings[-3:-1]:
            sims.append(cosine_sim(turn_embedding, prev))
        rep_score = float(np.mean(sims))
        self.frustration.repetition_scores.append(rep_score)

        # Specificity change
        if len(self.depth_history) >= 2:
            spec_change = abs(self.depth_history[-1] - self.depth_history[-2])
        else:
            spec_change = 0.0

        # Frustration fires?
        is_frustrated = (
            rep_score > self.hp.frustration_repetition_thresh
            and spec_change < self.hp.frustration_specificity_thresh
        )

        if is_frustrated:
            if self.frustration.stage == 0:
                self.frustration.stage = 1
                self.frustration.turns_in_stage = 1
                penalty = 0.5 * (
                    (rep_score - self.hp.frustration_repetition_thresh) / 0.25
                )
                self.frustration.blend_penalty = min(penalty, 0.5)
            else:
                self.frustration.turns_in_stage += 1
                if self.frustration.turns_in_stage >= 3 and self.frustration.stage == 1:
                    self.frustration.stage = 2
                    self.frustration.blend_penalty = 0.9
                elif self.frustration.turns_in_stage >= 6 and self.frustration.stage == 2:
                    self.frustration.stage = 3
                    self.frustration.blend_penalty = 1.0
        else:
            # Reset frustration if user breaks pattern
            if self.frustration.stage > 0:
                self.frustration.stage = 0
                self.frustration.turns_in_stage = 0
                self.frustration.blend_penalty = 0.0

        return rep_score

    def _prune_trees(self) -> int:
        """Remove dead trees."""
        before = len(self.trees)
        self.trees = [
            t for t in self.trees
            if not (
                t.confidence < self.hp.prune_min_confidence
                and t.turns_alive > self.hp.prune_min_turns
            )
        ]
        return before - len(self.trees)

    def _compute_feedback(
        self, current_turn_embedding: np.ndarray
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        One-bit feedback signal: did the user's current turn confirm
        or reject the schema direction from the PREVIOUS turn?

        Compares: similarity to converged tree's accumulated embedding
        vs similarity to global centroid of all turns.

        Returns (bit, continuous):
            bit: 1 if schema direction confirmed, 0 if rejected, None if N/A
            continuous: schema_sim - global_sim (positive = confirmed)
        """
        if self._prev_accumulated is None or self._prev_global_centroid is None:
            return None, None

        if not self._prev_converged:
            return None, None  # no schema to confirm/reject

        schema_sim = cosine_sim(current_turn_embedding, self._prev_accumulated)
        global_sim = cosine_sim(current_turn_embedding, self._prev_global_centroid)

        continuous = schema_sim - global_sim
        bit = 1 if continuous > 0 else 0

        return bit, round(continuous, 4)

    def _compute_global_centroid(self) -> Optional[np.ndarray]:
        """Mean of all turn embeddings so far."""
        if not self.turn_embeddings:
            return None
        return np.mean(self.turn_embeddings, axis=0)

    def get_state_snapshot(self) -> Dict:
        """Full state for temp file persistence."""
        return {
            "turn_count": self.turn_count,
            "is_converged": self.is_converged,
            "converged_tree_id": self.converged_tree_id,
            "turns_since_convergence": self.turns_since_convergence,
            "num_trees": len(self.trees),
            "convergence_history": [round(c, 4) for c in self.convergence_history],
            "divergence_history": [round(d, 4) for d in self.divergence_history],
            "depth_history": [round(d, 4) for d in self.depth_history],
            "direction_history": [round(d, 4) for d in self.direction_history],
            "frustration_stage": self.frustration.stage,
        }
