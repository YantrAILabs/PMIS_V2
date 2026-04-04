"""
Epistemic Action Scorer for PMIS v2.

When gamma is low (exploration/balanced mode) and the system isn't sure
which Context the user is working in, this scorer identifies the single
question whose answer would MOST reduce uncertainty.

Implements: argmax_q E[KL(posterior || prior)] over candidate questions.

Triggering condition:
  - gamma < 0.6 (not in pure exploit mode)
  - 2+ candidate Contexts with similar scores in retrieved results
  - NOT in orphan territory (no context to disambiguate against)
"""

import numpy as np
from typing import List, Dict, Any, Optional
from db.manager import DBManager
from core.surprise import compute_raw_surprise


class EpistemicScorer:
    """
    Given ambiguity between multiple plausible Contexts, identifies
    the clarifying question whose answer would most reduce uncertainty.
    """

    def __init__(self, db: DBManager, hyperparams: Optional[Dict] = None):
        self.db = db
        self.hp = hyperparams or {}

    def score(
        self,
        retrieved_memories: List[Dict[str, Any]],
        query_embedding: np.ndarray,
        gamma: float,
        is_orphan_territory: bool,
    ) -> List[Dict[str, Any]]:
        """
        Main entry point. Returns ranked epistemic questions (0-3).

        Only triggers when:
          1. gamma < 0.6 (exploration or balanced mode)
          2. Not orphan territory (need at least one Context to compare)
          3. 2+ Contexts appear in retrieved results with close scores
        """
        # Guard: don't trigger in exploit mode
        if gamma > 0.6:
            return []

        # Guard: orphan territory = no contexts to disambiguate
        if is_orphan_territory:
            return []

        # Find distinct Contexts among retrieved results
        candidate_contexts = self._extract_candidate_contexts(
            retrieved_memories, query_embedding
        )

        if len(candidate_contexts) < 2:
            return []

        # Check if the top contexts are actually ambiguous (close scores)
        if not self._is_ambiguous(candidate_contexts):
            return []

        # Compute prior distribution over candidate Contexts
        prior = self._compute_prior(candidate_contexts)

        # Find distinguishing anchors (belong to one Context but not others)
        distinguishing = self._find_distinguishing_anchors(candidate_contexts)

        if not distinguishing:
            return []

        # Score each distinguishing anchor by expected information gain
        scored = []
        for anchor in distinguishing:
            info_gain = self._compute_information_gain(
                anchor, candidate_contexts, prior
            )
            question = self._generate_question(anchor, candidate_contexts)
            scored.append({
                "question": question,
                "anchor_id": anchor["id"],
                "anchor_content": anchor.get("content", "")[:150],
                "parent_context_id": anchor["_parent_ctx_id"],
                "parent_context_name": anchor["_parent_ctx_name"],
                "information_gain": info_gain,
                "candidate_contexts": [
                    {"id": c["id"], "name": c.get("content", "")[:80], "prior": float(p)}
                    for c, p in zip(candidate_contexts, prior)
                ],
            })

        scored.sort(key=lambda x: x["information_gain"], reverse=True)
        return scored[:3]

    def _extract_candidate_contexts(
        self,
        retrieved_memories: List[Dict[str, Any]],
        query_embedding: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Extract distinct Contexts from retrieved results.

        Two sources:
          1. CTX-level nodes directly in results
          2. Parent contexts of ANC-level nodes in results
        """
        context_ids_seen = set()
        contexts = []

        for mem in retrieved_memories:
            # Direct CTX in results
            if mem.get("level") == "CTX" and mem["id"] not in context_ids_seen:
                context_ids_seen.add(mem["id"])
                mem["_retrieval_score"] = mem.get("final_score", 0.5)
                contexts.append(mem)

            # Parent of ANC in results
            elif mem.get("level") == "ANC":
                parent_ids = mem.get("parent_ids", "[]")
                if isinstance(parent_ids, str):
                    import json
                    try:
                        parent_ids = json.loads(parent_ids)
                    except (json.JSONDecodeError, TypeError):
                        parent_ids = []

                for pid in parent_ids:
                    if pid not in context_ids_seen:
                        ctx_node = self.db.get_node(pid)
                        if ctx_node and ctx_node.get("level") == "CTX":
                            context_ids_seen.add(pid)
                            # Score this context by its best child's score
                            ctx_node["_retrieval_score"] = mem.get("final_score", 0.5)
                            contexts.append(ctx_node)

        return contexts

    def _is_ambiguous(self, contexts: List[Dict[str, Any]]) -> bool:
        """
        Check if the top Contexts have similar enough scores
        to be genuinely ambiguous (not one clear winner).
        """
        if len(contexts) < 2:
            return False

        scores = sorted(
            [c.get("_retrieval_score", 0.5) for c in contexts],
            reverse=True,
        )

        # Ambiguous if top-2 scores are within 30% of each other
        gap = scores[0] - scores[1]
        threshold = scores[0] * 0.30
        return gap < threshold

    def _compute_prior(self, contexts: List[Dict[str, Any]]) -> np.ndarray:
        """
        Prior probability distribution over Contexts based on
        their retrieval scores (softmax).
        """
        scores = np.array([c.get("_retrieval_score", 0.5) for c in contexts])
        # Softmax with temperature
        exp_scores = np.exp((scores - np.max(scores)) * 3.0)
        prior = exp_scores / (exp_scores.sum() + 1e-10)
        return np.clip(prior, 0.01, 0.99)

    def _find_distinguishing_anchors(
        self, contexts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find Anchors that belong to exactly ONE candidate Context.
        These are the features that distinguish one Context from another.
        """
        # Get children of each context
        context_children: Dict[str, set] = {}
        all_anchors = []

        for ctx in contexts:
            children = self.db.get_children(ctx["id"])
            context_children[ctx["id"]] = set(c["id"] for c in children)
            for child in children:
                child["_parent_ctx_id"] = ctx["id"]
                child["_parent_ctx_name"] = ctx.get("content", "")[:80]
                all_anchors.append(child)

        # Keep only anchors that belong to exactly one candidate Context
        distinguishing = []
        for anchor in all_anchors:
            parent = anchor["_parent_ctx_id"]
            in_other = any(
                anchor["id"] in children_set
                for ctx_id, children_set in context_children.items()
                if ctx_id != parent
            )
            if not in_other:
                distinguishing.append(anchor)

        # Sort by access_count descending (prefer well-tested anchors)
        distinguishing.sort(
            key=lambda a: a.get("access_count", 0), reverse=True
        )
        return distinguishing[:10]  # Cap at 10 to limit computation

    def _compute_information_gain(
        self,
        anchor: Dict[str, Any],
        contexts: List[Dict[str, Any]],
        prior: np.ndarray,
    ) -> float:
        """
        Expected information gain if we asked about this anchor's topic.

        If anchor belongs to Context_i:
          P(yes) ≈ prior[i]  (user is in that context)
          P(no)  ≈ 1 - prior[i]

        If yes → posterior concentrates on Context_i
        If no  → posterior spreads away from Context_i

        Expected IG = P(yes) × KL(post_yes || prior) + P(no) × KL(post_no || prior)
        """
        # Find which context this anchor belongs to
        parent_idx = None
        for i, ctx in enumerate(contexts):
            if ctx["id"] == anchor.get("_parent_ctx_id"):
                parent_idx = i
                break

        if parent_idx is None:
            return 0.0

        n = len(contexts)
        p_yes = float(prior[parent_idx])
        p_no = 1.0 - p_yes

        # Posterior if user confirms (yes): strongly concentrate on parent context
        post_yes = np.full(n, 0.05 / max(n - 1, 1))
        post_yes[parent_idx] = 0.95
        post_yes = post_yes / (post_yes.sum() + 1e-10)

        # Posterior if user denies (no): shift away from parent context
        post_no = prior.copy()
        post_no[parent_idx] *= 0.1
        post_no = post_no / (post_no.sum() + 1e-10)

        # Expected KL divergence
        kl_yes = self._kl_divergence(post_yes, prior)
        kl_no = self._kl_divergence(post_no, prior)

        return p_yes * kl_yes + p_no * kl_no

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """KL(p || q) with smoothing."""
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        return float(np.sum(p * np.log(p / q)))

    @staticmethod
    def _generate_question(
        anchor: Dict[str, Any],
        contexts: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a clarifying question from an anchor.

        The question asks whether the user's topic relates to this
        specific anchor's content, which would disambiguate between Contexts.
        """
        content = anchor.get("content", "")[:100]
        parent_name = anchor.get("_parent_ctx_name", "")[:50]

        # Find the OTHER context(s) to contrast against
        other_names = [
            c.get("content", "")[:50]
            for c in contexts
            if c["id"] != anchor.get("_parent_ctx_id")
        ]
        other_str = other_names[0] if other_names else "something else"

        return (
            f"Are you working on \"{parent_name}\" "
            f"(e.g., {content}), or more about \"{other_str}\"?"
        )
