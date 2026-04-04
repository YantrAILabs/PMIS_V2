"""
Tree Resolver for PMIS v2.

Determines which tree(s) a query belongs to by examining:
  1. Explicit signals (user mentions a project/domain name)
  2. Session continuity (same tree as recent turns)
  3. Semantic proximity (which tree's root/Context is closest)
"""

import numpy as np
from typing import Optional, List, Dict, Any

from core.surprise import compute_raw_surprise
from db.manager import DBManager
from core.session_state import SessionState


class TreeResolver:
    def __init__(self, db: DBManager):
        self.db = db
        # Keywords that map to known trees (populated from tree registry)
        self._keyword_cache: Dict[str, str] = {}

    def resolve(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        session: SessionState,
    ) -> Optional[str]:
        """
        Resolve which tree a query belongs to.
        Returns tree_id or None if ambiguous.
        """
        # Priority 1: Explicit keyword match
        tree_id = self._match_keywords(query_text)
        if tree_id:
            return tree_id

        # Priority 2: Session continuity — if the session is already in a tree,
        # stay there unless surprise is very high (topic changed)
        if session.active_tree_id:
            # Check if last few turns had consistent tree
            recent_gammas = session.gamma_history[-3:]
            if recent_gammas and np.mean(recent_gammas) > 0.5:
                # Still in exploit mode → same tree
                return session.active_tree_id

        # Priority 3: Semantic proximity to tree roots
        tree_id = self._match_by_root_proximity(query_embedding)
        if tree_id:
            return tree_id

        return None

    def _match_keywords(self, text: str) -> Optional[str]:
        """Match query text against known tree keywords."""
        if not self._keyword_cache:
            self._build_keyword_cache()

        text_lower = text.lower()
        for keyword, tree_id in self._keyword_cache.items():
            if keyword in text_lower:
                return tree_id
        return None

    def _build_keyword_cache(self):
        """Build keyword → tree_id mapping from tree registry."""
        trees = self.db.get_all_trees()
        for tree in trees:
            name = tree.get("name", "").lower()
            tree_id = tree["tree_id"]
            # Add tree name and its words as keywords
            self._keyword_cache[name] = tree_id
            for word in name.split():
                if len(word) > 3:  # Skip short words
                    self._keyword_cache[word] = tree_id

    def _match_by_root_proximity(self, query_embedding: np.ndarray) -> Optional[str]:
        """Find the tree whose root node is semantically closest."""
        trees = self.db.get_all_trees()
        if not trees:
            return None

        best_tree = None
        best_dist = float("inf")

        for tree in trees:
            root_id = tree.get("root_node_id")
            if not root_id:
                continue

            embs = self.db.get_embeddings(root_id)
            root_emb = embs.get("euclidean")
            if root_emb is None:
                continue

            dist = compute_raw_surprise(query_embedding, root_emb)
            if dist < best_dist:
                best_dist = dist
                best_tree = tree["tree_id"]

        # Only return if reasonably close (< 0.6 cosine distance)
        if best_dist < 0.6:
            return best_tree
        return None

    def invalidate_cache(self):
        """Clear keyword cache (call after tree changes)."""
        self._keyword_cache = {}
