"""
Predictive Retrieval for PMIS v2.

Implements the "predictive code" from hippocampal research:
not just what associates with a query (associative), but what
typically follows it in a sequence (predictive).

Traverses FOLLOWED_BY edges to surface memories that historically
came after the current topic.
"""

from typing import List, Dict, Any, Optional
from db.manager import DBManager


class PredictiveRetriever:
    def __init__(self, db: DBManager):
        self.db = db

    def predict_next(
        self,
        current_node_id: str,
        depth: int = 2,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Given a current node, traverse FOLLOWED_BY edges to find
        what typically comes next in conversations.

        depth: how many hops to follow (1 = immediate next, 2 = next + one more)
        """
        results = []
        visited = {current_node_id}
        frontier = [current_node_id]

        for d in range(depth):
            next_frontier = []
            for node_id in frontier:
                successor = self.db.get_sequence_next(node_id)
                if successor and successor["id"] not in visited:
                    successor["_prediction_depth"] = d + 1
                    results.append(successor)
                    visited.add(successor["id"])
                    next_frontier.append(successor["id"])
            frontier = next_frontier

        return results[:max_results]

    def predict_from_context(
        self,
        context_id: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Given a Context, find the most common sequence patterns
        among its children. Returns nodes that most frequently
        follow any child of this Context.
        """
        children = self.db.get_children(context_id)
        if not children:
            return []

        # Collect all FOLLOWED_BY targets from children
        follower_counts: Dict[str, int] = {}
        follower_nodes: Dict[str, Dict[str, Any]] = {}

        for child in children:
            successor = self.db.get_sequence_next(child["id"])
            if successor and successor["id"] != context_id:
                fid = successor["id"]
                follower_counts[fid] = follower_counts.get(fid, 0) + 1
                follower_nodes[fid] = successor

        # Sort by frequency
        sorted_followers = sorted(
            follower_counts.items(), key=lambda x: x[1], reverse=True
        )

        results = []
        for fid, count in sorted_followers[:max_results]:
            node = follower_nodes[fid]
            node["_prediction_frequency"] = count
            results.append(node)

        return results

    def get_conversation_trajectory(
        self,
        start_node_id: str,
        max_length: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct a conversation trajectory by following FOLLOWED_BY
        edges from a starting node.
        """
        trajectory = []
        current_id = start_node_id
        visited = set()

        for _ in range(max_length):
            if current_id in visited:
                break
            visited.add(current_id)

            node = self.db.get_node(current_id)
            if not node:
                break
            trajectory.append(node)

            successor = self.db.get_sequence_next(current_id)
            if not successor:
                break
            current_id = successor["id"]

        return trajectory
