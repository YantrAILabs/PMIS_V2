"""
Sequence Linker for PMIS v2.

Creates PRECEDED_BY / FOLLOWED_BY temporal edges between
consecutively stored memory nodes within a conversation.
This implements the "predictive code" — not just what associates
with what, but what follows what.
"""

from typing import Optional
from db.manager import DBManager


def link_sequence(
    db: DBManager,
    current_node_id: str,
    previous_node_id: Optional[str],
    tree_id: str = "default",
):
    """
    Create bidirectional sequence edges between two consecutively
    stored memory nodes.

    current_node_id:  the node just stored
    previous_node_id: the node stored immediately before (from session state)
    """
    if previous_node_id is None:
        return

    if current_node_id == previous_node_id:
        return

    # Current is PRECEDED_BY previous
    db.create_relation(
        source_id=current_node_id,
        target_id=previous_node_id,
        relation_type="preceded_by",
        tree_id=tree_id,
    )

    # Previous is FOLLOWED_BY current
    db.create_relation(
        source_id=previous_node_id,
        target_id=current_node_id,
        relation_type="followed_by",
        tree_id=tree_id,
    )


def link_co_occurrence(
    db: DBManager,
    node_ids: list,
    conversation_id: str,
):
    """
    Create CO_OCCURRED edges between all nodes stored in the same conversation.
    Called at end of conversation or periodically.
    """
    for i, id_a in enumerate(node_ids):
        for id_b in node_ids[i + 1:]:
            db.create_relation(
                source_id=id_a,
                target_id=id_b,
                relation_type="co_occurred",
                tree_id=conversation_id,
                weight=0.5,
            )
