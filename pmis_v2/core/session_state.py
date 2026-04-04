"""
Session State for PMIS v2.

Maintains live, in-memory state for the current conversation.
Tracks surprise history, gamma history, active tree, last node IDs,
and the running sequence of turn embeddings for predictive linking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class TurnRecord:
    turn_number: int
    role: str                          # "user" or "assistant"
    content_preview: str               # First 200 chars
    embedding: Optional[np.ndarray]    # Euclidean embedding of this turn
    node_id: Optional[str]             # ID of stored MemoryNode (None if skipped)
    gamma: float
    effective_surprise: float
    mode: str
    timestamp: datetime = field(default_factory=datetime.now)


class SessionState:
    """
    Live state for one conversation session.
    Created fresh for each conversation. Not persisted between conversations
    (conversation_turns table handles persistence).
    """

    def __init__(self, conversation_id: str, buffer_size: int = 50):
        self.conversation_id = conversation_id
        self.buffer_size = buffer_size
        self.started_at = datetime.now()

        # Turn tracking
        self.turns: List[TurnRecord] = []
        self.turn_counter: int = 0

        # Surprise / gamma history for staleness detection
        self.surprise_history: List[float] = []
        self.gamma_history: List[float] = []

        # Gamma override (set by /memory explore and /memory exploit)
        self.gamma_override: Optional[float] = None
        self.gamma_override_turns_remaining: int = 0

        # Active tree context (auto-detected or manually set)
        self.active_tree_id: Optional[str] = None
        self.active_context_id: Optional[str] = None

        # Last stored node IDs (for sequence linking)
        self.last_stored_node_id: Optional[str] = None
        self.stored_node_ids: List[str] = []

    def record_turn(
        self,
        role: str,
        content: str,
        embedding: Optional[np.ndarray],
        node_id: Optional[str],
        gamma: float,
        effective_surprise: float,
        mode: str,
    ):
        """Record a conversation turn."""
        self.turn_counter += 1
        record = TurnRecord(
            turn_number=self.turn_counter,
            role=role,
            content_preview=content[:200],
            embedding=embedding,
            node_id=node_id,
            gamma=gamma,
            effective_surprise=effective_surprise,
            mode=mode,
        )
        self.turns.append(record)

        # Maintain buffer size
        if len(self.turns) > self.buffer_size:
            self.turns = self.turns[-self.buffer_size:]

        # Track surprise and gamma
        self.surprise_history.append(effective_surprise)
        self.gamma_history.append(gamma)
        if len(self.surprise_history) > self.buffer_size:
            self.surprise_history = self.surprise_history[-self.buffer_size:]
            self.gamma_history = self.gamma_history[-self.buffer_size:]

        # Track stored nodes for sequence linking
        if node_id:
            prev = self.last_stored_node_id
            self.last_stored_node_id = node_id
            self.stored_node_ids.append(node_id)
            return prev  # Return previous node ID for PRECEDED_BY linking

        return None

    @property
    def recent_surprises(self) -> List[float]:
        return self.surprise_history

    @property
    def avg_gamma(self) -> float:
        if not self.gamma_history:
            return 0.5
        return float(np.mean(self.gamma_history))

    @property
    def last_user_embedding(self) -> Optional[np.ndarray]:
        """Get the most recent user turn's embedding."""
        for turn in reversed(self.turns):
            if turn.role == "user" and turn.embedding is not None:
                return turn.embedding
        return None

    def set_active_tree(self, tree_id: str):
        self.active_tree_id = tree_id

    def set_active_context(self, context_id: str):
        self.active_context_id = context_id

    def to_log_dicts(self) -> List[Dict[str, Any]]:
        """Serialize turns for database persistence."""
        return [
            {
                "conversation_id": self.conversation_id,
                "turn_number": t.turn_number,
                "role": t.role,
                "content_hash": t.node_id,
                "node_id": t.node_id,
                "gamma": t.gamma,
                "effective_surprise": t.effective_surprise,
                "mode": t.mode,
                "timestamp": t.timestamp.isoformat(),
            }
            for t in self.turns
        ]
