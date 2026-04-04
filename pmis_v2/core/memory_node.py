"""
MemoryNode: The fundamental data unit of PMIS v2.

Each node stores a triple embedding (euclidean + hyperbolic + temporal),
structural metadata (level, parents, relations, trees), surprise/precision
scores, and access tracking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
import hashlib
import json


class MemoryLevel(Enum):
    SUPER_CONTEXT = "SC"
    CONTEXT = "CTX"
    ANCHOR = "ANC"


class RelationType(Enum):
    CHILD_OF = "child_of"
    RELATED_TO = "related_to"
    PRECEDED_BY = "preceded_by"
    FOLLOWED_BY = "followed_by"
    CO_OCCURRED = "co_occurred"
    SIMILAR_TO = "similar_to"


@dataclass
class Relation:
    source_id: str
    target_id: str
    relation_type: RelationType
    tree_id: str
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "tree_id": self.tree_id,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
        }

    @staticmethod
    def from_dict(d: dict) -> "Relation":
        return Relation(
            source_id=d["source_id"],
            target_id=d["target_id"],
            relation_type=RelationType(d["relation_type"]),
            tree_id=d["tree_id"],
            weight=d.get("weight", 1.0),
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now(),
        )


@dataclass
class AccessPattern:
    count: int = 0
    last_accessed: Optional[datetime] = None
    access_history: List[datetime] = field(default_factory=list)
    decay_rate: float = 0.5

    def record_access(self):
        self.count += 1
        now = datetime.now()
        self.last_accessed = now
        self.access_history.append(now)
        if len(self.access_history) > 100:
            self.access_history = self.access_history[-100:]

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_history": [a.isoformat() for a in self.access_history[-20:]],
            "decay_rate": self.decay_rate,
        }

    @staticmethod
    def from_dict(d: dict) -> "AccessPattern":
        history = []
        for ts in d.get("access_history", []):
            try:
                history.append(datetime.fromisoformat(ts))
            except (ValueError, TypeError):
                pass
        return AccessPattern(
            count=d.get("count", 0),
            last_accessed=datetime.fromisoformat(d["last_accessed"]) if d.get("last_accessed") else None,
            access_history=history,
            decay_rate=d.get("decay_rate", 0.5),
        )


@dataclass
class MemoryNode:
    # Identity
    id: str
    content: str
    source_conversation_id: str = ""

    # Triple embedding
    euclidean_embedding: Optional[np.ndarray] = None   # float[1536]
    hyperbolic_coords: Optional[np.ndarray] = None     # float[32]
    temporal_embedding: Optional[np.ndarray] = None    # float[16]

    # Structural metadata
    level: MemoryLevel = MemoryLevel.ANCHOR
    parent_ids: List[str] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    tree_ids: List[str] = field(default_factory=list)

    # Surprise & precision
    precision: float = 0.5
    surprise_at_creation: float = 0.0

    # Temporal metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    era: str = ""

    # Access tracking
    access_pattern: AccessPattern = field(default_factory=AccessPattern)

    # Status
    is_orphan: bool = False
    is_tentative: bool = False
    is_deleted: bool = False

    # --- Factory methods ---

    @staticmethod
    def generate_id(content: str) -> str:
        """SHA256 hash of content + timestamp, truncated to 16 chars.
        Timestamp ensures unique IDs even for identical content from different conversations."""
        unique_input = content + "|" + datetime.now().isoformat()
        return hashlib.sha256(unique_input.encode()).hexdigest()[:16]

    @classmethod
    def create(
        cls,
        content: str,
        level: MemoryLevel,
        euclidean_embedding: np.ndarray,
        hyperbolic_coords: np.ndarray,
        temporal_embedding: np.ndarray,
        source_conversation_id: str = "",
        surprise: float = 0.0,
        precision: float = 0.5,
        era: str = "",
    ) -> "MemoryNode":
        """Factory method for creating a new MemoryNode with all embeddings."""
        node_id = cls.generate_id(content)
        return cls(
            id=node_id,
            content=content,
            source_conversation_id=source_conversation_id,
            euclidean_embedding=euclidean_embedding,
            hyperbolic_coords=hyperbolic_coords,
            temporal_embedding=temporal_embedding,
            level=level,
            surprise_at_creation=surprise,
            precision=precision,
            era=era,
            is_orphan=(level == MemoryLevel.ANCHOR),
            is_tentative=(surprise > 0.5),
        )

    # --- Computed properties ---

    @property
    def hierarchy_level_from_norm(self) -> float:
        """Derive hierarchy level from hyperbolic norm. 0=abstract, 1=specific."""
        if self.hyperbolic_coords is None:
            return 0.5
        return float(np.linalg.norm(self.hyperbolic_coords))

    @property
    def temporal_weight(self) -> float:
        """Power-law decay + reactivation boost + surprise shield."""
        now = datetime.now()
        age_hours = max((now - self.created_at).total_seconds() / 3600, 0.001)

        # Base decay: power law
        base_decay = (1 + age_hours) ** (-self.access_pattern.decay_rate)

        # Reactivation: each access boosts weight
        reactivation = 0.0
        for access in self.access_pattern.access_history[-20:]:
            access_age = max((now - access).total_seconds() / 3600, 0.001)
            reactivation += (1 + access_age) ** (-0.3)

        # Surprise shield: high-surprise memories resist decay
        surprise_shield = self.surprise_at_creation * 0.5

        return base_decay + reactivation + surprise_shield

    # --- Serialization ---

    def to_db_dict(self) -> Dict[str, Any]:
        """Serialize for SQLite storage (embeddings stored separately)."""
        return {
            "id": self.id,
            "content": self.content,
            "source_conversation_id": self.source_conversation_id,
            "level": self.level.value,
            "parent_ids": json.dumps(self.parent_ids),
            "tree_ids": json.dumps(self.tree_ids),
            "precision": self.precision,
            "surprise_at_creation": self.surprise_at_creation,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "era": self.era,
            "access_count": self.access_pattern.count,
            "last_accessed": self.access_pattern.last_accessed.isoformat() if self.access_pattern.last_accessed else None,
            "decay_rate": self.access_pattern.decay_rate,
            "is_orphan": self.is_orphan,
            "is_tentative": self.is_tentative,
            "is_deleted": self.is_deleted,
        }
