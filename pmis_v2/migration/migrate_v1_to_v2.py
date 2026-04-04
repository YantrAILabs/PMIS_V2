"""
Migration: PMIS v1 → v2

Migrates existing ChromaDB data into the new schema:
  1. Read all existing memories from v1 ChromaDB
  2. For each memory: generate hyperbolic coords, temporal embedding
  3. Compute initial precision and surprise scores
  4. Assign to trees based on existing parent_id relationships
  5. Write to new SQLite + embeddings tables

Usage:
  python -m migration.migrate_v1_to_v2 --v1-chroma-path ./old_chroma --v2-db-path ./data/memory.db
"""

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# These imports assume pmis_v2 is on the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory_node import MemoryNode, MemoryLevel
from core.poincare import ProjectionManager, assign_hyperbolic_coords
from core.temporal import temporal_encode, compute_era
from core import config
from db.manager import DBManager


def migrate(
    v1_chroma_path: str,
    v2_db_path: str = "data/memory.db",
    config_path: str = None,
):
    """
    Main migration function.
    """
    hp = config.load_config(config_path) if config_path else config.get_all()

    print(f"[Migration] Loading v1 data from: {v1_chroma_path}")
    v1_data = load_v1_data(v1_chroma_path)
    print(f"[Migration] Found {len(v1_data)} memories to migrate")

    # Initialize v2 database
    db = DBManager(v2_db_path)

    # Initialize projection manager
    input_dim = hp.get("local_embedding_dimensions", 768) if hp.get("use_local", True) else hp.get("embedding_dimensions", 1536)
    projection = ProjectionManager(input_dim=input_dim, output_dim=hp.get("poincare_dimensions", 32))

    # Track trees and parents for relationship creation
    tree_registry: Dict[str, str] = {}  # tree_name → tree_id

    migrated = 0
    skipped = 0

    for item in v1_data:
        try:
            node = convert_v1_to_v2(item, projection, hp)
            if node is None:
                skipped += 1
                continue

            db.create_node(node)

            # Handle parent relationships
            parent_id = item.get("parent_id")
            if parent_id:
                tree_id = item.get("tree_id", "migrated_default")
                db.create_relation(node.id, parent_id, "child_of", tree_id)

                if tree_id not in tree_registry:
                    tree_registry[tree_id] = tree_id
                    db.create_tree(tree_id, name=tree_id, description="Migrated from v1")

            migrated += 1

            if migrated % 50 == 0:
                print(f"[Migration] Migrated {migrated}/{len(v1_data)}")

        except Exception as e:
            print(f"[Migration] Error migrating item {item.get('id', '?')}: {e}")
            skipped += 1

    # Save projection matrix
    proj_path = str(Path(v2_db_path).parent / "projection_matrix.npy")
    projection.save(proj_path)

    print(f"[Migration] Complete. Migrated: {migrated}, Skipped: {skipped}")
    print(f"[Migration] Trees created: {list(tree_registry.keys())}")
    print(f"[Migration] Projection matrix saved to: {proj_path}")

    return {"migrated": migrated, "skipped": skipped, "trees": list(tree_registry.keys())}


def load_v1_data(chroma_path: str) -> List[Dict[str, Any]]:
    """
    Load data from v1 ChromaDB.
    Supports both persistent ChromaDB and JSON export formats.
    """
    path = Path(chroma_path)

    # Case 1: JSON export file
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "memories" in data:
            return data["memories"]
        return []

    # Case 2: ChromaDB persistent directory
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(path))
        collections = client.list_collections()
        all_items = []

        for collection in collections:
            col = client.get_collection(collection.name)
            result = col.get(include=["embeddings", "metadatas", "documents"])

            for i in range(len(result["ids"])):
                item = {
                    "id": result["ids"][i],
                    "content": result["documents"][i] if result["documents"] else "",
                    "embedding": result["embeddings"][i] if result["embeddings"] else None,
                    "collection": collection.name,
                }
                if result["metadatas"] and result["metadatas"][i]:
                    item.update(result["metadatas"][i])
                all_items.append(item)

        return all_items

    except ImportError:
        print("[Migration] chromadb not installed. Export to JSON first.")
        return []
    except Exception as e:
        print(f"[Migration] Error reading ChromaDB: {e}")
        return []


def convert_v1_to_v2(
    item: Dict[str, Any],
    projection: ProjectionManager,
    hp: Dict[str, Any],
) -> Optional[MemoryNode]:
    """Convert a single v1 memory to a v2 MemoryNode."""
    content = item.get("content") or item.get("document") or item.get("text", "")
    if not content or len(content.strip()) < 5:
        return None

    # Determine level from v1 metadata
    v1_level = item.get("level", item.get("type", "anchor")).upper()
    if v1_level in ("SC", "SUPER_CONTEXT"):
        level = MemoryLevel.SUPER_CONTEXT
        level_str = "SC"
    elif v1_level in ("CTX", "CONTEXT"):
        level = MemoryLevel.CONTEXT
        level_str = "CTX"
    else:
        level = MemoryLevel.ANCHOR
        level_str = "ANC"

    # Euclidean embedding
    embedding = item.get("embedding")
    if embedding is None:
        return None  # Can't migrate without embedding
    euclidean = np.array(embedding, dtype=np.float32)

    # Handle dimension mismatch with projection manager
    if len(euclidean) != projection.input_dim:
        # Pad or truncate
        if len(euclidean) < projection.input_dim:
            euclidean = np.pad(euclidean, (0, projection.input_dim - len(euclidean)))
        else:
            euclidean = euclidean[:projection.input_dim]

    # Hyperbolic coordinates
    hyperbolic = assign_hyperbolic_coords(
        euclidean_embedding=euclidean,
        level=level_str,
        projection_manager=projection,
        hyperparams=hp,
    )

    # Temporal embedding
    created_str = item.get("created_at", item.get("timestamp"))
    if created_str:
        try:
            created_at = datetime.fromisoformat(str(created_str))
        except (ValueError, TypeError):
            created_at = datetime.now()
    else:
        created_at = datetime.now()

    temporal = temporal_encode(created_at, dim=hp.get("temporal_embedding_dim", 16))

    # Era
    era = compute_era(created_at, hp.get("era_boundaries", {}))

    # Create node
    node = MemoryNode.create(
        content=content,
        level=level,
        euclidean_embedding=euclidean,
        hyperbolic_coords=hyperbolic,
        temporal_embedding=temporal,
        source_conversation_id=item.get("conversation_id", "v1_migrated"),
        surprise=item.get("surprise", 0.3),    # Default mid-range for migrated
        precision=item.get("precision", 0.5),
        era=era,
    )
    node.is_orphan = (level == MemoryLevel.ANCHOR and not item.get("parent_id"))
    node.is_tentative = False  # Migrated data is trusted
    node.access_pattern.count = item.get("access_count", 1)

    return node


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate PMIS v1 to v2")
    parser.add_argument("--v1-chroma-path", required=True, help="Path to v1 ChromaDB or JSON export")
    parser.add_argument("--v2-db-path", default="data/memory.db", help="Path to v2 SQLite database")
    parser.add_argument("--config", default=None, help="Path to hyperparameters.yaml")
    args = parser.parse_args()

    migrate(args.v1_chroma_path, args.v2_db_path, args.config)
