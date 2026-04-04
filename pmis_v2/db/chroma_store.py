"""
ChromaDB Store for PMIS v2.

Parallel ANN index for euclidean embeddings. SQLite remains the source
of truth for metadata, hyperbolic coords, and temporal embeddings.
ChromaDB handles only the fast approximate nearest-neighbor search.

Auto-synced: every create_node/soft_delete in DBManager triggers
a corresponding add/delete here.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class ChromaStore:
    """Fast ANN index for euclidean embeddings using ChromaDB."""

    def __init__(self, persist_dir: str = "data/chroma", collection_name: str = "pmis_v2"):
        self.enabled = HAS_CHROMADB
        self._client = None
        self._collection = None
        self.persist_dir = persist_dir

        if self.enabled:
            try:
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=Settings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                print(f"[ChromaStore] Failed to initialize: {e}. Falling back to linear scan.")
                self.enabled = False

    def add(self, node_id: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Add or update a node's euclidean embedding in the ANN index."""
        if not self.enabled:
            return
        try:
            meta = {
                "level": str(metadata.get("level", "ANC")) if metadata else "ANC",
                "is_orphan": str(metadata.get("is_orphan", False)) if metadata else "False",
            }
            # Add tree_ids as comma-separated string (ChromaDB metadata must be scalar)
            tree_ids = metadata.get("tree_ids", []) if metadata else []
            if isinstance(tree_ids, list):
                meta["tree_ids"] = ",".join(str(t) for t in tree_ids) if tree_ids else ""
            elif isinstance(tree_ids, str):
                meta["tree_ids"] = tree_ids

            self._collection.upsert(
                ids=[node_id],
                embeddings=[embedding.tolist()],
                metadatas=[meta],
            )
        except Exception as e:
            print(f"[ChromaStore] Add error for {node_id}: {e}")

    def remove(self, node_id: str):
        """Remove a node from the ANN index."""
        if not self.enabled:
            return
        try:
            self._collection.delete(ids=[node_id])
        except Exception:
            pass  # Ignore if not found

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 20,
        level_filter: Optional[str] = None,
        tree_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fast ANN query. Returns candidate IDs + distances.
        
        This replaces the O(N) linear scan with O(log N) HNSW search.
        The caller then loads full metadata from SQLite for only these candidates.
        """
        if not self.enabled:
            return []

        try:
            if self.count() == 0:
                return []

            where_filters = {}
            if level_filter:
                where_filters["level"] = level_filter

            # ChromaDB where clause
            where = where_filters if where_filters else None

            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=max(1, min(n_results, self.count())),
                where=where,
                include=["distances", "metadatas"],
            )

            if not results or not results["ids"] or not results["ids"][0]:
                return []

            candidates = []
            for i, node_id in enumerate(results["ids"][0]):
                dist = results["distances"][0][i] if results["distances"] else 0.5
                meta = results["metadatas"][0][i] if results["metadatas"] else {}

                # Apply tree filter post-query (ChromaDB $contains not reliable)
                if tree_filter and meta.get("tree_ids", ""):
                    if tree_filter not in meta["tree_ids"]:
                        continue

                candidates.append({
                    "id": node_id,
                    "chroma_distance": dist,
                    "chroma_similarity": 1.0 - dist,
                    "chroma_metadata": meta,
                })

            return candidates

        except Exception as e:
            print(f"[ChromaStore] Query error: {e}")
            return []

    def count(self) -> int:
        if not self.enabled or not self._collection:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def rebuild_from_db(self, db_manager, batch_size: int = 100):
        """
        Rebuild the entire ChromaDB index from SQLite embeddings table.
        Used after migration or if index gets corrupted.
        """
        if not self.enabled:
            print("[ChromaStore] ChromaDB not available. Skipping rebuild.")
            return 0

        count = 0
        for level in ["SC", "CTX", "ANC"]:
            nodes = db_manager.get_nodes_by_level(level)
            ids_batch = []
            embs_batch = []
            metas_batch = []

            for node in nodes:
                if node.get("is_deleted"):
                    continue
                embs = db_manager.get_embeddings(node["id"])
                euc = embs.get("euclidean")
                if euc is None:
                    continue

                ids_batch.append(node["id"])
                embs_batch.append(euc.tolist())
                metas_batch.append({
                    "level": node.get("level", "ANC"),
                    "is_orphan": str(node.get("is_orphan", False)),
                    "tree_ids": ",".join(node.get("tree_ids", [])) if isinstance(node.get("tree_ids"), list) else str(node.get("tree_ids", "")),
                })

                if len(ids_batch) >= batch_size:
                    self._collection.upsert(ids=ids_batch, embeddings=embs_batch, metadatas=metas_batch)
                    count += len(ids_batch)
                    ids_batch, embs_batch, metas_batch = [], [], []

            # Flush remaining
            if ids_batch:
                self._collection.upsert(ids=ids_batch, embeddings=embs_batch, metadatas=metas_batch)
                count += len(ids_batch)

        print(f"[ChromaStore] Rebuilt index with {count} nodes")
        return count
