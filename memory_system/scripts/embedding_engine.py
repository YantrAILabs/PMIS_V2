#!/usr/bin/env python3
"""
PMIS Neural Embedding Engine
Uses all-MiniLM-L6-v2 (384-dim) for semantic search.
Stores embeddings in Vector_DB/ as numpy arrays.
"""

import os
import json
import numpy as np
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
VECTOR_DB = ROOT / "Vector_DB"
GRAPH_DB = ROOT / "Graph_DB" / "graph.db"
INDEX_FILE = VECTOR_DB / "embeddings.npz"
META_FILE = VECTOR_DB / "meta.json"

# Lazy load model
_model = None

def get_model():
    global _model
    if _model is None:
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def encode_texts(texts, batch_size=64):
    """Encode a list of texts into 384-dim vectors."""
    model = get_model()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)


def build_index():
    """Build vector index from all anchors + contexts + SCs in graph.db."""
    VECTOR_DB.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(GRAPH_DB))
    conn.row_factory = sqlite3.Row

    # Get all nodes with text
    rows = conn.execute("""
        SELECT id, type, title, content, description
        FROM nodes
        ORDER BY type, id
    """).fetchall()

    node_ids = []
    node_types = []
    texts = []

    for r in rows:
        nid = r["id"]
        ntype = r["type"]
        title = r["title"] or ""
        content = r["content"] or ""
        desc = r["description"] or ""

        # Build rich text for embedding
        if ntype == "anchor":
            text = f"{title}. {content}" if content and content != title else title
        elif ntype == "context":
            text = f"{title}. {desc}" if desc else title
        else:  # super_context
            text = f"{title}. {desc}" if desc else title

        node_ids.append(nid)
        node_types.append(ntype)
        texts.append(text)

    conn.close()

    if not texts:
        print(json.dumps({"error": "No nodes to index"}))
        return

    # Encode all texts
    embeddings = encode_texts(texts)

    # Save
    np.savez_compressed(str(INDEX_FILE),
                        embeddings=embeddings,
                        node_ids=np.array(node_ids),
                        node_types=np.array(node_types))

    # Save metadata
    meta = {
        "total_nodes": len(texts),
        "embedding_dim": embeddings.shape[1],
        "model": "all-MiniLM-L6-v2",
        "by_type": {
            "super_context": sum(1 for t in node_types if t == "super_context"),
            "context": sum(1 for t in node_types if t == "context"),
            "anchor": sum(1 for t in node_types if t == "anchor"),
        }
    }
    META_FILE.write_text(json.dumps(meta, indent=2))

    print(json.dumps({
        "indexed": True,
        "total": len(texts),
        "dim": int(embeddings.shape[1]),
        **meta["by_type"]
    }))
    return embeddings, node_ids, node_types


class EmbeddingSearch:
    """Load pre-built index and search by cosine similarity."""

    def __init__(self):
        self.embeddings = None
        self.node_ids = None
        self.node_types = None
        self.ready = False
        self._load()

    def _load(self):
        if INDEX_FILE.exists():
            data = np.load(str(INDEX_FILE), allow_pickle=True)
            self.embeddings = data["embeddings"]
            self.node_ids = list(data["node_ids"])
            self.node_types = list(data["node_types"])
            self.ready = True

    def search(self, query, top_k=20, type_filter=None):
        """Search for similar nodes. Returns list of (node_id, score)."""
        if not self.ready:
            return []

        # Encode query
        q_vec = encode_texts([query])[0]

        # Cosine similarity (embeddings are already normalized)
        scores = self.embeddings @ q_vec

        # Filter by type if specified
        if type_filter:
            mask = np.array([t in type_filter for t in self.node_types])
            scores = scores * mask

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.05:  # minimum threshold
                results.append((self.node_ids[idx], float(scores[idx]), self.node_types[idx]))

        return results

    def search_sessions(self, query, session_texts, top_k=10):
        """Search through session texts (for LongMemEval benchmark).
        Returns list of (session_index, score)."""
        if not session_texts:
            return []

        q_vec = encode_texts([query])[0]
        s_vecs = encode_texts(session_texts)

        scores = s_vecs @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((int(idx), float(scores[idx])))

        return results


def add_single(node_id, text):
    """Add a single new embedding to the index (called at store time)."""
    if not INDEX_FILE.exists():
        build_index()
        return

    data = np.load(str(INDEX_FILE), allow_pickle=True)
    existing_embs = data["embeddings"]
    existing_ids = list(data["node_ids"])
    existing_types = list(data["node_types"])

    # Skip if already indexed
    if node_id in existing_ids:
        return

    # Get node type from DB
    conn = sqlite3.connect(str(GRAPH_DB))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT type FROM nodes WHERE id=?", (node_id,)).fetchone()
    conn.close()
    ntype = row["type"] if row else "anchor"

    # Encode new text
    new_emb = encode_texts([text])

    # Append
    updated_embs = np.vstack([existing_embs, new_emb])
    updated_ids = np.array(existing_ids + [node_id])
    updated_types = np.array(existing_types + [ntype])

    np.savez_compressed(str(INDEX_FILE),
                        embeddings=updated_embs,
                        node_ids=updated_ids,
                        node_types=updated_types)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    elif len(sys.argv) > 2 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:])
        searcher = EmbeddingSearch()
        results = searcher.search(query, top_k=10)
        for nid, score, ntype in results:
            print(f"  {score:.3f}  [{ntype}]  {nid}")
    else:
        print("Usage:")
        print("  python3 embedding_engine.py build    # Build index from graph.db")
        print("  python3 embedding_engine.py search <query>  # Search")
