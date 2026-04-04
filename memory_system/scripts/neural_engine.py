#!/usr/bin/env python3
"""
PMIS Neural Embedding Engine
==============================
Handles:
  1. Encoding text → 384-dim vectors (all-MiniLM-L6-v2)
  2. Storing embeddings to disk (Vector_DB/)
  3. Cosine similarity search at retrieval
  4. Lazy model loading (only loads when first needed)

Zero-config: works on Apple M2 with MPS, falls back to CPU.
"""

import json
import os
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
VECTOR_DB = ROOT / "Vector_DB"
EMBEDDINGS_FILE = VECTOR_DB / "embeddings.npz"
INDEX_FILE = VECTOR_DB / "index.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# ── Lazy model loading ──
_model = None
_model_loaded = False

def _get_model():
    """Load sentence-transformers model lazily (first call only)."""
    global _model, _model_loaded
    if _model_loaded:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        _model_loaded = True
        return _model
    except ImportError:
        _model_loaded = True
        _model = None
        return None


def is_available():
    """Check if neural engine can be used."""
    return _get_model() is not None


def encode(texts, show_progress=False):
    """Encode list of texts → numpy array (N, 384)."""
    model = _get_model()
    if model is None:
        return None
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    embs = model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
    return embs.astype(np.float32)


def encode_single(text):
    """Encode a single text → numpy array (384,)."""
    model = _get_model()
    if model is None:
        return None
    emb = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
    return emb[0].astype(np.float32)


def cosine_similarity(query_vec, matrix):
    """Cosine similarity between query (384,) and matrix (N, 384)."""
    if matrix is None or len(matrix) == 0:
        return np.array([])
    # Normalize
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    m_normed = matrix / m_norms
    return np.dot(m_normed, q_norm)


# ══════════════════════════════════════════════
# PERSISTENT STORAGE (Vector_DB/)
# ══════════════════════════════════════════════

def save_embeddings(node_ids, embeddings):
    """Save embeddings to disk. node_ids[i] maps to embeddings[i]."""
    VECTOR_DB.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(EMBEDDINGS_FILE), embeddings=embeddings)
    with open(INDEX_FILE, 'w') as f:
        json.dump({"node_ids": node_ids, "model": MODEL_NAME, "dim": 384}, f)


def load_embeddings():
    """Load embeddings from disk. Returns (node_ids, embeddings) or (None, None)."""
    if not EMBEDDINGS_FILE.exists() or not INDEX_FILE.exists():
        return None, None
    try:
        with open(INDEX_FILE) as f:
            index = json.load(f)
        data = np.load(str(EMBEDDINGS_FILE))
        return index["node_ids"], data["embeddings"]
    except Exception:
        return None, None


def has_stored_embeddings():
    """Check if pre-computed embeddings exist on disk."""
    return EMBEDDINGS_FILE.exists() and INDEX_FILE.exists()


# ══════════════════════════════════════════════
# BUILD / UPDATE INDEX
# ══════════════════════════════════════════════

def build_full_index(conn):
    """Build embeddings for ALL nodes and save to disk.

    Call this once after integration, or after bulk imports.
    Returns count of nodes indexed.
    """
    if not is_available():
        return 0

    import memory as mem

    all_nodes = conn.execute("SELECT * FROM nodes").fetchall()

    # Build enriched text for each node (same as p9_retrieve uses)
    edges = conn.execute("SELECT src, tgt FROM edges WHERE type='parent_child'").fetchall()
    parent_of = {e["tgt"]: e["src"] for e in edges}
    nodes_by_id = {n["id"]: dict(n) for n in all_nodes}

    node_ids = []
    texts = []

    for n in all_nodes:
        nid = n["id"]
        node_ids.append(nid)

        # Enriched text: SC title + Context title + Node title + content
        t = n["title"]
        pid = parent_of.get(nid)
        if pid and pid in nodes_by_id:
            t = nodes_by_id[pid]["title"] + " " + t
            gpid = parent_of.get(pid)
            if gpid and gpid in nodes_by_id:
                t = nodes_by_id[gpid]["title"] + " " + t
        if n["content"] and n["content"] != n["title"]:
            t += " " + n["content"][:200]
        if n["description"]:
            t += " " + n["description"][:100]
        texts.append(t)

    # Encode all
    embeddings = encode(texts, show_progress=True)
    if embeddings is None:
        return 0

    # Save to disk
    save_embeddings(node_ids, embeddings)

    return len(node_ids)


def add_to_index(conn, node_id):
    """Add a single new node to the existing index.

    Call this at store time for each new node.
    Much faster than rebuild — just appends one vector.
    """
    if not is_available():
        return False

    # Load existing
    existing_ids, existing_embs = load_embeddings()
    if existing_ids is None:
        # No existing index — need full build first
        return False

    # Skip if already indexed
    if node_id in existing_ids:
        return True

    # Get node text
    nodes_by_id = {}
    for n in conn.execute("SELECT * FROM nodes").fetchall():
        nodes_by_id[n["id"]] = dict(n)

    if node_id not in nodes_by_id:
        return False

    n = nodes_by_id[node_id]
    edges = conn.execute("SELECT src, tgt FROM edges WHERE type='parent_child'").fetchall()
    parent_of = {e["tgt"]: e["src"] for e in edges}

    # Enriched text
    t = n["title"]
    pid = parent_of.get(node_id)
    if pid and pid in nodes_by_id:
        t = nodes_by_id[pid]["title"] + " " + t
        gpid = parent_of.get(pid)
        if gpid and gpid in nodes_by_id:
            t = nodes_by_id[gpid]["title"] + " " + t
    if n.get("content") and n["content"] != n["title"]:
        t += " " + n["content"][:200]
    if n.get("description"):
        t += " " + n["description"][:100]

    # Encode single node
    new_emb = encode_single(t)
    if new_emb is None:
        return False

    # Append to existing
    updated_ids = list(existing_ids) + [node_id]
    updated_embs = np.vstack([existing_embs, new_emb.reshape(1, -1)])

    save_embeddings(updated_ids, updated_embs)
    return True


# ══════════════════════════════════════════════
# NEURAL VECTOR SEARCH (replaces TF-IDF VectorEngine)
# ══════════════════════════════════════════════

class NeuralVectorEngine:
    """Drop-in replacement for VectorEngine in p9_retrieve.py.

    Uses pre-computed neural embeddings from disk.
    Falls back to real-time encoding if no stored embeddings.
    """

    def __init__(self):
        self.node_ids = None
        self.embeddings = None
        self.active = False
        self.node_id_to_idx = {}

    def build(self, texts=None, conn=None):
        """Load pre-computed embeddings from disk.

        'texts' param is kept for API compat with VectorEngine but ignored.
        If conn is provided and no stored embeddings, builds from scratch.
        """
        if not is_available():
            self.active = False
            return

        # Try loading pre-computed
        ids, embs = load_embeddings()
        if ids is not None:
            self.node_ids = ids
            self.embeddings = embs
            self.node_id_to_idx = {nid: i for i, nid in enumerate(ids)}
            self.active = True
            return

        # No stored embeddings — build from scratch if conn available
        if conn is not None:
            n = build_full_index(conn)
            if n > 0:
                ids, embs = load_embeddings()
                if ids is not None:
                    self.node_ids = ids
                    self.embeddings = embs
                    self.node_id_to_idx = {nid: i for i, nid in enumerate(ids)}
                    self.active = True
                    return

        self.active = False

    def search(self, query):
        """Search by query text. Returns list of (node_id, score) tuples."""
        if not self.active or self.embeddings is None:
            return []

        q_vec = encode_single(query)
        if q_vec is None:
            return []

        scores = cosine_similarity(q_vec, self.embeddings)
        results = []
        for i, score in enumerate(scores):
            if score > 0.1:  # minimum threshold
                results.append((self.node_ids[i], float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search_scores_by_index(self, query):
        """Return raw score array aligned with node_ids (for p9_retrieve compat)."""
        if not self.active or self.embeddings is None:
            return []

        q_vec = encode_single(query)
        if q_vec is None:
            return []

        return cosine_similarity(q_vec, self.embeddings).tolist()

    def get_node_score(self, query_vec_cache, node_id):
        """Get score for a specific node given cached query vector."""
        if not self.active or node_id not in self.node_id_to_idx:
            return 0.0
        idx = self.node_id_to_idx[node_id]
        return float(cosine_similarity(query_vec_cache, self.embeddings[idx:idx+1])[0])


# ══════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(SCRIPT_DIR))
    import memory as mem

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 neural_engine.py build       — Build full index")
        print("  python3 neural_engine.py search 'q'   — Search by query")
        print("  python3 neural_engine.py status        — Check index status")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "build":
        print("Building neural embedding index...")
        conn = mem.get_db()
        n = build_full_index(conn)
        conn.close()
        print(f"Indexed {n} nodes → {EMBEDDINGS_FILE}")

    elif cmd == "search":
        q = sys.argv[2] if len(sys.argv) > 2 else "test"
        conn = mem.get_db()
        engine = NeuralVectorEngine()
        engine.build(conn=conn)
        results = engine.search(q)
        nodes_by_id = {n["id"]: dict(n) for n in conn.execute("SELECT id, title, type FROM nodes").fetchall()}
        print(f"Query: {q}")
        print(f"Top 10 results:")
        for nid, score in results[:10]:
            n = nodes_by_id.get(nid, {})
            print(f"  {score:.3f}  [{n.get('type','?')}] {n.get('title','?')}")
        conn.close()

    elif cmd == "status":
        ids, embs = load_embeddings()
        if ids is not None:
            print(f"Index: {len(ids)} nodes, {embs.shape[1]}-dim")
            print(f"File: {EMBEDDINGS_FILE} ({EMBEDDINGS_FILE.stat().st_size/1024:.0f} KB)")
            print(f"Model: {MODEL_NAME}")
            types = {}
            # Count by type if we can read the db
            try:
                conn = mem.get_db()
                for nid in ids:
                    r = conn.execute("SELECT type FROM nodes WHERE id=?", (nid,)).fetchone()
                    if r:
                        types[r["type"]] = types.get(r["type"], 0) + 1
                conn.close()
                for t, c in sorted(types.items()):
                    print(f"  {t}: {c}")
            except:
                pass
        else:
            print("No stored embeddings. Run: python3 neural_engine.py build")
    else:
        print(f"Unknown command: {cmd}")
