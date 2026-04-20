"""PMIS public API — 5 async verbs + remember/ask sugar.

Thin async wrappers around the internal pmis_v2 subsystems.
Uses asyncio.to_thread so calls are non-blocking despite the
underlying engine being synchronous.

Environment variables:
    PMIS_DB_PATH     — path to SQLite memory.db (default: pmis_v2/data/memory.db)
    PMIS_CHROMA_DIR  — path to ChromaDB persist dir (default: pmis_v2/data/chroma)
    OPENAI_API_KEY   — for embeddings (if use_local=False in hyperparameters.yaml)
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the internal pmis_v2 package importable (its modules use top-level
# imports like `from core.surprise import ...` so pmis_v2 must be on sys.path).
_PMIS_V2 = Path(__file__).resolve().parent.parent / "pmis_v2"
if str(_PMIS_V2) not in sys.path:
    sys.path.insert(0, str(_PMIS_V2))


class _Engine:
    """Lazy-initialized singleton holding every PMIS subsystem."""

    _instance: Optional["_Engine"] = None

    @classmethod
    def get(cls) -> "_Engine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from db.manager import DBManager
        from db.chroma_store import ChromaStore
        from ingestion.embedder import Embedder
        from ingestion.pipeline import IngestionPipeline
        from retrieval.engine import RetrievalEngine
        from consolidation.nightly import NightlyConsolidation
        from core.session_state import SessionState
        from core import config

        self.hp = config.get_all()

        db_path = os.environ.get(
            "PMIS_DB_PATH", str(_PMIS_V2 / "data" / "memory.db")
        )
        chroma_dir = os.environ.get(
            "PMIS_CHROMA_DIR", str(Path(db_path).parent / "chroma")
        )
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.chroma = ChromaStore(persist_dir=chroma_dir)
        self.db = DBManager(db_path, chroma_store=self.chroma)
        self.embedder = Embedder(hyperparams=self.hp)
        self.pipeline = IngestionPipeline(self.db, self.embedder, self.hp)
        self.engine = RetrievalEngine(self.db, self.hp)
        self.nightly = NightlyConsolidation(self.db, self.hp)
        self.session = SessionState(conversation_id="pmis-api")


# ---------------------------------------------------------------------------
# Verb 1: ingest — embed + surprise-gated storage
# ---------------------------------------------------------------------------


def _ingest_sync(text: str, *, conversation_id: str = "", role: str = "user") -> Optional[str]:
    eng = _Engine.get()
    result = eng.pipeline.process_turn(
        content=text,
        session=eng.session,
        role=role,
        conversation_id=conversation_id or "pmis-api",
    )
    return result.node_id


async def ingest(
    text: str, *, conversation_id: str = "", role: str = "user"
) -> Optional[str]:
    """Ingest raw text into memory.

    Embeds, computes surprise + γ, applies the storage gate. Returns the new
    node_id, or None if surprise-gated away.
    """
    return await asyncio.to_thread(
        _ingest_sync, text, conversation_id=conversation_id, role=role
    )


# ---------------------------------------------------------------------------
# Verb 2: attach — promote orphan into the hierarchy
# ---------------------------------------------------------------------------


def _attach_sync(
    node_id: Optional[str] = None, *, project: Optional[str] = None
) -> Dict[str, Any]:
    eng = _Engine.get()

    if node_id is None:
        orphans = eng.db.get_orphan_nodes()
        if not orphans:
            return {"attached": False, "reason": "no orphan nodes"}
        node_id = orphans[0]["id"]

    node = eng.db.get_node(node_id)
    if not node:
        return {"attached": False, "reason": "node not found"}

    embs = eng.db.get_embeddings(node_id)
    query_embedding = embs.get("euclidean")
    if query_embedding is None:
        return {"attached": False, "reason": "no embedding on node"}

    nearest = eng.pipeline._find_nearest_context(query_embedding)
    if not nearest:
        return {"attached": False, "reason": "no context found"}

    tree_id = eng.session.active_tree_id or "default"
    eng.db.attach_to_parent(node_id, nearest["id"], tree_id=tree_id)

    return {
        "attached": True,
        "node_id": node_id,
        "parent_id": nearest["id"],
        "parent_preview": nearest.get("content", "")[:80],
        "project": project,
    }


async def attach(
    node_id: Optional[str] = None, *, project: Optional[str] = None
) -> Dict[str, Any]:
    """Attach an orphan node to its nearest Context in the hierarchy.

    If node_id is None, attaches the most recently-created orphan.
    """
    return await asyncio.to_thread(_attach_sync, node_id, project=project)


# ---------------------------------------------------------------------------
# Verb 3: retrieve — γ-blended search
# ---------------------------------------------------------------------------

_GAMMA_BY_MODE = {
    "associative": 0.85,
    "balanced": 0.5,
    "predictive": 0.2,
}


def _retrieve_sync(
    query: str, *, mode: str = "auto", k: int = 8
) -> List[Dict[str, Any]]:
    eng = _Engine.get()
    embs = eng.embedder.generate_triple_embedding(text=query, level="ANC")
    query_embedding = embs["euclidean"]

    if mode == "auto":
        gamma = getattr(eng.session, "last_gamma", None) or 0.5
    else:
        gamma = _GAMMA_BY_MODE.get(mode, 0.5)

    hits = eng.engine.retrieve(
        query_embedding=query_embedding, gamma=gamma, top_k=k
    )

    return [
        {
            "id": h.get("id"),
            "content": h.get("content"),
            "level": h.get("level"),
            "score": h.get("final_score"),
            "semantic": h.get("semantic_score"),
            "hierarchy": h.get("hierarchy_score"),
            "temporal": h.get("temporal_score"),
            "precision": h.get("precision_score"),
        }
        for h in hits
    ]


async def retrieve(
    query: str, *, mode: str = "auto", k: int = 8
) -> List[Dict[str, Any]]:
    """Retrieve memories matching the query.

    mode: "associative" (exploit, γ=0.85), "balanced" (γ=0.5),
          "predictive" (explore, γ=0.2), or "auto" (use session γ).
    """
    return await asyncio.to_thread(_retrieve_sync, query, mode=mode, k=k)


# ---------------------------------------------------------------------------
# Verb 4: consolidate — nightly 5-pass
# ---------------------------------------------------------------------------


def _consolidate_sync() -> Dict[str, Any]:
    return _Engine.get().nightly.run()


async def consolidate() -> Dict[str, Any]:
    """Run the 5-pass consolidation: compress, promote, birth, prune, RSGD/HGCN."""
    return await asyncio.to_thread(_consolidate_sync)


# ---------------------------------------------------------------------------
# Verb 5: delete — soft-delete node, or reset all
# ---------------------------------------------------------------------------


def _delete_sync(
    node_id: Optional[str] = None, *, all: bool = False
) -> Dict[str, Any]:
    eng = _Engine.get()

    if all:
        count = eng.db.count_nodes()
        with eng.db._connect() as conn:
            conn.execute("UPDATE memory_nodes SET is_deleted = 1")
            conn.execute("DELETE FROM relations")
        if getattr(eng.chroma, "enabled", False):
            try:
                eng.chroma.rebuild_from_db(eng.db)
            except Exception:
                pass
        return {"deleted": count, "reset": True}

    if node_id is None:
        return {"deleted": 0, "reason": "no node_id given (use all=True to reset)"}

    eng.db.soft_delete(node_id)
    return {"deleted": 1, "node_id": node_id}


async def delete(
    node_id: Optional[str] = None, *, all: bool = False
) -> Dict[str, Any]:
    """Soft-delete a single node, or reset the entire store with all=True."""
    return await asyncio.to_thread(_delete_sync, node_id, all=all)


# ---------------------------------------------------------------------------
# Convenience sugar: remember = ingest + attach, ask = retrieve(mode=auto)
# ---------------------------------------------------------------------------


async def remember(text: str, *, project: Optional[str] = None) -> Dict[str, Any]:
    """One-shot ingest + attach. Returns {node_id, stored, attached, ...}."""
    node_id = await ingest(text)
    if node_id is None:
        return {"node_id": None, "stored": False, "reason": "surprise-gated"}
    att = await attach(node_id, project=project)
    return {"node_id": node_id, "stored": True, **att}


async def ask(query: str, *, k: int = 5) -> List[Dict[str, Any]]:
    """One-shot retrieval with automatic mode selection."""
    return await retrieve(query, mode="auto", k=k)
