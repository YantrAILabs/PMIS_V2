"""
PMIS Team Memory — Sync Routes
Three-endpoint sync protocol: push, pull, bootstrap.
Plus: retrieve (team-wide P9+), feedback (ratings), health.
"""

import json
import uuid
import sqlite3
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel

import sys
PLATFORM_DIR = Path(__file__).parent
sys.path.insert(0, str(PLATFORM_DIR))
sys.path.insert(0, str(PLATFORM_DIR.parent / "scripts"))

from auth import get_user_from_token
from team_weights import recompute_team_weights

router = APIRouter(prefix="/api/sync", tags=["sync"])

# ── Central DB Path ──
CENTRAL_DIR = PLATFORM_DIR / "data" / "central"
CENTRAL_DB = CENTRAL_DIR / "graph_central.db"
_SERVER_TXN = 0  # monotonic counter, loaded from DB


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uuid():
    return uuid.uuid4().hex[:12]


# ── Central DB ──

def get_central_db():
    """Get connection to the central team memory database."""
    CENTRAL_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CENTRAL_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Create schema — same as memory.py base + team columns
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            content TEXT DEFAULT '',
            source TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            last_used TEXT NOT NULL,
            use_count INTEGER DEFAULT 1,
            quality REAL DEFAULT 0.0,
            weight REAL DEFAULT 0.5,
            initial_weight REAL DEFAULT 0.5,
            occurrence_log TEXT DEFAULT '[]',
            recency REAL DEFAULT 1.0,
            frequency REAL DEFAULT 0.0,
            consistency REAL DEFAULT 0.0,
            memory_stage TEXT DEFAULT 'impulse',
            discrimination_power REAL DEFAULT 0.0,
            mode_vector TEXT DEFAULT '{}',
            -- Team columns
            author TEXT NOT NULL DEFAULT '',
            source_folder TEXT DEFAULT '',
            node_hash TEXT DEFAULT '',
            server_txn INTEGER DEFAULT 0,
            pull_count INTEGER DEFAULT 0,
            unique_pullers TEXT DEFAULT '[]',
            rating_sum REAL DEFAULT 0.0,
            rating_count INTEGER DEFAULT 0,
            team_weight REAL DEFAULT 0.5,
            pushed_at TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            src TEXT NOT NULL,
            tgt TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'parent_child',
            weight REAL DEFAULT 1.0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sync_events (
            id TEXT PRIMARY KEY,
            author TEXT NOT NULL,
            action TEXT NOT NULL,
            node_count INTEGER DEFAULT 0,
            server_txn INTEGER,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS server_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cn_type ON nodes(type);
        CREATE INDEX IF NOT EXISTS idx_cn_hash ON nodes(node_hash);
        CREATE INDEX IF NOT EXISTS idx_cn_txn ON nodes(server_txn);
        CREATE INDEX IF NOT EXISTS idx_cn_author ON nodes(author);
        CREATE INDEX IF NOT EXISTS idx_ce_src ON edges(src);
        CREATE INDEX IF NOT EXISTS idx_ce_tgt ON edges(tgt);
    """)
    conn.commit()
    return conn


def _get_server_txn(conn):
    """Get current server transaction counter."""
    row = conn.execute(
        "SELECT value FROM server_state WHERE key='txn_counter'"
    ).fetchone()
    return int(row["value"]) if row else 0


def _increment_txn(conn):
    """Increment and return new server transaction counter."""
    current = _get_server_txn(conn)
    new_txn = current + 1
    conn.execute(
        "INSERT OR REPLACE INTO server_state (key, value) VALUES ('txn_counter', ?)",
        (str(new_txn),)
    )
    conn.commit()
    return new_txn


# ── Auth Dependency ──

async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    token = authorization.replace("Bearer ", "")
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(401, "Invalid or expired token")
    return user


# ── Request Models ──

class NodeData(BaseModel):
    id: str
    type: str
    title: str
    description: str = ""
    content: str = ""
    weight: float = 0.5
    node_hash: str = ""
    created_at: str = ""
    last_used: str = ""
    memory_stage: str = "impulse"

class EdgeData(BaseModel):
    src: str
    tgt: str
    type: str = "parent_child"

class PushRequest(BaseModel):
    author: str
    source_folder: str = ""
    nodes: List[dict] = []
    edges: List[dict] = []

class FeedbackRequest(BaseModel):
    author: str
    rating: str  # "up" or "down"
    anchor_ids: List[str] = []
    bad_anchors: List[str] = []


# ── Endpoint 1: PUSH ──

@router.post("/push")
async def push(req: PushRequest, user=Depends(get_current_user)):
    """Receive nodes from a local agent. LWW conflict resolution."""
    # Auth check: author must match token owner
    if req.author != user["username"]:
        raise HTTPException(403, f"Author mismatch: token belongs to '{user['username']}', not '{req.author}'")

    conn = get_central_db()
    try:
        server_txn = _increment_txn(conn)
        synced, skipped = 0, 0

        for node in req.nodes:
            node_hash = node.get("node_hash", "")
            node_id = node.get("id", "")
            updated_at = node.get("last_used", node.get("created_at", _now()))

            # Tier 1: Check by node_hash (fast, deterministic)
            existing = None
            if node_hash:
                existing = conn.execute(
                    "SELECT id, last_used FROM nodes WHERE node_hash=?",
                    (node_hash,)
                ).fetchone()

            # Tier 2: Check by title + type (fuzzy)
            if not existing:
                existing = conn.execute(
                    "SELECT id, last_used FROM nodes WHERE type=? AND title=?",
                    (node.get("type", ""), node.get("title", ""))
                ).fetchone()

            if existing:
                # LWW: keep newer
                if updated_at > (existing["last_used"] or ""):
                    conn.execute("""
                        UPDATE nodes SET
                            title=?, description=?, content=?, weight=?,
                            node_hash=?, last_used=?, author=?, source_folder=?,
                            server_txn=?, pushed_at=?
                        WHERE id=?
                    """, (
                        node.get("title", ""), node.get("description", ""),
                        node.get("content", ""), node.get("weight", 0.5),
                        node_hash, updated_at, req.author, req.source_folder,
                        server_txn, _now(), existing["id"]
                    ))
                    synced += 1
                else:
                    skipped += 1
            else:
                # New node — insert
                conn.execute("""
                    INSERT INTO nodes (id, type, title, description, content, source,
                        created_at, last_used, weight, initial_weight, node_hash,
                        author, source_folder, server_txn, pushed_at,
                        occurrence_log, memory_stage)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    node_id or _uuid(), node.get("type", "anchor"),
                    node.get("title", ""), node.get("description", ""),
                    node.get("content", ""), node.get("source", "claude_desktop"),
                    node.get("created_at", _now()), updated_at,
                    node.get("weight", 0.5), node.get("initial_weight", 0.5),
                    node_hash, req.author, req.source_folder,
                    server_txn, _now(), "[]", "impulse"
                ))
                synced += 1

        # Process edges
        for edge in req.edges:
            edge_id = _uuid()
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO edges (id, src, tgt, type, weight, created_at)
                    VALUES (?,?,?,?,1.0,?)
                """, (edge_id, edge.get("src"), edge.get("tgt"),
                      edge.get("type", "parent_child"), _now()))
            except Exception:
                pass

        # Log sync event
        conn.execute("""
            INSERT INTO sync_events (id, author, action, node_count, server_txn, timestamp)
            VALUES (?,?,?,?,?,?)
        """, (_uuid(), req.author, "push", synced, server_txn, _now()))

        conn.commit()
        return {"synced": synced, "skipped": skipped, "server_txn": server_txn}
    finally:
        conn.close()


# ── Endpoint 2: PULL ──

@router.get("/pull")
async def pull(since: int = 0, author: str = "", user=Depends(get_current_user)):
    """Return nodes updated since `since` txn, excluding requester's own nodes."""
    conn = get_central_db()
    try:
        rows = conn.execute("""
            SELECT * FROM nodes
            WHERE server_txn > ? AND author != ?
            ORDER BY server_txn ASC
        """, (since, author or user["username"])).fetchall()

        nodes = []
        for r in rows:
            nodes.append({k: r[k] for k in r.keys()})

        edges = conn.execute("""
            SELECT * FROM edges ORDER BY created_at ASC
        """).fetchall()
        edge_list = [{k: e[k] for k in e.keys()} for e in edges]

        return {
            "nodes": nodes,
            "edges": edge_list,
            "server_txn": _get_server_txn(conn),
            "count": len(nodes)
        }
    finally:
        conn.close()


# ── Endpoint 3: BOOTSTRAP ──

@router.get("/bootstrap")
async def bootstrap(since: int = 0, user=Depends(get_current_user)):
    """Return ALL nodes (including requester's). For initial setup or re-join."""
    conn = get_central_db()
    try:
        rows = conn.execute("""
            SELECT * FROM nodes WHERE server_txn > ?
            ORDER BY server_txn ASC
        """, (since,)).fetchall()

        nodes = [{k: r[k] for k in r.keys()} for r in rows]

        edges = conn.execute("SELECT * FROM edges").fetchall()
        edge_list = [{k: e[k] for k in e.keys()} for e in edges]

        return {
            "nodes": nodes,
            "edges": edge_list,
            "server_txn": _get_server_txn(conn),
            "count": len(nodes)
        }
    finally:
        conn.close()


# ── RETRIEVE (team-wide P9+) ──

@router.get("/retrieve")
async def retrieve(q: str, author: str = "", user=Depends(get_current_user)):
    """Team-wide retrieval with on-demand team_weight recompute."""
    conn = get_central_db()
    try:
        # Recompute team_weights on-demand
        recompute_team_weights(conn)

        requesting_author = author or user["username"]

        # Simple keyword retrieval (P9+ can be plugged in later)
        query_words = set(q.lower().split())
        all_nodes = conn.execute("""
            SELECT * FROM nodes WHERE author != ? AND type='anchor'
        """, (requesting_author,)).fetchall()

        scored = []
        for node in all_nodes:
            text = f"{node['title']} {node['content']}".lower()
            overlap = sum(1 for w in query_words if w in text)
            if overlap > 0:
                score = overlap * (node["team_weight"] or 0.5)
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = scored[:10]

        results = []
        for score, node in top_results:
            node_id = node["id"]
            # Track pull
            conn.execute("UPDATE nodes SET pull_count = pull_count + 1 WHERE id=?",
                         (node_id,))
            pullers = json.loads(node["unique_pullers"] or "[]")
            if requesting_author not in pullers:
                pullers.append(requesting_author)
                conn.execute("UPDATE nodes SET unique_pullers=? WHERE id=?",
                             (json.dumps(pullers), node_id))

            results.append({
                "id": node_id,
                "type": node["type"],
                "title": node["title"],
                "content": node["content"],
                "weight": node["weight"],
                "team_weight": node["team_weight"],
                "author": node["author"],
                "source_folder": node["source_folder"],
                "relevance_score": round(score, 3),
            })

        conn.commit()
        return {"results": results, "query": q, "count": len(results)}
    finally:
        conn.close()


# ── FEEDBACK (ratings) ──

@router.post("/feedback")
async def feedback(req: FeedbackRequest, user=Depends(get_current_user)):
    """Receive user ratings for team_weight computation."""
    conn = get_central_db()
    try:
        score = 4.0 if req.rating == "up" else 2.0

        updated = 0
        for aid in req.anchor_ids:
            conn.execute("""
                UPDATE nodes SET
                    rating_sum = rating_sum + ?,
                    rating_count = rating_count + 1
                WHERE id=?
            """, (score, aid))
            updated += 1

        # Penalize specific bad anchors
        for aid in (req.bad_anchors or []):
            conn.execute("""
                UPDATE nodes SET
                    rating_sum = rating_sum + 1.0,
                    rating_count = rating_count + 1
                WHERE id=?
            """, (aid,))
            updated += 1

        conn.commit()
        return {"updated": updated, "rating": req.rating}
    finally:
        conn.close()


# ── HEALTH ──

@router.get("/health")
async def health():
    """Central memory health report. No auth required for LAN discovery."""
    conn = get_central_db()
    try:
        total = conn.execute("SELECT COUNT(*) c FROM nodes").fetchone()["c"]
        authors = conn.execute(
            "SELECT DISTINCT author FROM nodes WHERE author != ''"
        ).fetchall()
        author_list = [a["author"] for a in authors]

        by_type = {}
        for r in conn.execute("SELECT type, COUNT(*) c FROM nodes GROUP BY type").fetchall():
            by_type[r["type"]] = r["c"]

        last_push = conn.execute(
            "SELECT MAX(timestamp) ts FROM sync_events WHERE action='push'"
        ).fetchone()["ts"]

        recent_syncs = conn.execute("""
            SELECT author, action, node_count, timestamp
            FROM sync_events ORDER BY timestamp DESC LIMIT 10
        """).fetchall()

        return {
            "status": "ok",
            "total_nodes": total,
            "authors": author_list,
            "nodes_by_type": by_type,
            "last_push": last_push,
            "server_txn": _get_server_txn(conn),
            "recent_syncs": [{k: s[k] for k in s.keys()} for s in recent_syncs],
        }
    finally:
        conn.close()
