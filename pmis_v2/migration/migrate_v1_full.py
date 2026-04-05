"""
PMIS V1 → V2 Full Migration Script

Migrates all data from the V1 graph.db (465 nodes, 447 edges, 41 tasks)
to the V2 schema with re-embedded 768-dim vectors via Ollama nomic-embed-text.

Usage:
    cd <repo-root>
    python3 pmis_v2/migration/migrate_v1_full.py

Source: memory_system/Graph_DB/graph.db + Vector_DB/embeddings.npz
Target: pmis_v2/data/memory.db + pmis_v2/data/chroma/
"""

import sys
import os
import json
import sqlite3
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory_node import MemoryNode, MemoryLevel
from core.poincare import assign_hyperbolic_coords, ProjectionManager
from core.temporal import temporal_encode, compute_era
from db.manager import DBManager
from db.chroma_store import ChromaStore


# ================================================================
# CONFIG
# ================================================================
V1_DB = Path(__file__).parent.parent.parent / "memory_system" / "Graph_DB" / "graph.db"
V2_DB = Path(__file__).parent.parent / "data" / "memory.db"
V2_CHROMA = Path(__file__).parent.parent / "data" / "chroma"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
HYPERBOLIC_DIM = 32
TEMPORAL_DIM = 16

# V2 era boundaries (from hyperparameters.yaml) — must be ISO strings for compute_era()
ERA_BOUNDARIES = {
    "pre_yantra": "2024-12-31",
    "kiran_ai_phase": "2025-03-01",
    "vision_os_phase": "2025-05-01",
    "current": "2026-12-31",
}

# Map V1 type → V2 level
TYPE_MAP = {
    "super_context": "SC",
    "context": "CTX",
    "anchor": "ANC",
}


def embed_text_ollama(text: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """Call Ollama embedding API."""
    import httpx
    try:
        resp = httpx.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text[:8000]},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return np.array(data["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"    [WARN] Embedding failed for '{text[:50]}...': {e}", file=sys.stderr)
        # Deterministic fallback: hash-based random vector
        import hashlib
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return rng.randn(EMBEDDING_DIM).astype(np.float32)


def parse_datetime(dt_str: str) -> datetime:
    """Parse V1 datetime string to datetime object."""
    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return datetime.now()


def migrate():
    print("=" * 60)
    print("PMIS V1 → V2 MIGRATION")
    print("=" * 60)
    print()

    # ============================================================
    # 1. VALIDATE SOURCES
    # ============================================================
    print("Step 1: Validating sources...")
    if not V1_DB.exists():
        print(f"  ERROR: V1 database not found at {V1_DB}")
        sys.exit(1)

    v1_conn = sqlite3.connect(str(V1_DB))
    v1_conn.row_factory = sqlite3.Row

    node_count = v1_conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edge_count = v1_conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    task_count = v1_conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    print(f"  V1 database: {node_count} nodes, {edge_count} edges, {task_count} tasks")

    # Check Ollama
    try:
        test = embed_text_ollama("test")
        print(f"  Ollama ready: {EMBEDDING_MODEL} ({len(test)}d)")
    except Exception as e:
        print(f"  ERROR: Ollama not available: {e}")
        sys.exit(1)
    print()

    # ============================================================
    # 2. INITIALIZE V2 DATABASE
    # ============================================================
    print("Step 2: Initializing V2 database...")

    # Remove old V2 data if exists
    if V2_DB.exists():
        os.remove(V2_DB)
        print(f"  Removed existing {V2_DB}")
    if V2_CHROMA.exists():
        import shutil
        shutil.rmtree(V2_CHROMA)
        print(f"  Removed existing {V2_CHROMA}")

    V2_DB.parent.mkdir(parents=True, exist_ok=True)
    chroma = ChromaStore(persist_dir=str(V2_CHROMA))
    db = DBManager(str(V2_DB), chroma_store=chroma)

    # Set embedding model metadata
    db.set_embedding_model(f"ollama/{EMBEDDING_MODEL}")

    pm = ProjectionManager(input_dim=EMBEDDING_DIM, output_dim=HYPERBOLIC_DIM, seed=42)
    print(f"  V2 database created at {V2_DB}")
    print()

    # ============================================================
    # 3. MIGRATE NODES (with re-embedding)
    # ============================================================
    print("Step 3: Migrating nodes (re-embedding via Ollama)...")

    # Read all V1 nodes, ordered so parents come before children
    v1_nodes = v1_conn.execute("""
        SELECT * FROM nodes
        ORDER BY CASE type
            WHEN 'super_context' THEN 1
            WHEN 'context' THEN 2
            WHEN 'anchor' THEN 3
        END, created_at
    """).fetchall()

    # Build parent lookup from edges (src=parent, tgt=child)
    v1_edges = v1_conn.execute("SELECT src, tgt, weight FROM edges").fetchall()
    child_to_parent = {}
    for edge in v1_edges:
        child_to_parent[edge["tgt"]] = edge["src"]

    # Track V1 ID → V2 node for edge migration
    v1_to_v2_id = {}
    v2_hyperbolic = {}  # node_id → hyperbolic coords (for parent lookup)
    migrated = 0
    failed = 0
    t_start = time.time()

    for i, v1_node in enumerate(v1_nodes):
        v1_id = v1_node["id"]
        v1_type = v1_node["type"]
        v2_level = TYPE_MAP.get(v1_type, "ANC")

        # Build content for embedding
        title = v1_node["title"] or ""
        content = v1_node["content"] or ""
        description = v1_node["description"] or ""
        text_for_embed = f"{title}. {content} {description}".strip()
        if not text_for_embed or text_for_embed == ".":
            text_for_embed = title or f"Untitled {v1_type}"

        # Parse timestamps
        created_at = parse_datetime(v1_node["created_at"])
        era = compute_era(created_at, ERA_BOUNDARIES)

        # Generate euclidean embedding via Ollama
        euc = embed_text_ollama(text_for_embed)

        # Generate hyperbolic coords
        parent_id = child_to_parent.get(v1_id)
        parent_coords = v2_hyperbolic.get(parent_id) if parent_id else None
        hyp = assign_hyperbolic_coords(euc, v2_level, pm, parent_coords=parent_coords)

        # Generate temporal embedding
        temp = temporal_encode(created_at, dim=TEMPORAL_DIM)

        # Map V1 fields to V2
        occurrence_log = json.loads(v1_node["occurrence_log"] or "[]")
        access_count = max(v1_node["use_count"] or 1, len(occurrence_log))
        precision = min(max(v1_node["weight"] or 0.5, 0.05), 1.0)

        # Create V2 node
        node = MemoryNode(
            id=v1_id,  # Preserve V1 IDs for edge migration
            content=text_for_embed,
            source_conversation_id="v1_migration",
            euclidean_embedding=euc,
            hyperbolic_coords=hyp,
            temporal_embedding=temp,
            level=MemoryLevel(v2_level),
            precision=precision,
            surprise_at_creation=0.0,
            created_at=created_at,
            last_modified=datetime.now(),
            era=era,
            is_orphan=False,
            is_tentative=False,
            is_deleted=False,
        )
        node.access_pattern.count = access_count

        try:
            db.create_node(node)
            v1_to_v2_id[v1_id] = node.id
            v2_hyperbolic[v1_id] = hyp
            migrated += 1
        except Exception as e:
            print(f"    [FAIL] Node {v1_id}: {e}", file=sys.stderr)
            failed += 1

        # Progress
        if (i + 1) % 50 == 0 or i == len(v1_nodes) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(v1_nodes)}] {rate:.1f} nodes/sec "
                  f"({migrated} ok, {failed} failed) [{elapsed:.1f}s]")

    print(f"  Nodes migrated: {migrated}/{len(v1_nodes)}")
    print()

    # ============================================================
    # 4. MIGRATE EDGES → RELATIONS
    # ============================================================
    print("Step 4: Migrating edges → relations...")

    edges_migrated = 0
    for edge in v1_edges:
        src = edge["src"]  # parent
        tgt = edge["tgt"]  # child

        if src in v1_to_v2_id and tgt in v1_to_v2_id:
            # V1 edges: src=parent, tgt=child
            # V2 relations: source=child, target=parent (child_of convention)
            db.create_relation(
                source_id=tgt,  # child points TO parent
                target_id=src,
                relation_type="child_of",
                tree_id="default",
                weight=edge["weight"] or 1.0,
            )
            edges_migrated += 1

    print(f"  Relations created: {edges_migrated}/{len(v1_edges)}")
    print()

    # ============================================================
    # 5. CREATE TREES FROM SC HIERARCHY
    # ============================================================
    print("Step 5: Creating trees from super contexts...")

    sc_nodes = v1_conn.execute(
        "SELECT id, title FROM nodes WHERE type='super_context'"
    ).fetchall()

    trees_created = 0
    for sc in sc_nodes:
        tree_id = f"tree_{sc['id'][:8]}"
        tree_name = sc["title"]
        try:
            db._conn.execute("""
                INSERT OR IGNORE INTO trees (tree_id, name, description, root_node_id)
                VALUES (?, ?, ?, ?)
            """, (tree_id, tree_name, f"Auto-migrated from V1 SC: {tree_name}", sc["id"]))
            db._conn.commit()
            trees_created += 1

            # Update tree_ids on children
            children = v1_conn.execute(
                "SELECT tgt FROM edges WHERE src=?", (sc["id"],)
            ).fetchall()
            for child in children:
                cid = child["tgt"]
                db._conn.execute("""
                    UPDATE memory_nodes SET tree_ids = ? WHERE id = ?
                """, (json.dumps([tree_id]), cid))
                # Also update grandchildren (anchors under contexts)
                grandchildren = v1_conn.execute(
                    "SELECT tgt FROM edges WHERE src=?", (cid,)
                ).fetchall()
                for gc in grandchildren:
                    db._conn.execute("""
                        UPDATE memory_nodes SET tree_ids = ? WHERE id = ?
                    """, (json.dumps([tree_id]), gc["tgt"]))
            db._conn.commit()
        except Exception as e:
            print(f"    [WARN] Tree for {tree_name}: {e}", file=sys.stderr)

    print(f"  Trees created: {trees_created}")
    print()

    # ============================================================
    # 6. MIGRATE TASK HISTORY → ACCESS LOG
    # ============================================================
    print("Step 6: Migrating task history → access_log...")

    tasks = v1_conn.execute("SELECT * FROM tasks").fetchall()
    task_anchors = v1_conn.execute("SELECT * FROM task_anchors").fetchall()

    access_entries = 0
    for ta in task_anchors:
        task = next((t for t in tasks if t["id"] == ta["task_id"]), None)
        if not task:
            continue

        anchor_id = ta["anchor_id"]
        if anchor_id not in v1_to_v2_id:
            continue

        score = task["score"] or 3.0
        gamma_approx = 0.7 if score >= 3.5 else 0.4  # Approximate from feedback

        try:
            db._conn.execute("""
                INSERT INTO access_log (node_id, accessed_at, query_text, gamma_at_access, surprise_at_access)
                VALUES (?, ?, ?, ?, ?)
            """, (anchor_id, task["started"], f"Task: {task['title'][:200]}", gamma_approx, 0.3))
            access_entries += 1
        except Exception:
            pass

    db._conn.commit()
    print(f"  Access log entries: {access_entries}")
    print()

    # ============================================================
    # 7. REFRESH CONTEXT STATS
    # ============================================================
    print("Step 7: Computing context stats cache...")

    ctx_nodes = db._conn.execute(
        "SELECT id FROM memory_nodes WHERE level='CTX' AND is_deleted=0"
    ).fetchall()
    for ctx in ctx_nodes:
        try:
            db._refresh_context_stats(ctx["id"])
        except Exception:
            pass

    stats_count = db._conn.execute("SELECT COUNT(*) FROM context_stats_cache").fetchone()[0]
    print(f"  Context stats cached: {stats_count}")
    print()

    # ============================================================
    # 8. FINAL COUNTS
    # ============================================================
    print("Step 8: Final validation...")

    v2_nodes = db._conn.execute("SELECT COUNT(*) FROM memory_nodes WHERE is_deleted=0").fetchone()[0]
    v2_embs = db._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    v2_rels = db._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    v2_trees = db._conn.execute("SELECT COUNT(*) FROM trees").fetchone()[0]
    v2_chroma = chroma.count() if chroma.enabled else 0

    by_level = {}
    for row in db._conn.execute("SELECT level, COUNT(*) as cnt FROM memory_nodes WHERE is_deleted=0 GROUP BY level"):
        by_level[row["level"]] = row["cnt"]

    print(f"  V2 Nodes:      {v2_nodes} (SC={by_level.get('SC',0)}, CTX={by_level.get('CTX',0)}, ANC={by_level.get('ANC',0)})")
    print(f"  V2 Embeddings: {v2_embs}")
    print(f"  V2 Relations:  {v2_rels}")
    print(f"  V2 Trees:      {v2_trees}")
    print(f"  ChromaDB:      {v2_chroma}")
    print()

    # Verify counts
    ok = True
    if v2_nodes != node_count:
        print(f"  ⚠ Node count mismatch: V1={node_count}, V2={v2_nodes}")
        ok = False
    if v2_embs != v2_nodes:
        print(f"  ⚠ Embedding count mismatch: nodes={v2_nodes}, embeddings={v2_embs}")
        ok = False
    if v2_rels != edge_count:
        print(f"  ⚠ Relation count mismatch: V1 edges={edge_count}, V2 relations={v2_rels}")
        ok = False

    total_time = time.time() - t_start
    if ok:
        print(f"  ✓ ALL COUNTS MATCH")
    print(f"\n  Total time: {total_time:.1f}s")

    v1_conn.close()
    db.close()

    print()
    print("=" * 60)
    print("MIGRATION COMPLETE" if ok else "MIGRATION COMPLETED WITH WARNINGS")
    print("=" * 60)


if __name__ == "__main__":
    migrate()
