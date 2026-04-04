"""
Database Manager for PMIS v2.

Unified interface for all SQLite operations. Every other module
talks to the database through this class.
"""

import sqlite3
import json
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager

from core.memory_node import MemoryNode, MemoryLevel, Relation, RelationType, AccessPattern


class DBManager:
    def __init__(self, db_path: str = "data/memory.db", chroma_store=None):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Persistent connection with WAL mode for concurrent reads
        self._conn = sqlite3.connect(self.db_path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # ChromaDB ANN index (optional, auto-synced)
        self._chroma = chroma_store

        self._init_db()

    def set_chroma(self, chroma_store):
        """Attach ChromaDB store after construction."""
        self._chroma = chroma_store

    def _init_db(self):
        """Initialize database with schema if tables don't exist."""
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            schema_sql = schema_path.read_text()
            # Strip comment-only lines that precede the first CREATE
            # and use executescript for reliable multi-statement execution
            try:
                self._conn.executescript(schema_sql)
            except Exception:
                # Fallback: execute statement by statement, skipping failures
                for statement in schema_sql.split(';'):
                    # Remove leading comment lines from each statement
                    lines = statement.split('\n')
                    clean_lines = []
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('--'):
                            continue
                        clean_lines.append(line)
                    stmt = '\n'.join(clean_lines).strip()
                    if stmt:
                        try:
                            self._conn.execute(stmt)
                        except Exception:
                            pass
                self._conn.commit()

        # P2a: Materialized context stats columns
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS context_stats_cache (
                context_id TEXT PRIMARY KEY,
                child_count INTEGER DEFAULT 0,
                avg_recency_hours REAL DEFAULT 720,
                internal_consistency REAL DEFAULT 0.5,
                last_updated TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (context_id) REFERENCES memory_nodes(id)
            );
            CREATE TABLE IF NOT EXISTS system_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self._conn.commit()

        # Migration: add unique index on conversation_turns if missing
        try:
            self._conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_turns_unique
                ON conversation_turns(conversation_id, turn_number)
            """)
            self._conn.commit()
        except Exception:
            pass  # Index may already exist or table has UNIQUE constraint

        # Migration: add rich turn logging columns to conversation_turns
        for col, typedef in [
            ("raw_surprise", "REAL"), ("cluster_precision", "REAL"),
            ("nearest_context_id", "TEXT"), ("nearest_context_name", "TEXT"),
            ("active_tree", "TEXT"), ("is_stale", "INTEGER DEFAULT 0"),
            ("storage_action", "TEXT"), ("system_prompt", "TEXT"),
            ("response_summary", "TEXT"),
        ]:
            try:
                self._conn.execute(f"SELECT {col} FROM conversation_turns LIMIT 1")
            except Exception:
                try:
                    self._conn.execute(f"ALTER TABLE conversation_turns ADD COLUMN {col} {typedef}")
                    self._conn.commit()
                except Exception:
                    pass

        # Migration: create turn detail tables (no FKs — historical records, nodes may be deleted)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS turn_retrieved_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL, memory_node_id TEXT,
                rank INTEGER NOT NULL, final_score REAL,
                semantic_score REAL, hierarchy_score REAL,
                temporal_score REAL, precision_score REAL,
                source TEXT, content_preview TEXT, node_level TEXT
            );
            CREATE TABLE IF NOT EXISTS turn_epistemic_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL, question_text TEXT NOT NULL,
                information_gain REAL, parent_context_id TEXT,
                parent_context_name TEXT, anchor_id TEXT, anchor_content TEXT
            );
            CREATE TABLE IF NOT EXISTS turn_predictive_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL, memory_node_id TEXT,
                content_preview TEXT, prediction_depth INTEGER,
                prediction_frequency INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_turn_retrieved ON turn_retrieved_memories(turn_id);
            CREATE INDEX IF NOT EXISTS idx_turn_epistemic ON turn_epistemic_questions(turn_id);
            CREATE INDEX IF NOT EXISTS idx_turn_predictive ON turn_predictive_memories(turn_id);
        """)
        self._conn.commit()

        # Migration: add RSGD columns to embeddings if missing
        for col, coltype in [("hyperbolic_norm", "REAL"), ("is_learned", "INTEGER DEFAULT 0"), ("last_trained", "TEXT")]:
            try:
                self._conn.execute(f"SELECT {col} FROM embeddings LIMIT 1")
            except Exception:
                try:
                    self._conn.execute(f"ALTER TABLE embeddings ADD COLUMN {col} {coltype}")
                    self._conn.commit()
                except Exception:
                    pass

        # Migration: create RSGD tables if missing
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS rsgd_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_type TEXT NOT NULL,
                epochs INTEGER, final_loss REAL, nodes_updated INTEGER,
                edges_used INTEGER, learning_rate REAL, wall_time_seconds REAL,
                run_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS context_centroids (
                context_id TEXT PRIMARY KEY,
                centroid BLOB NOT NULL, child_count INTEGER,
                last_computed TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (context_id) REFERENCES memory_nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_hyp_norm ON embeddings(hyperbolic_norm);
        """)
        self._conn.commit()

        # Migration: add semantic_distance column to access_log if missing
        try:
            self._conn.execute("SELECT semantic_distance FROM access_log LIMIT 1")
        except Exception:
            try:
                self._conn.execute("ALTER TABLE access_log ADD COLUMN semantic_distance REAL")
                self._conn.commit()
            except Exception:
                pass

        # Migration: add productivity columns to memory_nodes if missing
        for col, coltype in [
            ("productivity_time_mins", "REAL DEFAULT 0"),
            ("productivity_human_mins", "REAL DEFAULT 0"),
            ("productivity_ai_mins", "REAL DEFAULT 0"),
            ("last_productivity_sync", "TEXT DEFAULT ''"),
            ("is_project_node", "INTEGER DEFAULT 0"),
        ]:
            try:
                self._conn.execute(f"SELECT {col} FROM memory_nodes LIMIT 1")
            except Exception:
                try:
                    self._conn.execute(f"ALTER TABLE memory_nodes ADD COLUMN {col} {coltype}")
                    self._conn.commit()
                except Exception:
                    pass

        # Migration: create productivity/project tables if missing
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                company TEXT DEFAULT '',
                source TEXT DEFAULT 'manual',
                source_id TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                owner TEXT DEFAULT '',
                deadline TEXT DEFAULT '',
                sc_node_id TEXT DEFAULT '',
                expected_hours REAL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS deliverables (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                context_node_id TEXT DEFAULT '',
                anchor_node_ids TEXT DEFAULT '[]',
                status TEXT DEFAULT 'active',
                deadline TEXT DEFAULT '',
                expected_hours REAL DEFAULT 0,
                source TEXT DEFAULT 'manual',
                source_id TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );
            CREATE TABLE IF NOT EXISTS project_work_match_log (
                id TEXT PRIMARY KEY,
                segment_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                deliverable_id TEXT DEFAULT '',
                sc_node_id TEXT DEFAULT '',
                context_node_id TEXT DEFAULT '',
                anchor_node_id TEXT DEFAULT '',
                semantic_score REAL DEFAULT 0,
                hyperbolic_score REAL DEFAULT 0,
                combined_match_pct REAL DEFAULT 0,
                match_method TEXT DEFAULT '',
                work_description TEXT DEFAULT '',
                worker_type TEXT DEFAULT '',
                time_mins REAL DEFAULT 0,
                matched_at TEXT DEFAULT (datetime('now')),
                is_correct INTEGER DEFAULT -1
            );
            CREATE TABLE IF NOT EXISTS productivity_sync_log (
                id TEXT PRIMARY KEY,
                sync_type TEXT NOT NULL,
                triggered_at TEXT NOT NULL,
                segments_processed INTEGER DEFAULT 0,
                nodes_created INTEGER DEFAULT 0,
                nodes_updated INTEGER DEFAULT 0,
                matches_found INTEGER DEFAULT 0,
                avg_match_pct REAL DEFAULT 0,
                rsgd_epochs_run INTEGER DEFAULT 0,
                completed_at TEXT,
                status TEXT DEFAULT 'running'
            );
            CREATE INDEX IF NOT EXISTS idx_proj_status ON projects(status);
            CREATE INDEX IF NOT EXISTS idx_proj_sc ON projects(sc_node_id);
            CREATE INDEX IF NOT EXISTS idx_deliv_project ON deliverables(project_id);
            CREATE INDEX IF NOT EXISTS idx_pwm_segment ON project_work_match_log(segment_id);
            CREATE INDEX IF NOT EXISTS idx_pwm_project ON project_work_match_log(project_id);
            CREATE INDEX IF NOT EXISTS idx_pwm_score ON project_work_match_log(combined_match_pct);
            CREATE INDEX IF NOT EXISTS idx_pwm_date ON project_work_match_log(matched_at);
        """)
        self._conn.commit()

    @contextmanager
    def _connect(self):
        """Yield the persistent connection. Commits on success, rolls back on error."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self):
        """Close the persistent connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ---------------------------------------------------------------
    # MEMORY NODES — CRUD
    # ---------------------------------------------------------------

    def create_node(self, node: MemoryNode) -> str:
        """Insert a new memory node. Returns node ID."""
        data = node.to_db_dict()
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memory_nodes
                (id, content, source_conversation_id, level, parent_ids, tree_ids,
                 precision, surprise_at_creation, created_at, last_modified, era,
                 access_count, last_accessed, decay_rate, is_orphan, is_tentative, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"], data["content"], data["source_conversation_id"],
                data["level"], data["parent_ids"], data["tree_ids"],
                data["precision"], data["surprise_at_creation"],
                data["created_at"], data["last_modified"], data["era"],
                data["access_count"], data["last_accessed"],
                data["decay_rate"], data["is_orphan"], data["is_tentative"],
                data["is_deleted"],
            ))
            # Store embeddings
            self._store_embeddings(conn, node)

        # P1a: Auto-sync to ChromaDB ANN index
        if self._chroma and node.euclidean_embedding is not None:
            self._chroma.add(node.id, node.euclidean_embedding, {
                "level": node.level.value,
                "is_orphan": str(node.is_orphan),
                "tree_ids": node.tree_ids,
            })

        return node.id

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory node by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_nodes WHERE id = ? AND is_deleted = 0",
                (node_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)

    def get_nodes_by_level(self, level: str) -> List[Dict[str, Any]]:
        """Get all nodes at a given hierarchy level."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_nodes WHERE level = ? AND is_deleted = 0",
                (level,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_orphan_nodes(self) -> List[Dict[str, Any]]:
        """Get all orphan Anchors (not attached to any Context)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_nodes WHERE is_orphan = 1 AND is_deleted = 0"
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def update_node_access(self, node_id: str, query_text: str = "",
                           gamma: float = 0.5, surprise: float = 0.5,
                           semantic_distance: float = None):
        """Record an access to a node: bump count, update timestamp, log it."""
        with self._connect() as conn:
            conn.execute("""
                UPDATE memory_nodes
                SET access_count = access_count + 1,
                    last_accessed = datetime('now'),
                    last_modified = datetime('now')
                WHERE id = ?
            """, (node_id,))
            conn.execute("""
                INSERT INTO access_log (node_id, query_text, gamma_at_access, surprise_at_access, semantic_distance)
                VALUES (?, ?, ?, ?, ?)
            """, (node_id, query_text[:500], gamma, surprise, semantic_distance))

    def update_node_precision(self, node_id: str, precision: float):
        """Update a node's precision score."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE memory_nodes SET precision = ?, last_modified = datetime('now') WHERE id = ?",
                (precision, node_id)
            )

    def soft_delete(self, node_id: str):
        """Soft-delete a node (set is_deleted = 1). Cleans up relations. Auto-removes from ChromaDB."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE memory_nodes SET is_deleted = 1, last_modified = datetime('now') WHERE id = ?",
                (node_id,)
            )
            # Clean up all relations pointing to/from this node
            conn.execute(
                "DELETE FROM relations WHERE source_id = ? OR target_id = ?",
                (node_id, node_id)
            )
        # P1a: Remove from ChromaDB
        if self._chroma:
            self._chroma.remove(node_id)

    def attach_to_parent(self, child_id: str, parent_id: str, tree_id: str = "default"):
        """Attach an orphan to a parent Context. Clears orphan flag. Updates materialized stats."""
        with self._connect() as conn:
            # Update parent_ids
            row = conn.execute("SELECT parent_ids FROM memory_nodes WHERE id = ?", (child_id,)).fetchone()
            if row:
                parents = json.loads(row["parent_ids"])
                if parent_id not in parents:
                    parents.append(parent_id)
                conn.execute("""
                    UPDATE memory_nodes
                    SET parent_ids = ?, is_orphan = 0, last_modified = datetime('now')
                    WHERE id = ?
                """, (json.dumps(parents), child_id))

            # Create relation
            self.create_relation(child_id, parent_id, "child_of", tree_id)

        # P2a: Refresh materialized stats for the parent Context
        self._refresh_context_stats(parent_id)

    def merge_into_parent(self, child_id: str, parent_id: str):
        """
        Absorb a child Anchor into its parent Context.
        Appends child content summary to parent, then soft-deletes child.
        """
        with self._connect() as conn:
            child = conn.execute("SELECT content FROM memory_nodes WHERE id = ?", (child_id,)).fetchone()
            parent = conn.execute("SELECT content FROM memory_nodes WHERE id = ?", (parent_id,)).fetchone()
            if child and parent:
                merged = parent["content"] + "\n[Absorbed] " + child["content"][:200]
                conn.execute("""
                    UPDATE memory_nodes
                    SET content = ?, last_modified = datetime('now')
                    WHERE id = ?
                """, (merged, parent_id))
            self.soft_delete(child_id)

    # ---------------------------------------------------------------
    # RELATIONS
    # ---------------------------------------------------------------

    def create_relation(self, source_id: str, target_id: str,
                        relation_type: str, tree_id: str = "default",
                        weight: float = 1.0):
        """Create a relation between two nodes."""
        with self._connect() as conn:
            # Check for duplicate
            existing = conn.execute("""
                SELECT id FROM relations
                WHERE source_id = ? AND target_id = ? AND relation_type = ? AND tree_id = ?
            """, (source_id, target_id, relation_type, tree_id)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE relations SET weight = ?, created_at = datetime('now') WHERE id = ?",
                    (weight, existing["id"])
                )
            else:
                conn.execute("""
                    INSERT INTO relations (source_id, target_id, relation_type, tree_id, weight)
                    VALUES (?, ?, ?, ?, ?)
                """, (source_id, target_id, relation_type, tree_id, weight))

    def get_children(self, parent_id: str) -> List[Dict[str, Any]]:
        """Get all child nodes of a parent."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT mn.* FROM memory_nodes mn
                JOIN relations r ON mn.id = r.source_id
                WHERE r.target_id = ? AND r.relation_type = 'child_of'
                AND mn.is_deleted = 0
            """, (parent_id,)).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_parents(self, child_id: str) -> List[Dict[str, Any]]:
        """Get all parent nodes of a child."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT mn.* FROM memory_nodes mn
                JOIN relations r ON mn.id = r.target_id
                WHERE r.source_id = ? AND r.relation_type = 'child_of'
                AND mn.is_deleted = 0
            """, (child_id,)).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_sequence_next(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the node that follows this one in temporal sequence."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT mn.* FROM memory_nodes mn
                JOIN relations r ON mn.id = r.target_id
                WHERE r.source_id = ? AND r.relation_type = 'followed_by'
                AND mn.is_deleted = 0
                ORDER BY r.created_at DESC LIMIT 1
            """, (node_id,)).fetchone()
            return self._row_to_dict(row) if row else None

    def get_sequence_prev(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the node that precedes this one in temporal sequence."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT mn.* FROM memory_nodes mn
                JOIN relations r ON mn.id = r.target_id
                WHERE r.source_id = ? AND r.relation_type = 'preceded_by'
                AND mn.is_deleted = 0
                ORDER BY r.created_at DESC LIMIT 1
            """, (node_id,)).fetchone()
            return self._row_to_dict(row) if row else None

    # ---------------------------------------------------------------
    # TREES
    # ---------------------------------------------------------------

    def create_tree(self, tree_id: str, name: str, description: str = "",
                    root_node_id: str = None):
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trees (tree_id, name, description, root_node_id)
                VALUES (?, ?, ?, ?)
            """, (tree_id, name, description, root_node_id))

    def get_tree(self, tree_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM trees WHERE tree_id = ?", (tree_id,)).fetchone()
            return dict(row) if row else None

    def get_all_trees(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM trees").fetchall()
            return [dict(r) for r in rows]

    # ---------------------------------------------------------------
    # EMBEDDINGS
    # ---------------------------------------------------------------

    def _store_embeddings(self, conn: sqlite3.Connection, node: MemoryNode):
        """Store numpy embeddings as BLOBs with precomputed hyperbolic norm."""
        euc = node.euclidean_embedding.astype(np.float32).tobytes() if node.euclidean_embedding is not None else None
        hyp = node.hyperbolic_coords.astype(np.float32).tobytes() if node.hyperbolic_coords is not None else None
        tmp = node.temporal_embedding.astype(np.float32).tobytes() if node.temporal_embedding is not None else None
        hyp_norm = float(np.linalg.norm(node.hyperbolic_coords)) if node.hyperbolic_coords is not None else None
        conn.execute("""
            INSERT OR REPLACE INTO embeddings (node_id, euclidean, hyperbolic, temporal, hyperbolic_norm, is_learned)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (node.id, euc, hyp, tmp, hyp_norm))

    def get_embeddings(self, node_id: str) -> Dict[str, Optional[np.ndarray]]:
        """Load numpy embeddings from BLOBs."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM embeddings WHERE node_id = ?", (node_id,)).fetchone()
            if row is None:
                return {"euclidean": None, "hyperbolic": None, "temporal": None}
            return {
                "euclidean": np.frombuffer(row["euclidean"], dtype=np.float32) if row["euclidean"] else None,
                "hyperbolic": np.frombuffer(row["hyperbolic"], dtype=np.float32) if row["hyperbolic"] else None,
                "temporal": np.frombuffer(row["temporal"], dtype=np.float32) if row["temporal"] else None,
            }

    # ---------------------------------------------------------------
    # CONVERSATION TURNS
    # ---------------------------------------------------------------

    def log_turn(self, turn_data: Dict[str, Any]) -> int:
        """Log a conversation turn with full detail. Returns turn_id."""
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO conversation_turns
                (conversation_id, turn_number, role, content_hash, node_id,
                 gamma, effective_surprise, mode, timestamp,
                 raw_surprise, cluster_precision, nearest_context_id,
                 nearest_context_name, active_tree, is_stale,
                 storage_action, system_prompt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                turn_data["conversation_id"], turn_data["turn_number"],
                turn_data["role"], turn_data.get("content_hash"),
                turn_data.get("node_id"), turn_data.get("gamma"),
                turn_data.get("effective_surprise"), turn_data.get("mode"),
                turn_data.get("timestamp", datetime.now().isoformat()),
                turn_data.get("raw_surprise"),
                turn_data.get("cluster_precision"),
                turn_data.get("nearest_context_id"),
                turn_data.get("nearest_context_name"),
                turn_data.get("active_tree"),
                turn_data.get("is_stale", 0),
                turn_data.get("storage_action"),
                turn_data.get("system_prompt"),
            ))
            turn_id = cursor.lastrowid

            # Skip detail inserts if the INSERT was ignored (duplicate)
            if turn_id and turn_id > 0:
                # Retrieved memories detail
                for mem in turn_data.get("retrieved_memories", []):
                    conn.execute("""
                        INSERT INTO turn_retrieved_memories
                        (turn_id, memory_node_id, rank, final_score,
                         semantic_score, hierarchy_score, temporal_score,
                         precision_score, source, content_preview, node_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        turn_id, mem.get("memory_node_id"), mem.get("rank"),
                        mem.get("final_score"), mem.get("semantic_score"),
                        mem.get("hierarchy_score"), mem.get("temporal_score"),
                        mem.get("precision_score"), mem.get("source"),
                        mem.get("content_preview"), mem.get("node_level"),
                    ))

                # Epistemic questions detail
                for q in turn_data.get("epistemic_questions", []):
                    conn.execute("""
                        INSERT INTO turn_epistemic_questions
                        (turn_id, question_text, information_gain,
                         parent_context_id, parent_context_name,
                         anchor_id, anchor_content)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        turn_id, q.get("question_text"), q.get("information_gain"),
                        q.get("parent_context_id"), q.get("parent_context_name"),
                        q.get("anchor_id"), q.get("anchor_content"),
                    ))

                # Predictive memories detail
                for p in turn_data.get("predictive_memories", []):
                    conn.execute("""
                        INSERT INTO turn_predictive_memories
                        (turn_id, memory_node_id, content_preview,
                         prediction_depth, prediction_frequency)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        turn_id, p.get("memory_node_id"),
                        p.get("content_preview"),
                        p.get("prediction_depth"),
                        p.get("prediction_frequency"),
                    ))

            return turn_id or 0

    def update_turn_response(self, conversation_id: str, turn_number: int, response_summary: str):
        """Update a turn with Claude's response summary (called post-response)."""
        with self._connect() as conn:
            conn.execute("""
                UPDATE conversation_turns
                SET response_summary = ?
                WHERE conversation_id = ? AND turn_number = ?
            """, (response_summary[:2000], conversation_id, turn_number))

    # ---------------------------------------------------------------
    # CONSOLIDATION
    # ---------------------------------------------------------------

    def log_consolidation(self, action: Dict[str, Any]):
        """Log a nightly consolidation action."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO consolidation_log (action, source_node_ids, target_node_id, reason, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                action.get("action", ""),
                json.dumps(action.get("source_node_ids", action.get("orphan_ids", [action.get("child_id", action.get("node_id", ""))]))),
                action.get("target_node_id", action.get("parent_id", action.get("new_context_id", ""))),
                action.get("reason", ""),
                json.dumps(action),
            ))

    def get_prune_candidates(self, max_precision: float, max_access_count: int,
                             max_surprise: float, min_age_days: int) -> List[Dict[str, Any]]:
        """Find Anchors eligible for pruning."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM memory_nodes
                WHERE level = 'ANC'
                  AND is_deleted = 0
                  AND precision < ?
                  AND access_count < ?
                  AND surprise_at_creation < ?
                  AND julianday('now') - julianday(created_at) > ?
            """, (max_precision, max_access_count, max_surprise, min_age_days)).fetchall()
            return [self._row_to_dict(r) for r in rows]

    # ---------------------------------------------------------------
    # RSGD SUPPORT
    # ---------------------------------------------------------------

    def get_all_hyperbolic(self) -> Dict[str, np.ndarray]:
        """Load all hyperbolic embeddings in one query for RSGD."""
        with self._connect() as conn:
            rows = conn.execute("SELECT node_id, hyperbolic FROM embeddings WHERE hyperbolic IS NOT NULL").fetchall()
            result = {}
            for r in rows:
                try:
                    result[r["node_id"]] = np.frombuffer(r["hyperbolic"], dtype=np.float32).copy()
                except Exception:
                    pass
            return result

    def get_child_of_edges(self) -> List[tuple]:
        """Get all child_of edges as (child_id, parent_id) tuples for RSGD."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT r.source_id, r.target_id FROM relations r
                JOIN memory_nodes mn1 ON r.source_id = mn1.id
                JOIN memory_nodes mn2 ON r.target_id = mn2.id
                WHERE r.relation_type = 'child_of'
                  AND mn1.is_deleted = 0 AND mn2.is_deleted = 0
            """).fetchall()
            return [(r["source_id"], r["target_id"]) for r in rows]

    def get_node_levels(self) -> Dict[str, str]:
        """Get level for all active nodes."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, level FROM memory_nodes WHERE is_deleted = 0"
            ).fetchall()
            return {r["id"]: r["level"] for r in rows}

    def batch_update_hyperbolic(self, updates: Dict[str, np.ndarray]):
        """Bulk write updated hyperbolic embeddings after RSGD."""
        with self._connect() as conn:
            for node_id, coords in updates.items():
                blob = coords.astype(np.float32).tobytes()
                norm = float(np.linalg.norm(coords))
                conn.execute("""
                    UPDATE embeddings
                    SET hyperbolic = ?, hyperbolic_norm = ?, is_learned = 1,
                        last_trained = datetime('now')
                    WHERE node_id = ?
                """, (blob, norm, node_id))

    def store_centroid(self, context_id: str, centroid: np.ndarray, child_count: int):
        """Store/update a context's Frechet mean centroid."""
        blob = centroid.astype(np.float32).tobytes()
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO context_centroids
                (context_id, centroid, child_count, last_computed)
                VALUES (?, ?, ?, datetime('now'))
            """, (context_id, blob, child_count))

    def get_centroid(self, context_id: str) -> Optional[np.ndarray]:
        """Load a context's cached Frechet mean centroid."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT centroid FROM context_centroids WHERE context_id = ?",
                (context_id,)
            ).fetchone()
            if row and row["centroid"]:
                return np.frombuffer(row["centroid"], dtype=np.float32).copy()
            return None

    def log_rsgd_run(self, run_data: Dict[str, Any]):
        """Log an RSGD training run."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO rsgd_runs
                (run_type, epochs, final_loss, nodes_updated, edges_used, learning_rate, wall_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_data.get("run_type", "nightly"),
                run_data.get("epochs", 0),
                run_data.get("final_loss", 0),
                run_data.get("nodes_updated", 0),
                run_data.get("edges_used", 0),
                run_data.get("learning_rate", 0),
                run_data.get("wall_time_seconds", 0),
            ))

    # ---------------------------------------------------------------
    # CONTEXT STATS (for precision computation)
    # ---------------------------------------------------------------

    def get_context_stats(self, context_id: str) -> Dict[str, Any]:
        """
        Get stats for precision computation. P2a: reads from materialized cache first,
        falls back to live query if cache miss, then caches the result.
        """
        with self._connect() as conn:
            # Try cache first
            cached = conn.execute(
                "SELECT * FROM context_stats_cache WHERE context_id = ?",
                (context_id,)
            ).fetchone()

            if cached:
                return {
                    "id": context_id,
                    "num_anchors": cached["child_count"],
                    "avg_recency_hours": cached["avg_recency_hours"],
                    "internal_consistency": cached["internal_consistency"],
                }

        # Cache miss — compute live and cache
        return self._refresh_context_stats(context_id)

    def _refresh_context_stats(self, context_id: str) -> Dict[str, Any]:
        """Recompute and cache context stats. Called on child attach/detach."""
        with self._connect() as conn:
            child_count = conn.execute("""
                SELECT COUNT(*) as cnt FROM relations
                WHERE target_id = ? AND relation_type = 'child_of'
            """, (context_id,)).fetchone()["cnt"]

            avg_recency = conn.execute("""
                SELECT AVG(
                    (julianday('now') - julianday(mn.last_modified)) * 24
                ) as avg_hours
                FROM memory_nodes mn
                JOIN relations r ON mn.id = r.source_id
                WHERE r.target_id = ? AND r.relation_type = 'child_of'
                AND mn.is_deleted = 0
            """, (context_id,)).fetchone()["avg_hours"]

            stats = {
                "id": context_id,
                "num_anchors": child_count or 0,
                "avg_recency_hours": avg_recency or 720,
                "internal_consistency": 0.5,
            }

            # Write to cache
            conn.execute("""
                INSERT OR REPLACE INTO context_stats_cache
                (context_id, child_count, avg_recency_hours, internal_consistency, last_updated)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (context_id, stats["num_anchors"], stats["avg_recency_hours"],
                  stats["internal_consistency"]))

            return stats

    # ---------------------------------------------------------------
    # P1a: CHROMA ANN QUERY (pass-through)
    # ---------------------------------------------------------------

    def ann_query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 20,
        level_filter: str = None,
        tree_filter: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Fast ANN retrieval via ChromaDB. Returns candidate IDs + distances.
        Falls back to empty list if ChromaDB not available (caller does linear scan).
        """
        if self._chroma:
            return self._chroma.query(query_embedding, n_results, level_filter, tree_filter)
        return []

    @property
    def has_ann_index(self) -> bool:
        return self._chroma is not None and self._chroma.enabled

    # ---------------------------------------------------------------
    # P2b: EMBEDDING MODEL VERSION TRACKING
    # ---------------------------------------------------------------

    def get_embedding_model(self) -> Optional[str]:
        """Get the embedding model used for stored vectors."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM system_meta WHERE key = 'embedding_model'"
            ).fetchone()
            return row["value"] if row else None

    def set_embedding_model(self, model_name: str):
        """Record which embedding model is being used."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_meta (key, value) VALUES ('embedding_model', ?)",
                (model_name,)
            )

    def check_embedding_model_consistency(self, current_model: str) -> bool:
        """
        Check if current model matches stored model.
        Returns True if consistent, False if mismatch (needs re-embedding).
        On first run, sets the model and returns True.
        """
        stored = self.get_embedding_model()
        if stored is None:
            self.set_embedding_model(current_model)
            return True
        if stored != current_model:
            print(f"[WARNING] Embedding model mismatch: stored='{stored}', current='{current_model}'")
            print(f"[WARNING] Existing embeddings may be incompatible. Run re-embedding migration.")
            return False
        return True

    # ---------------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        # Parse JSON fields
        for json_field in ["parent_ids", "tree_ids"]:
            if json_field in d and isinstance(d[json_field], str):
                try:
                    d[json_field] = json.loads(d[json_field])
                except (json.JSONDecodeError, TypeError):
                    d[json_field] = []
        return d

    def count_nodes(self, level: str = None) -> int:
        """Count nodes, optionally filtered by level."""
        with self._connect() as conn:
            if level:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM memory_nodes WHERE level = ? AND is_deleted = 0",
                    (level,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM memory_nodes WHERE is_deleted = 0"
                ).fetchone()
            return row["cnt"]

    # ---------------------------------------------------------------
    # PROJECTS — CRUD
    # ---------------------------------------------------------------

    def create_project(self, project: Dict[str, Any]) -> str:
        """Insert a new project. Returns project ID."""
        pid = project.get("id", str(uuid.uuid4())[:10])
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO projects
                (id, name, description, company, source, source_id, status,
                 owner, deadline, sc_node_id, expected_hours, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, (
                pid, project["name"], project.get("description", ""),
                project.get("company", ""), project.get("source", "manual"),
                project.get("source_id", ""), project.get("status", "active"),
                project.get("owner", ""), project.get("deadline", ""),
                project.get("sc_node_id", ""), project.get("expected_hours", 0),
            ))
        return pid

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            return dict(row) if row else None

    def list_projects(self, status: str = None) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if status:
                rows = conn.execute("SELECT * FROM projects WHERE status = ?", (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM projects").fetchall()
            return [dict(r) for r in rows]

    def update_project(self, project_id: str, updates: Dict[str, Any]):
        with self._connect() as conn:
            set_clauses = []
            values = []
            for k, v in updates.items():
                if k != "id":
                    set_clauses.append(f"{k} = ?")
                    values.append(v)
            set_clauses.append("updated_at = datetime('now')")
            values.append(project_id)
            conn.execute(
                f"UPDATE projects SET {', '.join(set_clauses)} WHERE id = ?",
                values
            )

    def delete_project(self, project_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM deliverables WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    # ---------------------------------------------------------------
    # DELIVERABLES — CRUD
    # ---------------------------------------------------------------

    def create_deliverable(self, deliverable: Dict[str, Any]) -> str:
        did = deliverable.get("id", str(uuid.uuid4())[:10])
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO deliverables
                (id, project_id, name, description, context_node_id, anchor_node_ids,
                 status, deadline, expected_hours, source, source_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, (
                did, deliverable["project_id"], deliverable["name"],
                deliverable.get("description", ""),
                deliverable.get("context_node_id", ""),
                json.dumps(deliverable.get("anchor_node_ids", [])),
                deliverable.get("status", "active"),
                deliverable.get("deadline", ""),
                deliverable.get("expected_hours", 0),
                deliverable.get("source", "manual"),
                deliverable.get("source_id", ""),
            ))
        return did

    def get_deliverables(self, project_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM deliverables WHERE project_id = ?", (project_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def update_deliverable(self, deliverable_id: str, updates: Dict[str, Any]):
        with self._connect() as conn:
            set_clauses = []
            values = []
            for k, v in updates.items():
                if k != "id":
                    set_clauses.append(f"{k} = ?")
                    values.append(v if not isinstance(v, list) else json.dumps(v))
            set_clauses.append("updated_at = datetime('now')")
            values.append(deliverable_id)
            conn.execute(
                f"UPDATE deliverables SET {', '.join(set_clauses)} WHERE id = ?",
                values
            )

    def delete_deliverable(self, deliverable_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM deliverables WHERE id = ?", (deliverable_id,))

    # ---------------------------------------------------------------
    # PROJECT-WORK MATCH LOG
    # ---------------------------------------------------------------

    def log_match(self, match: Dict[str, Any]) -> str:
        mid = match.get("id", str(uuid.uuid4())[:10])
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO project_work_match_log
                (id, segment_id, project_id, deliverable_id,
                 sc_node_id, context_node_id, anchor_node_id,
                 semantic_score, hyperbolic_score, combined_match_pct,
                 match_method, work_description, worker_type, time_mins, matched_at, is_correct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
            """, (
                mid, match["segment_id"], match["project_id"],
                match.get("deliverable_id", ""),
                match.get("sc_node_id", ""), match.get("context_node_id", ""),
                match.get("anchor_node_id", ""),
                match.get("semantic_score", 0), match.get("hyperbolic_score", 0),
                match.get("combined_match_pct", 0),
                match.get("match_method", ""), match.get("work_description", ""),
                match.get("worker_type", ""), match.get("time_mins", 0),
                match.get("is_correct", -1),
            ))
        return mid

    def get_match_log(self, project_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if project_id:
                rows = conn.execute(
                    "SELECT * FROM project_work_match_log WHERE project_id = ? ORDER BY matched_at DESC LIMIT ?",
                    (project_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM project_work_match_log ORDER BY matched_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_match_quality_stats(self) -> Dict[str, Any]:
        """Aggregate match quality statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM project_work_match_log").fetchone()["cnt"]
            if total == 0:
                return {"total_matches": 0, "avg_match_pct": 0, "verified": 0, "correct": 0}
            avg = conn.execute("SELECT AVG(combined_match_pct) as avg FROM project_work_match_log").fetchone()["avg"]
            verified = conn.execute(
                "SELECT COUNT(*) as cnt FROM project_work_match_log WHERE is_correct != -1"
            ).fetchone()["cnt"]
            correct = conn.execute(
                "SELECT COUNT(*) as cnt FROM project_work_match_log WHERE is_correct = 1"
            ).fetchone()["cnt"]
            return {
                "total_matches": total,
                "avg_match_pct": round(avg or 0, 2),
                "verified": verified,
                "correct": correct,
                "accuracy_pct": round(correct / verified * 100, 1) if verified > 0 else 0,
            }

    # ---------------------------------------------------------------
    # PRODUCTIVITY SYNC LOG
    # ---------------------------------------------------------------

    def log_sync(self, sync_data: Dict[str, Any]) -> str:
        sid = sync_data.get("id", str(uuid.uuid4())[:10])
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO productivity_sync_log
                (id, sync_type, triggered_at, segments_processed,
                 nodes_created, nodes_updated, matches_found,
                 avg_match_pct, rsgd_epochs_run, completed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sid, sync_data.get("sync_type", "30min"),
                sync_data.get("triggered_at", datetime.now().isoformat()),
                sync_data.get("segments_processed", 0),
                sync_data.get("nodes_created", 0),
                sync_data.get("nodes_updated", 0),
                sync_data.get("matches_found", 0),
                sync_data.get("avg_match_pct", 0),
                sync_data.get("rsgd_epochs_run", 0),
                sync_data.get("completed_at"),
                sync_data.get("status", "running"),
            ))
        return sid

    def update_sync_status(self, sync_id: str, status: str, completed_at: str = None):
        with self._connect() as conn:
            conn.execute(
                "UPDATE productivity_sync_log SET status = ?, completed_at = ? WHERE id = ?",
                (status, completed_at or datetime.now().isoformat(), sync_id)
            )

    def get_sync_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM productivity_sync_log ORDER BY triggered_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ---------------------------------------------------------------
    # PRODUCTIVITY TIME UPDATES
    # ---------------------------------------------------------------

    def update_node_productivity_time(self, node_id: str, time_mins: float,
                                       human_mins: float, ai_mins: float):
        """Increment productivity time on a memory node."""
        with self._connect() as conn:
            conn.execute("""
                UPDATE memory_nodes SET
                    productivity_time_mins = productivity_time_mins + ?,
                    productivity_human_mins = productivity_human_mins + ?,
                    productivity_ai_mins = productivity_ai_mins + ?,
                    last_productivity_sync = datetime('now'),
                    last_modified = datetime('now')
                WHERE id = ?
            """, (time_mins, human_mins, ai_mins, node_id))

    def get_productivity_by_sc(self, date_filter: str = None) -> List[Dict[str, Any]]:
        """Get productivity time grouped by SC, with contexts and anchors.
        Deduplicates SCs with same content name by summing their time."""
        with self._connect() as conn:
            # Deduplicated: group by content, sum time, pick the id with most time
            scs = conn.execute("""
                SELECT id, content, level,
                       SUM(productivity_time_mins) as productivity_time_mins,
                       SUM(productivity_human_mins) as productivity_human_mins,
                       SUM(productivity_ai_mins) as productivity_ai_mins
                FROM memory_nodes
                WHERE level = 'SC' AND is_deleted = 0
                  AND productivity_time_mins > 0
                GROUP BY content
                ORDER BY SUM(productivity_time_mins) DESC
            """).fetchall()

            result = []
            for sc in scs:
                sc_dict = dict(sc)
                contexts = conn.execute("""
                    SELECT mn.id, mn.content, mn.productivity_time_mins,
                           mn.productivity_human_mins, mn.productivity_ai_mins
                    FROM memory_nodes mn
                    JOIN relations r ON mn.id = r.source_id
                    WHERE r.target_id = ? AND r.relation_type = 'child_of'
                      AND mn.is_deleted = 0 AND mn.productivity_time_mins > 0
                    ORDER BY mn.productivity_time_mins DESC
                """, (sc["id"],)).fetchall()

                ctx_list = []
                for ctx in contexts:
                    ctx_dict = dict(ctx)
                    anchors = conn.execute("""
                        SELECT mn.id, mn.content, mn.productivity_time_mins,
                               mn.productivity_human_mins, mn.productivity_ai_mins
                        FROM memory_nodes mn
                        JOIN relations r ON mn.id = r.source_id
                        WHERE r.target_id = ? AND r.relation_type = 'child_of'
                          AND mn.is_deleted = 0 AND mn.productivity_time_mins > 0
                        ORDER BY mn.productivity_time_mins DESC
                    """, (ctx["id"],)).fetchall()
                    ctx_dict["anchors"] = [dict(a) for a in anchors]
                    ctx_list.append(ctx_dict)

                sc_dict["contexts"] = ctx_list
                result.append(sc_dict)
            return result
