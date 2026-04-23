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

        # Migration: Phase 3 — value_score columns on memory_nodes
        for col, typedef in [
            ("value_score", "REAL DEFAULT 0.0"),
            ("value_goal", "REAL DEFAULT 0.0"),
            ("value_feedback", "REAL DEFAULT 0.0"),
            ("value_usage", "REAL DEFAULT 0.0"),
            ("value_recency", "REAL DEFAULT 0.0"),
            ("value_computed_at", "TEXT DEFAULT ''"),
            # Audit-fix: protect user-authored content from auto-rewrite
            ("is_user_edited", "INTEGER DEFAULT 0"),
        ]:
            try:
                self._conn.execute(f"SELECT {col} FROM memory_nodes LIMIT 1")
            except Exception:
                try:
                    self._conn.execute(f"ALTER TABLE memory_nodes ADD COLUMN {col} {typedef}")
                    self._conn.commit()
                except Exception:
                    pass
        try:
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_value_score ON memory_nodes(value_score)")
            self._conn.commit()
        except Exception:
            pass

        # Migration: Phase 5 — wiki staleness tracking
        for col, typedef in [
            ("value_score_avg_at_render", "REAL DEFAULT 0.0"),
            ("pitfall_count_at_render", "INTEGER DEFAULT 0"),
        ]:
            try:
                self._conn.execute(f"SELECT {col} FROM wiki_page_cache LIMIT 1")
            except Exception:
                try:
                    self._conn.execute(f"ALTER TABLE wiki_page_cache ADD COLUMN {col} {typedef}")
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

        # Migration: add value_score / value_multiplier columns to
        # turn_retrieved_memories so the per-retrieval multiplier effect is
        # introspectable at query time (not just baked into final_score).
        for col, coltype in [
            ("value_score", "REAL"),
            ("value_multiplier", "REAL"),
        ]:
            try:
                self._conn.execute(f"SELECT {col} FROM turn_retrieved_memories LIMIT 1")
            except Exception:
                try:
                    self._conn.execute(
                        f"ALTER TABLE turn_retrieved_memories ADD COLUMN {col} {coltype}"
                    )
                    self._conn.commit()
                except Exception:
                    pass

        # Migration: create restructure queue/log tables for LLM regen of
        # red-flagged nodes (audit-fix Item 7). Idempotent via IF NOT EXISTS.
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS restructure_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                reason TEXT DEFAULT '',
                queued_at TEXT DEFAULT (datetime('now')),
                processed_at TEXT,
                status TEXT DEFAULT 'queued'
            );
            CREATE TABLE IF NOT EXISTS restructure_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                trigger_reason TEXT DEFAULT '',
                before_content TEXT DEFAULT '',
                after_content TEXT DEFAULT '',
                applied_by TEXT DEFAULT '',
                run_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_restructure_queue_status ON restructure_queue(status);
            CREATE INDEX IF NOT EXISTS idx_restructure_queue_node ON restructure_queue(node_id);
            CREATE INDEX IF NOT EXISTS idx_restructure_log_node ON restructure_log(node_id);
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

        # Migration (2026-04-20, Phase 2): project-matching status columns
        for table, col, coltype in [
            ("project_work_match_log", "source", "TEXT DEFAULT 'semantic'"),
            ("activity_time_log", "project_id", "TEXT DEFAULT ''"),
            ("activity_time_log", "match_source", "TEXT DEFAULT ''"),
            ("activity_segments", "consolidated_into_node_id", "TEXT DEFAULT ''"),
        ]:
            try:
                self._conn.execute(f"SELECT {col} FROM {table} LIMIT 1")
            except Exception:
                try:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
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
                is_correct INTEGER DEFAULT -1,
                source TEXT DEFAULT 'semantic'  -- 'session_tag'|'semantic'|'manual'|'manual_consolidation'
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

        # Migration: telemetry tables for future-scope personalization training.
        # (Added 2026-04-15 — Track 2 prep work for personal LoRA / reranker training.)
        # Schema is intentionally simple and append-only; queries only happen at
        # training time, never at retrieval time, so no indexing concerns yet.
        self._conn.executescript("""
            -- Every retrieval becomes a training-data row. Pair (query, used vs unused
            -- nodes) is the supervised signal for a future reranker.
            CREATE TABLE IF NOT EXISTS retrieval_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT DEFAULT (datetime('now')),
                conversation_id TEXT,
                turn_number INTEGER,
                query_text TEXT,
                query_embedding BLOB,
                retrieved_node_ids TEXT,    -- JSON array
                retrieved_scores TEXT,      -- JSON array, parallel to node_ids
                actually_used_node_ids TEXT,-- JSON array of nodes referenced in response
                user_feedback TEXT,         -- 'up'|'down'|null
                response_excerpt TEXT       -- first 500 chars of response, for audit
            );
            CREATE INDEX IF NOT EXISTS idx_retrlog_conv ON retrieval_log(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_retrlog_ts ON retrieval_log(ts);

            -- Every parent-assignment decision becomes training data for a future
            -- "where does this memory go" classifier.
            CREATE TABLE IF NOT EXISTS classification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                ts TEXT DEFAULT (datetime('now')),
                chosen_parent_id TEXT,
                candidate_parent_ids TEXT,  -- JSON array, top-K considered
                candidate_scores TEXT,      -- JSON array, parallel
                decided_by TEXT,            -- 'consolidation'|'llm'|'manual'|'rules'
                user_corrected_to TEXT      -- if user later moves this node
            );
            CREATE INDEX IF NOT EXISTS idx_clslog_node ON classification_log(node_id);
        """)

        # Migration: tag content authorship on memory_nodes for future style adapter.
        # 'user' = user-authored anchor, 'llm' = LLM-generated (consolidation/auto-context),
        # 'consolidation' = produced by COMPRESS/BIRTH summary, 'manual' = direct entry.
        try:
            cols = [r[1] for r in self._conn.execute(
                "PRAGMA table_info(memory_nodes)"
            ).fetchall()]
            if "authored_by" not in cols:
                self._conn.execute(
                    "ALTER TABLE memory_nodes ADD COLUMN authored_by TEXT DEFAULT 'unknown'"
                )
        except Exception:
            pass

        # Migration (Phase 1 sync protocol): backfill blank match_source to 'nightly'
        # so pre-existing rows have a non-empty author tag, then enforce uniqueness on
        # (segment_id, date). This is the DB-level guard that prevents a manual run and
        # a nightly run from double-counting the same segment on the same date.
        try:
            self._conn.execute(
                "UPDATE activity_time_log SET match_source='nightly' "
                "WHERE match_source IS NULL OR match_source = ''"
            )
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_atl_segment_date "
                "ON activity_time_log(segment_id, date)"
            )
        except Exception:
            pass

        # Migration (Phase 3 review redesign): in-flight cluster proposals that
        # sit between the user clicking Consolidate and confirming/rejecting a
        # group. status transitions: draft -> confirmed|rejected|superseded.
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS review_proposals (
                id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT (datetime('now')),
                target_date TEXT NOT NULL,
                author TEXT NOT NULL DEFAULT 'user',
                status TEXT NOT NULL DEFAULT 'draft',
                proposed_content TEXT,
                segment_ids_json TEXT NOT NULL,
                project_probs_json TEXT,
                user_assigned_project_id TEXT DEFAULT '',
                user_assigned_deliverable_id TEXT DEFAULT '',
                anchor_node_id TEXT DEFAULT '',
                confirmed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_rp_date_status
                ON review_proposals(target_date, status);
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

    def get_all_trees(self, include_dead: bool = False) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if include_dead:
                rows = conn.execute("SELECT * FROM trees").fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT t.* FROM trees t
                    LEFT JOIN memory_nodes n ON n.id = t.root_node_id
                    WHERE t.root_node_id IS NULL OR n.is_deleted = 0
                    """
                ).fetchall()
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

    def refresh_node_embedding(self, node_id: str, new_euclidean: np.ndarray) -> bool:
        """
        Refresh a node's Euclidean embedding after content changes.
        Updates the embeddings BLOB + syncs ChromaDB ANN index. Hyperbolic
        and temporal embeddings are NOT touched (HGCN owns hyperbolic;
        temporal is timestamp-based and doesn't drift with content).

        Call this whenever a node's `content` is mutated (LLM regen, user
        edit) so semantic ranking + ChromaDB stay in sync. Returns True if
        the embedding was refreshed, False otherwise.
        """
        if new_euclidean is None:
            return False
        blob = new_euclidean.astype(np.float32).tobytes()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE embeddings SET euclidean = ? WHERE node_id = ?",
                (blob, node_id),
            )
            if cur.rowcount == 0:
                return False
            row = conn.execute(
                "SELECT level, is_orphan, tree_ids FROM memory_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()

        if self._chroma and row:
            try:
                tree_ids = row["tree_ids"] or "[]"
                if isinstance(tree_ids, str):
                    try:
                        tree_ids = json.loads(tree_ids)
                    except Exception:
                        tree_ids = []
                self._chroma.add(node_id, new_euclidean, metadata={
                    "level": row["level"],
                    "is_orphan": bool(row["is_orphan"]),
                    "tree_ids": tree_ids,
                })
            except Exception as e:
                # Don't fail the embedding refresh because Chroma is sad
                print(f"[refresh_node_embedding] ChromaDB sync error for {node_id}: {e}")
        return True

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
                         precision_score, source, content_preview, node_level,
                         value_score, value_multiplier)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        turn_id, mem.get("memory_node_id"), mem.get("rank"),
                        mem.get("final_score"), mem.get("semantic_score"),
                        mem.get("hierarchy_score"), mem.get("temporal_score"),
                        mem.get("precision_score"), mem.get("source"),
                        mem.get("content_preview"), mem.get("node_level"),
                        mem.get("value_score"), mem.get("value_multiplier"),
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
    # TURN DIAGNOSTICS
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # GOALS & FEEDBACK
    # ---------------------------------------------------------------

    def create_goal(self, goal_id: str, title: str, description: str = "",
                    status: str = "active") -> str:
        """Create a new goal. Returns goal_id."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO goals (id, title, description, status)
                VALUES (?, ?, ?, ?)
            """, (goal_id, title, description, status))
        return goal_id

    def update_goal(self, goal_id: str, **kwargs) -> bool:
        """Update goal fields (title, description, status)."""
        allowed = {"title", "description", "status"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [goal_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE goals SET {set_clause} WHERE id = ?", values)
        return True

    def get_goal(self, goal_id: str) -> Optional[Dict]:
        """Get a single goal by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()
            return dict(row) if row else None

    def list_goals(self, status: Optional[str] = None) -> List[Dict]:
        """List all goals, optionally filtered by status."""
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM goals WHERE status = ? ORDER BY created_at DESC", (status,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM goals ORDER BY created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def link_goal_to_node(self, goal_id: str, node_id: str,
                          link_type: str = "supports", weight: float = 0.5) -> None:
        """Create or update a goal-node link."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO goal_links (goal_id, node_id, link_type, weight)
                VALUES (?, ?, ?, ?)
            """, (goal_id, node_id, link_type, weight))

    def unlink_goal_from_node(self, goal_id: str, node_id: str) -> None:
        """Remove a goal-node link."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM goal_links WHERE goal_id = ? AND node_id = ?",
                (goal_id, node_id)
            )

    def get_goals_for_node(self, node_id: str) -> List[Dict]:
        """Get all goals linked to a node."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT g.*, gl.link_type, gl.weight
                FROM goals g
                JOIN goal_links gl ON gl.goal_id = g.id
                WHERE gl.node_id = ?
                ORDER BY gl.weight DESC
            """, (node_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_nodes_for_goal(self, goal_id: str) -> List[Dict]:
        """Get all nodes linked to a goal with their link info."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT mn.id, mn.content, mn.level, gl.link_type, gl.weight
                FROM memory_nodes mn
                JOIN goal_links gl ON gl.node_id = mn.id
                WHERE gl.goal_id = ? AND mn.is_deleted = 0
                ORDER BY gl.weight DESC
            """, (goal_id,)).fetchall()
            return [dict(r) for r in rows]

    def add_feedback(self, node_id: str, polarity: str, content: str = "",
                     goal_id: Optional[str] = None, source: str = "explicit",
                     strength: float = 1.0) -> int:
        """Add a feedback entry. Returns feedback ID."""
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO feedback (node_id, goal_id, polarity, content, source, strength)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (node_id, goal_id, polarity, content, source, strength))
            return cursor.lastrowid or 0

    def get_feedback_for_node(self, node_id: str, limit: int = 20) -> List[Dict]:
        """Get feedback entries for a node, most recent first."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT f.*, g.title as goal_title
                FROM feedback f
                LEFT JOIN goals g ON g.id = f.goal_id
                WHERE f.node_id = ?
                ORDER BY f.timestamp DESC
                LIMIT ?
            """, (node_id, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_feedback_score(self, node_id: str) -> float:
        """Compute net feedback score: positive*0.10 - negative*0.15, clamped."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COALESCE(SUM(CASE WHEN polarity='positive' THEN strength * 0.10 ELSE 0 END), 0)
                    - COALESCE(SUM(CASE WHEN polarity='negative' THEN strength * 0.15 ELSE 0 END), 0)
                    as score
                FROM feedback
                WHERE node_id = ?
            """, (node_id,)).fetchone()
            raw = row["score"] if row else 0.0
            return max(-1.0, min(1.0, raw))

    def get_feedback_summary(self, tree_id: Optional[str] = None,
                             node_id: Optional[str] = None) -> Dict:
        """Get feedback summary (positive/negative counts) for a tree or node."""
        with self._connect() as conn:
            if node_id:
                where = "f.node_id = ?"
                params = (node_id,)
            elif tree_id:
                where = """f.node_id IN (
                    SELECT mn.id FROM memory_nodes mn
                    WHERE mn.tree_ids LIKE ? AND mn.is_deleted = 0
                )"""
                params = (f"%{tree_id}%",)
            else:
                where = "1=1"
                params = ()

            row = conn.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN polarity='positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN polarity='negative' THEN 1 ELSE 0 END) as negative,
                    SUM(CASE WHEN polarity='correction' THEN 1 ELSE 0 END) as corrections
                FROM feedback f
                WHERE {where}
            """, params).fetchone()
            return dict(row) if row else {"total": 0, "positive": 0, "negative": 0, "corrections": 0}

    def log_diagnostics(self, row) -> None:
        """
        Log a complete diagnostic row for one turn.
        Accepts a DiagnosticRow dataclass from core.diagnostics.
        """
        from core.diagnostics import DIAGNOSTIC_COLUMNS, diagnostic_row_to_tuple

        placeholders = ", ".join(["?"] * len(DIAGNOSTIC_COLUMNS))
        columns = ", ".join(DIAGNOSTIC_COLUMNS)

        with self._connect() as conn:
            conn.execute(f"""
                INSERT OR REPLACE INTO turn_diagnostics ({columns})
                VALUES ({placeholders})
            """, diagnostic_row_to_tuple(row))

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

    def get_active_deliverable_candidates(self) -> List[Dict[str, Any]]:
        """
        Return every scoreable target for the project matcher: each active
        deliverable, plus every anchor attached to it, as flat rows with the
        fields the matcher needs (embeddings resolved by caller via get_embeddings).

        Rows contain: deliverable_id, project_id, anchor_node_id (or '' for
        deliverable-as-target), context_node_id, sc_node_id, name, deadline.
        Anchors are preferred targets (finer grain); deliverable rows are a
        fallback when no anchors are attached.
        """
        candidates: List[Dict[str, Any]] = []
        with self._connect() as conn:
            deliv_rows = conn.execute("""
                SELECT d.id as deliverable_id, d.project_id, d.name, d.deadline,
                       d.context_node_id, d.anchor_node_ids, p.sc_node_id
                FROM deliverables d
                JOIN projects p ON p.id = d.project_id
                WHERE d.status = 'active' AND p.status = 'active'
            """).fetchall()
            for r in deliv_rows:
                anchor_ids: List[str] = []
                try:
                    raw = r["anchor_node_ids"] or "[]"
                    anchor_ids = json.loads(raw) if isinstance(raw, str) else list(raw)
                except (ValueError, TypeError):
                    anchor_ids = []

                base = {
                    "deliverable_id": r["deliverable_id"],
                    "project_id": r["project_id"],
                    "context_node_id": r["context_node_id"] or "",
                    "sc_node_id": r["sc_node_id"] or "",
                    "name": r["name"],
                    "deadline": r["deadline"] or "",
                }
                if anchor_ids:
                    for aid in anchor_ids:
                        candidates.append({**base, "anchor_node_id": aid})
                else:
                    candidates.append({**base, "anchor_node_id": ""})
        return candidates

    # ---------------------------------------------------------------
    # WORK SESSIONS (Phase 1 — live matcher)
    # ---------------------------------------------------------------

    def create_work_session(
        self,
        project_id: str = "",
        deliverable_id: str = "",
        auto_assigned: int = 0,
        confirmed_by_user: int = 0,
        note: str = "",
    ) -> str:
        sid = str(uuid.uuid4())[:12]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO work_sessions
                   (id, project_id, deliverable_id, auto_assigned, confirmed_by_user, note)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (sid, project_id, deliverable_id, auto_assigned, confirmed_by_user, note),
            )
        return sid

    def update_work_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        sets, vals = [], []
        for k, v in updates.items():
            if k == "id":
                continue
            sets.append(f"{k} = ?")
            vals.append(v)
        if not sets:
            return
        vals.append(session_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE work_sessions SET {', '.join(sets)} WHERE id = ?", vals)

    def end_work_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE work_sessions SET ended_at = datetime('now') WHERE id = ? AND ended_at IS NULL",
                (session_id,),
            )

    def auto_end_stale_work_sessions(self, max_age_hours: float = 6.0) -> List[str]:
        """Close any work_session whose ended_at IS NULL and started_at is older
        than max_age_hours. Called on server startup to prevent stuck sessions
        (e.g. from crashed tracker or server) from blocking /api/work/start.
        Returns the list of session ids that were force-closed."""
        cutoff_expr = f"datetime('now', '-{int(max_age_hours * 3600)} seconds')"
        with self._connect() as conn:
            stale = conn.execute(
                f"""SELECT id FROM work_sessions
                    WHERE ended_at IS NULL AND started_at < {cutoff_expr}"""
            ).fetchall()
            if not stale:
                return []
            ids = [r["id"] for r in stale]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"""UPDATE work_sessions
                    SET ended_at = datetime('now'),
                        note = COALESCE(note,'') || ' [auto-closed: >'
                            || ? || 'h stale]'
                    WHERE id IN ({placeholders})""",
                [str(max_age_hours)] + ids,
            )
            return ids

    def get_active_work_session(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM work_sessions
                   WHERE ended_at IS NULL
                   ORDER BY started_at DESC LIMIT 1"""
            ).fetchone()
            return dict(row) if row else None

    def get_work_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM work_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_work_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM work_sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def upsert_segment_override(
        self,
        segment_id: str,
        session_id: str,
        project_id: str,
        deliverable_id: str,
        source: str = "session",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO segment_override_bindings
                   (segment_id, session_id, project_id, deliverable_id, source)
                   VALUES (?, ?, ?, ?, ?)""",
                (segment_id, session_id, project_id, deliverable_id, source),
            )

    def get_segment_override(self, segment_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM segment_override_bindings WHERE segment_id = ?",
                (segment_id,),
            ).fetchone()
            return dict(row) if row else None

    # ---------------------------------------------------------------
    # AGENT HARNESSES + TRAINING EVENTS (Phase 4)
    # ---------------------------------------------------------------

    def create_harness(self, record: Dict[str, Any]) -> str:
        hid = record.get("id") or str(uuid.uuid4())[:12]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO agent_harnesses
                   (id, deliverable_id, project_id, title, bundle_path,
                    problem_statement_md, anchors_used, pattern_signature,
                    trigger_source, pre_run_features, mode, model_used, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    hid,
                    record["deliverable_id"],
                    record.get("project_id", ""),
                    record.get("title", ""),
                    record["bundle_path"],
                    record.get("problem_statement_md", ""),
                    json.dumps(record.get("anchors_used") or []),
                    record.get("pattern_signature", ""),
                    record.get("trigger_source", "manual"),
                    json.dumps(record.get("pre_run_features") or {}),
                    record.get("mode", "template"),
                    record.get("model_used", ""),
                    record.get("status", "ready"),
                ),
            )
        return hid

    def get_harness(self, hid: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_harnesses WHERE id = ?", (hid,)
            ).fetchone()
            return dict(row) if row else None

    def list_harnesses(
        self, deliverable_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if deliverable_id:
                rows = conn.execute(
                    """SELECT * FROM agent_harnesses
                       WHERE deliverable_id = ? AND status != 'archived'
                       ORDER BY updated_at DESC LIMIT ?""",
                    (deliverable_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM agent_harnesses WHERE status != 'archived'
                       ORDER BY updated_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def record_harness_run(
        self,
        harness_id: str,
        thumb: Optional[str] = None,    # 'up' | 'down' | None (just record a run)
        outcome: Optional[str] = None,  # 'goal_achieved' | 'goal_unchanged'
        post_run_signals: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Increment run counters, update thumbs, recompute success_rate. Also
        writes a training_events row so Phase 6 corpus gets the label."""
        post_run_signals = post_run_signals or {}
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_count, thumbs_up, thumbs_down, post_run_signals FROM agent_harnesses WHERE id = ?",
                (harness_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"harness_not_found: {harness_id}")
            run_count = (row["run_count"] or 0) + 1
            thumbs_up = row["thumbs_up"] or 0
            thumbs_down = row["thumbs_down"] or 0
            if thumb == "up":
                thumbs_up += 1
            elif thumb == "down":
                thumbs_down += 1
            total_votes = thumbs_up + thumbs_down
            success_rate = thumbs_up / total_votes if total_votes else 0.0
            existing_signals = json.loads(row["post_run_signals"] or "{}")
            existing_signals.setdefault("runs", []).append({
                "thumb": thumb, "outcome": outcome, "at": datetime.now().isoformat(timespec="seconds"),
                **post_run_signals,
            })
            conn.execute(
                """UPDATE agent_harnesses
                   SET run_count = ?, thumbs_up = ?, thumbs_down = ?,
                       success_rate = ?, post_run_signals = ?,
                       last_run_at = datetime('now'), updated_at = datetime('now')
                   WHERE id = ?""",
                (run_count, thumbs_up, thumbs_down, success_rate,
                 json.dumps(existing_signals), harness_id),
            )

        # Log training event (harness_outcome)
        if thumb or outcome:
            self.log_training_event({
                "event_type": "harness_outcome",
                "harness_id": harness_id,
                "features": {"run_count": run_count},
                "label": {"thumb": thumb, "outcome": outcome, "success_rate": success_rate},
            })

    def log_training_event(self, record: Dict[str, Any]) -> str:
        tid = record.get("id") or str(uuid.uuid4())[:12]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO training_events
                   (id, event_type, segment_id, node_id, deliverable_id,
                    harness_id, features, label, pmis_version, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tid,
                    record["event_type"],
                    record.get("segment_id", ""),
                    record.get("node_id", ""),
                    record.get("deliverable_id", ""),
                    record.get("harness_id", ""),
                    json.dumps(record.get("features") or {}),
                    json.dumps(record.get("label") or {}),
                    record.get("pmis_version", "phase-4a"),
                    record.get("model_version", ""),
                ),
            )
        return tid

    # ---------------------------------------------------------------
    # SEGMENT ARTIFACTS (Step 2 — boilerplate detector)
    # ---------------------------------------------------------------

    def upsert_segment_artifact(self, record: Dict[str, Any]) -> str:
        aid = record.get("id") or str(uuid.uuid4())[:12]
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO segment_artifacts
                   (id, segment_id, artifact_type, path_or_uri, content_hash,
                    preview, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    aid,
                    record["segment_id"],
                    record["artifact_type"],
                    record.get("path_or_uri", ""),
                    record.get("content_hash", ""),
                    record.get("preview", ""),
                    record.get("source", "heuristic"),
                ),
            )
        return aid

    def list_artifact_clusters(
        self, min_repetitions: int = 3, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Boilerplate candidates: content_hash buckets with ≥N repetitions."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT content_hash, artifact_type,
                          COUNT(*) AS reps,
                          MIN(preview) AS sample_preview,
                          COUNT(DISTINCT segment_id) AS distinct_segments
                   FROM segment_artifacts
                   WHERE content_hash != ''
                   GROUP BY content_hash, artifact_type
                   HAVING reps >= ?
                   ORDER BY reps DESC LIMIT ?""",
                (min_repetitions, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_segment_artifacts(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT artifact_type, COUNT(*) AS n FROM segment_artifacts
                   GROUP BY artifact_type"""
            ).fetchall()
            return {r["artifact_type"]: r["n"] for r in rows}

    def count_training_events(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT event_type, COUNT(*) as n FROM training_events GROUP BY event_type"""
            ).fetchall()
            return {r["event_type"]: r["n"] for r in rows}

    def propagate_goal_achievement(
        self, goal_id: str, strength: float = 0.6, note: str = ""
    ) -> Dict[str, Any]:
        """Mark goal achieved + emit positive feedback on every linked anchor
        with source='outcome'. Also logs training_events (event_type='harness_outcome')
        for any harness whose anchors overlap with the goal's linked nodes —
        that's the outcome-triggered signal from Phase 4b locked decisions.

        Returns counts of what it touched.
        """
        with self._connect() as conn:
            goal_row = conn.execute(
                "SELECT id, title, status FROM goals WHERE id = ?", (goal_id,)
            ).fetchone()
            if not goal_row:
                raise ValueError(f"goal_not_found: {goal_id}")

            # 1. Transition goal to achieved
            conn.execute(
                "UPDATE goals SET status='achieved', updated_at=datetime('now') WHERE id=?",
                (goal_id,),
            )

            # 2. Fetch linked nodes
            link_rows = conn.execute(
                """SELECT node_id, weight, link_type FROM goal_links
                   WHERE goal_id = ? AND link_type = 'supports'""",
                (goal_id,),
            ).fetchall()
            linked_node_ids = [r["node_id"] for r in link_rows]

            # 3. Emit positive feedback per linked node, source=outcome
            fb_content = f"Goal achieved: {goal_row['title']}"
            if note:
                fb_content += f" — {note}"
            for r in link_rows:
                conn.execute(
                    """INSERT INTO feedback
                       (node_id, goal_id, polarity, content, source, strength)
                       VALUES (?, ?, 'positive', ?, 'session', ?)""",
                    (r["node_id"], goal_id, fb_content, float(strength) * float(r["weight"] or 0.5) * 2.0),
                )

            # 4. Find harnesses whose anchors_used overlaps with linked nodes;
            #    bump their post_run_signals + log training events.
            harness_touched = 0
            if linked_node_ids:
                harness_rows = conn.execute(
                    "SELECT id, anchors_used, post_run_signals FROM agent_harnesses"
                ).fetchall()
                for h in harness_rows:
                    try:
                        anchors = json.loads(h["anchors_used"] or "[]")
                    except Exception:
                        anchors = []
                    anchor_ids = {a.get("node_id") for a in anchors if isinstance(a, dict)}
                    overlap = anchor_ids & set(linked_node_ids)
                    if not overlap:
                        continue
                    existing = json.loads(h["post_run_signals"] or "{}")
                    existing.setdefault("outcome_propagations", []).append({
                        "goal_id": goal_id,
                        "overlap_count": len(overlap),
                        "at": datetime.now().isoformat(timespec="seconds"),
                    })
                    conn.execute(
                        """UPDATE agent_harnesses SET post_run_signals=?,
                           updated_at=datetime('now') WHERE id=?""",
                        (json.dumps(existing), h["id"]),
                    )
                    harness_touched += 1
                    # Training event
                    tid = str(uuid.uuid4())[:12]
                    conn.execute(
                        """INSERT INTO training_events
                           (id, event_type, deliverable_id, harness_id, features, label,
                            pmis_version, model_version)
                           VALUES (?, 'harness_outcome', '', ?, ?, ?, 'phase-4b', '')""",
                        (
                            tid,
                            h["id"],
                            json.dumps({"goal_id": goal_id, "overlap_anchors": list(overlap)}),
                            json.dumps({"outcome": "goal_achieved", "strength": strength}),
                        ),
                    )

            conn.commit()

        return {
            "goal_id": goal_id,
            "title": goal_row["title"],
            "linked_nodes": len(linked_node_ids),
            "feedback_written": len(linked_node_ids),
            "harnesses_touched": harness_touched,
        }

    def get_deliverable(self, deliverable_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT d.*, p.name as project_name, p.sc_node_id as project_sc_node_id
                   FROM deliverables d
                   LEFT JOIN projects p ON p.id = d.project_id
                   WHERE d.id = ?""",
                (deliverable_id,),
            ).fetchone()
            return dict(row) if row else None

    # ---------------------------------------------------------------

    def get_activity_anchors_for_date(self, target_date: str) -> List[Dict[str, Any]]:
        """
        Return anchor memory_nodes created by the activity-merge pass for
        target_date. Used by the project matcher to score today's work anchors
        against active deliverables.
        """
        tag = f"activity_merge_{target_date}"
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, content, source_conversation_id, created_at
                FROM memory_nodes
                WHERE source_conversation_id = ?
                  AND level = 'ANC'
                  AND is_deleted = 0
            """, (tag,)).fetchall()
            return [dict(r) for r in rows]

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
                 match_method, work_description, worker_type, time_mins,
                 matched_at, is_correct, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        datetime('now'), ?, ?)
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
                match.get("source", "semantic"),
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

    def set_match_correctness(self, match_id: str, is_correct: int) -> bool:
        """Mark a match row as user-confirmed (1) or user-rejected (0).
        Returns True if a row was updated."""
        if is_correct not in (0, 1):
            raise ValueError("is_correct must be 0 or 1")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE project_work_match_log SET is_correct = ? WHERE id = ?",
                (is_correct, match_id),
            )
            return cur.rowcount > 0

    def get_matches_for_goal(
        self, goal_id: str, min_score: float = 0.75, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return recent matches whose matched node (anchor/context/sc) is
        linked to this goal via goal_links. Used by the Goals wiki page to
        show surfaced work-to-deliverable matches for thumbs up/down."""
        with self._connect() as conn:
            # Collect goal-linked node ids
            linked = conn.execute(
                "SELECT node_id FROM goal_links WHERE goal_id = ?",
                (goal_id,),
            ).fetchall()
            linked_ids = {r["node_id"] for r in linked}
            if not linked_ids:
                return []

            placeholders = ",".join("?" * len(linked_ids))
            params: List[Any] = list(linked_ids) * 3 + [min_score, limit]
            rows = conn.execute(f"""
                SELECT m.*, mn.content as work_content, mn.level as work_level,
                       d.name as deliverable_name,
                       p.name as project_name
                FROM project_work_match_log m
                LEFT JOIN memory_nodes mn ON mn.id = m.segment_id
                LEFT JOIN deliverables d ON d.id = m.deliverable_id
                LEFT JOIN projects p ON p.id = m.project_id
                WHERE (
                    m.anchor_node_id IN ({placeholders})
                    OR m.context_node_id IN ({placeholders})
                    OR m.sc_node_id IN ({placeholders})
                )
                AND m.combined_match_pct >= ?
                ORDER BY m.matched_at DESC
                LIMIT ?
            """, params).fetchall()
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

    # ═══════════════════════════════════════════════════════════
    # Work pages — 30-min sync content layer
    # ═══════════════════════════════════════════════════════════

    def create_work_page(
        self,
        title: str,
        summary: str,
        date_local: str,
        sc_id: str = "",
        ctx_id: str = "",
        embedding: Optional[np.ndarray] = None,
        user_id: str = "local",
    ) -> str:
        """Create a new open work_page. Returns the generated page_id."""
        page_id = uuid.uuid4().hex[:16]
        blob = (
            embedding.astype(np.float32).tobytes()
            if embedding is not None
            else None
        )
        self._conn.execute(
            """
            INSERT INTO work_pages
                (id, user_id, title, summary, date_local, sc_id, ctx_id, embedding_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (page_id, user_id, title, summary, date_local, sc_id, ctx_id, blob),
        )
        self._conn.commit()
        return page_id

    def append_to_work_page(
        self,
        page_id: str,
        new_summary: str,
        new_embedding: Optional[np.ndarray] = None,
        new_title: Optional[str] = None,
    ) -> None:
        """Re-stitch a page after new segments have been added to its cluster."""
        blob = (
            new_embedding.astype(np.float32).tobytes()
            if new_embedding is not None
            else None
        )
        if new_title is not None and blob is not None:
            self._conn.execute(
                """UPDATE work_pages SET title=?, summary=?, embedding_blob=?,
                   last_sync_at=datetime('now') WHERE id=?""",
                (new_title, new_summary, blob, page_id),
            )
        elif new_title is not None:
            self._conn.execute(
                """UPDATE work_pages SET title=?, summary=?,
                   last_sync_at=datetime('now') WHERE id=?""",
                (new_title, new_summary, page_id),
            )
        elif blob is not None:
            self._conn.execute(
                """UPDATE work_pages SET summary=?, embedding_blob=?,
                   last_sync_at=datetime('now') WHERE id=?""",
                (new_summary, blob, page_id),
            )
        else:
            self._conn.execute(
                """UPDATE work_pages SET summary=?, last_sync_at=datetime('now')
                   WHERE id=?""",
                (new_summary, page_id),
            )
        self._conn.commit()

    def add_page_segment(
        self,
        page_id: str,
        segment_id: str,
        sync_turn: int,
        weight: float = 1.0,
    ) -> None:
        """Attach a tracker segment to a work_page (idempotent on PK)."""
        self._conn.execute(
            """INSERT OR IGNORE INTO work_page_anchors
               (page_id, segment_id, sync_turn, weight) VALUES (?, ?, ?, ?)""",
            (page_id, segment_id, sync_turn, weight),
        )
        self._conn.commit()

    def list_open_work_pages(
        self, date_local: str, user_id: str = "local"
    ) -> List[Dict[str, Any]]:
        """All state=open pages for a day. Decoded embedding on each row."""
        rows = self._conn.execute(
            """SELECT id, title, summary, sc_id, ctx_id, embedding_blob,
                      created_at, last_sync_at
               FROM work_pages
               WHERE date_local = ? AND user_id = ? AND state = 'open'
               ORDER BY last_sync_at DESC""",
            (date_local, user_id),
        ).fetchall()

        pages: List[Dict[str, Any]] = []
        for r in rows:
            page = dict(r)
            blob = page.pop("embedding_blob", None)
            page["embedding"] = (
                np.frombuffer(blob, dtype=np.float32)
                if blob is not None
                else None
            )
            pages.append(page)
        return pages

    def get_work_page(self, page_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM work_pages WHERE id = ?", (page_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_work_pages_by_state(
        self,
        state: str,
        date_local: Optional[str] = None,
        user_id: str = "local",
    ) -> List[Dict[str, Any]]:
        if date_local:
            rows = self._conn.execute(
                """SELECT * FROM work_pages
                   WHERE state = ? AND date_local = ? AND user_id = ?
                   ORDER BY created_at DESC""",
                (state, date_local, user_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM work_pages WHERE state = ? AND user_id = ?
                   ORDER BY created_at DESC""",
                (state, user_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_page_segments(self, page_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT segment_id, sync_turn, weight, added_at
               FROM work_page_anchors WHERE page_id = ?
               ORDER BY sync_turn ASC""",
            (page_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def set_work_page_tag(
        self,
        page_id: str,
        project_id: str,
        deliverable_id: str = "",
        tag_state: str = "confirmed",
        tag_source: str = "user",
    ) -> None:
        """Tag a page to a project. state becomes 'tagged' when confirmed."""
        new_state = "tagged" if tag_state == "confirmed" else "open"
        self._conn.execute(
            """UPDATE work_pages
               SET project_id=?, deliverable_id=?, tag_state=?, tag_source=?,
                   state=?, tagged_at=datetime('now')
               WHERE id=?""",
            (project_id, deliverable_id, tag_state, tag_source, new_state, page_id),
        )
        self._conn.commit()

    def get_last_sync_timestamp(
        self, user_id: str = "local"
    ) -> Optional[str]:
        key = f"last_sync_{user_id}"
        row = self._conn.execute(
            "SELECT value FROM system_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_last_sync_timestamp(
        self, ts: str, user_id: str = "local"
    ) -> None:
        key = f"last_sync_{user_id}"
        self._conn.execute(
            """INSERT INTO system_meta (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (key, ts),
        )
        self._conn.commit()

    def archive_work_page(
        self, page_id: str, tag_state: str = "rejected"
    ) -> None:
        """Soft-archive a page (user rejected, or TTL-expired). tag_state
        distinguishes 'rejected' (explicit) from '' (TTL)."""
        self._conn.execute(
            """UPDATE work_pages
               SET state='archived', tag_state=?, tagged_at=datetime('now')
               WHERE id=?""",
            (tag_state, page_id),
        )
        self._conn.commit()

    def get_next_sync_turn(
        self, date_local: str, user_id: str = "local"
    ) -> int:
        """Next sync_turn ordinal for today (1-indexed)."""
        row = self._conn.execute(
            """SELECT COALESCE(MAX(wpa.sync_turn), 0) AS max_turn
               FROM work_page_anchors wpa
               JOIN work_pages wp ON wp.id = wpa.page_id
               WHERE wp.date_local = ? AND wp.user_id = ?""",
            (date_local, user_id),
        ).fetchone()
        return (row["max_turn"] if row else 0) + 1

    # ═══════════════════════════════════════════════════════════
    # Project digests — employee-first per-project window summaries
    # ═══════════════════════════════════════════════════════════

    def get_confirmed_pages_for_project_window(
        self,
        project_id: str,
        window_start: str,
        window_end: str,
        user_id: str = "local",
    ) -> List[Dict[str, Any]]:
        """Confirmed work_pages tagged to a project within the date window.
        window_start/window_end are inclusive YYYY-MM-DD bounds."""
        rows = self._conn.execute(
            """SELECT id, title, summary, date_local, state, tag_state,
                      project_id, deliverable_id, created_at, tagged_at
               FROM work_pages
               WHERE project_id = ? AND user_id = ?
                 AND state = 'tagged' AND tag_state = 'confirmed'
                 AND date_local BETWEEN ? AND ?
               ORDER BY date_local ASC, created_at ASC""",
            (project_id, user_id, window_start, window_end),
        ).fetchall()
        pages: List[Dict[str, Any]] = []
        for r in rows:
            page = dict(r)
            page["segment_count"] = len(self.get_page_segments(page["id"]))
            pages.append(page)
        return pages

    def upsert_project_digest(
        self,
        project_id: str,
        window_type: str,
        window_start: str,
        window_end: str,
        summary_markdown: str,
        key_points: Optional[List[str]] = None,
        total_minutes: float = 0,
        source_page_ids: Optional[List[str]] = None,
        generated_by: str = "manual",
        user_id: str = "local",
    ) -> str:
        """Upsert a digest keyed by (user_id, project_id, window_type, window_start).
        Returns the digest id (reuses existing on conflict)."""
        digest_id = uuid.uuid4().hex[:16]
        self._conn.execute(
            """INSERT INTO project_digests
               (id, user_id, project_id, window_type, window_start, window_end,
                summary_markdown, key_points_json, total_minutes,
                source_page_ids_json, generated_by, is_stale,
                generated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, datetime('now'))
               ON CONFLICT(user_id, project_id, window_type, window_start) DO UPDATE SET
                   window_end = excluded.window_end,
                   summary_markdown = excluded.summary_markdown,
                   key_points_json = excluded.key_points_json,
                   total_minutes = excluded.total_minutes,
                   source_page_ids_json = excluded.source_page_ids_json,
                   generated_by = excluded.generated_by,
                   is_stale = 0,
                   generated_at = datetime('now')""",
            (
                digest_id, user_id, project_id, window_type,
                window_start, window_end, summary_markdown,
                json.dumps(key_points or []),
                total_minutes,
                json.dumps(source_page_ids or []),
                generated_by,
            ),
        )
        self._conn.commit()
        row = self._conn.execute(
            """SELECT id FROM project_digests
               WHERE user_id = ? AND project_id = ?
                 AND window_type = ? AND window_start = ?""",
            (user_id, project_id, window_type, window_start),
        ).fetchone()
        return row["id"] if row else digest_id

    def list_project_digests(
        self,
        project_id: str,
        window_type: Optional[str] = None,
        limit: int = 20,
        user_id: str = "local",
    ) -> List[Dict[str, Any]]:
        """Recent digests for a project, newest first."""
        if window_type:
            rows = self._conn.execute(
                """SELECT * FROM project_digests
                   WHERE project_id = ? AND user_id = ? AND window_type = ?
                   ORDER BY window_start DESC LIMIT ?""",
                (project_id, user_id, window_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM project_digests
                   WHERE project_id = ? AND user_id = ?
                   ORDER BY generated_at DESC LIMIT ?""",
                (project_id, user_id, limit),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["key_points"] = json.loads(d.pop("key_points_json") or "[]")
            except Exception:
                d["key_points"] = []
            try:
                d["source_page_ids"] = json.loads(d.pop("source_page_ids_json") or "[]")
            except Exception:
                d["source_page_ids"] = []
            out.append(d)
        return out

    def get_latest_project_digest(
        self,
        project_id: str,
        window_type: str = "day",
        user_id: str = "local",
    ) -> Optional[Dict[str, Any]]:
        rows = self.list_project_digests(
            project_id, window_type=window_type, limit=1, user_id=user_id
        )
        return rows[0] if rows else None

    def mark_project_digests_stale(
        self, project_id: str, user_id: str = "local"
    ) -> int:
        """Flag all digests for a project as stale (call when a source
        work_page is edited, re-tagged, or archived). Next Dream regenerates."""
        cur = self._conn.execute(
            """UPDATE project_digests
               SET is_stale = 1
               WHERE project_id = ? AND user_id = ?""",
            (project_id, user_id),
        )
        self._conn.commit()
        return cur.rowcount or 0

    # ═══════════════════════════════════════════════════════════
    # Dream auto-match (step 4) — proposed/confirmed bookkeeping
    # ═══════════════════════════════════════════════════════════

    def count_confirmed_page_tags(self, user_id: str = "local") -> int:
        """Total human-validated confirmed tags (user-tagged OR user-confirmed
        Dream proposals — both are training signal). Gates the bootstrap
        cold start: Dream won't auto-propose until this clears the threshold."""
        row = self._conn.execute(
            """SELECT COUNT(*) AS n FROM work_pages
               WHERE user_id = ? AND tag_state = 'confirmed'""",
            (user_id,),
        ).fetchone()
        return int(row["n"]) if row else 0

    def list_untagged_pages_for_matching(
        self,
        user_id: str = "local",
        since_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Pages that Dream's project matcher should consider.

        Includes state='open' with no tag_state (never proposed) AND
        state='open' with tag_state='proposed' (re-score allowed — HGCN
        may have moved). Excludes confirmed/rejected/archived/stale.
        """
        if since_date:
            rows = self._conn.execute(
                """SELECT * FROM work_pages
                   WHERE user_id = ? AND state = 'open'
                     AND tag_state IN ('', 'proposed')
                     AND date_local >= ?
                   ORDER BY date_local DESC, created_at DESC""",
                (user_id, since_date),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM work_pages
                   WHERE user_id = ? AND state = 'open'
                     AND tag_state IN ('', 'proposed')
                   ORDER BY date_local DESC, created_at DESC""",
                (user_id,),
            ).fetchall()
        pages: List[Dict[str, Any]] = []
        for r in rows:
            page = dict(r)
            blob = page.pop("embedding_blob", None)
            page["embedding"] = (
                np.frombuffer(blob, dtype=np.float32)
                if blob is not None else None
            )
            pages.append(page)
        return pages

    def set_work_page_proposal(
        self,
        page_id: str,
        project_id: str,
        deliverable_id: str = "",
        score: float = 0.0,
        user_id: str = "local",
    ) -> None:
        """Write a Dream-proposed match onto a still-open page. Keeps
        state='open' so the page stays in the Unassigned lane with a
        'proposed' banner — user still has to confirm before HGCN sees it."""
        self._conn.execute(
            """UPDATE work_pages
               SET project_id = ?, deliverable_id = ?,
                   tag_state = 'proposed', tag_source = 'dream_proposed',
                   rank_score = ?
               WHERE id = ? AND user_id = ?""",
            (project_id, deliverable_id, float(score), page_id, user_id),
        )
        self._conn.commit()

    def confirm_work_page_proposal(
        self, page_id: str, user_id: str = "local"
    ) -> Optional[Dict[str, Any]]:
        """User ✓ on a proposed tag. Flips to state='tagged' + confirmed,
        tag_source stays 'dream_proposed' so the lineage survives."""
        page = self.get_work_page(page_id)
        if not page:
            return None
        if (page.get("tag_state") or "") != "proposed":
            return None
        self._conn.execute(
            """UPDATE work_pages
               SET state = 'tagged', tag_state = 'confirmed',
                   tagged_at = datetime('now')
               WHERE id = ? AND user_id = ?""",
            (page_id, user_id),
        )
        self._conn.commit()
        return self.get_work_page(page_id)

    def expire_stale_proposals(
        self, max_age_days: int = 3, user_id: str = "local"
    ) -> int:
        """TTL the 'proposed' tag_state: any proposal older than max_age_days
        is cleared (tag_state='', project_id='') so Dream can re-score it,
        or it ages out of the lane. Returns rows affected."""
        cur = self._conn.execute(
            """UPDATE work_pages
               SET tag_state = '', tag_source = '',
                   project_id = '', deliverable_id = '', rank_score = 0
               WHERE user_id = ? AND state = 'open'
                 AND tag_state = 'proposed'
                 AND date_local <= date('now', ? || ' days')""",
            (user_id, f"-{int(max_age_days)}"),
        )
        self._conn.commit()
        return cur.rowcount or 0
