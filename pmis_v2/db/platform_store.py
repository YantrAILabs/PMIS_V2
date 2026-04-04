"""
Platform Store — READ-ONLY audit log for per-platform memory tracking.

Platform tables are NEVER used for retrieval, embedding lookups, or consolidation.
They exist solely to answer: "which platform contributed this memory?" and
"what's each platform's contribution rate?"

Main memory (memory_nodes + ChromaDB) is the SINGLE source of truth.
"""

import sqlite3
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any


class PlatformStore:
    """Manages the platforms and platform_memories audit tables."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._init_tables()

    def _init_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS platforms (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                api_key_hash TEXT,
                status TEXT DEFAULT 'active',
                last_seen TEXT,
                total_turns INTEGER DEFAULT 0,
                total_memories INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS platform_memories (
                id TEXT PRIMARY KEY,
                platform_id TEXT NOT NULL,
                conversation_id TEXT,
                content TEXT NOT NULL,
                level TEXT,
                merged_node_id TEXT,
                merge_status TEXT DEFAULT 'pending',
                raw_surprise REAL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (platform_id) REFERENCES platforms(id)
            );

            CREATE INDEX IF NOT EXISTS idx_pm_platform ON platform_memories(platform_id);
            CREATE INDEX IF NOT EXISTS idx_pm_status ON platform_memories(merge_status);
            CREATE INDEX IF NOT EXISTS idx_pm_created ON platform_memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_platforms_status ON platforms(status);
        """)
        self._conn.commit()

    # ── Platform CRUD ──

    def register_platform(self, platform_id: str, name: str, api_key_hash: str = "") -> Dict:
        self._conn.execute("""
            INSERT OR REPLACE INTO platforms (id, name, api_key_hash, status, last_seen, created_at)
            VALUES (?, ?, ?, 'active', datetime('now'), datetime('now'))
        """, (platform_id, name, api_key_hash))
        self._conn.commit()
        return self.get_platform(platform_id)

    def get_platform(self, platform_id: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM platforms WHERE id = ?", (platform_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_platforms(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM platforms ORDER BY last_seen DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_platform_seen(self, platform_id: str):
        """Update last_seen and increment turn count."""
        self._conn.execute("""
            UPDATE platforms
            SET last_seen = datetime('now'),
                total_turns = total_turns + 1,
                status = 'active'
            WHERE id = ?
        """, (platform_id,))
        self._conn.commit()

    def update_platform_status(self, platform_id: str, status: str):
        self._conn.execute(
            "UPDATE platforms SET status = ? WHERE id = ?",
            (status, platform_id)
        )
        self._conn.commit()

    def increment_memory_count(self, platform_id: str):
        self._conn.execute(
            "UPDATE platforms SET total_memories = total_memories + 1 WHERE id = ?",
            (platform_id,)
        )
        self._conn.commit()

    # ── Platform Memories (append-only audit log) ──

    def log_memory(
        self,
        platform_id: str,
        content: str,
        conversation_id: str = "",
        level: str = "",
    ) -> str:
        """Log a raw input to platform_memories. Returns the log entry ID."""
        entry_id = hashlib.sha256(
            f"{platform_id}:{content}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        self._conn.execute("""
            INSERT INTO platform_memories (id, platform_id, conversation_id, content, level)
            VALUES (?, ?, ?, ?, ?)
        """, (entry_id, platform_id, conversation_id, content, level))
        self._conn.commit()
        return entry_id

    def update_merge_status(
        self,
        entry_id: str,
        merged_node_id: Optional[str],
        merge_status: str,
        raw_surprise: Optional[float] = None,
    ):
        """Update a platform_memories entry after orchestrator processes the turn."""
        self._conn.execute("""
            UPDATE platform_memories
            SET merged_node_id = ?, merge_status = ?, raw_surprise = ?
            WHERE id = ?
        """, (merged_node_id, merge_status, raw_surprise, entry_id))
        self._conn.commit()

    def get_platform_memories(
        self,
        platform_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict]:
        rows = self._conn.execute("""
            SELECT * FROM platform_memories
            WHERE platform_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (platform_id, limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def get_platform_stats(self, platform_id: str) -> Dict:
        """Get memory stats for a specific platform."""
        row = self._conn.execute("""
            SELECT
                COUNT(*) as total_logged,
                SUM(CASE WHEN merge_status = 'merged' THEN 1 ELSE 0 END) as merged,
                SUM(CASE WHEN merge_status = 'skipped' THEN 1 ELSE 0 END) as skipped,
                SUM(CASE WHEN merge_status = 'pending' THEN 1 ELSE 0 END) as pending,
                AVG(raw_surprise) as avg_surprise
            FROM platform_memories WHERE platform_id = ?
        """, (platform_id,)).fetchone()
        return dict(row) if row else {}

    def get_recent_memories(self, limit: int = 50) -> List[Dict]:
        """Get recent platform_memories across all platforms."""
        rows = self._conn.execute("""
            SELECT pm.*, p.name as platform_name
            FROM platform_memories pm
            JOIN platforms p ON pm.platform_id = p.id
            ORDER BY pm.created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_all_platform_stats(self) -> List[Dict]:
        """Get summary stats for all platforms."""
        rows = self._conn.execute("""
            SELECT
                p.id, p.name, p.status, p.last_seen,
                p.total_turns, p.total_memories, p.created_at,
                COUNT(pm.id) as logged_count,
                SUM(CASE WHEN pm.merge_status = 'merged' THEN 1 ELSE 0 END) as merge_count,
                SUM(CASE WHEN pm.merge_status = 'skipped' THEN 1 ELSE 0 END) as skip_count
            FROM platforms p
            LEFT JOIN platform_memories pm ON p.id = pm.platform_id
            GROUP BY p.id
            ORDER BY p.last_seen DESC
        """).fetchall()
        return [dict(r) for r in rows]
