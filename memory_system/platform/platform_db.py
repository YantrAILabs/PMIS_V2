"""
PMIS Platform — Database Layer
Per-user isolated SQLite databases with encryption support.
"""

import sqlite3
import os
import hashlib
import uuid
from pathlib import Path
from datetime import datetime, timezone

PLATFORM_DIR = Path(__file__).parent
DATA_DIR = PLATFORM_DIR / "data"
USERS_DB = DATA_DIR / "users.db"
MEMORIES_DIR = DATA_DIR / "memories"


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uuid():
    return uuid.uuid4().hex[:12]


# ══════════════════════════════════════
# USERS DB (platform-level)
# ══════════════════════════════════════

def get_users_db():
    """Get connection to the platform users database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(USERS_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            display_name TEXT,
            password_hash TEXT NOT NULL,
            encryption_key_hash TEXT,
            created_at TEXT NOT NULL,
            last_active TEXT,
            role TEXT DEFAULT 'member'
        );

        CREATE TABLE IF NOT EXISTS api_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            scope TEXT DEFAULT 'full'
        );

        CREATE TABLE IF NOT EXISTS sharing_rules (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT,
            scope_type TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            scope_title TEXT,
            access_level TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT
        );

        CREATE TABLE IF NOT EXISTS access_requests (
            id TEXT PRIMARY KEY,
            requester_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            scope_type TEXT,
            scope_id TEXT,
            scope_title TEXT,
            task_context TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            resolved_by TEXT
        );

        CREATE TABLE IF NOT EXISTS sync_log (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            device_id TEXT,
            direction TEXT,
            items_synced INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_tokens_user ON api_tokens(user_id);
        CREATE INDEX IF NOT EXISTS idx_sharing_owner ON sharing_rules(owner_id);
        CREATE INDEX IF NOT EXISTS idx_sharing_target ON sharing_rules(target_id);
        CREATE INDEX IF NOT EXISTS idx_requests_owner ON access_requests(owner_id);
        CREATE INDEX IF NOT EXISTS idx_requests_requester ON access_requests(requester_id);
    """)
    conn.commit()
    return conn


# ══════════════════════════════════════
# PER-USER MEMORY DB
# ══════════════════════════════════════

def get_user_memory_path(username):
    """Get the path to a user's memory database."""
    user_dir = MEMORIES_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / "graph.db"


def get_user_memory_db(username):
    """Get a connection to a user's memory database.
    V1 memory.py retired — returns a basic SQLite connection for legacy compat."""
    import sqlite3
    db_path = get_user_memory_path(username)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def derive_encryption_key(password, salt):
    """Derive an encryption key from password using PBKDF2."""
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()


# ══════════════════════════════════════
# SHARING HELPERS
# ══════════════════════════════════════

def get_shared_scs_for_user(users_conn, user_id):
    """Get all SCs shared with a specific user (or org-wide)."""
    rows = users_conn.execute("""
        SELECT sr.*, u.username as owner_username, u.display_name as owner_name
        FROM sharing_rules sr
        JOIN users u ON u.id = sr.owner_id
        WHERE (sr.target_type = 'org'
               OR (sr.target_type = 'user' AND sr.target_id = ?)
               OR sr.target_type = 'everyone')
          AND (sr.expires_at IS NULL OR sr.expires_at > ?)
        ORDER BY sr.created_at DESC
    """, (user_id, _now())).fetchall()
    return [dict(r) for r in rows]


def check_access(users_conn, requester_id, owner_id, scope_id):
    """Check if requester has access to a specific scope owned by owner."""
    # Check direct sharing rules
    rule = users_conn.execute("""
        SELECT access_level FROM sharing_rules
        WHERE owner_id = ? AND scope_id = ?
          AND (target_type = 'org'
               OR (target_type = 'user' AND target_id = ?)
               OR target_type = 'everyone')
          AND (expires_at IS NULL OR expires_at > ?)
        ORDER BY
          CASE access_level WHEN 'full' THEN 1 WHEN 'read' THEN 2 WHEN 'request_based' THEN 3 END
        LIMIT 1
    """, (owner_id, scope_id, requester_id, _now())).fetchone()

    if not rule:
        return "private"
    return rule["access_level"]


def has_approved_request(users_conn, requester_id, owner_id, scope_id):
    """Check if there's an approved access request."""
    row = users_conn.execute("""
        SELECT id FROM access_requests
        WHERE requester_id = ? AND owner_id = ? AND scope_id = ?
          AND status = 'approved'
        LIMIT 1
    """, (requester_id, owner_id, scope_id)).fetchone()
    return row is not None
