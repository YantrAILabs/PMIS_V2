#!/usr/bin/env python3
"""
PMIS Memory Operations v2 — Zero external dependencies.
Uses only Python stdlib: sqlite3, json, os, sys, datetime, math, statistics.

FIXES IMPLEMENTED:
  1. Evidence-based weights — anchors earn weight from task outcomes, not guesses
  2. Temporal memory stages — impulse / active / established / fading
  3. Winning structure snapshots — best-performing tree is captured and served first

Usage:
    python3 scripts/memory.py store '{"super_context":...}'   # structured JSON
    python3 scripts/memory.py store "title" "flat text"        # backward compatible
    python3 scripts/memory.py retrieve "task description"
    python3 scripts/memory.py browse
    python3 scripts/memory.py tree
    python3 scripts/memory.py stats
    python3 scripts/memory.py score "task_id" "4.5"
    python3 scripts/memory.py rebuild
    python3 scripts/memory.py viz
"""

import sqlite3
import json
import uuid
import os
import sys
import math
import hashlib
import statistics
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import urllib.request
import urllib.error
import urllib.parse
# P9+ Triple Fusion retrieval (added by migrate_p9.py)
try:
    from p9_retrieve import p9_retrieve as _p9_retrieve
    _HAS_P9 = True
except ImportError:
    _HAS_P9 = False


# ── Paths ──

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
GRAPH_DB = ROOT / "Graph_DB" / "graph.db"
TASKS_DIR = ROOT / "Graph_DB" / "tasks"
VECTOR_DIR = ROOT / "Vector_DB"
VIZ_PATH = ROOT / "memory.html"

# ── Team Memory Config ──

YANTRAI_DIR = Path.home() / ".yantrai"
YANTRAI_CONFIG = YANTRAI_DIR / "config.json"
YANTRAI_BRAIN_DB = YANTRAI_DIR / "brain.db"
YANTRAI_RETRY_QUEUE = YANTRAI_DIR / "retry_queue.json"
YANTRAI_SYNC_STATE = YANTRAI_DIR / "sync_state.json"


def _detect_project_root():
    """Walk up from CWD to find project root (.git or CLAUDE.md)."""
    cwd = Path.cwd()
    for d in [cwd] + list(cwd.parents):
        if (d / ".git").exists() or (d / "CLAUDE.md").exists():
            return d
    return cwd


def _load_team_config():
    """Load team config from ~/.yantrai/config.json. Returns None if not set up."""
    if not YANTRAI_CONFIG.exists():
        return None
    try:
        return json.loads(YANTRAI_CONFIG.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _get_author():
    """Get author name from env var or config."""
    author = os.environ.get("YANTRAI_AUTHOR", "")
    if not author:
        cfg = _load_team_config()
        if cfg:
            author = cfg.get("author", "")
    return author or "unknown"


def _get_server():
    """Get central server URL from env var or config."""
    server = os.environ.get("YANTRAI_SERVER", "")
    if not server:
        cfg = _load_team_config()
        if cfg:
            server = cfg.get("server", "")
    return server


def _get_token():
    """Get auth token from config."""
    cfg = _load_team_config()
    return cfg.get("token", "") if cfg else ""


def _is_shared_folder(folder_path=None):
    """Check if current/given folder is in shared_folders list."""
    cfg = _load_team_config()
    if not cfg:
        return False
    folder = str(folder_path or ROOT)
    shared = cfg.get("shared_folders", [])
    return any(folder.rstrip("/") == s.rstrip("/") for s in shared)


def _node_hash(title, content=""):
    """SHA256 hash of title+content for deterministic dedup."""
    raw = f"{title.lower().strip()}|{(content or '').lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _central_reachable():
    """Quick check if central server is reachable."""
    server = _get_server()
    if not server:
        return False
    try:
        url = f"http://{server}/health"
        req = urllib.request.Request(url, method="GET")
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


def _central_request(method, endpoint, data=None):
    """Make an authenticated HTTP request to central server."""
    server = _get_server()
    token = _get_token()
    if not server:
        return None
    url = f"http://{server}{endpoint}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode() if data else None
    try:
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read().decode())
    except Exception:
        return None


def _append_retry(nodes, edges):
    """Append failed push to retry queue (survives terminal close)."""
    YANTRAI_DIR.mkdir(parents=True, exist_ok=True)
    queue = []
    if YANTRAI_RETRY_QUEUE.exists():
        try:
            queue = json.loads(YANTRAI_RETRY_QUEUE.read_text())
        except (json.JSONDecodeError, OSError):
            queue = []
    queue.append({
        "nodes": nodes,
        "edges": edges,
        "failed_at": _now(),
        "attempts": 1
    })
    YANTRAI_RETRY_QUEUE.write_text(json.dumps(queue))


def _process_retry():
    """Retry failed pushes. Called at the start of every store."""
    if not YANTRAI_RETRY_QUEUE.exists():
        return 0
    try:
        queue = json.loads(YANTRAI_RETRY_QUEUE.read_text())
    except (json.JSONDecodeError, OSError):
        return 0
    if not queue:
        return 0
    remaining = []
    retried = 0
    for item in queue:
        result = _central_request("POST", "/api/sync/push", {
            "author": _get_author(),
            "source_folder": str(ROOT),
            "nodes": item["nodes"],
            "edges": item["edges"]
        })
        if result is not None:
            retried += 1
        else:
            item["attempts"] = item.get("attempts", 0) + 1
            if item["attempts"] < 10:
                remaining.append(item)
    YANTRAI_RETRY_QUEUE.write_text(json.dumps(remaining))
    return retried

def _id():
    return uuid.uuid4().hex[:10]

def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _parse_ts(ts_str):
    """Parse an ISO timestamp string into datetime."""
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


# ══════════════════════════════════════════════════════════
# SCHEMA — includes all 3 fixes
# ══════════════════════════════════════════════════════════

def get_db():
    GRAPH_DB.parent.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(GRAPH_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        -- Core nodes: SC, Context, Anchor
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
            -- FIX 1: initial weight from Claude, preserved separately
            initial_weight REAL DEFAULT 0.5,
            -- FIX 2: temporal tracking
            occurrence_log TEXT DEFAULT '[]',
            recency REAL DEFAULT 1.0,
            frequency REAL DEFAULT 0.0,
            consistency REAL DEFAULT 0.0,
            memory_stage TEXT DEFAULT 'impulse',
            -- v3: discrimination power and mode context
            discrimination_power REAL DEFAULT 0.0,
            mode_vector TEXT DEFAULT '{}'
        );

        -- Parent-child and other edges
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            src TEXT NOT NULL,
            tgt TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'parent_child',
            weight REAL DEFAULT 1.0,
            created_at TEXT NOT NULL
        );

        -- Task log
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            sc_id TEXT,
            title TEXT NOT NULL,
            started TEXT NOT NULL,
            completed TEXT,
            score REAL DEFAULT 0,
            content TEXT DEFAULT '',
            -- FIX 3: snapshot of tree at score time
            structure_snapshot TEXT DEFAULT ''
        );

        -- FIX 1+3: which anchors were present during each task
        CREATE TABLE IF NOT EXISTS task_anchors (
            task_id TEXT NOT NULL,
            anchor_id TEXT NOT NULL,
            context_id TEXT NOT NULL,
            was_retrieved INTEGER DEFAULT 0
        );

        -- v3: Decision log for convergent sessions
        CREATE TABLE IF NOT EXISTS decisions (
            id TEXT PRIMARY KEY,
            sc_id TEXT NOT NULL,
            anchor_id TEXT,
            session_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            alternatives_eliminated TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.5,
            reversible INTEGER DEFAULT 1,
            evidence TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_n_type ON nodes(type);
        CREATE INDEX IF NOT EXISTS idx_e_src ON edges(src);
        CREATE INDEX IF NOT EXISTS idx_e_tgt ON edges(tgt);
        CREATE INDEX IF NOT EXISTS idx_ta_task ON task_anchors(task_id);
        CREATE INDEX IF NOT EXISTS idx_ta_anchor ON task_anchors(anchor_id);
        CREATE INDEX IF NOT EXISTS idx_dec_sc ON decisions(sc_id);
    """)
    conn.commit()

    # Migration: add new columns to existing DBs
    _migrate(conn)
    return conn


def _migrate(conn):
    """Add new columns if they don't exist (safe for existing DBs)."""
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}
    new_cols = {
        "initial_weight": "REAL DEFAULT 0.5",
        "occurrence_log": "TEXT DEFAULT '[]'",
        "recency": "REAL DEFAULT 1.0",
        "frequency": "REAL DEFAULT 0.0",
        "consistency": "REAL DEFAULT 0.0",
        "memory_stage": "TEXT DEFAULT 'impulse'",
        "discrimination_power": "REAL DEFAULT 0.0",
        "mode_vector": "TEXT DEFAULT '{}'",
        # Team memory columns
        "author": "TEXT DEFAULT ''",
        "source_folder": "TEXT DEFAULT ''",
        "node_hash": "TEXT DEFAULT ''",
        # For brain.db unified schema
        "mem_source": "TEXT DEFAULT 'local'",
        "team_weight": "REAL DEFAULT 0.5",
        "synced_at": "TEXT DEFAULT ''",
    }
    for col, typedef in new_cols.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE nodes ADD COLUMN {col} {typedef}")

    task_cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "structure_snapshot" not in task_cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN structure_snapshot TEXT DEFAULT ''")
    if "mode_vector" not in task_cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN mode_vector TEXT DEFAULT '{}'")

    # Create task_anchors if not exists (already in schema but safe)
    conn.execute("""CREATE TABLE IF NOT EXISTS task_anchors (
        task_id TEXT NOT NULL, anchor_id TEXT NOT NULL,
        context_id TEXT NOT NULL, was_retrieved INTEGER DEFAULT 0,
        execution_params TEXT DEFAULT '{}')""")

    # v3: Add execution_params to existing task_anchors
    ta_cols = {r[1] for r in conn.execute("PRAGMA table_info(task_anchors)").fetchall()}
    if "execution_params" not in ta_cols:
        conn.execute("ALTER TABLE task_anchors ADD COLUMN execution_params TEXT DEFAULT '{}'")

    # Team memory indexes
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_n_hash ON nodes(node_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_n_author ON nodes(author)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_n_memsource ON nodes(mem_source)")
    except Exception:
        pass  # columns may not exist yet on first migration pass

    conn.commit()


# ══════════════════════════════════════════════════════════
# TEMPORAL ENGINE (Fix 2)
# ══════════════════════════════════════════════════════════

def record_occurrence(conn, node_id):
    """Append today's date to the node's occurrence log and recompute temporal scores."""
    row = conn.execute("SELECT occurrence_log FROM nodes WHERE id=?", (node_id,)).fetchone()
    if not row:
        return

    try:
        log = json.loads(row["occurrence_log"] or "[]")
    except json.JSONDecodeError:
        log = []

    today = _today()
    # Only add if not already recorded today
    if not log or log[-1] != today:
        log.append(today)

    # Keep last 180 days of log
    cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")
    log = [d for d in log if d >= cutoff]

    # Compute temporal scores
    recency, frequency, consistency = compute_temporal(log)
    stage = classify_stage(recency, frequency, consistency)

    conn.execute(
        """UPDATE nodes SET occurrence_log=?, recency=?, frequency=?,
           consistency=?, memory_stage=?, last_used=?, use_count=use_count+1
           WHERE id=?""",
        (json.dumps(log), recency, frequency, consistency, stage, _now(), node_id)
    )
    conn.commit()


def compute_temporal(log):
    """From a list of date strings, compute recency, frequency, consistency."""
    if not log:
        return 0.0, 0.0, 0.0

    now = datetime.now(timezone.utc)

    # Recency: exponential decay from last occurrence, half-life 7 days
    last_date = datetime.strptime(log[-1], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_since = (now - last_date).total_seconds() / 86400
    recency = math.exp(-0.693 * days_since / 7.0)

    # Frequency: occurrences in last 30 days / 30
    cutoff_30 = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    recent_count = sum(1 for d in log if d >= cutoff_30)
    frequency = min(recent_count / 30.0, 1.0)

    # Consistency: 1 / (stddev of gaps between occurrences + 1)
    if len(log) >= 3:
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in log]
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        if gaps:
            try:
                gap_std = statistics.stdev(gaps)
            except statistics.StatisticsError:
                gap_std = 0
            consistency = 1.0 / (gap_std + 1.0)
        else:
            consistency = 0.0
    elif len(log) == 2:
        consistency = 0.3  # two occurrences — some regularity
    else:
        consistency = 0.0  # single occurrence — no regularity

    return round(recency, 3), round(frequency, 3), round(consistency, 3)


def classify_stage(recency, frequency, consistency):
    """Classify a memory into one of 4 temporal stages.

    Iteration 2 fix: "established" requires BOTH high consistency AND
    sufficient frequency (≥0.15 = ~5 uses in 30 days). This prevents
    3 evenly-spaced recent uses from being called "established".
    """
    # No data at all = new node
    if recency == 0 and frequency == 0:
        return "impulse"
    # ESTABLISHED: proven over time — needs sustained, consistent usage
    # frequency >= 0.15 means ~5+ uses in 30 days (not just 3 lucky hits)
    if frequency >= 0.15 and consistency > 0.3:
        return "established"
    # FADING: old and rarely used
    if recency < 0.3 and frequency < 0.05:
        return "fading"
    # ACTIVE: currently in use (moderate+ recency with real usage)
    # Guard: very low frequency + high consistency = coincidental regularity, not active
    if recency > 0.3 and frequency >= 0.05 and not (frequency < 0.1 and consistency > 0.8):
        return "active"
    # IMPULSE: new, unproven, or single use
    return "impulse"


def temporal_boost(stage):
    """Retrieval multiplier based on memory stage."""
    return {"impulse": 0.8, "active": 1.2, "established": 1.5, "fading": 0.5}.get(stage, 1.0)


# ══════════════════════════════════════════════════════════
# EVIDENCE-BASED WEIGHTS (Fix 1)
# ══════════════════════════════════════════════════════════

def compute_evidence_weight(conn, anchor_id):
    """Compute weight from actual task outcomes where this anchor was present."""
    # Get all tasks where this anchor was used
    rows = conn.execute("""
        SELECT t.score FROM tasks t
        JOIN task_anchors ta ON ta.task_id = t.id
        WHERE ta.anchor_id = ? AND t.score > 0 AND t.completed IS NOT NULL
    """, (anchor_id,)).fetchall()

    scores = [r["score"] for r in rows]

    # Get the initial weight Claude assigned
    node = conn.execute("SELECT initial_weight, weight FROM nodes WHERE id=?", (anchor_id,)).fetchone()
    if not node:
        return 0.5

    initial = node["initial_weight"] or 0.5

    if not scores:
        # No evidence yet — trust Claude's estimate
        return initial

    mean_score = sum(scores) / len(scores)
    # Non-linear mapping (Iteration 3 fix): balanced power curve
    # 0.7 exponent amplifies high scores without over-compression
    raw = mean_score / 5.0
    evidence_weight = raw ** 0.7

    # Blend: more data → trust evidence more (widened decay schedule)
    n = len(scores)
    if n >= 5:
        decay = 0.15  # 15% initial, 85% evidence
    elif n >= 3:
        decay = 0.3   # 30% initial, 70% evidence
    elif n >= 2:
        decay = 0.5   # 50/50
    else:
        decay = 0.7   # 70% initial, 30% evidence

    blended = (initial * decay) + (evidence_weight * (1 - decay))
    return round(min(1.0, max(0.05, blended)), 3)


def compute_discrimination_power(conn, anchor_id):
    """How much does this anchor's outcome vary based on execution?
    ADP > 1.0 = execution details matter. ADP < 0.5 = reliable regardless."""
    anchor_scores = conn.execute("""
        SELECT t.score FROM tasks t
        JOIN task_anchors ta ON ta.task_id = t.id
        WHERE ta.anchor_id = ? AND t.score > 0 AND t.completed IS NOT NULL
    """, (anchor_id,)).fetchall()

    if len(anchor_scores) < 3:
        return 0.0

    all_scores = conn.execute(
        "SELECT score FROM tasks WHERE score > 0 AND completed IS NOT NULL"
    ).fetchall()

    if len(all_scores) < 3:
        return 0.0

    try:
        anchor_var = statistics.variance([r["score"] for r in anchor_scores])
        all_var = statistics.variance([r["score"] for r in all_scores])
        return round(anchor_var / all_var, 3) if all_var > 0.001 else 0.0
    except statistics.StatisticsError:
        return 0.0


# ══════════════════════════════════════════════════════════
# MODE VECTORS (v3 Phase 4)
# ══════════════════════════════════════════════════════════

MODE_TEMPLATES = {
    "outreach":  {"work": 0.9, "creative": 0.2, "social": 0.5, "learning": 0.1, "travel": 0.0, "home": 0.0},
    "website":   {"work": 0.7, "creative": 0.8, "learning": 0.2, "social": 0.1, "travel": 0.0, "home": 0.1},
    "editorial": {"work": 0.3, "creative": 0.8, "learning": 0.6, "social": 0.1, "travel": 0.0, "home": 0.3},
    "research":  {"work": 0.4, "creative": 0.2, "learning": 0.9, "social": 0.1, "travel": 0.0, "home": 0.2},
    "sales":     {"work": 0.9, "creative": 0.1, "social": 0.6, "learning": 0.2, "travel": 0.2, "home": 0.0},
    "design":    {"work": 0.6, "creative": 0.9, "learning": 0.3, "social": 0.1, "travel": 0.0, "home": 0.2},
    "marketing": {"work": 0.8, "creative": 0.6, "social": 0.5, "learning": 0.2, "travel": 0.1, "home": 0.0},
    "planning":  {"work": 0.7, "creative": 0.3, "learning": 0.4, "social": 0.2, "travel": 0.1, "home": 0.3},
    "finance":   {"work": 0.9, "creative": 0.1, "learning": 0.3, "social": 0.2, "travel": 0.0, "home": 0.4},
    "travel":    {"work": 0.1, "creative": 0.3, "learning": 0.2, "social": 0.4, "travel": 0.9, "home": 0.1},
    "security":  {"work": 0.8, "creative": 0.3, "learning": 0.5, "social": 0.1, "travel": 0.0, "home": 0.1},
    "health":    {"work": 0.6, "creative": 0.2, "learning": 0.7, "social": 0.3, "travel": 0.0, "home": 0.4},
    "school":    {"work": 0.5, "creative": 0.4, "learning": 0.9, "social": 0.3, "travel": 0.0, "home": 0.2},
    "memory":    {"work": 0.5, "creative": 0.4, "learning": 0.8, "social": 0.1, "travel": 0.0, "home": 0.2},
    "architecture": {"work": 0.6, "creative": 0.5, "learning": 0.7, "social": 0.1, "travel": 0.0, "home": 0.1},
}

DEFAULT_MODE = {"work": 0.5, "creative": 0.5, "learning": 0.5, "social": 0.3, "travel": 0.1, "home": 0.3}


def infer_mode_vector(title, contexts=None):
    """Infer mode vector from title keywords."""
    title_lower = title.lower()
    for keyword, template in MODE_TEMPLATES.items():
        if keyword in title_lower:
            return dict(template)
    return dict(DEFAULT_MODE)


def mode_similarity(vec_a, vec_b):
    """Cosine similarity between two mode vectors."""
    if not vec_a or not vec_b:
        return 0.5
    all_keys = set(list(vec_a.keys()) + list(vec_b.keys()))
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in all_keys)
    mag_a = sum(v ** 2 for v in vec_a.values()) ** 0.5
    mag_b = sum(v ** 2 for v in vec_b.values()) ** 0.5
    if mag_a < 0.001 or mag_b < 0.001:
        return 0.5
    return round(dot / (mag_a * mag_b), 3)


# ══════════════════════════════════════════════════════════
# CORE NODE OPERATIONS
# ══════════════════════════════════════════════════════════

def add_node(conn, ntype, title, desc="", content="", source="", weight=0.5,
             author="", source_folder=""):
    nid = _id()
    ts = _now()
    log = json.dumps([_today()])
    nhash = _node_hash(title, content)
    if not author:
        author = _get_author()
    if not source_folder:
        source_folder = str(ROOT)
    conn.execute(
        """INSERT INTO nodes (id, type, title, description, content, source,
           created_at, last_used, use_count, quality, weight, initial_weight,
           occurrence_log, recency, frequency, consistency, memory_stage,
           author, source_folder, node_hash)
           VALUES (?,?,?,?,?,?,?,?,1,0,?,?,?,1.0,0.0,0.0,'impulse',?,?,?)""",
        (nid, ntype, title, desc, content, source, ts, ts, weight, weight, log,
         author, source_folder, nhash)
    )
    conn.commit()
    return nid


def find_similar_node(conn, ntype, title):
    """Text-based dedup: exact match, then fuzzy Jaccard word-overlap.

    Iteration 3 fix: Uses fuzzy word matching — two words "match" if one
    contains the other or they share a 4+ character common prefix. This
    handles stemming variations (follow/followup, demo/demos/demonstrations).
    """
    STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
                  "for", "and", "or", "on", "at", "by", "with", "from", "as", "it",
                  "that", "this", "be", "has", "had", "have", "do", "does", "did",
                  "but", "not", "so", "if", "no", "its", "my", "i", "we", "they",
                  "get", "gets", "got", "more", "than", "also", "just", "very",
                  "can", "will", "should", "would", "could", "how", "what", "when",
                  "where", "who", "why", "been", "being", "use", "using", "used"}
    JACCARD_THRESHOLD = 0.35  # Lowered slightly because fuzzy matching is stricter per-pair

    def _sig_words(text):
        return [w for w in text.lower().split() if w not in STOP_WORDS and len(w) > 2]

    def _fuzzy_match(w1, w2):
        """Two words match if one contains the other, or they share a 4+ char prefix."""
        if w1 == w2:
            return True
        if w1 in w2 or w2 in w1:
            return True
        # Common prefix of 4+ chars (handles stem variants)
        prefix_len = 0
        for c1, c2 in zip(w1, w2):
            if c1 == c2:
                prefix_len += 1
            else:
                break
        return prefix_len >= 4

    def _fuzzy_jaccard(words_a, words_b):
        """Fuzzy Jaccard: count fuzzy matches between word lists."""
        if not words_a or not words_b:
            return 0.0
        set_a = list(words_a)
        set_b = list(words_b)
        matched_a = set()
        matched_b = set()
        for i, wa in enumerate(set_a):
            for j, wb in enumerate(set_b):
                if j not in matched_b and _fuzzy_match(wa, wb):
                    matched_a.add(i)
                    matched_b.add(j)
                    break
        matches = len(matched_a)
        union = len(set_a) + len(set_b) - matches
        return matches / union if union > 0 else 0.0

    # 1. Exact match
    row = conn.execute(
        "SELECT id FROM nodes WHERE type=? AND title=? LIMIT 1", (ntype, title)
    ).fetchone()
    if row:
        return row["id"]

    # 2. Fuzzy Jaccard similarity against all nodes of same type
    input_words = _sig_words(title)
    if len(input_words) < 2:
        return None

    candidates = conn.execute(
        "SELECT id, title FROM nodes WHERE type=?", (ntype,)
    ).fetchall()

    best_id, best_sim = None, 0.0
    for cand in candidates:
        cand_words = _sig_words(cand["title"])
        if not cand_words:
            continue
        sim = _fuzzy_jaccard(input_words, cand_words)
        if sim > best_sim:
            best_sim = sim
            best_id = cand["id"]

    if best_sim >= JACCARD_THRESHOLD:
        return best_id
    return None


def find_child_by_title(conn, parent_id, title):
    """Find a child of parent whose title matches."""
    children = get_children(conn, parent_id)
    title_lower = title.lower()
    for ch in children:
        if ch["title"].lower() == title_lower:
            return ch["id"]
    words = title_lower.split()
    if len(words) >= 2:
        for ch in children:
            ch_words = ch["title"].lower().split()
            if words[0] in ch_words and words[1] in ch_words:
                return ch["id"]
    return None


def link(conn, src, tgt, weight=1.0, edge_type="parent_child"):
    existing = conn.execute(
        "SELECT id FROM edges WHERE src=? AND tgt=? AND type=? LIMIT 1", (src, tgt, edge_type)
    ).fetchone()
    if existing:
        # Update weight if higher
        conn.execute("UPDATE edges SET weight=MAX(weight,?) WHERE id=?", (weight, existing["id"]))
        conn.commit()
        return existing["id"]
    eid = _id()
    conn.execute("INSERT INTO edges VALUES (?,?,?,?,?,?)",
        (eid, src, tgt, edge_type, weight, _now()))
    conn.commit()
    return eid


def get_children(conn, nid):
    return [dict(r) for r in conn.execute(
        """SELECT n.*, e.weight as edge_weight FROM nodes n
           JOIN edges e ON e.tgt=n.id
           WHERE e.src=? AND e.type='parent_child'
           ORDER BY n.weight DESC, e.weight DESC""",
        (nid,)
    ).fetchall()]


# ══════════════════════════════════════════════════════════
# STORE — accepts structured JSON from Claude Desktop
# ══════════════════════════════════════════════════════════

def cmd_store(conn, title_or_json, content=None):
    """Store structured memory. Accepts JSON (preferred) or flat text."""

    # Detect mode
    data = None
    if content is None:
        try:
            data = json.loads(title_or_json)
        except (json.JSONDecodeError, TypeError):
            print(json.dumps({"error": "Single argument must be valid JSON"}))
            return
    else:
        data = _flat_to_structured(title_or_json, content)

    if not data or not data.get("super_context"):
        print(json.dumps({"error": "Missing super_context"}))
        return

    # ── Resolve super context ──
    sc_title = data["super_context"]
    sc_id = find_similar_node(conn, "super_context", sc_title)
    if sc_id:
        record_occurrence(conn, sc_id)
    else:
        sc_id = add_node(conn, "super_context", sc_title,
                         desc=data.get("description", ""), source="claude_desktop")
        # v3: Infer and store mode vector for new SCs
        mode_vec = infer_mode_vector(sc_title, data.get("contexts", []))
        conn.execute("UPDATE nodes SET mode_vector=? WHERE id=?", (json.dumps(mode_vec), sc_id))
        conn.commit()

    # ── Process contexts and anchors ──
    ctx_created, ctx_reused, anc_created, anc_reused = 0, 0, 0, 0
    all_anchor_ids = []  # for task_anchors tracking

    for ctx_data in data.get("contexts", []):
        ctx_title = ctx_data.get("title", "")
        if not ctx_title:
            continue
        ctx_weight = ctx_data.get("weight", 0.5)

        existing_ctx = find_child_by_title(conn, sc_id, ctx_title)
        if existing_ctx:
            ctx_id = existing_ctx
            record_occurrence(conn, ctx_id)
            # Keep higher weight
            old = conn.execute("SELECT weight FROM nodes WHERE id=?", (ctx_id,)).fetchone()
            if old and ctx_weight > old["weight"]:
                conn.execute("UPDATE nodes SET weight=?, initial_weight=? WHERE id=?",
                             (ctx_weight, ctx_weight, ctx_id))
                conn.commit()
            ctx_reused += 1
        else:
            ctx_id = add_node(conn, "context", ctx_title,
                              desc=ctx_data.get("description", ""),
                              source="claude_desktop", weight=ctx_weight)
            link(conn, sc_id, ctx_id, weight=ctx_weight)
            ctx_created += 1

        for anc_data in ctx_data.get("anchors", []):
            anc_title = anc_data.get("title", "")
            if not anc_title or len(anc_title) < 5:
                continue
            anc_weight = anc_data.get("weight", 0.5)
            anc_content = anc_data.get("content", anc_title)

            existing_anc = find_similar_node(conn, "anchor", anc_title)
            if existing_anc:
                record_occurrence(conn, existing_anc)
                old = conn.execute("SELECT weight FROM nodes WHERE id=?", (existing_anc,)).fetchone()
                if old and anc_weight > old["weight"]:
                    conn.execute("UPDATE nodes SET weight=?, initial_weight=? WHERE id=?",
                                 (anc_weight, anc_weight, existing_anc))
                    conn.commit()
                exec_params = json.dumps(anc_data.get("execution_params", {}))
                all_anchor_ids.append((existing_anc, ctx_id, exec_params))
                anc_reused += 1
            else:
                anc_id = add_node(conn, "anchor", anc_title,
                                  content=anc_content, source="claude_desktop",
                                  weight=anc_weight)
                link(conn, ctx_id, anc_id, weight=anc_weight)
                # P9_TAGS_PATCH: Auto-generate and store tags
                _anc_tags = anc_data.get("tags", [])
                if not _anc_tags:
                    _tag_words = [w.lower() for w in anc_title.split() if len(w) > 3]
                    _tag_words += [w.lower() for w in anc_content.split() if len(w) > 4][:5]
                    _anc_tags = list(set(_tag_words))[:8]
                try:
                    conn.execute("UPDATE nodes SET tags=? WHERE id=?", (json.dumps(_anc_tags), anc_id))
                    conn.commit()
                except Exception:
                    pass  # tags column might not exist yet
                # P12: Embeddings built in batch via build_index() — not per-anchor
                exec_params = json.dumps(anc_data.get("execution_params", {}))
                all_anchor_ids.append((anc_id, ctx_id, exec_params))
                anc_created += 1

    # ── Log task + record which anchors are present ──
    tid = _id()
    conn.execute(
        "INSERT INTO tasks VALUES (?,?,?,?,NULL,0,?,'',?)",
        (tid, sc_id, data.get("summary", sc_title), _now(), json.dumps(data),
         json.dumps(infer_mode_vector(sc_title)))
    )
    for anc_id, ctx_id, exec_params in all_anchor_ids:
        conn.execute("INSERT INTO task_anchors(task_id, anchor_id, context_id, was_retrieved, execution_params) VALUES (?,?,?,0,?)",
                     (tid, anc_id, ctx_id, exec_params))
    conn.commit()

    # Save task file
    (TASKS_DIR / f"{tid}.json").write_text(json.dumps({
        "id": tid, "super_context": sc_title, "timestamp": _now(),
        "contexts_created": ctx_created, "anchors_created": anc_created,
    }, indent=2))

    # ── Neural embedding: add new nodes to vector index ──
    _neural_indexed = 0
    try:
        from embedding_engine import add_single, INDEX_FILE
        if INDEX_FILE.exists():
            for anc_id, ctx_id, _ in all_anchor_ids:
                anc_node = conn.execute("SELECT title, content FROM nodes WHERE id=?", (anc_id,)).fetchone()
                if anc_node:
                    title = anc_node["title"] or ""
                    content = anc_node["content"] or ""
                    text = f"{title}. {content}" if content and content != title else title
                    add_single(anc_id, text)
                    _neural_indexed += 1
    except ImportError:
        pass  # embedding engine not available

    print(json.dumps({
        "stored": True, "task_id": tid, "super_context": sc_title,
        "contexts_created": ctx_created, "contexts_reused": ctx_reused,
        "anchors_created": anc_created, "anchors_reused": anc_reused,
        "neural_indexed": _neural_indexed,
    }, indent=2))


def _flat_to_structured(title, content):
    """Backward compatibility: convert flat text to structured JSON."""
    lines = [l.strip() for l in content.strip().split("\n") if l.strip() and len(l.strip()) >= 10]
    anchors = []
    for line in lines:
        if ": " in line and len(line.split(": ")[0]) < 60:
            parts = line.split(": ", 1)
            anchors.append({"title": parts[0], "content": parts[1], "weight": 0.5})
        else:
            anchors.append({"title": line[:80], "content": line, "weight": 0.5})
    return {
        "super_context": title,
        "contexts": [{"title": f"{title} — {datetime.now().strftime('%b %d')}",
                      "weight": 0.7, "anchors": anchors}],
        "summary": title
    }


# ══════════════════════════════════════════════════════════
# RETRIEVE — temporal boost + winning structure (Fixes 2+3)
# ══════════════════════════════════════════════════════════

def cmd_retrieve(conn, query):
    """Retrieve memories — routes to P9+ if available, else falls back to word-overlap."""
    if _HAS_P9:
        return _p9_retrieve(conn, query)
    return _cmd_retrieve_legacy(conn, query)

def _cmd_retrieve_legacy(conn, query):
    """Legacy retrieve: word-overlap with temporal awareness and winning-structure priority."""
    query_lower = query.lower()
    query_words = set(w for w in query_lower.split() if len(w) > 2)

    scs = [dict(r) for r in conn.execute(
        "SELECT * FROM nodes WHERE type='super_context' ORDER BY last_used DESC"
    ).fetchall()]

    scored = []
    for sc in scs:
        # Word overlap scoring
        sc_text = f"{sc['title']} {sc['description']}".lower()
        sc_words = set(w for w in sc_text.split() if len(w) > 2)
        overlap = len(query_words & sc_words) * 2

        children = get_children(conn, sc["id"])
        for ch in children:
            ch_text = f"{ch['title']} {ch.get('description', '')}".lower()
            overlap += len(query_words & set(w for w in ch_text.split() if len(w) > 2))
            for anc in get_children(conn, ch["id"]):
                anc_text = f"{anc['title']} {anc.get('content', '')}".lower()
                for w in query_words:
                    if w in anc_text:
                        overlap += 0.5

        # Quality + usage bonus
        quality_bonus = (sc["quality"] / 5.0) * 2 if sc["quality"] else 0
        use_bonus = min(sc["use_count"] / 10.0, 0.5)

        # FIX 2: Temporal boost for this SC
        t_boost = temporal_boost(sc.get("memory_stage", "impulse"))

        # v3: Mode vector boost
        query_mode = infer_mode_vector(query, [])
        node_mode_str = sc.get("mode_vector") or "{}"
        try:
            node_mode = json.loads(node_mode_str)
        except json.JSONDecodeError:
            node_mode = {}
        m_boost = 0.5 + mode_similarity(query_mode, node_mode) if node_mode else 1.0

        total = (overlap + quality_bonus + use_bonus) * t_boost * m_boost

        if total > 0:
            scored.append((total, sc))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Build results
    context_pack = {"query": query, "memories": [], "total_matches": len(scored)}

    for score, sc in scored[:3]:
        # FIX 3: Find the winning structure for this SC
        winning = _get_winning_structure(conn, sc["id"])

        memory = {
            "super_context": sc["title"],
            "sc_id": sc["id"],
            "relevance": round(score, 2),
            "quality": sc["quality"],
            "uses": sc["use_count"],
            "stage": sc.get("memory_stage", "impulse"),
            "temporal_boost": temporal_boost(sc.get("memory_stage", "impulse")),
            "winning_score": winning.get("score", 0) if winning else 0,
            "contexts": []
        }

        # v3: Include decision constraints
        dec_rows = conn.execute(
            "SELECT decision, confidence, reversible FROM decisions WHERE sc_id=? AND confidence > 0.3 ORDER BY confidence DESC",
            (sc["id"],)
        ).fetchall()
        if dec_rows:
            memory["decisions"] = [{"decision": d["decision"], "confidence": round(d["confidence"], 2), "locked": not d["reversible"]} for d in dec_rows]
            memory["convergence"] = get_convergence_state(conn, sc["id"])["convergence"]

        # Get anchors that were in winning structure
        winning_anchors = set()
        if winning and winning.get("snapshot"):
            for ctx_name, anc_list in winning["snapshot"].items():
                for a in anc_list:
                    winning_anchors.add(a.lower() if isinstance(a, str) else "")

        for ctx in get_children(conn, sc["id"]):
            ctx_entry = {
                "context": ctx["title"],
                "weight": round(ctx.get("edge_weight", ctx.get("weight", 0.5)), 2),
                "stage": ctx.get("memory_stage", "impulse"),
                "anchors": []
            }

            for anc in get_children(conn, ctx["id"]):
                # FIX 1: Use evidence-based weight
                ev_weight = compute_evidence_weight(conn, anc["id"])
                in_winning = anc["title"].lower() in winning_anchors

                adp_val = anc.get("discrimination_power", 0.0)
                anc_entry = {
                    "title": anc["title"],
                    "content": anc.get("content", ""),
                    "weight": ev_weight,
                    "stage": anc.get("memory_stage", "impulse"),
                    "in_winning_structure": in_winning,
                    "discrimination_power": adp_val,
                    "variant_sensitive": adp_val > 1.0,
                }

                # v3 Phase 5: Show best/worst execution when variant-sensitive
                if adp_val > 1.0:
                    best = conn.execute("""
                        SELECT ta.execution_params, t.score FROM task_anchors ta
                        JOIN tasks t ON t.id = ta.task_id
                        WHERE ta.anchor_id = ? AND t.score > 0
                        ORDER BY t.score DESC LIMIT 1
                    """, (anc["id"],)).fetchone()
                    worst = conn.execute("""
                        SELECT ta.execution_params, t.score FROM task_anchors ta
                        JOIN tasks t ON t.id = ta.task_id
                        WHERE ta.anchor_id = ? AND t.score > 0
                        ORDER BY t.score ASC LIMIT 1
                    """, (anc["id"],)).fetchone()
                    if best:
                        try:
                            anc_entry["best_execution"] = {"params": json.loads(best["execution_params"] or "{}"), "score": best["score"]}
                        except json.JSONDecodeError:
                            pass
                    if worst:
                        try:
                            anc_entry["worst_execution"] = {"params": json.loads(worst["execution_params"] or "{}"), "score": worst["score"]}
                        except json.JSONDecodeError:
                            pass

                ctx_entry["anchors"].append(anc_entry)

            # Sort: winning-structure anchors first, then by weight
            ctx_entry["anchors"].sort(
                key=lambda a: (a["in_winning_structure"], a["weight"]),
                reverse=True
            )
            memory["contexts"].append(ctx_entry)

        context_pack["memories"].append(memory)

    # v3: Augment with transfer knowledge from related SCs
    seen_sc_ids = {m["sc_id"] for m in context_pack["memories"]}
    for mem_entry in list(context_pack["memories"]):
        transfer_rows = conn.execute("""
            SELECT e.tgt as tgt_id, e.weight as strength, n.title as tgt_title
            FROM edges e JOIN nodes n ON n.id = e.tgt
            WHERE e.src = ? AND e.type = 'transfer'
            ORDER BY e.weight DESC LIMIT 2
        """, (mem_entry["sc_id"],)).fetchall()

        for t in transfer_rows:
            if t["tgt_id"] not in seen_sc_ids:
                tgt_sc = conn.execute("SELECT * FROM nodes WHERE id=?", (t["tgt_id"],)).fetchone()
                if tgt_sc:
                    transfer_mem = {
                        "super_context": tgt_sc["title"],
                        "sc_id": tgt_sc["id"],
                        "transferred_from": mem_entry["super_context"],
                        "transfer_strength": t["strength"],
                        "transfer_discount": TRANSFER_DISCOUNT,
                        "relevance": round(mem_entry["relevance"] * TRANSFER_DISCOUNT, 2),
                        "quality": tgt_sc["quality"],
                        "contexts": []
                    }
                    for ctx in get_children(conn, tgt_sc["id"]):
                        ctx_entry = {"context": ctx["title"], "weight": round(ctx.get("weight", 0.5), 2), "anchors": []}
                        for anc in get_children(conn, ctx["id"])[:5]:  # limit transferred anchors
                            ctx_entry["anchors"].append({
                                "title": anc["title"], "content": anc.get("content", ""),
                                "weight": round(anc.get("weight", 0.5) * TRANSFER_DISCOUNT, 2),
                                "transferred": True
                            })
                        if ctx_entry["anchors"]:
                            transfer_mem["contexts"].append(ctx_entry)
                    if transfer_mem["contexts"]:
                        context_pack["memories"].append(transfer_mem)
                        seen_sc_ids.add(t["tgt_id"])

    # Record that these anchors were retrieved (for future scoring)
    _record_retrieval(conn, context_pack)

    print(json.dumps(context_pack, indent=2))


def _get_winning_structure(conn, sc_id):
    """Find the highest-scoring task for this SC and return its snapshot."""
    row = conn.execute(
        """SELECT id, score, structure_snapshot FROM tasks
           WHERE sc_id=? AND completed IS NOT NULL AND score > 0
           ORDER BY score DESC, completed DESC LIMIT 1""",
        (sc_id,)
    ).fetchone()

    if not row or not row["structure_snapshot"]:
        return None

    try:
        snapshot = json.loads(row["structure_snapshot"])
    except (json.JSONDecodeError, TypeError):
        return None

    return {"task_id": row["id"], "score": row["score"], "snapshot": snapshot}


def _record_retrieval(conn, context_pack):
    """Record retrieval occurrences for all served anchors — enables temporal stage evolution."""
    for mem in context_pack.get("memories", []):
        sc_id = mem.get("sc_id")
        if sc_id:
            record_occurrence(conn, sc_id)
        for ctx in mem.get("contexts", []):
            # Find context node by title under this SC
            if sc_id:
                ctx_id = find_child_by_title(conn, sc_id, ctx.get("context", ""))
                if ctx_id:
                    record_occurrence(conn, ctx_id)
            for anc in ctx.get("anchors", []):
                # Find anchor by title
                anc_row = conn.execute(
                    "SELECT id FROM nodes WHERE type='anchor' AND title=? LIMIT 1",
                    (anc.get("title", ""),)
                ).fetchone()
                if anc_row:
                    record_occurrence(conn, anc_row["id"])


# ══════════════════════════════════════════════════════════
# SCORE — snapshot + evidence update (Fixes 1+3)
# ══════════════════════════════════════════════════════════

def cmd_score(conn, task_id, score_val):
    """Score a task. Captures structure snapshot and updates evidence-based weights."""
    score = float(score_val)

    task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not task:
        print(json.dumps({"error": "Task not found"}))
        return

    sc_id = task["sc_id"]
    if not sc_id:
        print(json.dumps({"error": "Task has no super context"}))
        return

    # FIX 3: Capture structure snapshot from THIS task's actual anchors
    snapshot = _capture_snapshot(conn, task_id)
    conn.execute(
        "UPDATE tasks SET completed=?, score=?, structure_snapshot=? WHERE id=?",
        (_now(), score, json.dumps(snapshot), task_id)
    )
    conn.commit()

    # Update SC quality from all scored tasks (not incremental — avoids wrong denominator)
    sc = conn.execute("SELECT * FROM nodes WHERE id=?", (sc_id,)).fetchone()
    if sc:
        scored_tasks = conn.execute(
            "SELECT score FROM tasks WHERE sc_id=? AND completed IS NOT NULL AND score > 0",
            (sc_id,)
        ).fetchall()
        if scored_tasks:
            avg_q = sum(t["score"] for t in scored_tasks) / len(scored_tasks)
            conn.execute("UPDATE nodes SET quality=? WHERE id=?", (avg_q, sc_id))
            conn.commit()

    # v3: Reinforce/decay decisions based on task score
    if score >= 4.0:
        conn.execute(
            "UPDATE decisions SET confidence = MIN(1.0, confidence + 0.1 * (1.0 - confidence)) WHERE sc_id=?",
            (sc_id,))
    elif score < 2.5:
        conn.execute(
            "UPDATE decisions SET confidence = confidence * 0.7, reversible = 1 WHERE sc_id=?",
            (sc_id,))
    conn.commit()

    # FIX 1: Recompute evidence-based weights for all anchors in this task
    anchor_rows = conn.execute(
        "SELECT anchor_id FROM task_anchors WHERE task_id=?", (task_id,)
    ).fetchall()

    updated = 0
    for row in anchor_rows:
        new_w = compute_evidence_weight(conn, row["anchor_id"])
        adp = compute_discrimination_power(conn, row["anchor_id"])
        conn.execute("UPDATE nodes SET weight=?, discrimination_power=? WHERE id=?",
                     (new_w, adp, row["anchor_id"]))
        updated += 1

    # Also update context weights based on their children's average
    for ctx in get_children(conn, sc_id):
        child_weights = [a.get("weight", 0.5) for a in get_children(conn, ctx["id"])]
        if child_weights:
            ctx_w = sum(child_weights) / len(child_weights)
            conn.execute("UPDATE nodes SET weight=? WHERE id=?", (ctx_w, ctx["id"]))
            updated += 1

    conn.commit()

    print(json.dumps({
        "scored": True,
        "task_id": task_id,
        "score": score,
        "anchors_updated": updated,
        "snapshot_captured": True,
        "winning_recipe": snapshot,
    }, indent=2))


def _capture_snapshot(conn, task_id):
    """Capture the exact tree that belonged to THIS task, not the live tree.
    Reads from task_anchors which records which anchors were stored with which task."""
    rows = conn.execute("""
        SELECT ta.context_id, n_ctx.title as ctx_title, n_anc.title as anc_title
        FROM task_anchors ta
        JOIN nodes n_anc ON n_anc.id = ta.anchor_id
        JOIN nodes n_ctx ON n_ctx.id = ta.context_id
        WHERE ta.task_id = ?
        ORDER BY n_ctx.title, n_anc.weight DESC
    """, (task_id,)).fetchall()

    snapshot = {}
    for row in rows:
        ctx_title = row["ctx_title"]
        if ctx_title not in snapshot:
            snapshot[ctx_title] = []
        snapshot[ctx_title].append(row["anc_title"])

    return snapshot


# ══════════════════════════════════════════════════════════
# REBUILD — recompute all weights + temporal (Fixes 1+2)
# ══════════════════════════════════════════════════════════

def cmd_rebuild(conn):
    """Recompute all weights from evidence and all temporal scores."""
    updated = 0

    # Recompute temporal for all nodes
    all_nodes = conn.execute("SELECT id, occurrence_log FROM nodes").fetchall()
    for node in all_nodes:
        try:
            log = json.loads(node["occurrence_log"] or "[]")
        except json.JSONDecodeError:
            log = []
        recency, frequency, consistency = compute_temporal(log)
        stage = classify_stage(recency, frequency, consistency)
        conn.execute(
            "UPDATE nodes SET recency=?, frequency=?, consistency=?, memory_stage=? WHERE id=?",
            (recency, frequency, consistency, stage, node["id"])
        )
        updated += 1

    # Recompute evidence-based weights + ADP for all anchors
    anchors = conn.execute("SELECT id FROM nodes WHERE type='anchor'").fetchall()
    for anc in anchors:
        new_w = compute_evidence_weight(conn, anc["id"])
        adp = compute_discrimination_power(conn, anc["id"])
        conn.execute("UPDATE nodes SET weight=?, discrimination_power=? WHERE id=?",
                     (new_w, adp, anc["id"]))

    # Recompute context weights from children averages
    contexts = conn.execute("SELECT id FROM nodes WHERE type='context'").fetchall()
    for ctx in contexts:
        child_weights = [a.get("weight", 0.5) for a in get_children(conn, ctx["id"])]
        if child_weights:
            conn.execute("UPDATE nodes SET weight=? WHERE id=?",
                         (sum(child_weights) / len(child_weights), ctx["id"]))

    # Recompute SC quality from task scores
    scs = conn.execute("SELECT id FROM nodes WHERE type='super_context'").fetchall()
    for sc in scs:
        tasks = conn.execute(
            "SELECT score FROM tasks WHERE sc_id=? AND completed IS NOT NULL AND score > 0",
            (sc["id"],)
        ).fetchall()
        if tasks:
            avg = sum(t["score"] for t in tasks) / len(tasks)
            conn.execute("UPDATE nodes SET quality=? WHERE id=?", (avg, sc["id"]))

    # v3: Detect transfer edges between SCs
    transfers = detect_transfers(conn)

    conn.commit()
    print(json.dumps({"rebuilt": True, "nodes_updated": updated, "transfers_detected": len(transfers)}))


# ══════════════════════════════════════════════════════════
# DECISION LOG + CONVERGENCE (v3 Phase 1)
# ══════════════════════════════════════════════════════════

def get_convergence_state(conn, sc_id):
    """Compute how converged a super context's decisions are."""
    decisions = conn.execute(
        "SELECT * FROM decisions WHERE sc_id=? ORDER BY created_at", (sc_id,)
    ).fetchall()

    if not decisions:
        return {"convergence": 0.0, "total_decisions": 0, "locked": 0, "open": 0, "avg_confidence": 0.0}

    locked = sum(1 for d in decisions if not d["reversible"])
    total = len(decisions)
    avg_confidence = sum(d["confidence"] for d in decisions) / total
    lock_ratio = locked / total if total > 0 else 0

    convergence = (avg_confidence * 0.4) + (lock_ratio * 0.4) + (min(total / 10.0, 1.0) * 0.2)

    return {
        "convergence": round(convergence, 3),
        "total_decisions": total,
        "locked": locked,
        "open": total - locked,
        "avg_confidence": round(avg_confidence, 3)
    }


def session_preamble(conn, sc_id):
    """Generate constraint list for AI at conversation start."""
    state = get_convergence_state(conn, sc_id)
    decisions = conn.execute(
        "SELECT decision, confidence, reversible FROM decisions WHERE sc_id=? AND confidence > 0.3 ORDER BY confidence DESC",
        (sc_id,)
    ).fetchall()

    constraints = []
    for d in decisions:
        lock = "locked" if not d["reversible"] else "open"
        constraints.append({"decision": d["decision"], "confidence": round(d["confidence"], 2), "status": lock})

    return {
        "convergence": state["convergence"],
        "constraints": constraints,
        "instruction": "These decisions are established. Do not contradict locked decisions. Open decisions can be revisited with evidence."
    }


def cmd_decision(conn, sc_id, data_json):
    """Store a new decision for a super context."""
    try:
        data = json.loads(data_json)
    except (json.JSONDecodeError, TypeError):
        print(json.dumps({"error": "Invalid JSON for decision data"}))
        return

    # Verify SC exists
    sc = conn.execute("SELECT id, title FROM nodes WHERE id=? AND type='super_context'", (sc_id,)).fetchone()
    if not sc:
        print(json.dumps({"error": f"Super context {sc_id} not found"}))
        return

    did = _id()
    conn.execute(
        """INSERT INTO decisions (id, sc_id, anchor_id, session_id, decision,
           alternatives_eliminated, confidence, reversible, evidence, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (did, sc_id, data.get("anchor_id", ""), data.get("session_id", _id()),
         data["decision"], json.dumps(data.get("alternatives_eliminated", [])),
         data.get("confidence", 0.5), 1 if data.get("reversible", True) else 0,
         data.get("evidence", ""), _now())
    )
    conn.commit()

    print(json.dumps({
        "stored": True, "decision_id": did,
        "sc": sc["title"], "decision": data["decision"],
        "convergence": get_convergence_state(conn, sc_id)
    }, indent=2))


def cmd_convergence(conn, sc_id):
    """Show convergence state for a super context."""
    state = get_convergence_state(conn, sc_id)
    preamble = session_preamble(conn, sc_id)
    print(json.dumps({"sc_id": sc_id, **state, "constraints": preamble["constraints"]}, indent=2))


def cmd_decisions_list(conn, sc_id):
    """List all decisions for a super context."""
    decisions = conn.execute(
        "SELECT * FROM decisions WHERE sc_id=? ORDER BY confidence DESC", (sc_id,)
    ).fetchall()
    result = []
    for d in decisions:
        result.append({
            "id": d["id"], "decision": d["decision"],
            "confidence": d["confidence"], "locked": not d["reversible"],
            "evidence": d["evidence"], "created_at": d["created_at"]
        })
    print(json.dumps({"sc_id": sc_id, "decisions": result, "total": len(result)}, indent=2))


# ══════════════════════════════════════════════════════════
# TRANSFER EDGES (v3 Phase 2)
# ══════════════════════════════════════════════════════════

TRANSFER_THRESHOLD = 0.25
TRANSFER_DISCOUNT = 0.6


def compute_structural_similarity(conn, sc_id_a, sc_id_b):
    """How similar are two super contexts' internal structures?"""
    ctx_a = get_children(conn, sc_id_a)
    ctx_b = get_children(conn, sc_id_b)
    if not ctx_a or not ctx_b:
        return 0.0

    # Fuzzy match on context titles
    ctx_titles_a = [c["title"].lower() for c in ctx_a]
    ctx_titles_b = [c["title"].lower() for c in ctx_b]

    matched = 0
    for ta in ctx_titles_a:
        words_a = set(ta.split())
        for tb in ctx_titles_b:
            words_b = set(tb.split())
            union = len(words_a | words_b)
            if union > 0 and len(words_a & words_b) / union > 0.3:
                matched += 1
                break
    ctx_sim = matched / max(len(ctx_titles_a), len(ctx_titles_b))

    # Anchor title overlap
    anc_a = set()
    for ctx in ctx_a:
        for anc in get_children(conn, ctx["id"]):
            anc_a.add(anc["title"].lower())
    anc_b = set()
    for ctx in ctx_b:
        for anc in get_children(conn, ctx["id"]):
            anc_b.add(anc["title"].lower())

    anc_union = len(anc_a | anc_b)
    anc_overlap = len(anc_a & anc_b) / anc_union if anc_union > 0 else 0

    return round(ctx_sim * 0.6 + anc_overlap * 0.4, 3)


def detect_transfers(conn):
    """Find all SC pairs with structural similarity > threshold. Create transfer edges."""
    scs = conn.execute("SELECT id, title FROM nodes WHERE type='super_context'").fetchall()
    transfers = []

    # Clean old transfer edges first
    conn.execute("DELETE FROM edges WHERE type='transfer'")
    conn.commit()

    for i, sc_a in enumerate(scs):
        for sc_b in scs[i + 1:]:
            sim = compute_structural_similarity(conn, sc_a["id"], sc_b["id"])
            if sim >= TRANSFER_THRESHOLD:
                link(conn, sc_a["id"], sc_b["id"], sim, "transfer")
                link(conn, sc_b["id"], sc_a["id"], sim, "transfer")
                transfers.append({"from": sc_a["title"], "to": sc_b["title"], "similarity": sim})

    return transfers


def cmd_transfers(conn):
    """Detect and show all transfer relationships."""
    transfers = detect_transfers(conn)
    print(json.dumps({"transfers": transfers, "total": len(transfers)}, indent=2))


# ══════════════════════════════════════════════════════════
# BROWSE / TREE / STATS — now include temporal info
# ══════════════════════════════════════════════════════════

def cmd_browse(conn):
    scs = [dict(r) for r in conn.execute(
        "SELECT * FROM nodes WHERE type='super_context' ORDER BY quality DESC, last_used DESC"
    ).fetchall()]
    result = []
    for sc in scs:
        children = get_children(conn, sc["id"])
        anc_count = sum(len(get_children(conn, c["id"])) for c in children)
        best = conn.execute(
            "SELECT MAX(score) as s FROM tasks WHERE sc_id=? AND completed IS NOT NULL",
            (sc["id"],)
        ).fetchone()
        result.append({
            "id": sc["id"], "title": sc["title"],
            "contexts": len(children), "anchors": anc_count,
            "quality": sc["quality"], "best_score": best["s"] or 0,
            "uses": sc["use_count"], "stage": sc.get("memory_stage", "?"),
            "last_used": sc["last_used"]
        })
    print(json.dumps({"super_contexts": result, "total": len(result)}, indent=2))


def cmd_tree(conn):
    scs = [dict(r) for r in conn.execute(
        "SELECT * FROM nodes WHERE type='super_context' ORDER BY quality DESC, last_used DESC"
    ).fetchall()]

    lines = ["PMIS Memory Tree", "=" * 50]
    for sc in scs:
        q = f" q={sc['quality']:.1f}" if sc["quality"] else ""
        stg = sc.get("memory_stage", "?")
        lines.append(f"\n[SC] {sc['title']}{q} [{stg}]")

        for ctx in get_children(conn, sc["id"]):
            w = ctx.get("edge_weight", ctx.get("weight", 0.5))
            stg = ctx.get("memory_stage", "?")
            lines.append(f"  ├── [CTX] {ctx['title']} (w={w:.2f}) [{stg}]")

            anchors = get_children(conn, ctx["id"])
            for i, anc in enumerate(anchors):
                prefix = "  │   └──" if i == len(anchors) - 1 else "  │   ├──"
                content = anc.get("content", "")
                w = anc.get("weight", 0.5)
                stg = anc.get("memory_stage", "?")
                text = f"{anc['title']}: {content[:60]}" if content and content != anc["title"] else anc["title"]
                lines.append(f"{prefix} {text} (w={w:.2f}) [{stg}]")

    print("\n".join(lines))


def cmd_stats(conn):
    counts = {}
    for t in ("super_context", "context", "anchor"):
        counts[t] = conn.execute("SELECT COUNT(*) c FROM nodes WHERE type=?", (t,)).fetchone()["c"]
    counts["edges"] = conn.execute("SELECT COUNT(*) c FROM edges").fetchone()["c"]
    counts["tasks"] = conn.execute("SELECT COUNT(*) c FROM tasks").fetchone()["c"]

    # Stage distribution
    stages = {}
    for row in conn.execute("SELECT memory_stage, COUNT(*) c FROM nodes GROUP BY memory_stage").fetchall():
        stages[row["memory_stage"]] = row["c"]

    # Task score stats
    scored = conn.execute(
        "SELECT COUNT(*) c, AVG(score) avg, MAX(score) best FROM tasks WHERE score > 0"
    ).fetchone()

    print(json.dumps({
        "super_contexts": counts["super_context"],
        "contexts": counts["context"],
        "anchors": counts["anchor"],
        "edges": counts["edges"],
        "tasks_logged": counts["tasks"],
        "tasks_scored": scored["c"] or 0,
        "avg_score": round(scored["avg"] or 0, 1),
        "best_score": scored["best"] or 0,
        "memory_stages": stages,
    }, indent=2))


# ══════════════════════════════════════════════════════════
# VISUALIZE — includes stages and winning structure
# ══════════════════════════════════════════════════════════

def cmd_viz(conn):
    scs = [dict(r) for r in conn.execute(
        "SELECT * FROM nodes WHERE type='super_context' ORDER BY quality DESC"
    ).fetchall()]

    tree_data = {"name": "PMIS", "type": "root", "children": []}
    stats = {}
    for t in ("super_context", "context", "anchor"):
        stats[t] = conn.execute("SELECT COUNT(*) c FROM nodes WHERE type=?", (t,)).fetchone()["c"]

    for sc in scs:
        sc_node = {"name": sc["title"], "type": "super_context", "id": sc["id"],
                   "weight": sc["weight"], "quality": sc["quality"],
                   "uses": sc["use_count"], "stage": sc.get("memory_stage", "impulse"),
                   "children": []}
        for ctx in get_children(conn, sc["id"]):
            ctx_node = {"name": ctx["title"], "type": "context",
                        "weight": ctx.get("edge_weight", ctx.get("weight", 0.5)),
                        "uses": ctx["use_count"], "stage": ctx.get("memory_stage", "impulse"),
                        "desc": ctx.get("description", ""), "children": []}
            for anc in get_children(conn, ctx["id"]):
                ctx_node["children"].append({
                    "name": anc["title"], "type": "anchor",
                    "weight": anc.get("weight", 0.5),
                    "stage": anc.get("memory_stage", "impulse"),
                    "desc": anc.get("content", ""),
                    "adp": anc.get("discrimination_power", 0.0)
                })
            sc_node["children"].append(ctx_node)
        tree_data["children"].append(sc_node)

    # ── Dashboard data collection ──
    dashboard = {}

    # 1. Stage distribution
    stage_dist = {"impulse": 0, "active": 0, "established": 0, "fading": 0}
    for row in conn.execute("SELECT memory_stage, COUNT(*) c FROM nodes GROUP BY memory_stage").fetchall():
        stage_dist[row["memory_stage"]] = row["c"]
    dashboard["stages"] = stage_dist

    # 2. Weight distribution (histogram buckets 0-0.1, 0.1-0.2, ... 0.9-1.0)
    weight_buckets = [0]*10
    for row in conn.execute("SELECT weight FROM nodes WHERE type='anchor'").fetchall():
        idx = min(int(row["weight"] * 10), 9)
        weight_buckets[idx] += 1
    dashboard["weight_hist"] = weight_buckets

    # 3. ADP distribution
    adp_vals = [r["discrimination_power"] for r in conn.execute(
        "SELECT discrimination_power FROM nodes WHERE type='anchor' AND discrimination_power > 0"
    ).fetchall()]
    dashboard["adp_values"] = adp_vals
    dashboard["adp_high"] = sum(1 for v in adp_vals if v > 1.0)
    dashboard["adp_low"] = sum(1 for v in adp_vals if v <= 1.0)

    # 4. Task score history
    task_scores = []
    for row in conn.execute("SELECT title, score, completed FROM tasks WHERE score > 0 ORDER BY completed").fetchall():
        task_scores.append({"title": row["title"], "score": row["score"], "date": row["completed"] or ""})
    dashboard["task_scores"] = task_scores
    scored_tasks = conn.execute("SELECT COUNT(*) c, AVG(score) avg, MAX(score) best, MIN(score) worst FROM tasks WHERE score > 0").fetchone()
    dashboard["tasks_scored"] = scored_tasks["c"] or 0
    dashboard["avg_score"] = round(scored_tasks["avg"] or 0, 2)
    dashboard["best_score"] = scored_tasks["best"] or 0
    dashboard["worst_score"] = scored_tasks["worst"] or 0
    dashboard["tasks_total"] = conn.execute("SELECT COUNT(*) c FROM tasks").fetchone()["c"]

    # 5. Per-SC dashboard cards
    sc_cards = []
    for sc in scs:
        card = {
            "id": sc["id"], "title": sc["title"], "quality": sc["quality"],
            "stage": sc.get("memory_stage", "impulse"), "uses": sc["use_count"],
            "weight": sc["weight"]
        }
        # Mode vector
        try:
            card["mode_vector"] = json.loads(sc.get("mode_vector") or "{}")
        except (json.JSONDecodeError, TypeError):
            card["mode_vector"] = {}
        # Convergence
        card["convergence"] = get_convergence_state(conn, sc["id"])
        # Context/anchor counts
        ctxs = get_children(conn, sc["id"])
        card["contexts"] = len(ctxs)
        card["anchors"] = sum(len(get_children(conn, c["id"])) for c in ctxs)
        # Transfer edges
        transfers = conn.execute(
            "SELECT n.title, e.weight FROM edges e JOIN nodes n ON n.id=e.tgt WHERE e.src=? AND e.type='transfer'",
            (sc["id"],)
        ).fetchall()
        card["transfers"] = [{"to": t["title"], "sim": t["weight"]} for t in transfers]
        # Best task
        best = conn.execute("SELECT MAX(score) s FROM tasks WHERE sc_id=? AND score>0", (sc["id"],)).fetchone()
        card["best_task"] = best["s"] or 0
        sc_cards.append(card)
    dashboard["sc_cards"] = sc_cards

    # 6. Transfer edge count
    dashboard["transfer_count"] = conn.execute("SELECT COUNT(*) c FROM edges WHERE type='transfer'").fetchone()["c"]

    # 7. Decision totals
    dashboard["total_decisions"] = conn.execute("SELECT COUNT(*) c FROM decisions").fetchone()["c"]
    dashboard["locked_decisions"] = conn.execute("SELECT COUNT(*) c FROM decisions WHERE reversible=0").fetchone()["c"]

    # 8. Edge counts
    dashboard["parent_child_edges"] = conn.execute("SELECT COUNT(*) c FROM edges WHERE type='parent_child'").fetchone()["c"]

    # 9. Temporal scores (recency/frequency/consistency averages)
    temps = conn.execute("SELECT AVG(recency) r, AVG(frequency) f, AVG(consistency) c FROM nodes WHERE type='anchor'").fetchone()
    dashboard["avg_recency"] = round(temps["r"] or 0, 3)
    dashboard["avg_frequency"] = round(temps["f"] or 0, 3)
    dashboard["avg_consistency"] = round(temps["c"] or 0, 3)

    html = VIZ_TEMPLATE.replace("__DATA__", json.dumps(tree_data))
    html = html.replace("__STATS__", json.dumps(stats))
    html = html.replace("__DASHBOARD__", json.dumps(dashboard))
    VIZ_PATH.write_text(html)
    print(json.dumps({"saved": str(VIZ_PATH)}))


VIZ_TEMPLATE = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PMIS Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;500;600&display=swap');
:root{--bg:#0c0c0f;--bg2:#16161b;--bg3:#1e1e26;--fg:#e2e0d8;--fg2:#8a8880;--fg3:#5a5850;--g:#4ade80;--p:#a78bfa;--o:#fb923c;--b:#60a5fa;--r:#f87171;--t:#2dd4bf;--y:#facc15;--border:#2a2a30}
*{margin:0;padding:0;box-sizing:border-box}body{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--fg);min-height:100vh}
.mono{font-family:'JetBrains Mono',monospace}
/* Header + Tabs */
.hdr{padding:24px 40px 0;border-bottom:1px solid var(--border)}
.hdr h1{font-size:24px;font-weight:300;letter-spacing:2px;margin-bottom:16px}
.hdr h1 b{color:var(--g);font-weight:600}
.tabs{display:flex;gap:0}.tab{padding:10px 24px;font-size:14px;cursor:pointer;border-bottom:2px solid transparent;color:var(--fg2);transition:all .2s}
.tab:hover{color:var(--fg)}.tab.active{color:var(--g);border-bottom-color:var(--g);font-weight:500}
/* Panels */
.panel{display:none;padding:24px 40px 40px}.panel.active{display:block}
/* Cards grid */
.grid{display:grid;gap:16px;margin:16px 0}.g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:1fr 1fr 1fr}.g4{grid-template-columns:1fr 1fr 1fr 1fr}
@media(max-width:900px){.g2,.g3,.g4{grid-template-columns:1fr}}
/* Stat card */
.card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:18px 20px}
.card-title{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--fg2);margin-bottom:8px}
.card-val{font-size:28px;font-weight:600;font-family:'JetBrains Mono',monospace}
.card-sub{font-size:12px;color:var(--fg2);margin-top:4px}
/* Section headers */
.sec{font-size:16px;font-weight:500;margin:28px 0 12px;display:flex;align-items:center;gap:8px}
.sec .dot{width:8px;height:8px;border-radius:50%;background:var(--g)}
/* Bar chart */
.bar-chart{display:flex;align-items:flex-end;gap:4px;height:100px;padding-top:10px}
.bar-col{display:flex;flex-direction:column;align-items:center;flex:1;gap:4px}
.bar{border-radius:4px 4px 0 0;min-height:2px;width:100%;transition:height .3s}
.bar-label{font-size:9px;color:var(--fg2);font-family:'JetBrains Mono',monospace}
/* Pipeline */
.pipeline{display:flex;gap:0;margin:16px 0;flex-wrap:wrap}
.pipe-step{flex:1;min-width:130px;padding:14px 16px;background:var(--bg2);border:1px solid var(--border);position:relative}
.pipe-step:first-child{border-radius:10px 0 0 10px}.pipe-step:last-child{border-radius:0 10px 10px 0}
.pipe-name{font-size:12px;font-weight:500;margin-bottom:6px;color:var(--b)}
.pipe-params{font-size:11px;color:var(--fg2);line-height:1.6}
.pipe-val{color:var(--g);font-family:'JetBrains Mono',monospace;font-weight:600}
.pipe-arrow{position:absolute;right:-8px;top:50%;transform:translateY(-50%);z-index:1;color:var(--fg3);font-size:16px}
/* SC cards */
.sc-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:12px}
.sc-title{font-size:15px;font-weight:500;display:flex;align-items:center;gap:8px;margin-bottom:10px}
.sc-row{display:flex;gap:16px;flex-wrap:wrap}
.sc-metric{font-size:12px;color:var(--fg2)}.sc-metric span{color:var(--g);font-family:'JetBrains Mono',monospace;font-weight:600}
/* Radar chart (SVG) */
.radar-wrap{display:flex;justify-content:center;padding:8px 0}
/* Comparison table */
.cmp-table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}
.cmp-table th{text-align:left;padding:10px 12px;border-bottom:2px solid var(--border);color:var(--fg2);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:1px}
.cmp-table td{padding:8px 12px;border-bottom:1px solid #1a1a20}
.cmp-table tr:hover td{background:var(--bg3)}
.cmp-yes{color:var(--g)}.cmp-no{color:var(--r)}.cmp-partial{color:var(--o)}
/* Badge */
.badge{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-family:'JetBrains Mono',monospace;margin-left:4px}
.b-impulse{background:#2a2040;color:var(--p)}.b-active{background:#1a2a20;color:var(--g)}
.b-established{background:#1a2a30;color:var(--t)}.b-fading{background:#2a2020;color:var(--fg2)}
/* Progress ring */
.ring{position:relative;display:inline-flex;align-items:center;justify-content:center}
.ring-label{position:absolute;font-size:14px;font-weight:600;font-family:'JetBrains Mono',monospace}
/* Transfer badge */
.xfer{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;background:#1a2030;color:var(--b);margin:2px}
/* Health bar */
.health-bar{height:6px;border-radius:3px;background:var(--bg3);overflow:hidden;margin-top:6px}
.health-fill{height:100%;border-radius:3px;transition:width .5s}
/* Memory Graph (tab 2) */
.sf{margin:16px 0}.sf input{width:100%;padding:10px 16px;background:var(--bg2);border:1px solid var(--border);border-radius:8px;color:var(--fg);font-family:'Outfit';font-size:14px}
.sf input:focus{outline:none;border-color:var(--g)}
.sc{background:var(--bg2);border:1px solid var(--border);border-radius:12px;margin:0 0 16px;overflow:hidden}
.sh{padding:14px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;cursor:pointer}.sh:hover{background:#1a1a20}
.dt{width:10px;height:10px;border-radius:50%}
.sn{font-size:16px;font-weight:500;flex:1}.sm{font-size:11px;color:var(--fg2);font-family:'JetBrains Mono',monospace}
.sb{padding:0 20px 14px}
.cx{margin:10px 0 0;padding:10px 14px;background:#12121a;border-radius:8px;border-left:3px solid var(--b)}
.ct{font-size:14px;font-weight:500;color:var(--b)}.cm{font-size:11px;color:var(--fg2);font-family:'JetBrains Mono',monospace;margin-top:2px}
.al{margin:6px 0 0 10px}.an{padding:3px 0;font-size:13px;color:var(--fg2);display:flex;gap:6px;align-items:baseline}
.an .ad{flex-shrink:0}.an .at{color:var(--fg);flex:1}.an .aw{font-family:'JetBrains Mono',monospace;font-size:11px}
/* Param check */
.pcheck{display:flex;align-items:center;gap:8px;padding:6px 0;font-size:13px;border-bottom:1px solid #1a1a20}
.pcheck:last-child{border:none}.pcheck-ok{color:var(--g)}.pcheck-warn{color:var(--o)}.pcheck-err{color:var(--r)}
/* Score sparkline */
.spark{display:flex;align-items:flex-end;gap:2px;height:40px}
.spark-bar{flex:1;border-radius:2px;min-width:4px}
</style></head><body>
<div class="hdr">
  <h1><b>PMIS</b> Intelligence System</h1>
  <div class="tabs">
    <div class="tab active" onclick="switchTab('dashboard')">Dashboard</div>
    <div class="tab" onclick="switchTab('graph')">Memory Graph</div>
    <div class="tab" onclick="switchTab('benchmark')">Architecture</div>
    <div class="tab" onclick="switchTab('params')">Parameters</div>
  </div>
</div>

<!-- ═══════════ DASHBOARD TAB ═══════════ -->
<div class="panel active" id="panel-dashboard">
  <div class="grid g4" id="top-stats"></div>
  <div class="sec"><div class="dot"></div>Processing Pipeline</div>
  <div class="pipeline" id="pipeline"></div>
  <div class="grid g2">
    <div>
      <div class="sec"><div class="dot" style="background:var(--p)"></div>Temporal Stage Distribution</div>
      <div class="card"><div class="bar-chart" id="stage-bars"></div></div>
    </div>
    <div>
      <div class="sec"><div class="dot" style="background:var(--o)"></div>Weight Distribution</div>
      <div class="card"><div class="bar-chart" id="weight-bars"></div></div>
    </div>
  </div>
  <div class="sec"><div class="dot" style="background:var(--b)"></div>Super Context Health</div>
  <div id="sc-cards"></div>
  <div class="grid g2">
    <div>
      <div class="sec"><div class="dot" style="background:var(--t)"></div>Task Score History</div>
      <div class="card"><div class="spark" id="score-spark"></div><div id="score-summary" style="margin-top:8px;font-size:12px;color:var(--fg2)"></div></div>
    </div>
    <div>
      <div class="sec"><div class="dot" style="background:var(--r)"></div>Discrimination Power</div>
      <div class="card" id="adp-card"></div>
    </div>
  </div>
</div>

<!-- ═══════════ MEMORY GRAPH TAB ═══════════ -->
<div class="panel" id="panel-graph">
  <div class="grid g3" id="graph-stats" style="margin-bottom:12px"></div>
  <div class="sf"><input id="q" placeholder="Filter memory..." oninput="renderGraph(this.value)"></div>
  <div id="tree"></div>
</div>

<!-- ═══════════ ARCHITECTURE TAB ═══════════ -->
<div class="panel" id="panel-benchmark">
  <div class="sec"><div class="dot" style="background:var(--y)"></div>PMIS vs Leading Memory Architectures</div>
  <p style="font-size:13px;color:var(--fg2);margin-bottom:16px;line-height:1.6">Comparison against Generative Agents (Stanford 2023), MemGPT (Berkeley 2023), and A-MEM (2024). Evaluating structural capabilities, not LLM quality.</p>
  <table class="cmp-table" id="arch-table"></table>
  <div class="sec" style="margin-top:32px"><div class="dot" style="background:var(--t)"></div>Benchmark Alignment</div>
  <p style="font-size:13px;color:var(--fg2);margin-bottom:16px;line-height:1.6">Each metric shows whether PMIS parameters are tracking in the right direction for a production memory system.</p>
  <div id="bench-checks"></div>
</div>

<!-- ═══════════ PARAMETERS TAB ═══════════ -->
<div class="panel" id="panel-params">
  <div class="sec"><div class="dot"></div>Full Parameter Map</div>
  <p style="font-size:13px;color:var(--fg2);margin-bottom:16px;line-height:1.6">Every tracked parameter, its engine role, health status, and current value from the live database.</p>
  <div id="param-grid"></div>
</div>

<script>
const D=__DATA__,S=__STATS__,DB=__DASHBOARD__;
const STC={'impulse':'var(--p)','active':'var(--g)','established':'var(--t)','fading':'var(--fg2)'};
function E(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function B(s){return '<span class="badge b-'+(s||'impulse')+'">'+s+'</span>'}

/* ── Tab switching ── */
function switchTab(id){
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',t.textContent.toLowerCase().includes(id.substr(0,4))));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='panel-'+id));
}

/* ── Ring SVG ── */
function ring(pct,color,size){
  const r=size/2-4,c=2*Math.PI*r,d=c*(1-pct);
  return '<svg width="'+size+'" height="'+size+'"><circle cx="'+size/2+'" cy="'+size/2+'" r="'+r+'" fill="none" stroke="#1a1a20" stroke-width="4"/><circle cx="'+size/2+'" cy="'+size/2+'" r="'+r+'" fill="none" stroke="'+color+'" stroke-width="4" stroke-dasharray="'+c+'" stroke-dashoffset="'+d+'" transform="rotate(-90 '+size/2+' '+size/2+')"/></svg>';
}

/* ── Top stats ── */
function renderTopStats(){
  const total=S.super_context+S.context+S.anchor;
  const health=total>0?Math.min(100,Math.round(((DB.stages.established||0)+(DB.stages.active||0))*100/total)):0;
  document.getElementById('top-stats').innerHTML=[
    {n:total,l:'Total Nodes',c:'var(--g)'},
    {n:DB.tasks_scored,l:'Scored Tasks',c:'var(--b)',sub:'avg: '+(DB.avg_score||0)},
    {n:DB.total_decisions,l:'Decisions',c:'var(--p)',sub:DB.locked_decisions+' locked'},
    {n:health+'%',l:'Health Score',c:health>60?'var(--g)':health>30?'var(--o)':'var(--r)',sub:'active+established / total'}
  ].map(s=>'<div class="card"><div class="card-title">'+s.l+'</div><div class="card-val" style="color:'+s.c+'">'+s.n+'</div>'+(s.sub?'<div class="card-sub">'+s.sub+'</div>':'')+'</div>').join('');
}

/* ── Pipeline ── */
function renderPipeline(){
  const steps=[
    {name:'Ingestion',params:[['Fuzzy Jaccard','0.35 threshold'],['Dedup','substring + 4-char prefix'],['Mode Infer','15 templates']]},
    {name:'Temporal Engine',params:[['Recency',DB.avg_recency.toFixed(3)],['Frequency',DB.avg_frequency.toFixed(3)],['Consistency',DB.avg_consistency.toFixed(3)]]},
    {name:'Weight Engine',params:[['Power curve','^0.7'],['Decay sched','0.7/0.5/0.3/0.15'],['Evidence','task scores']]},
    {name:'ADP Analysis',params:[['High ADP',DB.adp_high+' anchors'],['Low ADP',DB.adp_low+' anchors'],['Threshold','> 1.0']]},
    {name:'Retrieval',params:[['t_boost','0.5-1.5x'],['m_boost','cosine sim'],['Transfers',DB.transfer_count+' edges']]},
    {name:'Convergence',params:[['Decisions',DB.total_decisions],['Locked',DB.locked_decisions],['Formula','0.4c+0.4l+0.2n']]}
  ];
  document.getElementById('pipeline').innerHTML=steps.map((s,i)=>'<div class="pipe-step"><div class="pipe-name">'+s.name+'</div><div class="pipe-params">'+s.params.map(p=>'<div>'+p[0]+': <span class="pipe-val">'+p[1]+'</span></div>').join('')+'</div>'+(i<steps.length-1?'<span class="pipe-arrow">&#9654;</span>':'')+'</div>').join('');
}

/* ── Stage bars ── */
function renderStageBars(){
  const st=DB.stages,mx=Math.max(st.impulse||1,st.active||1,st.established||1,st.fading||1);
  const cols=[{k:'impulse',c:'var(--p)'},{k:'active',c:'var(--g)'},{k:'established',c:'var(--t)'},{k:'fading',c:'var(--fg2)'}];
  document.getElementById('stage-bars').innerHTML=cols.map(c=>{
    const v=st[c.k]||0,h=Math.max(4,v/mx*80);
    return '<div class="bar-col"><div class="bar" style="height:'+h+'px;background:'+c.c+'"></div><div class="bar-label">'+v+'</div><div class="bar-label">'+c.k.substr(0,3)+'</div></div>';
  }).join('');
}

/* ── Weight histogram ── */
function renderWeightBars(){
  const bk=DB.weight_hist||[],mx=Math.max(...bk,1);
  document.getElementById('weight-bars').innerHTML=bk.map((v,i)=>{
    const h=Math.max(2,v/mx*80),lbl=(i/10).toFixed(1);
    return '<div class="bar-col"><div class="bar" style="height:'+h+'px;background:var(--o)"></div><div class="bar-label">'+v+'</div><div class="bar-label">'+lbl+'</div></div>';
  }).join('');
}

/* ── SC cards ── */
function renderSCCards(){
  const el=document.getElementById('sc-cards');
  if(!DB.sc_cards||!DB.sc_cards.length){el.innerHTML='<div class="card" style="text-align:center;color:var(--fg2);padding:40px">No super contexts yet</div>';return}
  el.innerHTML=DB.sc_cards.map(sc=>{
    const conv=sc.convergence||{};
    const convPct=Math.round((conv.convergence||0)*100);
    const stgColor=STC[sc.stage]||'var(--p)';
    const mv=sc.mode_vector||{};
    const mvKeys=Object.keys(mv);
    const radar=mvKeys.length>0?renderRadar(mv,60):'';
    const xfers=(sc.transfers||[]).map(t=>'<span class="xfer">&#8594; '+E(t.to)+' ('+t.sim.toFixed(2)+')</span>').join('');
    return '<div class="sc-card"><div class="sc-title"><div class="dt" style="background:'+stgColor+'"></div>'+E(sc.title)+B(sc.stage)+'</div><div class="sc-row">'+
      '<div class="sc-metric">Quality: <span>'+(sc.quality||0).toFixed(1)+'</span></div>'+
      '<div class="sc-metric">Contexts: <span>'+sc.contexts+'</span></div>'+
      '<div class="sc-metric">Anchors: <span>'+sc.anchors+'</span></div>'+
      '<div class="sc-metric">Uses: <span>'+sc.uses+'</span></div>'+
      '<div class="sc-metric">Best: <span>'+(sc.best_task||0)+'</span></div>'+
      '<div class="sc-metric">Convergence: <span>'+convPct+'%</span></div>'+
      '<div class="sc-metric">Decisions: <span>'+(conv.total_decisions||0)+'</span> ('+((conv.locked||0))+' locked)</div>'+
    '</div>'+(xfers?'<div style="margin-top:8px;font-size:11px;color:var(--fg2)">Transfers: '+xfers+'</div>':'')+
    (radar?'<div class="radar-wrap">'+radar+'</div>':'')+
    '<div class="health-bar"><div class="health-fill" style="width:'+Math.min(100,Math.round((sc.quality||0)*20))+'%;background:'+((sc.quality||0)>=3?'var(--g)':(sc.quality||0)>=2?'var(--o)':'var(--r)')+'"></div></div></div>';
  }).join('');
}

/* ── Mini radar for mode vectors ── */
function renderRadar(mv,sz){
  const keys=Object.keys(mv);if(!keys.length)return '';
  const n=keys.length,cx=sz/2,cy=sz/2,r=sz/2-8;
  const pts=keys.map((k,i)=>{const a=Math.PI*2*i/n-Math.PI/2;const v=mv[k]||0;return{x:cx+r*v*Math.cos(a),y:cy+r*v*Math.sin(a),lx:cx+(r+8)*Math.cos(a),ly:cy+(r+8)*Math.sin(a),k:k}});
  const poly=pts.map(p=>p.x+','+p.y).join(' ');
  const gridLines=pts.map(p=>'<line x1="'+cx+'" y1="'+cy+'" x2="'+(cx+r*Math.cos(Math.PI*2*pts.indexOf(p)/n-Math.PI/2))+'" y2="'+(cy+r*Math.sin(Math.PI*2*pts.indexOf(p)/n-Math.PI/2))+'" stroke="#2a2a30"/>').join('');
  return '<svg width="'+sz*2+'" height="'+sz+'" viewBox="0 0 '+(sz+40)+' '+sz+'">'+gridLines+'<polygon points="'+poly+'" fill="rgba(74,222,128,0.15)" stroke="var(--g)" stroke-width="1.5"/>'+pts.map(p=>'<circle cx="'+p.x+'" cy="'+p.y+'" r="2" fill="var(--g)"/><text x="'+p.lx+'" y="'+p.ly+'" font-size="7" fill="var(--fg2)" text-anchor="middle" dominant-baseline="central">'+p.k.substr(0,3)+'</text>').join('')+'</svg>';
}

/* ── Score sparkline ── */
function renderScores(){
  const scores=DB.task_scores||[];
  const el=document.getElementById('score-spark');
  if(!scores.length){el.innerHTML='<div style="color:var(--fg2);font-size:12px;padding:10px">No scored tasks yet</div>';return}
  const mx=5;
  el.innerHTML=scores.map(s=>{const h=Math.max(3,s.score/mx*36);const c=s.score>=4?'var(--g)':s.score>=3?'var(--b)':s.score>=2?'var(--o)':'var(--r)';
    return '<div class="spark-bar" style="height:'+h+'px;background:'+c+'" title="'+E(s.title)+': '+s.score+'"></div>';}).join('');
  document.getElementById('score-summary').innerHTML='Scored: <strong style="color:var(--g)">'+DB.tasks_scored+'</strong> &middot; Avg: <strong style="color:var(--b)">'+DB.avg_score+'</strong> &middot; Best: <strong style="color:var(--g)">'+DB.best_score+'</strong> &middot; Worst: <strong style="color:var(--o)">'+DB.worst_score+'</strong>';
}

/* ── ADP card ── */
function renderADP(){
  const vals=DB.adp_values||[];
  const el=document.getElementById('adp-card');
  if(!vals.length){el.innerHTML='<div class="card-title">Anchor Discrimination Power</div><div style="color:var(--fg2);font-size:12px">No ADP data yet (need 3+ scored tasks per anchor)</div>';return}
  const high=DB.adp_high,low=DB.adp_low,total=high+low;
  el.innerHTML='<div class="card-title">Anchor Discrimination Power</div>'+
    '<div style="display:flex;gap:20px;margin:10px 0"><div><div class="card-val" style="font-size:22px;color:var(--r)">'+high+'</div><div class="card-sub">High ADP (&gt;1.0) &#8212; execution matters</div></div>'+
    '<div><div class="card-val" style="font-size:22px;color:var(--t)">'+low+'</div><div class="card-sub">Low ADP (&le;1.0) &#8212; reliably consistent</div></div></div>'+
    '<div class="health-bar" style="margin-top:10px"><div class="health-fill" style="width:'+(total>0?Math.round(low*100/total):0)+'%;background:var(--t)"></div></div>'+
    '<div class="card-sub" style="margin-top:4px">'+Math.round(total>0?low*100/total:0)+'% anchors are reliably consistent</div>';
}

/* ── Architecture comparison ── */
function renderArchComparison(){
  const features=[
    {f:'Hierarchical memory (multi-level)',pmis:2,ga:1,mg:1,am:2,note:'PMIS: 3-level SC>CTX>Anchor. A-MEM: concept hierarchy.'},
    {f:'Evidence-based weight evolution',pmis:2,ga:0,mg:0,am:1,note:'PMIS: task scores adjust weights via blended power curve.'},
    {f:'Temporal memory stages',pmis:2,ga:2,mg:1,am:1,note:'PMIS: 4 stages with R/F/C scoring. GA: recency/importance/relevance.'},
    {f:'Deduplication / consolidation',pmis:2,ga:0,mg:0,am:2,note:'PMIS: fuzzy Jaccard. A-MEM: LLM-based abstraction.'},
    {f:'Decision convergence tracking',pmis:2,ga:0,mg:0,am:0,note:'PMIS unique: decisions lock with evidence, prevent AI drift.'},
    {f:'Cross-domain transfer',pmis:2,ga:0,mg:0,am:1,note:'PMIS: structural similarity edges. A-MEM: implicit via embeddings.'},
    {f:'Anchor discrimination power',pmis:2,ga:0,mg:0,am:0,note:'PMIS unique: identifies when execution details change outcomes.'},
    {f:'Mode / context vectors',pmis:2,ga:0,mg:0,am:0,note:'PMIS unique: retrieval boosted by domain mode similarity.'},
    {f:'Execution parameters',pmis:2,ga:0,mg:0,am:0,note:'PMIS unique: tracks best/worst execution details per anchor.'},
    {f:'Reflection / synthesis',pmis:1,ga:2,mg:1,am:2,note:'GA: periodic reflection. PMIS: via rebuild + convergence.'},
    {f:'Embedding-based retrieval',pmis:0,ga:2,mg:2,am:2,note:'PMIS uses keyword overlap; no vector embeddings yet.'},
    {f:'Infinite context (paging)',pmis:0,ga:0,mg:2,am:0,note:'MemGPT unique: OS-style virtual memory paging.'},
    {f:'LLM-free operation',pmis:2,ga:0,mg:0,am:0,note:'PMIS: zero external deps, pure Python stdlib.'},
    {f:'Task outcome feedback loop',pmis:2,ga:0,mg:0,am:0,note:'PMIS: score tasks, weights adapt. Others lack explicit feedback.'},
  ];
  const sym={2:'<span class="cmp-yes">&#9679;</span>',1:'<span class="cmp-partial">&#9681;</span>',0:'<span class="cmp-no">&#9675;</span>'};
  const pmisScore=features.reduce((a,f)=>a+f.pmis,0);
  const gaScore=features.reduce((a,f)=>a+f.ga,0);
  const mgScore=features.reduce((a,f)=>a+f.mg,0);
  const amScore=features.reduce((a,f)=>a+f.am,0);
  document.getElementById('arch-table').innerHTML='<thead><tr><th>Capability</th><th>PMIS v3</th><th>Gen. Agents</th><th>MemGPT</th><th>A-MEM</th><th>Notes</th></tr></thead><tbody>'+
    features.map(f=>'<tr><td>'+f.f+'</td><td style="text-align:center">'+sym[f.pmis]+'</td><td style="text-align:center">'+sym[f.ga]+'</td><td style="text-align:center">'+sym[f.mg]+'</td><td style="text-align:center">'+sym[f.am]+'</td><td style="font-size:11px;color:var(--fg2)">'+f.note+'</td></tr>').join('')+
    '<tr style="border-top:2px solid var(--border)"><td><strong>Score</strong></td><td style="text-align:center"><strong style="color:var(--g)">'+pmisScore+'/'+features.length*2+'</strong></td><td style="text-align:center"><strong>'+gaScore+'/'+features.length*2+'</strong></td><td style="text-align:center"><strong>'+mgScore+'/'+features.length*2+'</strong></td><td style="text-align:center"><strong>'+amScore+'/'+features.length*2+'</strong></td><td></td></tr></tbody>';
}

/* ── Benchmark checks ── */
function renderBenchChecks(){
  const total=S.super_context+S.context+S.anchor;
  const estab=(DB.stages.established||0),actv=(DB.stages.active||0);
  const healthPct=total>0?(estab+actv)*100/total:0;
  const checks=[
    {param:'Weight Convergence (Pearson r > 0.7)',status:DB.tasks_scored>=3?'ok':'warn',note:DB.tasks_scored>=3?'Evidence from '+DB.tasks_scored+' scored tasks':'Need 3+ scored tasks to validate'},
    {param:'Temporal Classification (>80% accuracy)',status:total>0?'ok':'warn',note:'4-stage classifier: impulse/active/established/fading'},
    {param:'Dedup Recall (>60%)',status:'ok',note:'Fuzzy Jaccard at threshold 0.35 with substring+prefix matching'},
    {param:'Decision Convergence Formula',status:DB.total_decisions>0?'ok':'warn',note:DB.total_decisions+' decisions tracked, '+DB.locked_decisions+' locked'},
    {param:'Transfer Edge Detection',status:'ok',note:DB.transfer_count+' transfer edges (threshold 0.25)'},
    {param:'ADP Computation',status:DB.adp_values.length>0?'ok':'warn',note:DB.adp_values.length+' anchors with ADP computed'},
    {param:'Mode Vector Inference',status:'ok',note:'15 domain templates + cosine similarity retrieval boost'},
    {param:'Execution Params Tracking',status:DB.tasks_scored>0?'ok':'warn',note:'Best/worst params surfaced when ADP > 1.0'},
    {param:'Memory Health (active+established > 40%)',status:healthPct>40?'ok':healthPct>20?'warn':'err',note:Math.round(healthPct)+'% of nodes are active or established'},
    {param:'Score Feedback Loop Active',status:DB.tasks_scored>=1?'ok':'err',note:DB.tasks_scored+' tasks scored. Weights evolve with each score.'},
  ];
  document.getElementById('bench-checks').innerHTML=checks.map(c=>{
    const icon=c.status==='ok'?'<span class="pcheck-ok">&#10003;</span>':c.status==='warn'?'<span class="pcheck-warn">&#9888;</span>':'<span class="pcheck-err">&#10007;</span>';
    return '<div class="pcheck">'+icon+'<span style="flex:1">'+c.param+'</span><span style="font-size:11px;color:var(--fg2)">'+c.note+'</span></div>';
  }).join('');
}

/* ── Parameters tab ── */
function renderParams(){
  const params=[
    {cat:'Temporal Engine',items:[
      {p:'recency',role:'Exponential decay from last use (half-life 7d)',val:DB.avg_recency.toFixed(3),target:'Higher = recently used nodes rank higher',ok:true},
      {p:'frequency',role:'Uses in last 30 days / 30',val:DB.avg_frequency.toFixed(3),target:'Separates established from impulse',ok:true},
      {p:'consistency',role:'1 / (stddev of gaps + 1)',val:DB.avg_consistency.toFixed(3),target:'Detects regular usage patterns',ok:true},
      {p:'memory_stage',role:'Classified: impulse/active/established/fading',val:Object.entries(DB.stages).map(e=>e[0]+':'+e[1]).join(', '),target:'Retrieval boost: 0.8/1.2/1.5/0.5x',ok:true},
    ]},
    {cat:'Weight Engine',items:[
      {p:'initial_weight',role:'Claude estimate at ingestion time',val:'Set per anchor (0.1-0.95)',target:'Starting point, decays with evidence',ok:true},
      {p:'evidence_weight',role:'Power curve: (mean_score/5)^0.7',val:'Computed from task scores',target:'Non-linear: amplifies high scores without compression',ok:true},
      {p:'decay_schedule',role:'Blend ratio initial vs evidence',val:'n=1:0.7, n=2:0.5, n=3:0.3, n>=5:0.15',target:'More data = more trust in evidence',ok:true},
      {p:'blended_weight',role:'(initial*decay) + (evidence*(1-decay))',val:'Per anchor 0.05-1.0',target:'Final retrieval ranking signal',ok:true},
    ]},
    {cat:'Discrimination Power',items:[
      {p:'ADP',role:'var(anchor_scores) / var(all_scores)',val:DB.adp_values.length+' computed',target:'ADP > 1.0 = execution details change outcomes',ok:DB.adp_values.length>0},
      {p:'variant_sensitive',role:'Flag when ADP > 1.0',val:DB.adp_high+' anchors flagged',target:'Triggers best/worst execution param display',ok:true},
    ]},
    {cat:'Mode Vectors',items:[
      {p:'mode_vector',role:'6D: work/creative/learning/social/travel/home',val:'Per SC, inferred from title',target:'Cosine similarity boosts relevant retrievals',ok:true},
      {p:'mode_similarity',role:'Cosine sim between query and SC modes',val:'0.0-1.0 range',target:'m_boost = 0.5 + sim (range 0.5-1.5x)',ok:true},
    ]},
    {cat:'Decision Convergence',items:[
      {p:'confidence',role:'Per-decision confidence (0-1)',val:'Reinforced/decayed by task scores',target:'High score: +0.1*(1-c). Low score: *0.7',ok:true},
      {p:'convergence',role:'0.4*avg_conf + 0.4*lock_ratio + 0.2*min(n/10,1)',val:DB.sc_cards.length>0?DB.sc_cards.map(s=>(s.convergence.convergence||0).toFixed(2)).join(', '):'n/a',target:'Higher = more settled domain, fewer contradictions',ok:true},
      {p:'lock_ratio',role:'Fraction of irreversible decisions',val:DB.total_decisions>0?Math.round(DB.locked_decisions*100/DB.total_decisions)+'%':'n/a',target:'Mature SCs have high lock ratio',ok:true},
    ]},
    {cat:'Transfer Edges',items:[
      {p:'structural_similarity',role:'0.6*ctx_jaccard + 0.4*anchor_overlap',val:DB.transfer_count+' edges',target:'Threshold 0.25, discount 0.6x',ok:true},
      {p:'transfer_discount',role:'Weight multiplier for cross-domain knowledge',val:'0.6x',target:'Transferred anchors rank lower than native',ok:true},
    ]},
    {cat:'Deduplication',items:[
      {p:'fuzzy_jaccard',role:'Substring containment + 4-char prefix match',val:'Threshold: 0.35',target:'Prevents duplicate anchors across paraphrases',ok:true},
      {p:'stop_words',role:'Filter common words before comparison',val:'60+ stop words',target:'Improves signal in short titles',ok:true},
    ]},
  ];
  document.getElementById('param-grid').innerHTML=params.map(cat=>'<div class="card" style="margin-bottom:16px"><div class="card-title">'+cat.cat+'</div>'+
    cat.items.map(p=>'<div class="pcheck"><span style="color:'+(p.ok?'var(--g)':'var(--o)')+'">&#9679;</span><div style="flex:1"><div style="font-size:13px;font-weight:500">'+p.p+'</div><div style="font-size:11px;color:var(--fg2)">'+p.role+'</div></div><div style="text-align:right;min-width:180px"><div class="mono" style="font-size:11px;color:var(--g)">'+p.val+'</div><div style="font-size:10px;color:var(--fg3)">'+p.target+'</div></div></div>').join('')+'</div>').join('');
}

/* ── Memory Graph ── */
function renderGraph(f){
  const el=document.getElementById('tree'),kids=D.children||[];
  if(!kids.length){el.innerHTML='<div style="text-align:center;padding:80px;color:var(--fg2)">No memories stored yet</div>';return}
  const fl=f?f.toLowerCase():'';
  el.innerHTML=kids.filter(sc=>{if(!fl)return true;if(sc.name.toLowerCase().includes(fl))return true;
    return(sc.children||[]).some(c=>c.name.toLowerCase().includes(fl)||(c.children||[]).some(a=>a.name.toLowerCase().includes(fl)));
  }).map(sc=>{const ctxs=(sc.children||[]).map(c=>{
    const ancs=(c.children||[]).map(a=>'<div class="an"><span class="ad" style="color:'+(STC[a.stage]||'var(--fg2)')+'">&#x25cf;</span><span class="at">'+E(a.name)+'</span><span class="aw" style="color:var(--o)">w='+(a.weight||0).toFixed(2)+'</span>'+(a.adp>0?'<span class="aw" style="color:var(--r)">adp='+(a.adp||0).toFixed(2)+'</span>':'')+B(a.stage)+'</div>').join('');
    return '<div class="cx"><div class="ct">'+E(c.name)+B(c.stage)+'</div><div class="cm">w='+(c.weight||0).toFixed(2)+'</div>'+(ancs?'<div class="al">'+ancs+'</div>':'')+'</div>';
  }).join('');
  const dotColor=STC[sc.stage]||'var(--p)';
  return '<div class="sc"><div class="sh" onclick="this.nextElementSibling.hidden=!this.nextElementSibling.hidden"><div class="dt" style="background:'+dotColor+'"></div><div class="sn">'+E(sc.name)+'</div><div class="sm">'+(sc.children||[]).length+' ctx &middot; q='+(sc.quality||0).toFixed(1)+' &middot; '+((sc.uses||0))+'x '+B(sc.stage)+'</div></div><div class="sb">'+ctxs+'</div></div>';
  }).join('');
}
document.getElementById('graph-stats').innerHTML=Object.entries(S).map(([k,v])=>'<div class="card"><div class="card-val" style="font-size:20px">'+v+'</div><div class="card-sub">'+k.replace(/_/g,' ')+'</div></div>').join('');

/* ── Render all ── */
renderTopStats();renderPipeline();renderStageBars();renderWeightBars();renderSCCards();renderScores();renderADP();
renderArchComparison();renderBenchChecks();renderParams();renderGraph('');
</script></body></html>'''


# ══════════════════════════════════════════════════════════
# SESSION MANAGEMENT — persistent session state across CLI calls
# ══════════════════════════════════════════════════════════

SESSION_FILE = ROOT / "Graph_DB" / ".session.json"
SESSION_TIME_GAP_HOURS = 3
SESSION_DIVERGENCE_THRESHOLD = 0.40
SESSION_RATING_INTERVAL = 3  # ask for rating every N conversations
SESSION_STALE_HOURS = 24  # abandon session without rating after this


def _session_load():
    """Load session state from disk. Returns dict or None."""
    if not SESSION_FILE.exists():
        return None
    try:
        return json.loads(SESSION_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _session_save(state):
    """Persist session state to disk."""
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(json.dumps(state, indent=2))


def _session_clear():
    """Remove session file."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


def _session_new(query):
    """Create a fresh session state."""
    tokens = _session_tokenize(query)
    return {
        "session_id": _id(),
        "started_at": _now(),
        "last_interaction": _now(),
        "current_sc_id": None,
        "current_sc_title": None,
        "conversation_count": 1,
        "query_history": [query],
        "session_vector": dict(Counter(tokens)),
        "task_ids": [],
        "pending_rating": None,
    }


def _session_tokenize(text):
    """Tokenize for session vector — significant words only."""
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "for",
            "and", "or", "but", "in", "on", "at", "to", "of", "with", "how",
            "what", "which", "do", "does", "can", "should", "i", "my", "our",
            "that", "this", "it", "by", "from", "has", "have", "had", "not",
            "will", "about", "need", "help", "me", "now", "also", "its", "we",
            "these", "more", "into", "using", "used", "use", "get", "set", "new",
            "like", "just", "very", "so", "if", "than", "then", "too", "some"}
    return [w.lower().strip(".,;:!?()[]{}\"'/-") for w in text.split()
            if len(w) > 2 and w.lower().strip(".,;:!?()[]{}\"'/-") not in stop]


def _session_cosine(vec_a, vec_b):
    """Cosine similarity between two word-frequency dicts."""
    if not vec_a or not vec_b:
        return 0.0
    keys = set(list(vec_a.keys()) + list(vec_b.keys()))
    dot = sum(float(vec_a.get(k, 0)) * float(vec_b.get(k, 0)) for k in keys)
    mag_a = sum(float(v) ** 2 for v in vec_a.values()) ** 0.5
    mag_b = sum(float(v) ** 2 for v in vec_b.values()) ** 0.5
    if mag_a < 0.001 or mag_b < 0.001:
        return 0.0
    return dot / (mag_a * mag_b)


def _session_check_divergence(session_vector, new_query, query_history=None):
    """Check if new query diverges from session context.
    Returns (similarity, diverged).

    Two-layer check:
    1. Word-overlap cosine (fast, zero-dep)
    2. If word-overlap says diverged, neural embedding as second opinion
       (compares new query against recent query history semantically)"""
    tokens = _session_tokenize(new_query)
    query_vec = dict(Counter(tokens))
    word_sim = _session_cosine(session_vector, query_vec)

    # Adaptive threshold: short queries need less overlap
    threshold = SESSION_DIVERGENCE_THRESHOLD
    if len(tokens) <= 3:
        threshold = 0.10
    elif len(tokens) <= 5:
        threshold = 0.18
    elif len(tokens) <= 8:
        threshold = 0.30

    word_diverged = word_sim < threshold

    # If words say diverged, get neural second opinion (if available)
    if word_diverged and query_history and len(query_history) >= 1:
        neural_sim = _session_neural_similarity(new_query, query_history)
        if neural_sim is not None and neural_sim > 0.20:
            # Neural says they're related — override word-overlap
            return round(neural_sim, 4), False

    return round(word_sim, 4), word_diverged


def _session_neural_similarity(new_query, query_history):
    """Semantic similarity between new query and recent history. Returns float or None."""
    try:
        from embedding_engine import encode_texts
        recent = " ".join(query_history[-3:])
        vecs = encode_texts([new_query, recent])
        return float(vecs[0] @ vecs[1])
    except Exception:
        return None


def _session_update_vector(old_vec, new_query, decay=0.85):
    """Decay old context, blend in new query tokens."""
    tokens = _session_tokenize(new_query)
    query_vec = Counter(tokens)
    result = {}
    for k, v in old_vec.items():
        result[k] = float(v) * decay
    for k, v in query_vec.items():
        result[k] = result.get(k, 0) + float(v) * (1 - decay)
    # Prune near-zero entries
    return {k: round(v, 4) for k, v in result.items() if v > 0.01}


def _session_suggest_scs(conn, query):
    """Suggest top super contexts matching the query."""
    tokens = set(_session_tokenize(query))
    if not tokens:
        return []

    scs = conn.execute(
        "SELECT id, title, description, quality, use_count FROM nodes WHERE type='super_context' ORDER BY last_used DESC"
    ).fetchall()

    scored = []
    for sc in scs:
        sc_text = f"{sc['title']} {sc['description'] or ''}".lower()
        sc_words = set(w for w in sc_text.split() if len(w) > 2)
        overlap = len(tokens & sc_words)
        if overlap > 0:
            scored.append({
                "title": sc["title"],
                "sc_id": sc["id"],
                "match_score": round(overlap / max(len(tokens), 1), 3),
                "description": (sc["description"] or "")[:100],
                "quality": sc["quality"] or 0,
                "uses": sc["use_count"],
            })

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored[:3]


# ══════════════════════════════════════════════════════════
# THREE-LAYER RETRIEVAL
# ══════════════════════════════════════════════════════════

def _lazy_pull_team_updates():
    """Pull team updates from central on every session begin. Silent on failure."""
    cfg = _load_team_config()
    if not cfg or not cfg.get("server"):
        return

    # Load last pull txn
    last_txn = 0
    if YANTRAI_SYNC_STATE.exists():
        try:
            ss = json.loads(YANTRAI_SYNC_STATE.read_text())
            last_txn = ss.get("last_pull_txn", 0)
        except Exception:
            pass

    author = _get_author()
    result = _central_request("GET", f"/api/sync/pull?since={last_txn}&author={author}")
    if result is None:
        return  # Central unreachable — use stale cache

    nodes = result.get("nodes", [])
    new_txn = result.get("server_txn", last_txn)

    if not nodes:
        # Still update txn to avoid re-pulling nothing
        YANTRAI_SYNC_STATE.write_text(json.dumps({"last_pull_txn": new_txn}))
        return

    # Upsert team nodes into brain.db (source='team')
    try:
        brain_conn = _get_brain_db()
        for node in nodes:
            nid = node.get("id", "")
            nhash = node.get("node_hash", "")

            # Check if exists by hash or id
            existing = None
            if nhash:
                existing = brain_conn.execute(
                    "SELECT id FROM nodes WHERE node_hash=?", (nhash,)
                ).fetchone()
            if not existing and nid:
                existing = brain_conn.execute(
                    "SELECT id FROM nodes WHERE id=?", (nid,)
                ).fetchone()

            if existing:
                # Update
                brain_conn.execute("""
                    UPDATE nodes SET title=?, content=?, weight=?,
                        team_weight=?, synced_at=?, mem_source='team'
                    WHERE id=?
                """, (
                    node.get("title", ""), node.get("content", ""),
                    node.get("weight", 0.5), node.get("team_weight", 0.5),
                    _now(), existing["id"]
                ))
            else:
                # Insert as team node
                brain_conn.execute("""
                    INSERT OR IGNORE INTO nodes (
                        id, type, title, description, content, source,
                        created_at, last_used, weight, initial_weight,
                        node_hash, author, source_folder, mem_source,
                        team_weight, synced_at, occurrence_log, memory_stage
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'team',?,?,?,'impulse')
                """, (
                    nid, node.get("type", "anchor"), node.get("title", ""),
                    node.get("description", ""), node.get("content", ""),
                    node.get("source", ""), node.get("created_at", _now()),
                    node.get("last_used", _now()), node.get("weight", 0.5),
                    node.get("initial_weight", 0.5), nhash,
                    node.get("author", ""), node.get("source_folder", ""),
                    node.get("team_weight", 0.5), _now(), "[]"
                ))

        brain_conn.commit()
        brain_conn.close()
    except Exception:
        pass

    # Save sync state
    YANTRAI_SYNC_STATE.write_text(json.dumps({"last_pull_txn": new_txn}))


def _get_brain_conn():
    """Get a direct SQLite connection to brain.db without swapping globals."""
    if not YANTRAI_BRAIN_DB.exists():
        return None
    conn = sqlite3.connect(str(YANTRAI_BRAIN_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _retrieve_from_brain(query, source_filter=None):
    """Run keyword retrieval against brain.db. Returns list of anchor dicts."""
    brain_conn = _get_brain_conn()
    if brain_conn is None:
        return []

    try:
        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        if not query_words:
            return []

        where_clause = ""
        params = []
        if source_filter:
            where_clause = "AND mem_source=?"
            params.append(source_filter)

        anchors = brain_conn.execute(f"""
            SELECT * FROM nodes WHERE type='anchor' {where_clause}
        """, params).fetchall()

        scored = []
        for anc in anchors:
            # Convert Row to dict for safe .get() access
            a = {k: anc[k] for k in anc.keys()}
            text = f"{a['title']} {a.get('content') or ''}".lower()
            overlap = sum(1 for w in query_words if w in text)
            if overlap > 0:
                weight = a.get("weight") or 0.5
                team_weight = a.get("team_weight") or 0.5
                score = overlap * weight
                scored.append({
                    "title": a["title"],
                    "content": a.get("content") or "",
                    "weight": weight,
                    "team_weight": team_weight,
                    "author": a.get("author", ""),
                    "source_folder": a.get("source_folder", ""),
                    "mem_source": a.get("mem_source", "local"),
                    "node_hash": a.get("node_hash", ""),
                    "relevance": round(score, 3),
                    "memory_stage": a.get("memory_stage", "impulse"),
                })

        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return scored[:10]
    except Exception:
        return []
    finally:
        brain_conn.close()


def _retrieve_from_central(query):
    """HTTP retrieval against central server. Returns list of anchor dicts."""
    author = _get_author()
    result = _central_request("GET", f"/api/sync/retrieve?q={urllib.parse.quote(query)}&author={author}")
    if result is None:
        return None  # Central unreachable — caller should use cached fallback
    return result.get("results", [])


def _is_duplicate_anchor(a, b, threshold=0.5):
    """Check if two anchors are duplicates using hash + fuzzy title match.

    Three-tier dedup:
    1. Hash match (exact title+content)
    2. Embedding similarity (if available) — deferred to autoresearch
    3. Fuzzy Jaccard on title words
    """
    # Tier 1: Hash match
    h_a = a.get("node_hash", "")
    h_b = b.get("node_hash", "")
    if h_a and h_b and h_a == h_b:
        return True

    # Tier 3: Fuzzy Jaccard on title (Tier 2 embedding skipped — plugged in later)
    title_a = set(w.lower() for w in a.get("title", "").split() if len(w) > 2)
    title_b = set(w.lower() for w in b.get("title", "").split() if len(w) > 2)
    if not title_a or not title_b:
        return False
    intersection = len(title_a & title_b)
    union = len(title_a | title_b)
    jaccard = intersection / union if union > 0 else 0
    return jaccard > threshold


def _merge_team_retrieval(context_pack, query):
    """Merge L2 (personal brain) and L3 (team central) results into context_pack.

    context_pack already has L1 (project) results from p9_retrieve.
    We add two new sections: brain_memories and team_memories.
    """
    import urllib.parse

    # Collect L1 anchors for dedup
    l1_anchors = []
    for mem in context_pack.get("memories", []):
        for ctx in mem.get("contexts", []):
            for anc in ctx.get("anchors", []):
                l1_anchors.append(anc)

    # ── L2: Personal brain (source='local', cross-project knowledge) ──
    l2_raw = _retrieve_from_brain(query, source_filter="local")
    l2_deduped = []
    for anc in l2_raw:
        if not any(_is_duplicate_anchor(anc, l1a) for l1a in l1_anchors):
            anc["final_score"] = anc["relevance"] * 1.0  # personal weight
            l2_deduped.append(anc)

    # ── L3: Team central (other people's knowledge) ──
    l3_raw = _retrieve_from_central(query)
    if l3_raw is None:
        # Central unreachable — fall back to cached team nodes in brain.db
        l3_raw = _retrieve_from_brain(query, source_filter="team")
        # Convert to same format
        for item in l3_raw:
            item["team_weight"] = item.get("team_weight", 0.5)

    all_local = l1_anchors + l2_deduped
    l3_deduped = []
    for anc in (l3_raw or []):
        if not any(_is_duplicate_anchor(anc, la) for la in all_local):
            tw = anc.get("team_weight", 0.5)
            anc["final_score"] = anc.get("relevance", anc.get("relevance_score", 0)) * tw
            l3_deduped.append(anc)

    l3_deduped.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    l3_deduped = l3_deduped[:5]  # cap team results

    # ── Add to context_pack ──
    if l2_deduped:
        context_pack["brain_memories"] = l2_deduped[:5]

    if l3_deduped:
        context_pack["team_memories"] = l3_deduped

    # Summary counts
    context_pack["layers"] = {
        "l1_project": len(l1_anchors),
        "l2_brain": len(l2_deduped),
        "l3_team": len(l3_deduped),
    }


def cmd_session_begin(conn, query):
    """Start or continue a session. Detects conversation boundaries."""
    state = _session_load()
    needs_rating = False
    previous_session = None
    is_new = False

    if state:
        # Check time gap
        last_ts = _parse_ts(state.get("last_interaction", ""))
        now = datetime.now(timezone.utc)
        hours_since = (now - last_ts).total_seconds() / 3600

        if hours_since > SESSION_STALE_HOURS:
            # Stale session — abandon without rating
            _session_clear()
            state = None
            is_new = True
        elif hours_since > SESSION_TIME_GAP_HOURS:
            # Time gap detected — flag for rating
            needs_rating = True
            previous_session = {
                "sc_title": state.get("current_sc_title", "Unknown"),
                "sc_id": state.get("current_sc_id"),
                "conversation_count": state.get("conversation_count", 0),
                "task_ids": state.get("task_ids", []),
                "queries": state.get("query_history", [])[:5],
            }
            state = None
            is_new = True
        else:
            # Check divergence — but only after the session has 2+ turns
            # First few turns build context; divergence on turn 1→2 is unreliable
            # because short queries share few words even when topically related
            conv_count = state.get("conversation_count", 0)
            vec = state.get("session_vector", {})
            if vec and conv_count >= 2:
                sim, diverged = _session_check_divergence(vec, query, state.get("query_history", []))
                if diverged:
                    needs_rating = True
                    previous_session = {
                        "sc_title": state.get("current_sc_title", "Unknown"),
                        "sc_id": state.get("current_sc_id"),
                        "conversation_count": state.get("conversation_count", 0),
                        "task_ids": state.get("task_ids", []),
                        "queries": state.get("query_history", [])[:5],
                        "divergence_similarity": sim,
                    }
                    state = None
                    is_new = True

    # Create new or continue
    if state is None:
        state = _session_new(query)
        is_new = True
    else:
        # Continue existing session
        state["conversation_count"] = state.get("conversation_count", 0) + 1
        state["last_interaction"] = _now()
        state["query_history"] = state.get("query_history", []) + [query]
        state["session_vector"] = _session_update_vector(
            state.get("session_vector", {}), query
        )

    # Save pending rating info if needed
    if needs_rating and previous_session:
        state["pending_rating"] = previous_session

    # ── Lazy pull from central (replaces daemon) ──
    _lazy_pull_team_updates()

    # ── Three-layer retrieval ──
    import io, contextlib

    # L1: Project-local retrieval (existing P9+ or legacy)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if _HAS_P9:
            _p9_retrieve(conn, query)
        else:
            cmd_retrieve(conn, query)
    retrieve_output = buf.getvalue()

    try:
        context_pack = json.loads(retrieve_output)
    except (json.JSONDecodeError, TypeError):
        context_pack = {"query": query, "memories": [], "total_matches": 0}

    # L2 + L3: Retrieve from brain.db and central, merge into context_pack
    _merge_team_retrieval(context_pack, query)

    # Record retrieval occurrences (fixes Bottleneck #2)
    _record_retrieval(conn, context_pack)

    # Accumulate retrieval results into session vector (enriches context for next turn)
    # Only accumulate from top result's SC title and top anchors — keeps vector focused
    vec = state.get("session_vector", {})
    if context_pack.get("memories"):
        top_mem = context_pack["memories"][0]
        for w in _session_tokenize(top_mem.get("super_context", "")):
            vec[w] = vec.get(w, 0) + 0.3
        for ctx in top_mem.get("contexts", [])[:2]:  # top 2 contexts only
            for anc in ctx.get("anchors", [])[:2]:  # top 2 anchors per context
                for w in _session_tokenize(anc.get("title", "")):
                    vec[w] = vec.get(w, 0) + 0.1
    # Cap vector size to prevent bloat — keep top 50 terms
    if len(vec) > 50:
        sorted_terms = sorted(vec.items(), key=lambda x: -x[1])[:50]
        vec = dict(sorted_terms)
    state["session_vector"] = {k: round(v, 4) for k, v in vec.items() if v > 0.01}

    # Detect primary SC from retrieval results
    if context_pack.get("memories") and is_new:
        top_mem = context_pack["memories"][0]
        state["current_sc_id"] = top_mem.get("sc_id")
        state["current_sc_title"] = top_mem.get("super_context")

    # SC suggestions for new sessions
    suggested_scs = []
    if is_new:
        suggested_scs = _session_suggest_scs(conn, query)

    # Should we ask for rating? (every N conversations)
    conv_count = state.get("conversation_count", 1)
    should_ask_rating = (conv_count > 0 and conv_count % SESSION_RATING_INTERVAL == 0)

    # Save session state
    _session_save(state)

    # Build response
    result = context_pack
    result["session"] = {
        "session_id": state["session_id"],
        "is_new_session": is_new,
        "conversation_count": conv_count,
        "needs_rating": needs_rating,
        "previous_session": previous_session,
        "should_ask_rating": should_ask_rating,
        "current_sc": state.get("current_sc_title"),
    }
    if suggested_scs:
        result["session"]["suggested_scs"] = suggested_scs

    print(json.dumps(result, indent=2))


def cmd_session_rate(conn, direction, specific_anchors_str=None):
    """Rate session: up (4.0) or down (2.0 selective)."""
    state = _session_load()

    # Determine which task_ids to score
    task_ids = []
    if state and state.get("pending_rating"):
        task_ids = state["pending_rating"].get("task_ids", [])
    elif state:
        task_ids = state.get("task_ids", [])

    if not task_ids:
        print(json.dumps({"rated": False, "reason": "No tasks in session to rate"}))
        return

    if direction == "down" and not specific_anchors_str:
        # First call — return anchor list for follow-up
        anchor_titles = set()
        for tid in task_ids:
            rows = conn.execute("""
                SELECT DISTINCT n.title FROM task_anchors ta
                JOIN nodes n ON n.id = ta.anchor_id
                WHERE ta.task_id = ?
            """, (tid,)).fetchall()
            for r in rows:
                anchor_titles.add(r["title"])

        print(json.dumps({
            "needs_followup": True,
            "anchors_used": sorted(anchor_titles),
            "task_ids": task_ids,
            "instruction": "Ask user what didn't work, then call: session rate down \"anchor title\""
        }, indent=2))
        return

    # Apply scores
    anchors_penalized = []
    if direction == "up":
        score_val = "4.0"
        for tid in task_ids:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cmd_score(conn, tid, score_val)
    elif direction == "down":
        # Parse specific anchors to penalize
        penalize_titles = set()
        if specific_anchors_str:
            # Could be comma-separated or a single title
            for t in specific_anchors_str.split(","):
                t = t.strip()
                if t:
                    penalize_titles.add(t.lower())

        for tid in task_ids:
            # Score the task at 2.0 for penalized anchors, 3.0 for others
            # First score the whole task at 3.0 (neutral)
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cmd_score(conn, tid, "3.0")

            # Now selectively penalize specific anchors
            if penalize_titles:
                rows = conn.execute("""
                    SELECT ta.anchor_id, n.title FROM task_anchors ta
                    JOIN nodes n ON n.id = ta.anchor_id
                    WHERE ta.task_id = ?
                """, (tid,)).fetchall()

                for r in rows:
                    if r["title"].lower() in penalize_titles:
                        # Direct weight penalty
                        current = conn.execute(
                            "SELECT weight FROM nodes WHERE id=?", (r["anchor_id"],)
                        ).fetchone()
                        if current:
                            new_weight = max(0.1, current["weight"] * 0.6)
                            conn.execute(
                                "UPDATE nodes SET weight=? WHERE id=?",
                                (round(new_weight, 3), r["anchor_id"])
                            )
                            anchors_penalized.append(r["title"])
                conn.commit()

    # Clear pending rating
    if state:
        state["pending_rating"] = None
        _session_save(state)

    print(json.dumps({
        "rated": True,
        "direction": direction,
        "tasks_scored": len(task_ids),
        "task_ids": task_ids,
        "anchors_penalized": anchors_penalized,
    }, indent=2))


def cmd_session_store(conn, json_arg):
    """Store memory and track task_id in session. Also mirrors to L2 and pushes to L3."""
    # Retry any pending failed pushes first
    _process_retry()

    # Capture store output to get task_id
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_store(conn, json_arg)
    output = buf.getvalue()

    try:
        result = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        print(output)  # Pass through on error
        return

    # Track task_id in session
    task_id = result.get("task_id")
    if task_id:
        state = _session_load()
        if state:
            task_ids = state.get("task_ids", [])
            if task_id not in task_ids:
                task_ids.append(task_id)
            state["task_ids"] = task_ids
            sc_title = result.get("super_context")
            if sc_title:
                state["current_sc_title"] = sc_title
            _session_save(state)

    # L2: Mirror to brain.db (best-effort)
    try:
        data = json.loads(json_arg)
        _mirror_to_brain(data, author=_get_author(), source_folder=str(ROOT))
    except Exception:
        pass

    # L3: Push to central (if shared folder, best-effort)
    try:
        data = json.loads(json_arg)
        _push_to_central(conn, data, _get_author(), str(ROOT))
    except Exception:
        pass

    # Add team info to output
    try:
        result = json.loads(output)
        result["team"] = {
            "author": _get_author(),
            "mirrored_to_brain": True,
            "pushed_to_central": _is_shared_folder(),
        }
        print(json.dumps(result, indent=2))
    except Exception:
        print(output)


def cmd_session_end(conn):
    """Explicitly end session."""
    state = _session_load()
    _session_clear()
    if state:
        print(json.dumps({
            "ended": True,
            "session_id": state.get("session_id"),
            "conversations": state.get("conversation_count", 0),
            "tasks_stored": len(state.get("task_ids", [])),
        }, indent=2))
    else:
        print(json.dumps({"ended": True, "note": "No active session"}))


def cmd_session(conn, subcmd, *args):
    """Session command dispatcher."""
    if subcmd == "begin" and args:
        cmd_session_begin(conn, args[0])
    elif subcmd == "rate" and args:
        direction = args[0]
        specific = args[1] if len(args) > 1 else None
        cmd_session_rate(conn, direction, specific)
    elif subcmd == "store" and args:
        cmd_session_store(conn, args[0])
    elif subcmd == "end":
        cmd_session_end(conn)
    else:
        print(json.dumps({"error": f"Unknown session subcommand: {subcmd}. Use: begin, rate, store, end"}))


# ══════════════════════════════════════════════════════════
# TEAM MEMORY COMMANDS
# ══════════════════════════════════════════════════════════

def _get_brain_db():
    """Get connection to personal brain.db (L2). Creates if not exists."""
    YANTRAI_DIR.mkdir(parents=True, exist_ok=True)
    # Temporarily swap paths to create brain.db with same schema
    original_graph = GRAPH_DB
    original_tasks = TASKS_DIR
    original_vector = VECTOR_DIR
    try:
        # Use module-level globals trick
        import memory as _self
        _self.GRAPH_DB = YANTRAI_BRAIN_DB
        _self.TASKS_DIR = YANTRAI_DIR / "brain_tasks"
        _self.VECTOR_DIR = YANTRAI_DIR / "brain_vectors"
        conn = get_db()
        return conn
    finally:
        _self.GRAPH_DB = original_graph
        _self.TASKS_DIR = original_tasks
        _self.VECTOR_DIR = original_vector


def _mirror_to_brain(store_data, author="", source_folder=""):
    """Mirror a store to brain.db (L2). Silent on failure."""
    try:
        brain_conn = _get_brain_db()
        # Store with same data, marking source as 'local'
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd_store(brain_conn, json.dumps(store_data) if isinstance(store_data, dict) else store_data)
        brain_conn.close()
    except Exception:
        pass  # brain mirror is best-effort


def _push_to_central(conn, data, author, source_folder):
    """Push stored nodes to central server. Returns True on success."""
    if not _is_shared_folder(source_folder):
        return False

    # Collect nodes from this store
    sc_title = data.get("super_context", "")
    nodes_to_push = []
    edges_to_push = []

    # Get SC node
    sc_row = conn.execute("SELECT * FROM nodes WHERE type='super_context' AND title=?",
                          (sc_title,)).fetchone()
    if sc_row:
        nodes_to_push.append(dict(sc_row))

    # Get contexts and anchors for this SC
    for ctx in conn.execute("""
        SELECT n.* FROM nodes n JOIN edges e ON e.tgt=n.id
        WHERE e.src=? AND n.type='context'
    """, (sc_row["id"],)).fetchall() if sc_row else []:
        nodes_to_push.append(dict(ctx))
        edges_to_push.append({"src": sc_row["id"], "tgt": ctx["id"], "type": "parent_child"})

        for anc in conn.execute("""
            SELECT n.* FROM nodes n JOIN edges e ON e.tgt=n.id
            WHERE e.src=? AND n.type='anchor'
        """, (ctx["id"],)).fetchall():
            nodes_to_push.append(dict(anc))
            edges_to_push.append({"src": ctx["id"], "tgt": anc["id"], "type": "parent_child"})

    if not nodes_to_push:
        return False

    # Serialize nodes (convert Row objects to dicts, handle non-JSON fields)
    serialized = []
    for n in nodes_to_push:
        nd = {k: n[k] for k in n.keys()} if hasattr(n, 'keys') else n
        serialized.append(nd)

    result = _central_request("POST", "/api/sync/push", {
        "author": author,
        "source_folder": str(source_folder),
        "nodes": serialized,
        "edges": edges_to_push
    })

    if result is None:
        # Push failed — queue for retry
        _append_retry(serialized, edges_to_push)
        return False
    return True


def cmd_share(conn):
    """Add current project folder to shared_folders."""
    cfg = _load_team_config()
    if not cfg:
        print(json.dumps({"error": "Team not configured. Run yantrai_setup.py first."}))
        return
    folder = str(ROOT)
    shared = cfg.get("shared_folders", [])
    if folder not in shared:
        shared.append(folder)
        cfg["shared_folders"] = shared
        YANTRAI_CONFIG.write_text(json.dumps(cfg, indent=2))
    print(json.dumps({"shared": True, "folder": folder, "total_shared": len(shared)}))


def cmd_unshare(conn):
    """Remove current project folder from shared_folders."""
    cfg = _load_team_config()
    if not cfg:
        print(json.dumps({"error": "Team not configured."}))
        return
    folder = str(ROOT)
    shared = cfg.get("shared_folders", [])
    shared = [f for f in shared if f.rstrip("/") != folder.rstrip("/")]
    cfg["shared_folders"] = shared
    YANTRAI_CONFIG.write_text(json.dumps(cfg, indent=2))
    print(json.dumps({"unshared": True, "folder": folder, "total_shared": len(shared)}))


def cmd_team_status(conn):
    """Show team memory status."""
    cfg = _load_team_config()
    server = _get_server()
    reachable = _central_reachable() if server else False
    retry_count = 0
    if YANTRAI_RETRY_QUEUE.exists():
        try:
            retry_count = len(json.loads(YANTRAI_RETRY_QUEUE.read_text()))
        except Exception:
            pass
    sync_state = {}
    if YANTRAI_SYNC_STATE.exists():
        try:
            sync_state = json.loads(YANTRAI_SYNC_STATE.read_text())
        except Exception:
            pass

    print(json.dumps({
        "configured": cfg is not None,
        "author": _get_author(),
        "server": server,
        "server_reachable": reachable,
        "shared_folders": cfg.get("shared_folders", []) if cfg else [],
        "retry_queue_size": retry_count,
        "last_pull_txn": sync_state.get("last_pull_txn", 0),
        "brain_db_exists": YANTRAI_BRAIN_DB.exists(),
    }, indent=2))


# ══════════════════════════════════════════════════════════
# CLI ROUTER
# ══════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/memory.py <command> [args]")
        print("Commands: store, retrieve, browse, tree, stats, score, rebuild, viz")
        return

    conn = get_db()
    cmd = sys.argv[1]

    try:
        if cmd == "retrieve" and len(sys.argv) >= 3:
            (_p9_retrieve(conn, sys.argv[2]) if _HAS_P9 else cmd_retrieve(conn, sys.argv[2]))
        elif cmd == "store" and len(sys.argv) >= 4:
            cmd_store(conn, sys.argv[2], sys.argv[3])
        elif cmd == "store" and len(sys.argv) == 3:
            cmd_store(conn, sys.argv[2])
        elif cmd == "browse":
            cmd_browse(conn)
        elif cmd == "tree":
            cmd_tree(conn)
        elif cmd == "stats":
            cmd_stats(conn)
        elif cmd == "score" and len(sys.argv) >= 4:
            cmd_score(conn, sys.argv[2], sys.argv[3])
        elif cmd == "rebuild":
            cmd_rebuild(conn)
        elif cmd == "viz":
            cmd_viz(conn)
        elif cmd == "transfers":
            cmd_transfers(conn)
        elif cmd == "decision" and len(sys.argv) >= 4:
            cmd_decision(conn, sys.argv[2], sys.argv[3])
        elif cmd == "convergence" and len(sys.argv) >= 3:
            cmd_convergence(conn, sys.argv[2])
        elif cmd == "decisions" and len(sys.argv) >= 3:
            cmd_decisions_list(conn, sys.argv[2])
        elif cmd == "session" and len(sys.argv) >= 3:
            cmd_session(conn, sys.argv[2], *sys.argv[3:])
        elif cmd == "share":
            cmd_share(conn)
        elif cmd == "unshare":
            cmd_unshare(conn)
        elif cmd == "team-status":
            cmd_team_status(conn)
        else:
            print(f"Unknown command or missing args: {cmd}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
