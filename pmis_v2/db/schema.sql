-- PMIS v2 Database Schema
-- Run once to initialize: sqlite3 data/memory.db < db/schema.sql

CREATE TABLE IF NOT EXISTS memory_nodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_conversation_id TEXT DEFAULT '',
    level TEXT NOT NULL CHECK(level IN ('SC', 'CTX', 'ANC')),
    parent_ids TEXT DEFAULT '[]',        -- JSON list
    tree_ids TEXT DEFAULT '[]',          -- JSON list
    precision REAL DEFAULT 0.5,
    surprise_at_creation REAL DEFAULT 0.0,
    created_at TEXT DEFAULT (datetime('now')),
    last_modified TEXT DEFAULT (datetime('now')),
    era TEXT DEFAULT '',
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT,
    decay_rate REAL DEFAULT 0.5,
    is_orphan INTEGER DEFAULT 0,
    is_tentative INTEGER DEFAULT 0,
    is_deleted INTEGER DEFAULT 0,
    -- Productivity tracking columns
    productivity_time_mins REAL DEFAULT 0,
    productivity_human_mins REAL DEFAULT 0,
    productivity_ai_mins REAL DEFAULT 0,
    last_productivity_sync TEXT DEFAULT '',
    is_project_node INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    tree_id TEXT NOT NULL DEFAULT 'default',
    weight REAL DEFAULT 1.0,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (source_id) REFERENCES memory_nodes(id),
    FOREIGN KEY (target_id) REFERENCES memory_nodes(id)
);

CREATE TABLE IF NOT EXISTS trees (
    tree_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    root_node_id TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (root_node_id) REFERENCES memory_nodes(id)
);

CREATE TABLE IF NOT EXISTS access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    accessed_at TEXT DEFAULT (datetime('now')),
    query_text TEXT DEFAULT '',
    gamma_at_access REAL,
    surprise_at_access REAL,
    semantic_distance REAL,
    FOREIGN KEY (node_id) REFERENCES memory_nodes(id)
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content_hash TEXT,
    node_id TEXT,
    gamma REAL,
    effective_surprise REAL,
    mode TEXT,
    timestamp TEXT DEFAULT (datetime('now')),
    raw_surprise REAL,
    cluster_precision REAL,
    nearest_context_id TEXT,
    nearest_context_name TEXT,
    active_tree TEXT,
    is_stale INTEGER DEFAULT 0,
    storage_action TEXT,
    system_prompt TEXT,
    response_summary TEXT,
    FOREIGN KEY (node_id) REFERENCES memory_nodes(id),
    UNIQUE(conversation_id, turn_number)
);

-- Detail: which memories were retrieved at each turn
CREATE TABLE IF NOT EXISTS turn_retrieved_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL,
    memory_node_id TEXT,
    rank INTEGER NOT NULL,
    final_score REAL,
    semantic_score REAL,
    hierarchy_score REAL,
    temporal_score REAL,
    precision_score REAL,
    source TEXT,
    content_preview TEXT,
    node_level TEXT
);

-- Detail: what epistemic questions were generated at each turn
CREATE TABLE IF NOT EXISTS turn_epistemic_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL,
    question_text TEXT NOT NULL,
    information_gain REAL,
    parent_context_id TEXT,
    parent_context_name TEXT,
    anchor_id TEXT,
    anchor_content TEXT,
    FOREIGN KEY (turn_id) REFERENCES conversation_turns(id)
);

-- Detail: what predictive memories were surfaced at each turn
CREATE TABLE IF NOT EXISTS turn_predictive_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL,
    memory_node_id TEXT,
    content_preview TEXT,
    prediction_depth INTEGER,
    prediction_frequency INTEGER,
    FOREIGN KEY (turn_id) REFERENCES conversation_turns(id)
);

CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT DEFAULT (datetime('now')),
    action TEXT NOT NULL,
    source_node_ids TEXT DEFAULT '[]',
    target_node_id TEXT,
    reason TEXT DEFAULT '',
    details TEXT DEFAULT '{}'
);

-- Embeddings stored as BLOBs (numpy arrays serialized)
CREATE TABLE IF NOT EXISTS embeddings (
    node_id TEXT PRIMARY KEY,
    euclidean BLOB,                    -- numpy float32 array
    hyperbolic BLOB,                   -- numpy float32 array
    temporal BLOB,                     -- numpy float32 array
    hyperbolic_norm REAL,              -- precomputed ||hyperbolic|| for indexed queries
    is_learned INTEGER DEFAULT 0,      -- 0=random projection, 1=RSGD-refined
    last_trained TEXT,                  -- ISO timestamp of last RSGD update
    FOREIGN KEY (node_id) REFERENCES memory_nodes(id)
);

-- RSGD training run history
CREATE TABLE IF NOT EXISTS rsgd_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,            -- 'nightly', 'incremental', 'burn_in'
    epochs INTEGER,
    final_loss REAL,
    nodes_updated INTEGER,
    edges_used INTEGER,
    learning_rate REAL,
    wall_time_seconds REAL,
    run_at TEXT DEFAULT (datetime('now'))
);

-- Context centroids in hyperbolic space (Frechet mean of children)
CREATE TABLE IF NOT EXISTS context_centroids (
    context_id TEXT PRIMARY KEY,
    centroid BLOB NOT NULL,            -- numpy float32 array (Frechet mean)
    child_count INTEGER,
    last_computed TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (context_id) REFERENCES memory_nodes(id)
);

-- ═══════════════════════════════════════════════════════════
-- PRODUCTIVITY + PROJECT MANAGEMENT TABLES
-- ═══════════════════════════════════════════════════════════

-- Projects — company/initiative level (maps to SC)
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    company TEXT DEFAULT '',
    source TEXT DEFAULT 'manual',
    source_id TEXT DEFAULT '',
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'on_hold', 'cancelled')),
    owner TEXT DEFAULT '',
    deadline TEXT DEFAULT '',
    sc_node_id TEXT DEFAULT '',
    expected_hours REAL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Deliverables — under projects (maps to Context/Anchor)
CREATE TABLE IF NOT EXISTS deliverables (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    context_node_id TEXT DEFAULT '',
    anchor_node_ids TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'on_hold', 'cancelled')),
    deadline TEXT DEFAULT '',
    expected_hours REAL DEFAULT 0,
    source TEXT DEFAULT 'manual',
    source_id TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- Project-Work Match Log — tracks matching effectiveness
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

-- Productivity Sync Log — audit trail for 30-min auto-syncs
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
    status TEXT DEFAULT 'running' CHECK(status IN ('running', 'completed', 'failed'))
);

-- ═══════════════════════════════════════════════════════════
-- GOALS & FEEDBACK — External intent + validation layer
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'achieved', 'paused', 'abandoned')),
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS goal_links (
    goal_id TEXT NOT NULL REFERENCES goals(id),
    node_id TEXT NOT NULL REFERENCES memory_nodes(id),
    link_type TEXT NOT NULL DEFAULT 'supports'
        CHECK(link_type IN ('supports', 'blocks', 'neutral')),
    weight REAL DEFAULT 0.5,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (goal_id, node_id)
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL REFERENCES memory_nodes(id),
    goal_id TEXT REFERENCES goals(id),
    polarity TEXT NOT NULL CHECK(polarity IN ('positive', 'negative', 'correction')),
    content TEXT DEFAULT '',
    source TEXT NOT NULL DEFAULT 'explicit'
        CHECK(source IN ('explicit', 'session', 'implicit')),
    strength REAL DEFAULT 1.0,
    timestamp TEXT DEFAULT (datetime('now'))
);

-- ═══════════════════════════════════════════════════════════
-- WIKI PAGE CACHE — LLM-generated prose pages
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS wiki_page_cache (
    node_id TEXT PRIMARY KEY,
    prose_markdown TEXT,
    context_hash TEXT,
    generated_at TEXT DEFAULT (datetime('now')),
    llm_model TEXT,
    word_count INTEGER DEFAULT 0,
    FOREIGN KEY (node_id) REFERENCES memory_nodes(id)
);

-- ═══════════════════════════════════════════════════════════
-- ACTIVITY TIME TRACKING — Links activity segments to tree branches
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS activity_time_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id TEXT,
    memory_node_id TEXT,
    matched_ctx_id TEXT,
    matched_sc_id TEXT,
    duration_seconds INTEGER,
    date TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS node_time_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT,
    date TEXT,
    total_duration_mins REAL,
    segment_count INTEGER,
    project_id TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ═══════════════════════════════════════════════════════════
-- TURN DIAGNOSTICS — End-to-end pipeline instrumentation
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS turn_diagnostics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    timestamp TEXT NOT NULL,

    -- 1. EMBEDDING STAGE
    embed_model TEXT,
    embed_dim INTEGER,
    embed_latency_ms REAL,

    -- 2. NEAREST CONTEXT STAGE
    nearest_ctx_id TEXT,
    nearest_ctx_name TEXT,
    nearest_ctx_distance REAL,
    second_ctx_distance REAL,
    contexts_searched INTEGER,

    -- 3. SURPRISE STAGE
    raw_surprise REAL,
    cluster_precision REAL,
    precision_anchor_factor REAL,
    precision_recency_factor REAL,
    precision_consistency_factor REAL,
    effective_surprise REAL,
    is_orphan_territory INTEGER DEFAULT 0,

    -- 4. GAMMA STAGE
    gamma_input_surprise REAL,
    gamma_temperature REAL,
    gamma_bias REAL,
    gamma_staleness_penalty REAL,
    gamma_session_boost REAL,
    gamma_raw REAL,
    gamma_final REAL,
    gamma_mode TEXT,
    gamma_override_active INTEGER DEFAULT 0,

    -- 5. TREE RESOLUTION STAGE
    tree_resolution_method TEXT,
    tree_id TEXT,
    tree_name TEXT,
    tree_root_distance REAL,

    -- 6. RETRIEVAL STAGE
    narrow_k INTEGER,
    narrow_threshold REAL,
    narrow_candidates_found INTEGER,
    broad_k INTEGER,
    broad_threshold REAL,
    broad_candidates_found INTEGER,
    total_candidates_scored INTEGER,
    retrieval_latency_ms REAL,

    -- 7. SCORING BREAKDOWN (top-1 result)
    top1_node_id TEXT,
    top1_final_score REAL,
    top1_semantic_score REAL,
    top1_hierarchy_score REAL,
    top1_temporal_score REAL,
    top1_precision_score REAL,
    top1_node_level TEXT,
    top1_source TEXT,

    -- 8. SCORING SPREAD (across all results)
    avg_semantic REAL,
    avg_hierarchy REAL,
    avg_temporal REAL,
    avg_precision REAL,
    std_semantic REAL,
    std_hierarchy REAL,
    std_temporal REAL,
    std_precision REAL,
    score_range REAL,

    -- 9. STORAGE DECISION
    storage_action TEXT,
    storage_reason TEXT,
    stored_node_id TEXT,
    stored_as_orphan INTEGER DEFAULT 0,
    stored_as_tentative INTEGER DEFAULT 0,
    dedup_blocked INTEGER DEFAULT 0,

    -- 10. EPISTEMIC & PREDICTIVE
    epistemic_questions_count INTEGER DEFAULT 0,
    top_epistemic_info_gain REAL,
    predictive_memories_count INTEGER DEFAULT 0,

    -- 11. SESSION STATE
    session_turn_count INTEGER,
    session_avg_gamma REAL,
    session_precision_accumulator REAL,
    session_is_stale INTEGER DEFAULT 0,

    -- 12. POINCARÉ HEALTH
    avg_poincare_norm REAL,
    poincare_norm_spread REAL,
    hierarchy_score_discriminative REAL,

    UNIQUE(conversation_id, turn_number)
);

-- ═══════════════════════════════════════════════════════════
-- INDEXES
-- ═══════════════════════════════════════════════════════════

-- Core indexes
CREATE INDEX IF NOT EXISTS idx_nodes_level ON memory_nodes(level);
CREATE INDEX IF NOT EXISTS idx_nodes_orphan ON memory_nodes(is_orphan);
CREATE INDEX IF NOT EXISTS idx_nodes_deleted ON memory_nodes(is_deleted);
CREATE INDEX IF NOT EXISTS idx_nodes_era ON memory_nodes(era);
CREATE INDEX IF NOT EXISTS idx_nodes_project ON memory_nodes(is_project_node);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_relations_tree ON relations(tree_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_access_node ON access_log(node_id);
CREATE INDEX IF NOT EXISTS idx_turns_conv ON conversation_turns(conversation_id);
CREATE INDEX IF NOT EXISTS idx_hyp_norm ON embeddings(hyperbolic_norm);
CREATE INDEX IF NOT EXISTS idx_turn_retrieved ON turn_retrieved_memories(turn_id);
CREATE INDEX IF NOT EXISTS idx_turn_epistemic ON turn_epistemic_questions(turn_id);
CREATE INDEX IF NOT EXISTS idx_turn_predictive ON turn_predictive_memories(turn_id);

-- Goals & feedback indexes
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goal_links_goal ON goal_links(goal_id);
CREATE INDEX IF NOT EXISTS idx_goal_links_node ON goal_links(node_id);
CREATE INDEX IF NOT EXISTS idx_feedback_node ON feedback(node_id);
CREATE INDEX IF NOT EXISTS idx_feedback_goal ON feedback(goal_id);
CREATE INDEX IF NOT EXISTS idx_feedback_polarity ON feedback(polarity);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp);

-- Diagnostics indexes
CREATE INDEX IF NOT EXISTS idx_diag_conv ON turn_diagnostics(conversation_id);
CREATE INDEX IF NOT EXISTS idx_diag_timestamp ON turn_diagnostics(timestamp);
CREATE INDEX IF NOT EXISTS idx_diag_gamma_mode ON turn_diagnostics(gamma_mode);

-- Project/productivity indexes
CREATE INDEX IF NOT EXISTS idx_proj_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_proj_sc ON projects(sc_node_id);
CREATE INDEX IF NOT EXISTS idx_deliv_project ON deliverables(project_id);
CREATE INDEX IF NOT EXISTS idx_deliv_status ON deliverables(status);
CREATE INDEX IF NOT EXISTS idx_pwm_segment ON project_work_match_log(segment_id);
CREATE INDEX IF NOT EXISTS idx_pwm_project ON project_work_match_log(project_id);
CREATE INDEX IF NOT EXISTS idx_pwm_score ON project_work_match_log(combined_match_pct);
CREATE INDEX IF NOT EXISTS idx_pwm_date ON project_work_match_log(matched_at);
CREATE INDEX IF NOT EXISTS idx_psync_type ON productivity_sync_log(sync_type);
CREATE INDEX IF NOT EXISTS idx_psync_status ON productivity_sync_log(status);
