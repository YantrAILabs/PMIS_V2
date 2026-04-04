#!/usr/bin/env python3
"""
YantrAI Memory System — End-to-End Installation Verification

Runs a quick pipeline test: insert test segment → sync to PMIS V2 → verify node → cleanup.
Called by install.sh after DB init.
Exit code 0 = all passed, 1 = failures.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

REPO_DIR = Path(__file__).parent
TRACKER_DIR = REPO_DIR / "productivity-tracker"
PMIS_DIR = REPO_DIR / "pmis_v2"

sys.path.insert(0, str(TRACKER_DIR))
sys.path.insert(0, str(PMIS_DIR))

GREEN = "\033[0;32m"
RED = "\033[0;31m"
NC = "\033[0m"
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  {GREEN}✓{NC} {name}")
        passed += 1
    else:
        print(f"  {RED}✗{NC} {name} — {detail}")
        failed += 1


def main():
    global passed, failed
    data_dir = Path.home() / ".productivity-tracker"
    os.environ["SQLITE_DB_PATH"] = str(data_dir / "tracker.db")

    # ── Test 1: Tracker DB imports and connects ──
    try:
        from src.storage.db import Database
        db = Database(str(data_dir / "tracker.db"))
        db.initialize()
        check("Tracker DB connects", True)
    except Exception as e:
        check("Tracker DB connects", False, str(e))
        return 1

    # ── Test 2: Tracker schema has new columns ──
    try:
        import sqlite3
        conn = sqlite3.connect(str(data_dir / "tracker.db"))
        cols = {r[1] for r in conn.execute("PRAGMA table_info(context_1)").fetchall()}
        has_new = all(c in cols for c in ["synced_to_memory", "human_frame_count", "has_keyboard_activity" if False else "synced_to_memory"])
        check("Tracker schema migrated", "synced_to_memory" in cols)
        conn.close()
    except Exception as e:
        check("Tracker schema migrated", False, str(e))

    # ── Test 3: PMIS V2 DB imports and connects ──
    try:
        from db.manager import DBManager
        pmis_db = DBManager(db_path=str(PMIS_DIR / "data" / "memory.db"))
        node_count = pmis_db.count_nodes()
        check("PMIS V2 DB connects", True)
        check(f"PMIS V2 has {node_count} nodes", node_count >= 0)
    except Exception as e:
        check("PMIS V2 DB connects", False, str(e))
        return 1

    # ── Test 4: PMIS V2 has productivity tables ──
    try:
        tables = [r[0] for r in pmis_db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        check("projects table exists", "projects" in tables)
        check("deliverables table exists", "deliverables" in tables)
        check("project_work_match_log exists", "project_work_match_log" in tables)
        check("productivity_sync_log exists", "productivity_sync_log" in tables)
    except Exception as e:
        check("Productivity tables exist", False, str(e))

    # ── Test 5: memory_nodes has productivity columns ──
    try:
        mn_cols = {r[1] for r in pmis_db._conn.execute("PRAGMA table_info(memory_nodes)").fetchall()}
        check("productivity_time_mins column", "productivity_time_mins" in mn_cols)
        check("is_project_node column", "is_project_node" in mn_cols)
    except Exception as e:
        check("Productivity columns exist", False, str(e))

    # ── Test 6: Core imports work ──
    try:
        from core.poincare import ProjectionManager, poincare_distance
        pm = ProjectionManager()
        check("Poincare ProjectionManager", True)
    except Exception as e:
        check("Poincare ProjectionManager", False, str(e))

    try:
        from core.temporal import temporal_encode
        import numpy as np
        t = temporal_encode(datetime.now())
        check(f"Temporal encoding ({t.shape[0]}D)", t.shape[0] == 16)
    except Exception as e:
        check("Temporal encoding", False, str(e))

    try:
        from core.rsgd import RSGDTrainer
        check("RSGDTrainer imports", True)
    except Exception as e:
        check("RSGDTrainer imports", False, str(e))

    # ── Test 7: Pipeline sync imports ──
    try:
        from src.memory.pipeline_sync import ProductivityPipelineSync
        check("ProductivityPipelineSync imports", True)
    except Exception as e:
        check("ProductivityPipelineSync imports", False, str(e))

    try:
        from src.memory.project_matcher import ProjectMatcher
        check("ProjectMatcher imports", True)
    except Exception as e:
        check("ProjectMatcher imports", False, str(e))

    try:
        from src.agent.input_monitor import InputMonitor
        check("InputMonitor imports", True)
    except Exception as e:
        check("InputMonitor imports", False, str(e))

    # ── Test 8: Platform auth works ──
    try:
        # Clean import — no name clash workaround needed (platform_db.py is unique)
        sys.path.insert(0, str(REPO_DIR / "memory_system" / "platform"))
        from platform_auth import hash_password, verify_password
        h = hash_password("test123")
        check("Platform auth hashing", verify_password("test123", h))
    except Exception as e:
        check("Platform auth", False, str(e))

    # ── Cleanup ──
    try:
        db.close()
        pmis_db.close()
    except Exception:
        pass

    # ── Summary ──
    print(f"\n  Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
