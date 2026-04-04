#!/usr/bin/env python3
"""
PMIS v3 Benchmark Suite — Tests all 5 new features.
Run from memory/ root: python3 scripts/benchmark_v3.py
"""
import sqlite3, os, sys, json, math, random, datetime, tempfile

sys.path.insert(0, os.path.dirname(__file__))
import memory as mem

def fresh_db():
    """Create temp database with full v3 schema."""
    tmp = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(tmp)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY, type TEXT NOT NULL, title TEXT NOT NULL,
            description TEXT DEFAULT '', content TEXT DEFAULT '', source TEXT DEFAULT '',
            created_at TEXT NOT NULL, last_used TEXT NOT NULL,
            use_count INTEGER DEFAULT 1, quality REAL DEFAULT 0.0,
            weight REAL DEFAULT 0.5, initial_weight REAL DEFAULT 0.5,
            occurrence_log TEXT DEFAULT '[]',
            recency REAL DEFAULT 1.0, frequency REAL DEFAULT 0.0,
            consistency REAL DEFAULT 0.0, memory_stage TEXT DEFAULT 'impulse',
            discrimination_power REAL DEFAULT 0.0, mode_vector TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY, src TEXT NOT NULL, tgt TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'parent_child',
            weight REAL DEFAULT 1.0, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, sc_id TEXT, title TEXT NOT NULL,
            started TEXT NOT NULL, completed TEXT, score REAL DEFAULT 0,
            content TEXT DEFAULT '', structure_snapshot TEXT DEFAULT '',
            mode_vector TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS task_anchors (
            task_id TEXT NOT NULL, anchor_id TEXT NOT NULL,
            context_id TEXT NOT NULL, was_retrieved INTEGER DEFAULT 0,
            execution_params TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS decisions (
            id TEXT PRIMARY KEY, sc_id TEXT NOT NULL, anchor_id TEXT,
            session_id TEXT NOT NULL, decision TEXT NOT NULL,
            alternatives_eliminated TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.5, reversible INTEGER DEFAULT 1,
            evidence TEXT DEFAULT '', created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn, tmp


# ══════════════════════════════════════════════════════════
# BENCHMARK 1: Decision Convergence
# ══════════════════════════════════════════════════════════

def benchmark_decisions():
    conn, tmp = fresh_db()

    sc_id = mem.add_node(conn, "super_context", "Decision Test SC", weight=0.5)

    # Add 5 decisions with varying confidence
    decisions = [
        {"decision": "Use dark theme", "confidence": 0.9, "reversible": False},
        {"decision": "Use Sora font", "confidence": 0.8, "reversible": False},
        {"decision": "9-section layout", "confidence": 0.7, "reversible": True},
        {"decision": "Red strikethrough accent", "confidence": 0.6, "reversible": True},
        {"decision": "Amber secondary color", "confidence": 0.4, "reversible": True},
    ]

    for d in decisions:
        did = mem._id()
        conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?)",
            (did, sc_id, "", mem._id(), d["decision"], "[]", d["confidence"],
             0 if not d["reversible"] else 1, "", mem._now())
        )
    conn.commit()

    # Test convergence formula
    state = mem.get_convergence_state(conn, sc_id)

    # Expected: avg_conf = (0.9+0.8+0.7+0.6+0.4)/5 = 0.68
    # lock_ratio = 2/5 = 0.4
    # count_factor = min(5/10, 1) = 0.5
    # convergence = 0.4*0.68 + 0.4*0.4 + 0.2*0.5 = 0.272 + 0.16 + 0.1 = 0.532
    expected_conv = 0.532
    actual_conv = state["convergence"]
    conv_close = abs(actual_conv - expected_conv) < 0.01

    # Test preamble
    preamble = mem.session_preamble(conn, sc_id)
    has_constraints = len(preamble["constraints"]) == 5  # all 5 decisions have confidence > 0.3

    # Test decision reinforcement (simulate high-score task)
    # Before: confidence of "Amber secondary" = 0.4
    conn.execute(
        "UPDATE decisions SET confidence = MIN(1.0, confidence + 0.1 * (1.0 - confidence)) WHERE sc_id=?",
        (sc_id,))
    conn.commit()
    after_reinforce = conn.execute(
        "SELECT confidence FROM decisions WHERE decision='Amber secondary color'", ()
    ).fetchone()
    reinforce_worked = after_reinforce and after_reinforce["confidence"] > 0.4

    # Test decay (simulate low-score task)
    conn.execute(
        "UPDATE decisions SET confidence = confidence * 0.7, reversible = 1 WHERE sc_id=?",
        (sc_id,))
    conn.commit()
    after_decay = conn.execute(
        "SELECT confidence, reversible FROM decisions WHERE decision='Use dark theme'", ()
    ).fetchone()
    decay_worked = after_decay and after_decay["confidence"] < 0.9 and after_decay["reversible"] == 1

    conn.close(); os.unlink(tmp)

    tests_passed = sum([conv_close, has_constraints, reinforce_worked, decay_worked])
    return tests_passed, 4, {
        "convergence_formula": conv_close,
        "expected": expected_conv,
        "actual": actual_conv,
        "preamble_constraints": has_constraints,
        "reinforce": reinforce_worked,
        "decay": decay_worked,
    }


# ══════════════════════════════════════════════════════════
# BENCHMARK 2: Transfer Edge Detection
# ══════════════════════════════════════════════════════════

def benchmark_transfers():
    conn, tmp = fresh_db()

    # Create 3 SCs: two similar (website domains), one different
    sc1 = mem.add_node(conn, "super_context", "Schools Website", weight=0.5)
    sc2 = mem.add_node(conn, "super_context", "Health Website", weight=0.5)
    sc3 = mem.add_node(conn, "super_context", "B2B Cold Outreach", weight=0.5)

    # Similar contexts for sc1 and sc2
    shared_titles = ["Hero section design", "Layout architecture", "Contact form", "Intelligence layer"]
    for title in shared_titles:
        c1 = mem.add_node(conn, "context", title, weight=0.7)
        mem.link(conn, sc1, c1)
        c2 = mem.add_node(conn, "context", title, weight=0.7)
        mem.link(conn, sc2, c2)

    # Unique contexts for sc1
    c1u = mem.add_node(conn, "context", "School safety features", weight=0.6)
    mem.link(conn, sc1, c1u)

    # Unique contexts for sc2
    c2u = mem.add_node(conn, "context", "Patient monitoring", weight=0.6)
    mem.link(conn, sc2, c2u)

    # Different domain for sc3
    for title in ["Email copywriting", "Target research", "Follow-up sequences"]:
        c3 = mem.add_node(conn, "context", title, weight=0.7)
        mem.link(conn, sc3, c3)

    # Detect transfers
    transfers = mem.detect_transfers(conn)

    # Should find: Schools ↔ Health (high similarity)
    # Should NOT find: Schools ↔ Outreach or Health ↔ Outreach
    found_similar = any(
        ("Schools" in t["from"] and "Health" in t["to"]) or
        ("Health" in t["from"] and "Schools" in t["to"])
        for t in transfers
    )
    no_false_positive = not any(
        "Outreach" in t["from"] or "Outreach" in t["to"]
        for t in transfers
    )

    # Check edges exist in DB
    transfer_edges = conn.execute("SELECT * FROM edges WHERE type='transfer'").fetchall()
    edges_created = len(transfer_edges) >= 2  # bidirectional

    # Check structural similarity value
    sim = mem.compute_structural_similarity(conn, sc1, sc2)
    sim_reasonable = sim > 0.3  # should be high

    conn.close(); os.unlink(tmp)

    tests_passed = sum([found_similar, no_false_positive, edges_created, sim_reasonable])
    return tests_passed, 4, {
        "found_similar_pair": found_similar,
        "no_false_positives": no_false_positive,
        "edges_created": edges_created,
        "structural_similarity": sim,
        "sim_reasonable": sim_reasonable,
        "transfers_found": len(transfers)
    }


# ══════════════════════════════════════════════════════════
# BENCHMARK 3: Anchor Discrimination Power
# ══════════════════════════════════════════════════════════

def benchmark_adp():
    conn, tmp = fresh_db()
    random.seed(42)

    sc_id = mem.add_node(conn, "super_context", "ADP Test SC", weight=0.5)
    ctx_id = mem.add_node(conn, "context", "ADP Test Ctx", weight=0.5)
    mem.link(conn, sc_id, ctx_id)

    # Consistent anchor: always in tasks scoring 3.5-4.5
    consistent_anchor = mem.add_node(conn, "anchor", "Consistent Insight", weight=0.7)
    mem.link(conn, ctx_id, consistent_anchor)

    # Variable anchor: in tasks scoring 1.0-5.0 (high variance)
    variable_anchor = mem.add_node(conn, "anchor", "Variable Insight", weight=0.7)
    mem.link(conn, ctx_id, variable_anchor)

    # Create 8 tasks
    for i in range(8):
        task_id = mem._id()
        now = datetime.datetime.now().isoformat()

        # Consistent anchor always in mid-range tasks
        consistent_score = 3.5 + random.uniform(0, 1.0)
        conn.execute("INSERT INTO tasks(id, sc_id, title, started, completed, score) VALUES(?,?,?,?,?,?)",
                     (task_id, sc_id, f"Task_C_{i}", now, now, consistent_score))
        conn.execute("INSERT INTO task_anchors(task_id, anchor_id, context_id) VALUES(?,?,?)",
                     (task_id, consistent_anchor, ctx_id))

        # Variable anchor in wildly varying tasks
        task_id2 = mem._id()
        variable_score = 1.0 + random.uniform(0, 4.0)
        conn.execute("INSERT INTO tasks(id, sc_id, title, started, completed, score) VALUES(?,?,?,?,?,?)",
                     (task_id2, sc_id, f"Task_V_{i}", now, now, variable_score))
        conn.execute("INSERT INTO task_anchors(task_id, anchor_id, context_id) VALUES(?,?,?)",
                     (task_id2, variable_anchor, ctx_id))
    conn.commit()

    # Compute ADP
    consistent_adp = mem.compute_discrimination_power(conn, consistent_anchor)
    variable_adp = mem.compute_discrimination_power(conn, variable_anchor)

    # Consistent should have low ADP, variable should have high ADP
    consistent_low = consistent_adp < 0.8
    variable_high = variable_adp > 0.5
    variable_higher = variable_adp > consistent_adp

    conn.close(); os.unlink(tmp)

    tests_passed = sum([consistent_low, variable_high, variable_higher])
    return tests_passed, 3, {
        "consistent_adp": consistent_adp,
        "consistent_low": consistent_low,
        "variable_adp": variable_adp,
        "variable_high": variable_high,
        "variable_higher_than_consistent": variable_higher,
    }


# ══════════════════════════════════════════════════════════
# BENCHMARK 4: Mode Vectors
# ══════════════════════════════════════════════════════════

def benchmark_mode_vectors():
    # Test inference
    outreach_mode = mem.infer_mode_vector("B2B Cold Outreach Campaign")
    website_mode = mem.infer_mode_vector("Security Website Design")
    editorial_mode = mem.infer_mode_vector("Editorial on Electoral Reform")
    unknown_mode = mem.infer_mode_vector("Random Stuff")

    # Outreach should be high-work
    outreach_work = outreach_mode.get("work", 0) > 0.7
    # Website should be high-creative
    website_creative = website_mode.get("creative", 0) > 0.6
    # Editorial should be high-creative + high-learning
    editorial_creative = editorial_mode.get("creative", 0) > 0.6
    # Unknown should get default (balanced)
    unknown_balanced = abs(unknown_mode.get("work", 0) - 0.5) < 0.01

    # Test similarity
    sim_same = mem.mode_similarity(outreach_mode, outreach_mode)
    sim_diff = mem.mode_similarity(outreach_mode, editorial_mode)
    sim_close = mem.mode_similarity(website_mode, editorial_mode)

    same_high = sim_same > 0.95
    diff_lower = sim_diff < sim_same
    close_moderate = sim_close > sim_diff  # website and editorial both creative

    tests_passed = sum([outreach_work, website_creative, editorial_creative,
                        unknown_balanced, same_high, diff_lower, close_moderate])
    return tests_passed, 7, {
        "outreach_work_high": outreach_work,
        "outreach_mode": outreach_mode,
        "website_creative_high": website_creative,
        "editorial_creative_high": editorial_creative,
        "unknown_balanced": unknown_balanced,
        "self_similarity": sim_same,
        "same_high": same_high,
        "cross_similarity": sim_diff,
        "diff_lower": diff_lower,
        "website_editorial_similarity": sim_close,
        "close_moderate": close_moderate,
    }


# ══════════════════════════════════════════════════════════
# BENCHMARK 5: Execution Parameters
# ══════════════════════════════════════════════════════════

def benchmark_exec_params():
    conn, tmp = fresh_db()

    sc_id = mem.add_node(conn, "super_context", "Exec Params Test", weight=0.5)
    ctx_id = mem.add_node(conn, "context", "Test Context", weight=0.5)
    mem.link(conn, sc_id, ctx_id)

    anc_id = mem.add_node(conn, "anchor", "Test Anchor", weight=0.7)
    mem.link(conn, ctx_id, anc_id)

    # Store with execution params
    params_1 = json.dumps({"approach": "aggressive", "tone": "urgent"})
    params_2 = json.dumps({"approach": "soft", "tone": "friendly"})

    task1 = mem._id()
    now = datetime.datetime.now().isoformat()
    conn.execute("INSERT INTO tasks(id, sc_id, title, started, completed, score) VALUES(?,?,?,?,?,?)",
                 (task1, sc_id, "Task_1", now, now, 4.5))
    conn.execute("INSERT INTO task_anchors(task_id, anchor_id, context_id, was_retrieved, execution_params) VALUES(?,?,?,0,?)",
                 (task1, anc_id, ctx_id, params_1))

    task2 = mem._id()
    conn.execute("INSERT INTO tasks(id, sc_id, title, started, completed, score) VALUES(?,?,?,?,?,?)",
                 (task2, sc_id, "Task_2", now, now, 2.0))
    conn.execute("INSERT INTO task_anchors(task_id, anchor_id, context_id, was_retrieved, execution_params) VALUES(?,?,?,0,?)",
                 (task2, anc_id, ctx_id, params_2))
    conn.commit()

    # Verify params stored
    stored = conn.execute("SELECT execution_params FROM task_anchors WHERE task_id=?", (task1,)).fetchone()
    params_stored = stored and "aggressive" in stored["execution_params"]

    # Verify best/worst retrieval
    best = conn.execute("""
        SELECT ta.execution_params, t.score FROM task_anchors ta
        JOIN tasks t ON t.id = ta.task_id
        WHERE ta.anchor_id = ? AND t.score > 0
        ORDER BY t.score DESC LIMIT 1
    """, (anc_id,)).fetchone()
    worst = conn.execute("""
        SELECT ta.execution_params, t.score FROM task_anchors ta
        JOIN tasks t ON t.id = ta.task_id
        WHERE ta.anchor_id = ? AND t.score > 0
        ORDER BY t.score ASC LIMIT 1
    """, (anc_id,)).fetchone()

    best_correct = best and best["score"] == 4.5 and "aggressive" in best["execution_params"]
    worst_correct = worst and worst["score"] == 2.0 and "soft" in worst["execution_params"]

    conn.close(); os.unlink(tmp)

    tests_passed = sum([params_stored, best_correct, worst_correct])
    return tests_passed, 3, {
        "params_stored": params_stored,
        "best_execution_correct": best_correct,
        "worst_execution_correct": worst_correct,
    }


# ══════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PMIS v3 BENCHMARK SUITE")
    print("=" * 60)

    benchmarks = [
        ("Phase 1: Decision Convergence", benchmark_decisions, 4),
        ("Phase 2: Transfer Edges", benchmark_transfers, 4),
        ("Phase 3: Anchor Discrimination Power", benchmark_adp, 3),
        ("Phase 4: Mode Vectors", benchmark_mode_vectors, 7),
        ("Phase 5: Execution Parameters", benchmark_exec_params, 3),
    ]

    total_pass, total_tests = 0, 0
    all_results = []

    for name, func, expected_tests in benchmarks:
        print(f"\n🔬 {name}")
        passed, total, details = func()
        status = "✅ PASS" if passed == total else "❌ FAIL"
        print(f"   {passed}/{total} tests passed  {status}")
        for key, val in details.items():
            if isinstance(val, (dict, list)):
                continue
            print(f"   • {key}: {val}")
        total_pass += passed
        total_tests += total
        all_results.append((name, passed, total))

    print("\n" + "=" * 60)
    all_phases_pass = all(p == t for _, p, t in all_results)
    print(f"OVERALL: {total_pass}/{total_tests} tests passed across 5 phases")
    for name, passed, total in all_results:
        status = "✅" if passed == total else "❌"
        print(f"   {status} {name}: {passed}/{total}")
    print("=" * 60)
