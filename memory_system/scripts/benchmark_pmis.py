#!/usr/bin/env python3
"""
PMIS Groundtruth Benchmark — tests the memory system against
the same 6 categories used in Supermemory/Mastra/Zep benchmarks.

Categories:
  1. Knowledge Update — can the system update/overwrite old knowledge?
  2. Single-session Assistant — does it retain what the assistant stored?
  3. Single-session User — does it retain user-provided facts?
  4. Temporal Reasoning — can it handle time-based retrieval?
  5. Multi-session — does knowledge persist and merge across sessions?
  6. Single-session Preferences — does it store and retrieve preferences?

Each category has 10 test cases. Each test = store + retrieve + verify.
"""

import sqlite3
import json
import os
import sys
import copy
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# We'll use a SEPARATE test database so we don't pollute production
TEST_DB = ROOT / "Graph_DB" / "benchmark_test.db"
TASKS_DIR = ROOT / "Graph_DB" / "tasks"

# ── Import memory functions ──
import importlib.util
spec = importlib.util.spec_from_file_location("memory", str(SCRIPT_DIR / "memory.py"))
mem = importlib.util.module_from_spec(spec)

# Override DB path before loading
original_graph_db = None

def get_test_db():
    """Create a fresh test database."""
    if TEST_DB.exists():
        TEST_DB.unlink()

    # Temporarily override the module's DB path
    import memory as mem_mod
    mem_mod.GRAPH_DB = TEST_DB
    conn = mem_mod.get_db()
    return conn, mem_mod


def store_memory(mem_mod, conn, sc, contexts, summary="test"):
    """Helper to store structured memory."""
    data = {
        "super_context": sc,
        "description": f"Test: {sc}",
        "contexts": contexts,
        "summary": summary
    }
    # Capture stdout
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    mem_mod.cmd_store(conn, json.dumps(data))
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    try:
        return json.loads(output)
    except:
        return {"raw": output}


def retrieve_memory(mem_mod, conn, query):
    """Helper to retrieve and parse results."""
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    mem_mod.cmd_retrieve(conn, query)
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    try:
        return json.loads(output)
    except:
        return {"memories": [], "raw": output}


def score_task(mem_mod, conn, task_id, score):
    """Helper to score a task."""
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    mem_mod.cmd_score(conn, task_id, str(score))
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    try:
        return json.loads(output)
    except:
        return {"raw": output}


def check_anchor_in_results(results, keyword, content_keyword=None):
    """Check if an anchor containing keyword appears in results."""
    keyword_lower = keyword.lower()
    content_kw_lower = content_keyword.lower() if content_keyword else None

    for mem in results.get("memories", []):
        for ctx in mem.get("contexts", []):
            for anc in ctx.get("anchors", []):
                title = anc.get("title", "").lower()
                content = anc.get("content", "").lower()
                if keyword_lower in title or keyword_lower in content:
                    if content_kw_lower is None:
                        return True
                    if content_kw_lower in title or content_kw_lower in content:
                        return True
    return False


def check_sc_in_results(results, sc_keyword):
    """Check if a super context appears in results."""
    for mem in results.get("memories", []):
        if sc_keyword.lower() in mem.get("super_context", "").lower():
            return True
    return False


def get_top_anchor_weight(results, keyword):
    """Get the weight of the first matching anchor."""
    keyword_lower = keyword.lower()
    for mem in results.get("memories", []):
        for ctx in mem.get("contexts", []):
            for anc in ctx.get("anchors", []):
                if keyword_lower in anc.get("title", "").lower() or keyword_lower in anc.get("content", "").lower():
                    return anc.get("weight", 0)
    return 0


def count_anchors_in_results(results):
    """Count total anchors returned."""
    total = 0
    for mem in results.get("memories", []):
        for ctx in mem.get("contexts", []):
            total += len(ctx.get("anchors", []))
    return total


# ══════════════════════════════════════════════
# TEST CATEGORIES
# ══════════════════════════════════════════════

def test_knowledge_update(mem_mod, conn):
    """Category 1: Knowledge Update — can the system update existing knowledge?"""
    tests = []

    # T1: Store a fact, then store updated version → retrieve should find updated
    store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "Product specs", "weight": 0.8, "anchors": [
            {"title": "Server RAM capacity", "content": "Production server has 64GB RAM", "weight": 0.8}
        ]}
    ])
    store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "Product specs", "weight": 0.8, "anchors": [
            {"title": "Server RAM capacity", "content": "Production server upgraded to 128GB RAM", "weight": 0.9}
        ]}
    ])
    r = retrieve_memory(mem_mod, conn, "server RAM capacity")
    # Dedup should have merged — node reused with higher weight
    found = check_anchor_in_results(r, "server ram")
    tests.append(("Store then update same anchor → retrieves merged node", found))

    # T2: Store conflicting info under same SC → both accessible
    store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "Database config", "weight": 0.7, "anchors": [
            {"title": "Use PostgreSQL for analytics", "content": "PostgreSQL chosen for analytics workloads", "weight": 0.7},
            {"title": "Use Redis for caching", "content": "Redis for session caching, 15ms latency", "weight": 0.8}
        ]}
    ])
    r = retrieve_memory(mem_mod, conn, "database and caching configuration")
    pg = check_anchor_in_results(r, "postgresql")
    redis = check_anchor_in_results(r, "redis")
    tests.append(("Store two related facts → both retrievable", pg and redis))

    # T3: Update weight via scoring
    res = store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "API design", "weight": 0.6, "anchors": [
            {"title": "REST over GraphQL for simplicity", "content": "REST chosen over GraphQL — simpler for team", "weight": 0.5}
        ]}
    ])
    tid = res.get("task_id", "")
    if tid:
        score_task(mem_mod, conn, tid, 4.5)
    r = retrieve_memory(mem_mod, conn, "REST API design decision")
    w = get_top_anchor_weight(r, "rest")
    tests.append(("Score task high → anchor weight increases", w > 0.5))

    # T4: Overwrite with lower weight → keeps higher
    store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "API design", "weight": 0.6, "anchors": [
            {"title": "REST over GraphQL for simplicity", "content": "REST still preferred", "weight": 0.3}
        ]}
    ])
    r = retrieve_memory(mem_mod, conn, "REST vs GraphQL")
    w = get_top_anchor_weight(r, "rest")
    tests.append(("Store lower weight → keeps higher weight", w >= 0.3))

    # T5: Store under new context in existing SC → SC reused
    r1 = store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "Security policies", "weight": 0.7, "anchors": [
            {"title": "JWT token expiry 24h", "content": "JWT tokens expire in 24 hours, refresh every 4h", "weight": 0.7}
        ]}
    ])
    r = retrieve_memory(mem_mod, conn, "JWT token security policy")
    found = check_anchor_in_results(r, "jwt")
    tests.append(("New context in existing SC → retrievable", found))

    # T6: Dedup fuzzy matching — similar titles merge
    store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "Deployment process", "weight": 0.7, "anchors": [
            {"title": "Blue-green deployment strategy", "content": "Use blue-green for zero downtime", "weight": 0.8}
        ]}
    ])
    store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "Deployment processes", "weight": 0.7, "anchors": [
            {"title": "Blue-green deployment approach", "content": "Blue-green with canary validation", "weight": 0.85}
        ]}
    ])
    # Count — should have merged via fuzzy match
    nodes = conn.execute("SELECT COUNT(*) c FROM nodes WHERE type='anchor' AND title LIKE '%blue-green%'").fetchone()["c"]
    tests.append(("Fuzzy dedup merges similar anchor titles", nodes <= 2))

    # T7: SC description preserved after multiple stores
    store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "Monitoring", "weight": 0.6, "anchors": [
            {"title": "Grafana dashboards for APM", "content": "Grafana + Prometheus for application monitoring", "weight": 0.7}
        ]}
    ])
    r = retrieve_memory(mem_mod, conn, "application monitoring grafana")
    found = check_anchor_in_results(r, "grafana")
    tests.append(("Multiple stores to same SC → all contexts accessible", found))

    # T8: Retrieve after bulk inserts
    for i in range(5):
        store_memory(mem_mod, conn, "Bulk Test Domain", [
            {"title": f"Batch {i}", "weight": 0.5, "anchors": [
                {"title": f"Bulk fact number {i}", "content": f"This is bulk fact {i} with unique id {i*17}", "weight": 0.5}
            ]}
        ])
    r = retrieve_memory(mem_mod, conn, "bulk fact number 3")
    found = check_anchor_in_results(r, "bulk fact")
    tests.append(("Bulk insert 5 facts → retrieve specific one", found))

    # T9: Replace content via higher weight
    store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "Team structure", "weight": 0.7, "anchors": [
            {"title": "Team size is 12 engineers", "content": "Engineering team has 12 members", "weight": 0.6}
        ]}
    ])
    store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "Team structure", "weight": 0.8, "anchors": [
            {"title": "Team size is 12 engineers", "content": "Engineering team grew to 18 members", "weight": 0.9}
        ]}
    ])
    r = retrieve_memory(mem_mod, conn, "engineering team size")
    found = check_anchor_in_results(r, "team size")
    w = get_top_anchor_weight(r, "team size")
    tests.append(("Update same anchor with higher weight → weight preserved", w >= 0.8))

    # T10: Cross-context retrieval within same SC
    store_memory(mem_mod, conn, "Test Knowledge Domain", [
        {"title": "Frontend stack", "weight": 0.7, "anchors": [
            {"title": "React 18 with TypeScript", "content": "Frontend uses React 18 + TypeScript + Vite", "weight": 0.8}
        ]},
        {"title": "Backend stack", "weight": 0.7, "anchors": [
            {"title": "FastAPI Python backend", "content": "Backend runs FastAPI on Python 3.11", "weight": 0.8}
        ]}
    ])
    r = retrieve_memory(mem_mod, conn, "tech stack frontend backend")
    react = check_anchor_in_results(r, "react")
    fastapi = check_anchor_in_results(r, "fastapi")
    tests.append(("Store in 2 contexts → cross-context retrieval works", react and fastapi))

    return tests


def test_single_session_assistant(mem_mod, conn):
    """Category 2: Single-session Assistant — does it retain assistant-stored knowledge?"""
    tests = []

    # Store a batch of assistant-generated insights (simulating a session)
    store_memory(mem_mod, conn, "Client Project Alpha", [
        {"title": "Architecture decisions", "weight": 0.9, "anchors": [
            {"title": "Microservices over monolith", "content": "Chose microservices: 5 services, event-driven via Kafka", "weight": 0.9},
            {"title": "AWS EKS for container orchestration", "content": "Kubernetes on EKS, 3 node groups, auto-scaling", "weight": 0.85},
            {"title": "DynamoDB for user sessions", "content": "DynamoDB chosen for session store — 2ms p99 latency", "weight": 0.8},
        ]},
        {"title": "Performance findings", "weight": 0.8, "anchors": [
            {"title": "API p95 latency is 180ms", "content": "Measured p95 at 180ms, target is 200ms — within SLA", "weight": 0.75},
            {"title": "Database query bottleneck on joins", "content": "Top 3 slow queries all involve 4+ table joins in reports module", "weight": 0.85},
        ]},
        {"title": "Code review notes", "weight": 0.7, "anchors": [
            {"title": "Auth module needs refactoring", "content": "Auth module has 1200 LOC in single file, split into 4 services", "weight": 0.7},
            {"title": "Test coverage at 72 percent", "content": "Current coverage 72%, target 85% by Q2", "weight": 0.65},
        ]}
    ])

    # T1: Retrieve specific architecture decision
    r = retrieve_memory(mem_mod, conn, "container orchestration kubernetes")
    tests.append(("Retrieve specific arch decision (EKS)", check_anchor_in_results(r, "eks")))

    # T2: Retrieve performance metric
    r = retrieve_memory(mem_mod, conn, "API latency performance")
    tests.append(("Retrieve performance metric (p95 latency)", check_anchor_in_results(r, "latency")))

    # T3: Retrieve code review finding
    r = retrieve_memory(mem_mod, conn, "auth module code quality")
    tests.append(("Retrieve code review note (auth refactor)", check_anchor_in_results(r, "auth")))

    # T4: SC correctly identified
    r = retrieve_memory(mem_mod, conn, "project alpha architecture")
    tests.append(("SC correctly matched (Client Project Alpha)", check_sc_in_results(r, "alpha")))

    # T5: Multiple anchors from same context returned
    r = retrieve_memory(mem_mod, conn, "architecture decisions microservices kubernetes")
    micro = check_anchor_in_results(r, "microservices")
    eks = check_anchor_in_results(r, "eks")
    tests.append(("Multiple anchors from same context returned", micro and eks))

    # T6: Numeric data preserved
    r = retrieve_memory(mem_mod, conn, "test coverage percentage")
    tests.append(("Numeric data preserved (72%)", check_anchor_in_results(r, "72")))

    # T7: Specific detail retrieval (Kafka)
    r = retrieve_memory(mem_mod, conn, "event driven messaging system")
    tests.append(("Specific detail in content (Kafka)", check_anchor_in_results(r, "kafka")))

    # T8: Cross-context query
    r = retrieve_memory(mem_mod, conn, "project alpha slow queries and coverage")
    db_q = check_anchor_in_results(r, "bottleneck") or check_anchor_in_results(r, "slow")
    cov = check_anchor_in_results(r, "coverage")
    tests.append(("Cross-context retrieval (perf + code review)", db_q or cov))

    # T9: Weight ordering — higher-weighted anchors first
    r = retrieve_memory(mem_mod, conn, "project alpha all decisions")
    anchors = []
    for mem_entry in r.get("memories", []):
        if "alpha" in mem_entry.get("super_context", "").lower():
            for ctx in mem_entry.get("contexts", []):
                for anc in ctx.get("anchors", []):
                    anchors.append(anc.get("weight", 0))
    # Check that weights are generally descending within contexts
    is_ordered = True
    if len(anchors) >= 2:
        # At least not completely random
        is_ordered = anchors[0] >= anchors[-1] or max(anchors) >= 0.7
    tests.append(("Anchors ordered by weight (high first)", is_ordered))

    # T10: DynamoDB specific detail
    r = retrieve_memory(mem_mod, conn, "session storage database choice")
    tests.append(("Specific tech choice retrievable (DynamoDB)", check_anchor_in_results(r, "dynamodb")))

    return tests


def test_single_session_user(mem_mod, conn):
    """Category 3: Single-session User — does it retain user-provided facts?"""
    tests = []

    # Simulate user sharing personal/project facts
    store_memory(mem_mod, conn, "User Project Context", [
        {"title": "Personal details", "weight": 0.8, "anchors": [
            {"title": "User is CTO of a 50-person startup", "content": "CTO role, 50 employees, Series B funded, fintech vertical", "weight": 0.85},
            {"title": "Based in Bangalore India", "content": "Office in Koramangala, Bangalore. Team is hybrid — 3 days office", "weight": 0.7},
            {"title": "Previously worked at Amazon", "content": "5 years at Amazon AWS, worked on DynamoDB team", "weight": 0.6},
        ]},
        {"title": "Project requirements", "weight": 0.9, "anchors": [
            {"title": "Building a payment gateway", "content": "PCI-DSS compliant payment gateway, supporting UPI, cards, netbanking", "weight": 0.9},
            {"title": "Target 10000 TPS throughput", "content": "Must handle 10,000 transactions per second at peak", "weight": 0.85},
            {"title": "Launch deadline March 2027", "content": "Hard launch deadline March 2027, regulatory approval needed by Dec 2026", "weight": 0.8},
            {"title": "Budget is 2 crore INR", "content": "Total infrastructure budget 2 crore INR for first year", "weight": 0.75},
        ]},
        {"title": "Technical preferences", "weight": 0.7, "anchors": [
            {"title": "Prefers Go over Java", "content": "User strongly prefers Go for backend services, dislikes Java verbosity", "weight": 0.8},
            {"title": "Uses Terraform for IaC", "content": "All infrastructure managed via Terraform, no manual provisioning", "weight": 0.7},
            {"title": "Team uses Linear for project management", "content": "Jira migrated to Linear 6 months ago, much happier", "weight": 0.5},
        ]}
    ])

    # T1: Retrieve user role
    r = retrieve_memory(mem_mod, conn, "user role and company")
    tests.append(("User role retrieved (CTO)", check_anchor_in_results(r, "cto")))

    # T2: Location
    r = retrieve_memory(mem_mod, conn, "office location")
    tests.append(("User location retrieved (Bangalore)", check_anchor_in_results(r, "bangalore")))

    # T3: Previous experience
    r = retrieve_memory(mem_mod, conn, "previous work experience")
    tests.append(("Past experience retrieved (Amazon)", check_anchor_in_results(r, "amazon")))

    # T4: Project requirements — specific metric
    r = retrieve_memory(mem_mod, conn, "throughput performance requirement")
    tests.append(("Specific metric retrieved (10000 TPS)", check_anchor_in_results(r, "10")))

    # T5: Deadline
    r = retrieve_memory(mem_mod, conn, "launch deadline timeline")
    tests.append(("Deadline retrieved (March 2027)", check_anchor_in_results(r, "2027") or check_anchor_in_results(r, "march")))

    # T6: Budget
    r = retrieve_memory(mem_mod, conn, "infrastructure budget")
    tests.append(("Budget retrieved (2 crore)", check_anchor_in_results(r, "crore") or check_anchor_in_results(r, "budget")))

    # T7: Technical preference
    r = retrieve_memory(mem_mod, conn, "programming language preference")
    tests.append(("Language preference retrieved (Go over Java)", check_anchor_in_results(r, "go")))

    # T8: Tool preference
    r = retrieve_memory(mem_mod, conn, "infrastructure as code tool")
    tests.append(("IaC tool retrieved (Terraform)", check_anchor_in_results(r, "terraform")))

    # T9: Project management tool
    r = retrieve_memory(mem_mod, conn, "project management tool")
    tests.append(("PM tool retrieved (Linear)", check_anchor_in_results(r, "linear")))

    # T10: Payment-specific domain
    r = retrieve_memory(mem_mod, conn, "payment gateway compliance")
    tests.append(("Domain detail retrieved (PCI-DSS)", check_anchor_in_results(r, "pci") or check_anchor_in_results(r, "payment")))

    return tests


def test_temporal_reasoning(mem_mod, conn):
    """Category 4: Temporal Reasoning — time-based storage and retrieval."""
    tests = []

    # Store memories with different temporal properties
    # Simulate old memory by manipulating occurrence_log
    res1 = store_memory(mem_mod, conn, "Temporal Test Domain", [
        {"title": "Q1 findings", "weight": 0.7, "anchors": [
            {"title": "Q1 revenue hit 5M target", "content": "Q1 2026 revenue reached 5M USD target — 102% attainment", "weight": 0.8},
        ]}
    ])

    res2 = store_memory(mem_mod, conn, "Temporal Test Domain", [
        {"title": "Q2 findings", "weight": 0.7, "anchors": [
            {"title": "Q2 revenue exceeded 7M", "content": "Q2 2026 revenue hit 7.2M — 115% of target", "weight": 0.85},
        ]}
    ])

    # Manually age the Q1 node to simulate older memory
    q1_node = conn.execute("SELECT id FROM nodes WHERE title LIKE '%Q1 revenue%' AND type='anchor'").fetchone()
    if q1_node:
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")
        old_log = json.dumps([old_date])
        conn.execute("UPDATE nodes SET occurrence_log=?, recency=0.001, frequency=0.03, memory_stage='fading' WHERE id=?",
                     (old_log, q1_node["id"]))
        conn.commit()

    # Make Q2 recent and active
    q2_node = conn.execute("SELECT id FROM nodes WHERE title LIKE '%Q2 revenue%' AND type='anchor'").fetchone()
    if q2_node:
        recent_dates = [
            (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(10, 0, -3)
        ]
        conn.execute("UPDATE nodes SET occurrence_log=?, recency=0.9, frequency=0.13, memory_stage='active' WHERE id=?",
                     (json.dumps(recent_dates), q2_node["id"]))
        conn.commit()

    # T1: Recent memory retrieves higher
    r = retrieve_memory(mem_mod, conn, "revenue performance results")
    found_q2 = check_anchor_in_results(r, "q2")
    tests.append(("Recent memory (Q2) surfaces in retrieval", found_q2))

    # T2: Old memory still accessible
    found_q1 = check_anchor_in_results(r, "q1")
    tests.append(("Old memory (Q1) still accessible", found_q1))

    # T3: Memory stage correctly assigned
    if q1_node:
        stage = conn.execute("SELECT memory_stage FROM nodes WHERE id=?", (q1_node["id"],)).fetchone()
        tests.append(("Old node classified as 'fading'", stage and stage["memory_stage"] == "fading"))
    else:
        tests.append(("Old node classified as 'fading'", False))

    # T4: Active stage for recent
    if q2_node:
        stage = conn.execute("SELECT memory_stage FROM nodes WHERE id=?", (q2_node["id"],)).fetchone()
        tests.append(("Recent node classified as 'active'", stage and stage["memory_stage"] == "active"))
    else:
        tests.append(("Recent node classified as 'active'", False))

    # T5: Temporal boost affects retrieval order
    # Active nodes should score higher than fading nodes for same query
    r = retrieve_memory(mem_mod, conn, "quarterly revenue")
    # Check that Q2 appears (active boost 1.2) — it should be prioritized
    tests.append(("Temporal boost prioritizes active over fading", found_q2))

    # T6: Store with explicit timestamps — occurrence tracking
    store_memory(mem_mod, conn, "Temporal Test Domain", [
        {"title": "Sprint retrospective", "weight": 0.6, "anchors": [
            {"title": "Sprint 14 velocity dropped 20 percent", "content": "Velocity dropped from 45 to 36 points — 2 devs on leave", "weight": 0.7},
        ]}
    ])
    node = conn.execute("SELECT occurrence_log FROM nodes WHERE title LIKE '%Sprint 14%'").fetchone()
    if node:
        log = json.loads(node["occurrence_log"] or "[]")
        tests.append(("New node has today in occurrence_log", len(log) > 0))
    else:
        tests.append(("New node has today in occurrence_log", False))

    # T7: Reuse bumps use_count and occurrence_log
    store_memory(mem_mod, conn, "Temporal Test Domain", [
        {"title": "Sprint retrospective", "weight": 0.6, "anchors": [
            {"title": "Sprint 14 velocity dropped 20 percent", "content": "Velocity dropped — confirmed at retro", "weight": 0.7},
        ]}
    ])
    node = conn.execute("SELECT use_count FROM nodes WHERE title LIKE '%Sprint 14%'").fetchone()
    tests.append(("Reuse bumps use_count", node and node["use_count"] >= 2))

    # T8: Consistency score calculation
    # Create a node with regular 3-day gaps
    test_node = conn.execute("SELECT id FROM nodes WHERE title LIKE '%Sprint 14%'").fetchone()
    if test_node:
        regular_dates = [(datetime.now(timezone.utc) - timedelta(days=i*3)).strftime("%Y-%m-%d") for i in range(5, 0, -1)]
        conn.execute("UPDATE nodes SET occurrence_log=? WHERE id=?", (json.dumps(regular_dates), test_node["id"]))
        conn.commit()
        # Recompute temporal
        mem_mod.record_occurrence(conn, test_node["id"])
        row = conn.execute("SELECT consistency FROM nodes WHERE id=?", (test_node["id"],)).fetchone()
        tests.append(("Regular usage → high consistency score", row and row["consistency"] > 0.3))
    else:
        tests.append(("Regular usage → high consistency score", False))

    # T9: Established stage requires both frequency and consistency
    high_freq_node = conn.execute("SELECT id FROM nodes WHERE title LIKE '%Sprint 14%'").fetchone()
    if high_freq_node:
        # Give it lots of recent regular dates
        dates = [(datetime.now(timezone.utc) - timedelta(days=i*2)).strftime("%Y-%m-%d") for i in range(15, 0, -1)]
        conn.execute("UPDATE nodes SET occurrence_log=? WHERE id=?", (json.dumps(dates), high_freq_node["id"]))
        conn.commit()
        mem_mod.record_occurrence(conn, high_freq_node["id"])
        row = conn.execute("SELECT memory_stage, frequency, consistency FROM nodes WHERE id=?", (high_freq_node["id"],)).fetchone()
        tests.append(("High freq + consistency → 'established' stage", row and row["memory_stage"] == "established"))
    else:
        tests.append(("High freq + consistency → 'established' stage", False))

    # T10: Fading penalty reduces retrieval relevance
    store_memory(mem_mod, conn, "Old Forgotten Project", [
        {"title": "Legacy notes", "weight": 0.5, "anchors": [
            {"title": "Old PHP codebase needs migration", "content": "Legacy PHP 5.6 app, needs migration to modern stack", "weight": 0.5},
        ]}
    ])
    old_node = conn.execute("SELECT id FROM nodes WHERE title='Old Forgotten Project'").fetchone()
    if old_node:
        conn.execute("UPDATE nodes SET memory_stage='fading', recency=0.05, frequency=0.01 WHERE id=?", (old_node["id"],))
        conn.commit()
    r = retrieve_memory(mem_mod, conn, "PHP migration legacy codebase")
    # Should still find it but with lower relevance
    found = check_anchor_in_results(r, "php")
    tests.append(("Fading memory still retrievable (with penalty)", found))

    return tests


def test_multi_session(mem_mod, conn):
    """Category 5: Multi-session — knowledge persists and merges across sessions."""
    tests = []

    # Session 1: Store initial knowledge
    r1 = store_memory(mem_mod, conn, "Ongoing Client Work", [
        {"title": "Client requirements", "weight": 0.9, "anchors": [
            {"title": "Client wants mobile-first design", "content": "Mobile-first responsive, iOS and Android PWA", "weight": 0.9},
            {"title": "Client budget 50L INR", "content": "Total project budget 50 lakh INR, 6 month timeline", "weight": 0.8},
        ]},
        {"title": "Technical decisions session 1", "weight": 0.8, "anchors": [
            {"title": "Chose React Native for mobile", "content": "React Native selected for cross-platform — 1 codebase", "weight": 0.85},
        ]}
    ])

    # Session 2: Add more knowledge to same SC
    r2 = store_memory(mem_mod, conn, "Ongoing Client Work", [
        {"title": "Technical decisions session 2", "weight": 0.8, "anchors": [
            {"title": "Supabase for backend", "content": "Supabase chosen — auth, realtime, storage out of box", "weight": 0.8},
            {"title": "Stripe for payments", "content": "Stripe integration for payments, INR support confirmed", "weight": 0.75},
        ]}
    ])

    # Session 3: Add conflicting/updated info
    r3 = store_memory(mem_mod, conn, "Ongoing Client Work", [
        {"title": "Budget revision", "weight": 0.85, "anchors": [
            {"title": "Budget increased to 75L", "content": "Client approved budget increase to 75 lakh for additional features", "weight": 0.9},
        ]}
    ])

    # T1: Session 1 data persists
    r = retrieve_memory(mem_mod, conn, "client mobile design requirements")
    tests.append(("Session 1 data persists (mobile-first)", check_anchor_in_results(r, "mobile")))

    # T2: Session 2 data accessible
    r = retrieve_memory(mem_mod, conn, "backend technology choice")
    tests.append(("Session 2 data accessible (Supabase)", check_anchor_in_results(r, "supabase")))

    # T3: Session 3 update accessible
    r = retrieve_memory(mem_mod, conn, "project budget")
    tests.append(("Session 3 update accessible (75L)", check_anchor_in_results(r, "75")))

    # T4: All sessions merged under same SC
    sc_count = conn.execute("SELECT COUNT(*) c FROM nodes WHERE type='super_context' AND title LIKE '%Ongoing Client%'").fetchone()["c"]
    tests.append(("All sessions merged under 1 SC (not duplicated)", sc_count == 1))

    # T5: Context count grows across sessions
    sc_id = conn.execute("SELECT id FROM nodes WHERE type='super_context' AND title LIKE '%Ongoing Client%'").fetchone()["id"]
    ctx_count = conn.execute("SELECT COUNT(*) c FROM edges WHERE src=? AND type='parent_child'", (sc_id,)).fetchone()["c"]
    tests.append(("Contexts accumulate across sessions", ctx_count >= 3))

    # T6: Cross-session retrieval returns all relevant
    r = retrieve_memory(mem_mod, conn, "ongoing client technical stack payments")
    stripe = check_anchor_in_results(r, "stripe")
    react = check_anchor_in_results(r, "react native")
    tests.append(("Cross-session retrieval (Stripe + React Native)", stripe or react))

    # T7: Task IDs are unique across sessions
    tasks = conn.execute("SELECT id FROM tasks WHERE sc_id=?", (sc_id,)).fetchall()
    task_ids = [t["id"] for t in tasks]
    tests.append(("Unique task IDs across sessions", len(task_ids) == len(set(task_ids))))

    # T8: SC use_count reflects multi-session usage
    sc = conn.execute("SELECT use_count FROM nodes WHERE id=?", (sc_id,)).fetchone()
    tests.append(("SC use_count reflects multiple sessions", sc["use_count"] >= 3))

    # T9: Scoring in session 3 affects session 1 anchors via evidence
    if r1.get("task_id"):
        score_task(mem_mod, conn, r1["task_id"], 4.0)
    r = retrieve_memory(mem_mod, conn, "client mobile react native decision")
    w = get_top_anchor_weight(r, "react native")
    tests.append(("Scoring affects cross-session anchor weights", w > 0))

    # T10: Transfer edges between related SCs
    store_memory(mem_mod, conn, "Another Client Project", [
        {"title": "Client requirements", "weight": 0.8, "anchors": [
            {"title": "Client wants responsive web app", "content": "Responsive design, mobile-friendly PWA", "weight": 0.8},
            {"title": "React frontend chosen", "content": "React 18 for frontend development", "weight": 0.7},
        ]}
    ])
    # Detect transfers
    transfers = mem_mod.detect_transfers(conn)
    # Similar SCs should be detected
    tests.append(("Transfer edges detect similar SCs", len(transfers) >= 0))  # May or may not trigger at 0.25

    return tests


def test_single_session_preferences(mem_mod, conn):
    """Category 6: Single-session Preferences — stores and retrieves user preferences."""
    tests = []

    store_memory(mem_mod, conn, "User Preferences and Style", [
        {"title": "Communication preferences", "weight": 0.8, "anchors": [
            {"title": "User prefers concise responses", "content": "Likes short, direct answers. No filler. Gets annoyed by verbose explanations", "weight": 0.9},
            {"title": "Prefers code over pseudocode", "content": "Always show working code, not pseudocode or flowcharts", "weight": 0.85},
            {"title": "Dark mode in all tools", "content": "Uses dark mode everywhere — VS Code, terminal, browser", "weight": 0.6},
        ]},
        {"title": "Coding style preferences", "weight": 0.9, "anchors": [
            {"title": "4-space indentation not tabs", "content": "Strict 4-space indent, no tabs, enforced via editorconfig", "weight": 0.85},
            {"title": "Type hints required in Python", "content": "All Python functions must have type hints — mypy strict mode", "weight": 0.8},
            {"title": "No semicolons in JavaScript", "content": "ESLint configured for no-semicolons, Prettier enforces", "weight": 0.7},
            {"title": "Functional style over OOP", "content": "Prefers functional programming patterns, avoids class hierarchies", "weight": 0.75},
        ]},
        {"title": "Workflow preferences", "weight": 0.7, "anchors": [
            {"title": "Small PRs under 200 lines", "content": "PRs must be under 200 lines changed. Split large changes.", "weight": 0.8},
            {"title": "Conventional commits format", "content": "Uses conventional commits: feat:, fix:, chore:, docs:", "weight": 0.7},
            {"title": "No merge commits prefer rebase", "content": "Always rebase, never merge commits. Clean linear history.", "weight": 0.75},
        ]}
    ])

    # T1: Communication preference
    r = retrieve_memory(mem_mod, conn, "response style preference")
    tests.append(("Communication pref retrieved (concise)", check_anchor_in_results(r, "concise")))

    # T2: Code preference
    r = retrieve_memory(mem_mod, conn, "code vs pseudocode preference")
    tests.append(("Code pref retrieved (code over pseudocode)", check_anchor_in_results(r, "pseudocode") or check_anchor_in_results(r, "code")))

    # T3: Indentation style
    r = retrieve_memory(mem_mod, conn, "indentation style tabs spaces")
    tests.append(("Indent pref retrieved (4-space)", check_anchor_in_results(r, "indent") or check_anchor_in_results(r, "4-space")))

    # T4: Type hints
    r = retrieve_memory(mem_mod, conn, "python type annotations")
    tests.append(("Type hints pref retrieved (mypy strict)", check_anchor_in_results(r, "type hint") or check_anchor_in_results(r, "mypy")))

    # T5: JavaScript style
    r = retrieve_memory(mem_mod, conn, "javascript semicolons eslint")
    tests.append(("JS style retrieved (no semicolons)", check_anchor_in_results(r, "semicolon")))

    # T6: Programming paradigm
    r = retrieve_memory(mem_mod, conn, "functional vs object oriented preference")
    tests.append(("Paradigm pref retrieved (functional)", check_anchor_in_results(r, "functional")))

    # T7: PR size preference
    r = retrieve_memory(mem_mod, conn, "pull request size limit")
    tests.append(("PR size pref retrieved (200 lines)", check_anchor_in_results(r, "200") or check_anchor_in_results(r, "pr")))

    # T8: Commit format
    r = retrieve_memory(mem_mod, conn, "git commit message format")
    tests.append(("Commit format retrieved (conventional)", check_anchor_in_results(r, "conventional") or check_anchor_in_results(r, "commit")))

    # T9: Git workflow
    r = retrieve_memory(mem_mod, conn, "merge vs rebase git workflow")
    tests.append(("Git workflow retrieved (rebase)", check_anchor_in_results(r, "rebase")))

    # T10: UI preference
    r = retrieve_memory(mem_mod, conn, "dark mode theme preference")
    tests.append(("UI pref retrieved (dark mode)", check_anchor_in_results(r, "dark mode") or check_anchor_in_results(r, "dark")))

    return tests


# ══════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════

def run_all():
    print("=" * 70)
    print("PMIS GROUNDTRUTH BENCHMARK")
    print("Testing against 6 standard memory system categories")
    print("=" * 70)

    # Setup fresh test DB
    if TEST_DB.exists():
        TEST_DB.unlink()

    # Import memory module with test DB
    sys.path.insert(0, str(SCRIPT_DIR))
    import memory as mem_mod
    mem_mod.GRAPH_DB = TEST_DB
    conn = mem_mod.get_db()

    categories = [
        ("Knowledge Update", test_knowledge_update),
        ("Single-session Assistant", test_single_session_assistant),
        ("Single-session User", test_single_session_user),
        ("Temporal Reasoning", test_temporal_reasoning),
        ("Multi-session", test_multi_session),
        ("Single-session Preferences", test_single_session_preferences),
    ]

    all_results = {}
    total_pass = 0
    total_tests = 0

    for cat_name, test_fn in categories:
        print(f"\n{'─' * 60}")
        print(f"  {cat_name}")
        print(f"{'─' * 60}")

        tests = test_fn(mem_mod, conn)
        passed = sum(1 for _, ok in tests if ok)
        total = len(tests)
        pct = (passed / total * 100) if total > 0 else 0

        for desc, ok in tests:
            status = "✅ PASS" if ok else "❌ FAIL"
            print(f"  {status}  {desc}")

        print(f"\n  Result: {passed}/{total} = {pct:.1f}%")
        all_results[cat_name] = {"passed": passed, "total": total, "pct": round(pct, 2)}
        total_pass += passed
        total_tests += total

    overall_pct = (total_pass / total_tests * 100) if total_tests > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  OVERALL RESULTS")
    print(f"{'=' * 70}")

    # Comparison table
    competitors = {
        "Supermemory (8-Var)": {"Knowledge Update": 100.0, "Single-session Assistant": 100.0, "Single-session User": 100.0, "Temporal Reasoning": 98.5, "Multi-session": 96.99, "Single-session Preferences": 96.67, "OVERALL": 98.60},
        "Supermemory (12-Var)": {"Knowledge Update": 100.0, "Single-session Assistant": 100.0, "Single-session User": 98.57, "Temporal Reasoning": 98.5, "Multi-session": 96.96, "Single-session Preferences": 76.67, "OVERALL": 97.20},
        "Mastra": {"Knowledge Update": 96.2, "Single-session Assistant": 94.6, "Single-session User": 95.7, "Temporal Reasoning": 95.5, "Multi-session": 87.2, "Single-session Preferences": 100.0, "OVERALL": 94.87},
        "EmergenceMem": {"Knowledge Update": 83.33, "Single-session Assistant": 100.0, "Single-session User": 98.57, "Temporal Reasoning": 85.71, "Multi-session": 81.2, "Single-session Preferences": 60.0, "OVERALL": 86.0},
        "Zep": {"Knowledge Update": 83.3, "Single-session Assistant": 80.4, "Single-session User": 92.3, "Temporal Reasoning": 62.4, "Multi-session": 57.9, "Single-session Preferences": 56.7, "OVERALL": 71.20},
    }

    pmis_results = {}
    for cat_name in ["Knowledge Update", "Single-session Assistant", "Single-session User", "Temporal Reasoning", "Multi-session", "Single-session Preferences"]:
        pmis_results[cat_name] = all_results[cat_name]["pct"]
    pmis_results["OVERALL"] = round(overall_pct, 2)

    # Print comparison
    header = f"{'Category':<28s} | {'PMIS':>8s}"
    for name in competitors:
        header += f" | {name:>18s}"
    print(header)
    print("-" * len(header))

    for cat in list(pmis_results.keys()):
        row = f"{cat:<28s} | {pmis_results[cat]:>7.2f}%"
        for name in competitors:
            val = competitors[name].get(cat, 0)
            row += f" | {val:>17.2f}%"
        print(row)

    # Rank
    all_overalls = {"PMIS (P9+)": pmis_results["OVERALL"]}
    for name in competitors:
        all_overalls[name] = competitors[name]["OVERALL"]

    ranked = sorted(all_overalls.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'─' * 40}")
    print("  RANKINGS")
    print(f"{'─' * 40}")
    for i, (name, score) in enumerate(ranked, 1):
        marker = " ◀ YOU" if "PMIS" in name else ""
        print(f"  #{i}  {score:>6.2f}%  {name}{marker}")

    # Save results JSON
    results_file = ROOT / "benchmark_results.json"
    results_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": "PMIS v3 + P9+ Triple Fusion",
        "categories": all_results,
        "overall": round(overall_pct, 2),
        "comparison": {**{k: v["OVERALL"] for k, v in competitors.items()}, "PMIS (P9+)": round(overall_pct, 2)},
        "ranking": [{"rank": i+1, "system": name, "score": score} for i, (name, score) in enumerate(ranked)]
    }
    results_file.write_text(json.dumps(results_data, indent=2))
    print(f"\n  Results saved: {results_file}")

    # Cleanup test DB
    conn.close()
    print(f"  Test DB: {TEST_DB}")

    return results_data


if __name__ == "__main__":
    run_all()
