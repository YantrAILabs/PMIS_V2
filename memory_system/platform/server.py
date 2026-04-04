#!/usr/bin/env python3
"""
PMIS Platform — FastAPI Server
Multi-user memory platform with sharing and access control.

Usage:
    cd platform && python3 server.py
    → http://localhost:8000
    → http://localhost:8000/docs (API docs)
"""

import sys
import os
import json
import io
import sqlite3
from pathlib import Path
from contextlib import redirect_stdout
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Add paths — no name clash workarounds needed since platform/db.py → platform_db.py
PLATFORM_DIR = Path(__file__).parent
PMIS_V2_DIR = PLATFORM_DIR.parent.parent / "pmis_v2"
if not PMIS_V2_DIR.exists():
    PMIS_V2_DIR = Path.home() / "Desktop" / "memory" / "pmis_v2"

sys.path.insert(0, str(PLATFORM_DIR))
sys.path.insert(0, str(PLATFORM_DIR.parent / "scripts"))
sys.path.insert(0, str(PMIS_V2_DIR))

from platform_db import (get_users_db, get_user_memory_db, get_user_memory_path,
                          get_shared_scs_for_user, check_access, has_approved_request,
                          _now, _uuid, MEMORIES_DIR)
from platform_auth import register_user, login_user, get_user_from_token, list_users

# PMIS V2 imports (sole backend) — clean import, no name clash
_pmis_v2_available = False
PMISV2DBManager = None
try:
    from db.manager import DBManager as PMISV2DBManager
    _pmis_v2_available = True
except ImportError:
    pass


def _get_pmis_v2_db(username: str = "default"):
    """Get a PMIS V2 DBManager for a user."""
    if not _pmis_v2_available:
        return None
    db_path = str(PMIS_V2_DIR / "data" / "memory.db")
    return PMISV2DBManager(db_path=db_path)

app = FastAPI(title="PMIS Platform", version="1.0", description="Multi-user Memory Intelligence")

# Mount team sync routes
try:
    from sync_routes import router as sync_router
    app.include_router(sync_router)
except ImportError:
    pass  # sync_routes not available


# ══════════════════════════════════════
# AUTH DEPENDENCY
# ══════════════════════════════════════

async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    token = authorization.replace("Bearer ", "")
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(401, "Invalid or expired token")
    return user


# ══════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════

class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class StoreRequest(BaseModel):
    super_context: str
    description: Optional[str] = ""
    contexts: list = []
    summary: Optional[str] = ""

class ScoreRequest(BaseModel):
    task_id: str
    score: float

class ShareRequest(BaseModel):
    scope_type: str  # full_sc | context | anchor
    scope_id: str
    scope_title: Optional[str] = ""
    target_type: str = "org"  # org | user
    target_id: Optional[str] = None
    access_level: str = "read"  # read | request_based | full

class AccessRequestCreate(BaseModel):
    owner_id: str
    scope_id: str
    scope_title: Optional[str] = ""
    task_context: Optional[str] = ""

class AccessRequestResolve(BaseModel):
    status: str  # approved | denied

class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    company: str = ""
    deadline: str = ""
    expected_hours: float = 0
    owner: str = ""

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    company: Optional[str] = None
    status: Optional[str] = None
    deadline: Optional[str] = None
    expected_hours: Optional[float] = None
    sc_node_id: Optional[str] = None

class DeliverableCreate(BaseModel):
    name: str
    description: str = ""
    deadline: str = ""
    expected_hours: float = 0


# ══════════════════════════════════════
# HEALTH (no auth, for LAN discovery)
# ══════════════════════════════════════

@app.get("/health")
async def root_health():
    """Quick health check for LAN discovery. No auth needed."""
    return {"status": "ok", "service": "yantrai-memory-hub"}


# ══════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════

@app.post("/api/auth/register")
async def api_register(req: RegisterRequest):
    result = register_user(req.username, req.password, req.display_name)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result

@app.post("/api/auth/login")
async def api_login(req: LoginRequest):
    result = login_user(req.username, req.password)
    if "error" in result:
        raise HTTPException(401, result["error"])
    return result

@app.get("/api/auth/me")
async def api_me(user=Depends(get_current_user)):
    return user

@app.get("/api/auth/users")
async def api_users(user=Depends(get_current_user)):
    return {"users": list_users()}


# ══════════════════════════════════════
# MEMORY ROUTES (user-scoped)
# ══════════════════════════════════════

def _run_memory_cmd(username, cmd_fn, *args):
    """Run a memory.py command against a user's DB."""
    import memory as mem

    conn = get_user_memory_db(username)
    f = io.StringIO()
    try:
        with redirect_stdout(f):
            cmd_fn(conn, *args)
        output = f.getvalue()
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"output": output}
    finally:
        conn.close()


@app.post("/api/memory/store")
async def api_store(req: StoreRequest, user=Depends(get_current_user)):
    import memory as mem
    data = req.model_dump()
    result = _run_memory_cmd(user["username"], mem.cmd_store, json.dumps(data))
    return result


@app.get("/api/memory/retrieve")
async def api_retrieve(q: str, user=Depends(get_current_user)):
    import memory as mem
    result = _run_memory_cmd(user["username"], mem.cmd_retrieve, q)
    return result


@app.get("/api/memory/browse")
async def api_browse(user=Depends(get_current_user)):
    """Browse memory tree — uses PMIS V2 if available, falls back to V1."""
    v2 = _get_pmis_v2_db()
    if v2:
        try:
            scs = v2.get_nodes_by_level("SC")
            result = []
            for sc in scs:
                children = v2.get_children(sc["id"])
                anc_count = 0
                ctx_list = []
                for ctx in children:
                    anchors = v2.get_children(ctx["id"])
                    anc_count += len(anchors)
                    ctx_list.append({
                        "id": ctx["id"],
                        "title": ctx["content"][:80],
                        "anchors": [{"id": a["id"], "title": a["content"][:100],
                                     "precision": a.get("precision", 0),
                                     "access_count": a.get("access_count", 0)}
                                    for a in anchors],
                    })
                result.append({
                    "id": sc["id"], "title": sc["content"][:80],
                    "contexts": len(children), "anchors": anc_count,
                    "quality": sc.get("precision", 0),
                    "uses": sc.get("access_count", 0),
                    "stage": "established" if sc.get("access_count", 0) > 5 else "active",
                    "prod_time": sc.get("productivity_time_mins", 0),
                    "prod_human": sc.get("productivity_human_mins", 0),
                    "prod_ai": sc.get("productivity_ai_mins", 0),
                    "context_list": ctx_list,
                })
            result.sort(key=lambda x: x.get("prod_time", 0) + x.get("uses", 0), reverse=True)
            v2.close()
            return {"super_contexts": result, "total": len(result), "source": "pmis_v2"}
        except Exception as e:
            v2.close()
            pass
    # Fallback to V1
    import memory as mem
    result = _run_memory_cmd(user["username"], mem.cmd_browse)
    return result


@app.get("/api/memory/tree")
async def api_tree(user=Depends(get_current_user)):
    import memory as mem
    result = _run_memory_cmd(user["username"], mem.cmd_tree)
    return result


@app.get("/api/memory/stats")
async def api_stats(user=Depends(get_current_user)):
    import memory as mem
    result = _run_memory_cmd(user["username"], mem.cmd_stats)
    return result


@app.post("/api/memory/score")
async def api_score(req: ScoreRequest, user=Depends(get_current_user)):
    import memory as mem
    result = _run_memory_cmd(user["username"], mem.cmd_score, req.task_id, str(req.score))
    return result


@app.get("/api/memory/health")
async def api_health(user=Depends(get_current_user)):
    """Full memory health report: stages, weights, activity, versions, benchmarks."""
    import memory as mem
    conn = get_user_memory_db(user["username"])
    try:
        # Node counts
        counts = {}
        for t in ("super_context", "context", "anchor"):
            counts[t] = conn.execute("SELECT COUNT(*) c FROM nodes WHERE type=?", (t,)).fetchone()["c"]

        # Stage distribution
        stages = {}
        for r in conn.execute("SELECT memory_stage, COUNT(*) c FROM nodes GROUP BY memory_stage").fetchall():
            stages[r["memory_stage"]] = r["c"]

        # Weight distribution (anchors only)
        weight_buckets = [0] * 10
        for r in conn.execute("SELECT weight FROM nodes WHERE type='anchor'").fetchall():
            idx = min(int(r["weight"] * 10), 9)
            weight_buckets[idx] += 1

        # Task activity log
        tasks = []
        for r in conn.execute("SELECT id, title, started, completed, score FROM tasks ORDER BY started DESC LIMIT 20").fetchall():
            tasks.append(dict(r))

        # Task score stats
        scored = conn.execute("SELECT COUNT(*) c, AVG(score) avg, MAX(score) best, MIN(score) worst FROM tasks WHERE score > 0").fetchone()

        # Top anchors by weight
        top_anchors = []
        for r in conn.execute("SELECT title, weight, memory_stage, use_count FROM nodes WHERE type='anchor' ORDER BY weight DESC LIMIT 10").fetchall():
            top_anchors.append(dict(r))

        # Weakest anchors
        weak_anchors = []
        for r in conn.execute("SELECT title, weight, memory_stage, use_count FROM nodes WHERE type='anchor' AND weight < 0.3 ORDER BY weight ASC LIMIT 10").fetchall():
            weak_anchors.append(dict(r))

        # Temporal health
        avg_temps = conn.execute("SELECT AVG(recency) r, AVG(frequency) f, AVG(consistency) c FROM nodes WHERE type='anchor'").fetchone()

        result = {
            "counts": counts,
            "stages": stages,
            "weight_distribution": weight_buckets,
            "tasks": tasks,
            "scored_tasks": scored["c"] or 0,
            "avg_score": round(scored["avg"] or 0, 2),
            "best_score": scored["best"] or 0,
            "worst_score": scored["worst"] or 0,
            "top_anchors": top_anchors,
            "weak_anchors": weak_anchors,
            "avg_recency": round(avg_temps["r"] or 0, 3),
            "avg_frequency": round(avg_temps["f"] or 0, 3),
            "avg_consistency": round(avg_temps["c"] or 0, 3),
        }

        # Versions (from main memory folder)
        ver_file = PLATFORM_DIR.parent / "versions" / "VERSION_REGISTRY.json"
        if ver_file.exists():
            result["versions"] = json.loads(ver_file.read_text()).get("versions", [])

        # Benchmark results
        bench_file = PLATFORM_DIR.parent / "longmemeval_results.json"
        if bench_file.exists():
            bench = json.loads(bench_file.read_text())
            result["benchmark"] = {
                "overall": bench.get("overall_hit_rate", 0),
                "recall_at_5": bench.get("overall_recall_at_5", 0),
                "categories": bench.get("by_category", {}),
                "total_questions": bench.get("n_questions", 0),
            }

        return result
    finally:
        conn.close()


class SyncToPortalRequest(BaseModel):
    source_db_path: Optional[str] = None

@app.post("/api/memory/sync-from-local")
async def api_sync_from_local(user=Depends(get_current_user)):
    """Sync/push the main local memory DB into the user's platform memory."""
    import memory as mem

    local_db = PLATFORM_DIR.parent / "Graph_DB" / "graph.db"
    if not local_db.exists():
        raise HTTPException(404, "Local graph.db not found")

    # Read from local DB (read-only)
    local_conn = sqlite3.connect(str(local_db))
    local_conn.row_factory = sqlite3.Row

    scs = [dict(r) for r in local_conn.execute("SELECT * FROM nodes WHERE type='super_context'").fetchall()]
    edges = [dict(r) for r in local_conn.execute("SELECT * FROM edges WHERE type='parent_child'").fetchall()]
    all_nodes = {r["id"]: dict(r) for r in local_conn.execute("SELECT * FROM nodes").fetchall()}
    local_conn.close()

    children_map = {}
    for e in edges:
        children_map.setdefault(e["src"], []).append(e["tgt"])

    # Build structured data for each SC
    sc_payloads = []
    for sc in scs:
        contexts = []
        for ctx_id in children_map.get(sc["id"], []):
            ctx = all_nodes.get(ctx_id)
            if not ctx: continue
            anchors = []
            for anc_id in children_map.get(ctx_id, []):
                anc = all_nodes.get(anc_id)
                if not anc: continue
                anchors.append({"title": anc["title"], "content": anc.get("content", ""), "weight": anc.get("weight", 0.5)})
            if anchors:
                contexts.append({"title": ctx["title"], "weight": ctx.get("weight", 0.5), "anchors": anchors})
        if contexts:
            sc_payloads.append({
                "super_context": sc["title"], "description": sc.get("description", ""),
                "contexts": contexts, "summary": sc["title"]
            })

    # Store each SC into user's platform DB
    # Use direct DB connection with proper path isolation
    user_db_path = get_user_memory_path(user["username"])
    original_graph = mem.GRAPH_DB
    original_tasks = mem.TASKS_DIR

    synced = 0
    errors = []
    try:
        mem.GRAPH_DB = user_db_path
        mem.TASKS_DIR = user_db_path.parent / "tasks"
        mem.TASKS_DIR.mkdir(parents=True, exist_ok=True)
        user_conn = mem.get_db()

        for payload in sc_payloads:
            try:
                f = io.StringIO()
                with redirect_stdout(f):
                    mem.cmd_store(user_conn, json.dumps(payload))
                synced += 1
            except Exception as e:
                errors.append(str(e)[:50])

        user_conn.close()
    finally:
        mem.GRAPH_DB = original_graph
        mem.TASKS_DIR = original_tasks

    # Log the sync
    udb = get_users_db()
    udb.execute("INSERT INTO sync_log (id, user_id, direction, items_synced, timestamp) VALUES (?,?,?,?,?)",
                (_uuid(), user["user_id"], "push", synced, _now()))
    udb.commit()
    udb.close()

    return {"synced": True, "super_contexts_pushed": synced, "errors": len(errors)}


@app.post("/api/memory/rebuild")
async def api_rebuild(user=Depends(get_current_user)):
    import memory as mem
    result = _run_memory_cmd(user["username"], mem.cmd_rebuild)
    return result


# ══════════════════════════════════════
# SHARING ROUTES
# ══════════════════════════════════════

@app.post("/api/sharing/share")
async def api_share(req: ShareRequest, user=Depends(get_current_user)):
    conn = get_users_db()
    rule_id = _uuid()
    conn.execute(
        """INSERT INTO sharing_rules
           (id, owner_id, target_type, target_id, scope_type, scope_id, scope_title, access_level, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (rule_id, user["user_id"], req.target_type, req.target_id,
         req.scope_type, req.scope_id, req.scope_title, req.access_level, _now())
    )
    conn.commit()
    conn.close()
    return {"shared": True, "rule_id": rule_id, "access_level": req.access_level}


@app.get("/api/sharing/my-shares")
async def api_my_shares(user=Depends(get_current_user)):
    conn = get_users_db()
    rules = conn.execute(
        "SELECT * FROM sharing_rules WHERE owner_id=? ORDER BY created_at DESC",
        (user["user_id"],)
    ).fetchall()
    conn.close()
    return {"shares": [dict(r) for r in rules]}


@app.get("/api/sharing/shared-with-me")
async def api_shared_with_me(user=Depends(get_current_user)):
    conn = get_users_db()
    shares = get_shared_scs_for_user(conn, user["user_id"])
    conn.close()
    return {"shares": shares}


@app.delete("/api/sharing/{rule_id}")
async def api_revoke_share(rule_id: str, user=Depends(get_current_user)):
    conn = get_users_db()
    conn.execute("DELETE FROM sharing_rules WHERE id=? AND owner_id=?", (rule_id, user["user_id"]))
    conn.commit()
    conn.close()
    return {"revoked": True}


# ══════════════════════════════════════
# ACCESS REQUEST ROUTES
# ══════════════════════════════════════

@app.post("/api/requests/create")
async def api_create_request(req: AccessRequestCreate, user=Depends(get_current_user)):
    conn = get_users_db()
    req_id = _uuid()
    conn.execute(
        """INSERT INTO access_requests
           (id, requester_id, owner_id, scope_type, scope_id, scope_title, task_context, status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (req_id, user["user_id"], req.owner_id, "full_sc", req.scope_id,
         req.scope_title, req.task_context, "pending", _now())
    )
    conn.commit()
    conn.close()
    return {"request_id": req_id, "status": "pending"}


@app.get("/api/requests/inbox")
async def api_request_inbox(user=Depends(get_current_user)):
    conn = get_users_db()
    requests = conn.execute(
        """SELECT ar.*, u.username as requester_username, u.display_name as requester_name
           FROM access_requests ar
           JOIN users u ON u.id = ar.requester_id
           WHERE ar.owner_id=? AND ar.status='pending'
           ORDER BY ar.created_at DESC""",
        (user["user_id"],)
    ).fetchall()
    conn.close()
    return {"requests": [dict(r) for r in requests]}


@app.post("/api/requests/{request_id}/resolve")
async def api_resolve_request(request_id: str, req: AccessRequestResolve, user=Depends(get_current_user)):
    conn = get_users_db()
    conn.execute(
        "UPDATE access_requests SET status=?, resolved_at=?, resolved_by=? WHERE id=? AND owner_id=?",
        (req.status, _now(), user["user_id"], request_id, user["user_id"])
    )
    conn.commit()
    conn.close()
    return {"resolved": True, "status": req.status}


# ══════════════════════════════════════
# ORG SEARCH (cross-user)
# ══════════════════════════════════════

@app.get("/api/org/search")
async def api_org_search(q: str, user=Depends(get_current_user)):
    """Search across all shared memories in the org."""
    conn = get_users_db()
    shares = get_shared_scs_for_user(conn, user["user_id"])
    conn.close()

    results = []
    # Group shares by owner
    owners = {}
    for s in shares:
        oid = s["owner_id"]
        if oid not in owners:
            owners[oid] = {"username": s["owner_username"], "name": s["owner_name"], "scopes": []}
        owners[oid]["scopes"].append(s)

    # Search each owner's shared memories
    for owner_id, info in owners.items():
        try:
            import memory as mem
            owner_conn = get_user_memory_db(info["username"])

            f = io.StringIO()
            with redirect_stdout(f):
                mem.cmd_retrieve(owner_conn, q)
            output = f.getvalue()
            owner_conn.close()

            try:
                data = json.loads(output)
                for m in data.get("memories", []):
                    # Filter: only include SCs that are actually shared
                    shared_sc_ids = {s["scope_id"] for s in info["scopes"]}
                    sc_id = m.get("sc_id", "")

                    # Check access level
                    access = "private"
                    for s in info["scopes"]:
                        if s["scope_id"] == sc_id or s["scope_type"] == "full_sc":
                            access = s["access_level"]
                            break

                    if access == "private":
                        continue

                    m["owner"] = info["name"]
                    m["owner_username"] = info["username"]
                    m["access_level"] = access

                    if access == "request_based":
                        # Hide content, show only SC title
                        m["contexts"] = []
                        m["locked"] = True
                        m["message"] = f"🔒 {info['name']} has knowledge about '{m['super_context']}' — request access"

                    results.append(m)
            except json.JSONDecodeError:
                pass
        except Exception as e:
            pass

    results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return {"query": q, "org_results": results[:10]}


@app.get("/api/org/stats")
async def api_org_stats(user=Depends(get_current_user)):
    """Org-wide memory statistics."""
    conn = get_users_db()
    users = conn.execute("SELECT id, username, display_name FROM users").fetchall()
    conn.close()

    stats = {"total_users": len(users), "users": []}
    for u in users:
        try:
            mem_conn = get_user_memory_db(u["username"])
            n = mem_conn.execute("SELECT COUNT(*) c FROM nodes").fetchone()["c"]
            sc = mem_conn.execute("SELECT COUNT(*) c FROM nodes WHERE type='super_context'").fetchone()["c"]
            anc = mem_conn.execute("SELECT COUNT(*) c FROM nodes WHERE type='anchor'").fetchone()["c"]
            mem_conn.close()
            stats["users"].append({
                "username": u["username"], "display_name": u["display_name"],
                "nodes": n, "super_contexts": sc, "anchors": anc
            })
        except Exception:
            stats["users"].append({"username": u["username"], "nodes": 0})

    return stats


# ══════════════════════════════════════
# PRODUCTIVITY ROUTES
# ══════════════════════════════════════

@app.get("/api/productivity/dashboard")
async def api_productivity_dashboard(date: str = None, user=Depends(get_current_user)):
    """Full productivity dashboard data for a given date."""
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")

    try:
        # Get productivity breakdown by SC
        sc_breakdown = v2.get_productivity_by_sc()

        # Compute totals
        total_productive = 0
        total_human = 0
        total_ai = 0
        top_sc = []
        top_ctx = []
        top_anc = []

        for sc in sc_breakdown:
            sc_time = sc.get("productivity_time_mins", 0)
            sc_human = sc.get("productivity_human_mins", 0)
            sc_ai = sc.get("productivity_ai_mins", 0)
            total_productive += sc_time
            total_human += sc_human
            total_ai += sc_ai
            top_sc.append({"name": sc["content"], "time_mins": sc_time})

            for ctx in sc.get("contexts", []):
                ctx_time = ctx.get("productivity_time_mins", 0)
                top_ctx.append({"name": ctx["content"], "time_mins": ctx_time})
                for anc in ctx.get("anchors", []):
                    anc_time = anc.get("productivity_time_mins", 0)
                    top_anc.append({"name": anc["content"], "time_mins": anc_time})

        # Sort and take top 5
        top_sc.sort(key=lambda x: x["time_mins"], reverse=True)
        top_ctx.sort(key=lambda x: x["time_mins"], reverse=True)
        top_anc.sort(key=lambda x: x["time_mins"], reverse=True)

        return {
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "total_productive_mins": round(total_productive, 1),
            "productive_human_mins": round(total_human, 1),
            "productive_ai_mins": round(total_ai, 1),
            "sc_breakdown": sc_breakdown,
            "top_5_sc": top_sc[:5],
            "top_5_context": top_ctx[:5],
            "top_5_anchor": top_anc[:5],
        }
    finally:
        v2.close()


@app.get("/api/productivity/sync-log")
async def api_productivity_sync_log(limit: int = 20, user=Depends(get_current_user)):
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        return {"sync_log": v2.get_sync_log(limit)}
    finally:
        v2.close()


@app.get("/api/productivity/match-quality")
async def api_match_quality(user=Depends(get_current_user)):
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        return v2.get_match_quality_stats()
    finally:
        v2.close()


# ══════════════════════════════════════
# PROJECT ROUTES
# ══════════════════════════════════════

@app.get("/api/projects")
async def api_list_projects(status: Optional[str] = None, user=Depends(get_current_user)):
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        projects = v2.list_projects(status)
        # Enrich with productivity time from linked SC nodes
        for p in projects:
            sc_id = p.get("sc_node_id", "")
            if sc_id:
                node = v2.get_node(sc_id)
                if node:
                    p["total_time_mins"] = node.get("productivity_time_mins", 0)
                    p["human_time_mins"] = node.get("productivity_human_mins", 0)
                    p["ai_time_mins"] = node.get("productivity_ai_mins", 0)
        return {"projects": projects}
    finally:
        v2.close()


@app.post("/api/projects")
async def api_create_project(req: ProjectCreate, user=Depends(get_current_user)):
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        project_data = req.model_dump()
        project_data["source"] = "manual"
        pid = v2.create_project(project_data)
        return {"project_id": pid, "created": True}
    finally:
        v2.close()


@app.put("/api/projects/{project_id}")
async def api_update_project(project_id: str, req: ProjectUpdate, user=Depends(get_current_user)):
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        v2.update_project(project_id, updates)
        return {"updated": True}
    finally:
        v2.close()


@app.delete("/api/projects/{project_id}")
async def api_delete_project(project_id: str, user=Depends(get_current_user)):
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        v2.delete_project(project_id)
        return {"deleted": True}
    finally:
        v2.close()


@app.get("/api/projects/{project_id}/deliverables")
async def api_list_deliverables(project_id: str, user=Depends(get_current_user)):
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        return {"deliverables": v2.get_deliverables(project_id)}
    finally:
        v2.close()


@app.post("/api/projects/{project_id}/deliverables")
async def api_create_deliverable(project_id: str, req: DeliverableCreate, user=Depends(get_current_user)):
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        data = req.model_dump()
        data["project_id"] = project_id
        did = v2.create_deliverable(data)
        return {"deliverable_id": did, "created": True}
    finally:
        v2.close()


@app.get("/api/projects/{project_id}/match-log")
async def api_project_match_log(project_id: str, limit: int = 50, user=Depends(get_current_user)):
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        return {"matches": v2.get_match_log(project_id, limit)}
    finally:
        v2.close()


@app.post("/api/projects/import/asana")
async def api_import_asana(user=Depends(get_current_user)):
    """Import projects from Asana. Expects JSON body with token and workspace_gid."""
    raise HTTPException(501, "Asana import not yet implemented")


# ══════════════════════════════════════
# DAEMON LIVE STATUS
# ══════════════════════════════════════

@app.get("/api/daemon/status")
async def api_daemon_status(user=Depends(get_current_user)):
    """Live daemon status — reads tracker DB directly for real-time segments."""
    tracker_db_path = Path.home() / ".productivity-tracker" / "tracker.db"
    if not tracker_db_path.exists():
        return {"is_active": False, "total_segments_today": 0,
                "total_unique_windows": 0, "top_platforms": [], "segments": []}

    conn = sqlite3.connect(str(tracker_db_path))
    conn.row_factory = sqlite3.Row

    today = datetime.now().strftime("%Y-%m-%d")

    segments = conn.execute("""
        SELECT target_segment_id, window_name, platform, supercontext,
               context, anchor, worker, target_segment_length_secs,
               detailed_summary, human_frame_count, ai_frame_count,
               timestamp_start
        FROM context_1
        WHERE date(timestamp_start) = ?
        ORDER BY timestamp_start DESC
    """, (today,)).fetchall()

    # Is active? Check if latest segment started within last 5 minutes
    # (segments stay open while frames accumulate, so 2 min is too tight)
    is_active = False
    if segments:
        try:
            latest_ts = str(segments[0]["timestamp_start"])
            from datetime import datetime as _dt
            ts = _dt.fromisoformat(latest_ts)
            is_active = (datetime.now() - ts).total_seconds() < 300  # 5 min window
        except Exception:
            pass
    # Also check if there are recent frames (even if segment not yet finalized)
    if not is_active:
        try:
            cutoff = (datetime.now() - __import__('datetime').timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M:%S")
            recent = conn.execute(
                "SELECT COUNT(*) as cnt FROM context_2 WHERE frame_timestamp > ?",
                (cutoff,)
            ).fetchone()
            is_active = (recent["cnt"] or 0) > 0
        except Exception:
            pass

    # Unique windows
    windows = set()
    platform_counts = {}
    for s in segments:
        p = s["platform"] or s["window_name"] or "Unknown"
        windows.add(p)
        platform_counts[p] = platform_counts.get(p, 0) + 1
    top_platforms = sorted(platform_counts.items(), key=lambda x: -x[1])[:3]

    # Build segment list
    seg_list = []
    for s in segments:
        seg_id = s["target_segment_id"] or ""
        seg_num = seg_id.split("-")[-1] if seg_id else "0"
        seg_list.append({
            "sl_no": int(seg_num) if seg_num.isdigit() else 0,
            "segment_id": seg_id,
            "window": s["platform"] or s["window_name"] or "Unknown",
            "detailed_topic": s["detailed_summary"] or "",
            "worker": s["worker"] or "human",
            "duration_secs": s["target_segment_length_secs"] or 0,
            "human_frames": s["human_frame_count"] or 0,
            "ai_frames": s["ai_frame_count"] or 0,
            "timestamp": str(s["timestamp_start"]),
            "supercontext": s["supercontext"] or "",
        })

    conn.close()
    return {
        "is_active": is_active,
        "total_segments_today": len(segments),
        "total_unique_windows": len(windows),
        "top_platforms": [{"name": p, "count": c} for p, c in top_platforms],
        "segments": seg_list,
    }


# ══════════════════════════════════════
# PORTAL (static files)
# ══════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def portal_root():
    portal_path = PLATFORM_DIR / "portal" / "index.html"
    if portal_path.exists():
        return HTMLResponse(portal_path.read_text())
    return HTMLResponse("<h1>PMIS Platform</h1><p>Portal not built yet.</p>")

@app.get("/health", response_class=HTMLResponse)
async def health_page():
    health_path = PLATFORM_DIR / "portal" / "health.html"
    if health_path.exists():
        return HTMLResponse(health_path.read_text())
    return HTMLResponse("<h1>Health page not found</h1>")

app.mount("/portal", StaticFiles(directory=str(PLATFORM_DIR / "portal")), name="portal")


# ══════════════════════════════════════
# STARTUP
# ══════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  PMIS Platform")
    print("=" * 50)
    print(f"  Portal:   http://localhost:8000")
    print(f"  API Docs: http://localhost:8000/docs")
    print(f"  Data:     {PLATFORM_DIR / 'data'}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
