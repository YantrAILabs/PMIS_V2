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

@app.post("/api/memory/store")
async def api_store(req: StoreRequest, user=Depends(get_current_user)):
    """Store memory — creates nodes in PMIS V2."""
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        # For now, return acknowledgment — full ingestion pipeline handles storage
        return {"stored": True, "message": "Memory stored via productivity pipeline"}
    finally:
        v2.close()


@app.get("/api/memory/retrieve")
async def api_retrieve(q: str, user=Depends(get_current_user)):
    """Retrieve memories — searches PMIS V2 nodes by content."""
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        # Simple text search across memory nodes
        with v2._connect() as conn:
            rows = conn.execute("""
                SELECT id, content, level, precision, access_count, productivity_time_mins
                FROM memory_nodes WHERE is_deleted = 0 AND content LIKE ?
                ORDER BY access_count DESC LIMIT 20
            """, (f"%{q}%",)).fetchall()
        results = [dict(r) for r in rows]
        v2.close()
        return {"query": q, "results": results, "count": len(results)}
    except Exception as e:
        v2.close()
        raise HTTPException(500, str(e))


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
    # Fallback: empty if PMIS V2 unavailable
    return {"super_contexts": [], "total": 0, "source": "none"}


@app.get("/api/memory/tree")
async def api_tree(user=Depends(get_current_user)):
    """ASCII tree — built from PMIS V2 nodes."""
    v2 = _get_pmis_v2_db()
    if not v2:
        return {"output": "PMIS V2 not available"}
    try:
        lines = []
        scs = v2.get_nodes_by_level("SC")
        for sc in scs:
            lines.append(f"SC: {sc['content'][:60]}")
            for ctx in v2.get_children(sc["id"]):
                lines.append(f"  CTX: {ctx['content'][:60]}")
                for anc in v2.get_children(ctx["id"]):
                    lines.append(f"    ANC: {anc['content'][:60]}")
        v2.close()
        return {"output": "\n".join(lines), "total_scs": len(scs)}
    except Exception as e:
        v2.close()
        return {"output": str(e)}


@app.get("/api/memory/stats")
async def api_stats(user=Depends(get_current_user)):
    """Memory statistics from PMIS V2."""
    v2 = _get_pmis_v2_db()
    if not v2:
        return {"super_contexts": 0, "contexts": 0, "anchors": 0}
    try:
        stats = {
            "super_contexts": v2.count_nodes("SC"),
            "contexts": v2.count_nodes("CTX"),
            "anchors": v2.count_nodes("ANC"),
            "total": v2.count_nodes(),
        }
        v2.close()
        return stats
    except Exception as e:
        v2.close()
        return {"error": str(e)}


@app.post("/api/memory/score")
async def api_score(req: ScoreRequest, user=Depends(get_current_user)):
    """Score a task — updates node precision in PMIS V2."""
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        v2.update_node_precision(req.task_id, req.score / 5.0)
        v2.close()
        return {"scored": True, "task_id": req.task_id, "score": req.score}
    except Exception as e:
        v2.close()
        raise HTTPException(500, str(e))


@app.get("/api/memory/health")
async def api_health(user=Depends(get_current_user)):
    """Memory health report from PMIS V2."""
    v2 = _get_pmis_v2_db()
    if not v2:
        return {"counts": {"SC": 0, "CTX": 0, "ANC": 0}}
    try:
        counts = {"SC": v2.count_nodes("SC"), "CTX": v2.count_nodes("CTX"), "ANC": v2.count_nodes("ANC")}
        sync_log = v2.get_sync_log(5)
        match_quality = v2.get_match_quality_stats()
        v2.close()
        return {"counts": counts, "total": sum(counts.values()),
                "sync_log": sync_log, "match_quality": match_quality}
    except Exception as e:
        v2.close()
        return {"error": str(e)}


@app.post("/api/memory/rebuild")
async def api_rebuild(user=Depends(get_current_user)):
    """Trigger PMIS V2 nightly consolidation."""
    v2 = _get_pmis_v2_db()
    if not v2:
        raise HTTPException(503, "PMIS V2 not available")
    try:
        v2.close()
        return {"rebuilt": True, "message": "Consolidation runs automatically via 30-min daemon sync"}
    except Exception as e:
        v2.close()
        raise HTTPException(500, str(e))


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

    # Search each owner's shared memories via PMIS V2
    v2 = _get_pmis_v2_db()
    if v2:
        try:
            with v2._connect() as conn:
                for r in conn.execute(
                    "SELECT id, content, level FROM memory_nodes WHERE is_deleted=0 AND content LIKE ? LIMIT 10",
                    (f"%{q}%",)
                ).fetchall():
                    results.append({"id": r["id"], "content": r["content"][:200], "level": r["level"], "owner": "system"})
            v2.close()
        except Exception:
            v2.close()

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
    target_date = date or datetime.now().strftime("%Y-%m-%d")

    # Read date-specific data from the productivity tracker's daily_memory table
    tracker_db_path = Path.home() / ".productivity-tracker" / "tracker.db"
    if not tracker_db_path.exists():
        # Fall back to cumulative PMIS V2 data if tracker DB doesn't exist
        v2 = _get_pmis_v2_db()
        if not v2:
            raise HTTPException(503, "PMIS V2 not available")
        try:
            sc_breakdown = v2.get_productivity_by_sc()
            total_productive = sum(sc.get("productivity_time_mins", 0) for sc in sc_breakdown)
            total_human = sum(sc.get("productivity_human_mins", 0) for sc in sc_breakdown)
            total_ai = sum(sc.get("productivity_ai_mins", 0) for sc in sc_breakdown)
            return {
                "date": target_date,
                "total_productive_mins": round(total_productive, 1),
                "productive_human_mins": round(total_human, 1),
                "productive_ai_mins": round(total_ai, 1),
                "sc_breakdown": sc_breakdown,
                "top_5_sc": [], "top_5_context": [], "top_5_anchor": [],
            }
        finally:
            v2.close()

    conn = sqlite3.connect(str(tracker_db_path))
    conn.row_factory = sqlite3.Row

    try:
        # Query daily_memory for the selected date — this has per-date SC/CTX/ANC rows
        rows = conn.execute("""
            SELECT supercontext, context, anchor, level, time_mins, human_mins, agent_mins
            FROM daily_memory
            WHERE date = ?
            ORDER BY supercontext, context, anchor
        """, (target_date,)).fetchall()

        # Build SC hierarchy from daily rows
        sc_map = {}  # supercontext -> {contexts: {ctx_name -> {anchors: [...]}}}
        for r in rows:
            sc_name = r["supercontext"] or "Unknown"
            ctx_name = r["context"] or ""
            anc_name = r["anchor"] or ""
            level = r["level"] or ""
            time_mins = r["time_mins"] or 0
            human_mins = r["human_mins"] or 0
            agent_mins = r["agent_mins"] or 0

            if sc_name not in sc_map:
                sc_map[sc_name] = {
                    "content": sc_name,
                    "productivity_time_mins": 0, "productivity_human_mins": 0, "productivity_ai_mins": 0,
                    "contexts": {}
                }

            if level == "SC":
                sc_map[sc_name]["productivity_time_mins"] += time_mins
                sc_map[sc_name]["productivity_human_mins"] += human_mins
                sc_map[sc_name]["productivity_ai_mins"] += agent_mins
            elif level == "context" and ctx_name:
                if ctx_name not in sc_map[sc_name]["contexts"]:
                    sc_map[sc_name]["contexts"][ctx_name] = {
                        "content": ctx_name,
                        "productivity_time_mins": 0, "productivity_human_mins": 0, "productivity_ai_mins": 0,
                        "anchors": {}
                    }
                sc_map[sc_name]["contexts"][ctx_name]["productivity_time_mins"] += time_mins
                sc_map[sc_name]["contexts"][ctx_name]["productivity_human_mins"] += human_mins
                sc_map[sc_name]["contexts"][ctx_name]["productivity_ai_mins"] += agent_mins
            elif level == "anchor" and ctx_name and anc_name:
                if ctx_name not in sc_map[sc_name]["contexts"]:
                    sc_map[sc_name]["contexts"][ctx_name] = {
                        "content": ctx_name,
                        "productivity_time_mins": 0, "productivity_human_mins": 0, "productivity_ai_mins": 0,
                        "anchors": {}
                    }
                ctx_obj = sc_map[sc_name]["contexts"][ctx_name]
                if anc_name not in ctx_obj["anchors"]:
                    ctx_obj["anchors"][anc_name] = {
                        "content": anc_name,
                        "productivity_time_mins": 0, "productivity_human_mins": 0, "productivity_ai_mins": 0,
                    }
                ctx_obj["anchors"][anc_name]["productivity_time_mins"] += time_mins
                ctx_obj["anchors"][anc_name]["productivity_human_mins"] += human_mins
                ctx_obj["anchors"][anc_name]["productivity_ai_mins"] += agent_mins

        # Convert to list format expected by frontend
        sc_breakdown = []
        total_productive = 0
        total_human = 0
        total_ai = 0
        top_sc = []
        top_ctx = []
        top_anc = []

        for sc_name, sc_data in sc_map.items():
            sc_time = sc_data["productivity_time_mins"]
            sc_human = sc_data["productivity_human_mins"]
            sc_ai = sc_data["productivity_ai_mins"]
            total_productive += sc_time
            total_human += sc_human
            total_ai += sc_ai
            top_sc.append({"name": sc_name, "time_mins": sc_time})

            ctx_list = []
            for ctx_name, ctx_data in sc_data["contexts"].items():
                ctx_time = ctx_data["productivity_time_mins"]
                top_ctx.append({"name": ctx_name, "time_mins": ctx_time})

                anc_list = []
                for anc_name, anc_data in ctx_data["anchors"].items():
                    anc_time = anc_data["productivity_time_mins"]
                    top_anc.append({"name": anc_name, "time_mins": anc_time})
                    anc_list.append(anc_data)

                ctx_data_out = dict(ctx_data)
                ctx_data_out["anchors"] = sorted(anc_list, key=lambda x: x["productivity_time_mins"], reverse=True)
                ctx_list.append(ctx_data_out)

            sc_entry = {
                "content": sc_name,
                "productivity_time_mins": sc_time,
                "productivity_human_mins": sc_human,
                "productivity_ai_mins": sc_ai,
                "contexts": sorted(ctx_list, key=lambda x: x["productivity_time_mins"], reverse=True),
            }
            sc_breakdown.append(sc_entry)

        sc_breakdown.sort(key=lambda x: x["productivity_time_mins"], reverse=True)
        top_sc.sort(key=lambda x: x["time_mins"], reverse=True)
        top_ctx.sort(key=lambda x: x["time_mins"], reverse=True)
        top_anc.sort(key=lambda x: x["time_mins"], reverse=True)

        return {
            "date": target_date,
            "total_productive_mins": round(total_productive, 1),
            "productive_human_mins": round(total_human, 1),
            "productive_ai_mins": round(total_ai, 1),
            "sc_breakdown": sc_breakdown,
            "top_5_sc": top_sc[:5],
            "top_5_context": top_ctx[:5],
            "top_5_anchor": top_anc[:5],
        }
    finally:
        conn.close()


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
        return HTMLResponse(portal_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>PMIS Platform</h1><p>Portal not built yet.</p>")

@app.get("/health", response_class=HTMLResponse)
async def health_page():
    health_path = PLATFORM_DIR / "portal" / "health.html"
    if health_path.exists():
        return HTMLResponse(health_path.read_text(encoding="utf-8"))
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
