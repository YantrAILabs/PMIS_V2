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
from pathlib import Path
from contextlib import redirect_stdout
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Add scripts to path for memory.py imports
PLATFORM_DIR = Path(__file__).parent
sys.path.insert(0, str(PLATFORM_DIR))
sys.path.insert(0, str(PLATFORM_DIR.parent / "scripts"))

from db import (get_users_db, get_user_memory_db, get_user_memory_path,
                get_shared_scs_for_user, check_access, has_approved_request,
                _now, _uuid, MEMORIES_DIR)
from auth import register_user, login_user, get_user_from_token, list_users

app = FastAPI(title="PMIS Platform", version="1.0", description="Multi-user Memory Intelligence")


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
# PORTAL (static files)
# ══════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def portal_root():
    portal_path = PLATFORM_DIR / "portal" / "index.html"
    if portal_path.exists():
        return HTMLResponse(portal_path.read_text())
    return HTMLResponse("<h1>PMIS Platform</h1><p>Portal not built yet.</p>")

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
