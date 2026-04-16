"""
PMIS V2 HTTP Server

FastAPI server on port 8100 with:
- REST API for memory operations
- Server-rendered dashboard via Jinja2
- Cross-platform authentication (API keys)
- Universal webhook for any AI platform
- OpenAPI Actions schema for ChatGPT Custom GPTs
- Platform memory tracking (audit log)
- Desktop agent (health checker + integration assistant)
- SSE streaming for live dashboard updates
- Integration dashboard with setup wizard

Usage:
    python3 pmis_v2/server.py
    # or: uvicorn pmis_v2.server:app --port 8100 --reload
"""

import sys
import json
import asyncio
import threading
import time as _time
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# Add pmis_v2 to path
PMIS_DIR = Path(__file__).parent
sys.path.insert(0, str(PMIS_DIR))

from fastapi import FastAPI, Request, Query, Header, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import yaml

from orchestrator import Orchestrator
from db.platform_store import PlatformStore
from auth import validate_key, create_key, list_keys, revoke_key

# ================================================================
# APP SETUP
# ================================================================

_orch: Optional[Orchestrator] = None
_platform_store: Optional[PlatformStore] = None
_agent_thread: Optional[threading.Thread] = None
_server_start_time: Optional[datetime] = None
_sse_clients: List[asyncio.Queue] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize orchestrator, platform store, and agent on startup."""
    global _orch, _platform_store, _agent_thread, _server_start_time
    db_path = str(PMIS_DIR / "data" / "memory.db")
    _orch = Orchestrator(db_path=db_path)
    _platform_store = PlatformStore(_orch.db._conn)
    _server_start_time = datetime.now()
    print(f"[PMIS V2] Orchestrator initialized. DB: {db_path}")
    print(f"[PMIS V2] Platform store initialized.")

    # Start agent health checker background thread
    _agent_thread = threading.Thread(target=_agent_health_loop, daemon=True)
    _agent_thread.start()
    print("[PMIS V2] Agent health checker started.")

    yield

    # Cleanup: close any open sessions
    for sid in list(_orch._sessions.keys()):
        _orch.close_session(sid)
    _orch.db.close()
    print("[PMIS V2] Server shutdown complete.")


app = FastAPI(
    title="PMIS V2 Memory System",
    version="2.0",
    lifespan=lifespan,
)

# CORS — allow ChatGPT, Claude Web, and LAN clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates_dir = PMIS_DIR / "templates"
static_dir = PMIS_DIR / "static"
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = Jinja2Templates(directory=str(templates_dir))


# ================================================================
# AUTH DEPENDENCY
# ================================================================

async def optional_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Extract platform from API key if present. Does not require auth."""
    if x_api_key:
        platform = validate_key(x_api_key)
        return platform
    return None


async def require_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Require valid API key. Returns platform name."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    platform = validate_key(x_api_key)
    if not platform:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return platform


# ================================================================
# REQUEST MODELS
# ================================================================

class TurnRequest(BaseModel):
    content: str
    conversation_id: Optional[str] = None
    role: str = "user"
    platform: Optional[str] = None

class WebhookRequest(BaseModel):
    message: str
    platform: str = "unknown"
    conversation_id: Optional[str] = None

class StoreRequest(BaseModel):
    super_context: str
    description: str = ""
    contexts: list = []
    summary: str = ""

class RateRequest(BaseModel):
    direction: str  # "up" or "down"
    anchors: Optional[List[str]] = None

class SessionEndRequest(BaseModel):
    conversation_id: str

class CommandRequest(BaseModel):
    command: str

class HyperparamUpdate(BaseModel):
    updates: Dict[str, Any]

class PlatformRegister(BaseModel):
    id: str
    name: str


# ================================================================
# HEALTH
# ================================================================

@app.get("/health")
async def health():
    stats = _orch.get_stats() if _orch else {}
    return {"status": "ok", "timestamp": datetime.now().isoformat(), **stats}


# ================================================================
# CORE API
# ================================================================

def _process_turn_internal(content: str, conversation_id: str = None, role: str = "user", platform: str = None) -> dict:
    """Internal turn processor with platform tracking."""
    session = _orch.get_or_create_session(conversation_id)
    is_new = session.turn_counter == 0

    # Log to platform_memories (audit only) if platform known
    pm_entry_id = None
    if platform and _platform_store:
        # Auto-register platform if not known (prevents FK constraint failures)
        if not _platform_store.get_platform(platform):
            _platform_store.register_platform(platform, platform.replace("-", " ").title())
        _platform_store.update_platform_seen(platform)
        pm_entry_id = _platform_store.log_memory(
            platform_id=platform,
            content=content[:500],
            conversation_id=session.conversation_id,
        )

    result = _orch.process_turn(
        content=content,
        conversation_id=session.conversation_id,
        role=role,
    )

    # Update platform_memories merge status
    if pm_entry_id and _platform_store:
        surprise_val = result.surprise_result.effective_surprise if result.surprise_result else None
        if result.stored_node_id:
            _platform_store.update_merge_status(pm_entry_id, result.stored_node_id, "merged", surprise_val)
            _platform_store.increment_memory_count(platform)
        else:
            _platform_store.update_merge_status(pm_entry_id, None, "skipped", surprise_val)

    # Push SSE event
    _push_sse_event({
        "type": "turn",
        "platform": platform or "local",
        "stored": result.stored_node_id is not None,
        "mode": result.gamma_result.mode_label if result.gamma_result else "BALANCED",
    })

    return {
        "memories": result.system_prompt,
        "session": {
            "conversation_id": session.conversation_id,
            "turn_count": result.turn_number,
            "is_new_session": is_new,
        },
        "mode": result.gamma_result.mode_label if result.gamma_result else "BALANCED",
        "gamma": result.gamma_result.gamma if result.gamma_result else 0.5,
        "surprise": result.surprise_result.effective_surprise if result.surprise_result else 0.0,
        "retrieved_count": len(result.retrieved_memories),
        "stored": result.stored_node_id is not None,
        "stored_node_id": result.stored_node_id,
        "storage_action": result.storage_action,
        "active_tree": result.active_tree,
        "is_stale": result.is_stale,
        "epistemic_questions": result.epistemic_questions or [],
        "predictive": [
            {"content": m.get("content", "")[:150], "depth": m.get("_prediction_depth")}
            for m in (result.predictive_memories or [])
        ],
    }


@app.post("/api/turn")
async def process_turn(req: TurnRequest, platform: Optional[str] = Depends(optional_api_key)):
    """Process a conversation turn — the main entry point."""
    plat = req.platform or platform or "local"
    return _process_turn_internal(req.content, req.conversation_id, req.role, plat)


@app.post("/api/webhook")
async def webhook(req: WebhookRequest, platform: Optional[str] = Depends(optional_api_key)):
    """Universal webhook — any AI platform can call this with minimal structure.
    Returns memory_context as plain text for easy system prompt injection."""
    plat = platform or req.platform or "unknown"

    # Auto-register platform if not known
    if _platform_store and not _platform_store.get_platform(plat):
        _platform_store.register_platform(plat, plat.replace("-", " ").title())

    result = _process_turn_internal(req.message, req.conversation_id, "user", plat)

    return {
        "memory_context": result["memories"],
        "mode": result["mode"],
        "gamma": result["gamma"],
        "surprise": result["surprise"],
        "retrieved_count": result["retrieved_count"],
        "stored": result["stored"],
        "platform": plat,
    }


@app.post("/api/store")
async def store_memory(req: StoreRequest):
    """Manual memory store."""
    content_parts = [f"[{req.super_context}]"]
    for ctx in req.contexts:
        title = ctx.get("title", "") if isinstance(ctx, dict) else str(ctx)
        content_parts.append(f"  {title}")
    content = "\n".join(content_parts)

    session = _orch.get_or_create_session()
    result = _orch.process_turn(content=content, conversation_id=session.conversation_id)

    return {
        "stored": result.stored_node_id is not None,
        "stored_node_id": result.stored_node_id,
        "storage_action": result.storage_action,
    }


@app.post("/api/rate")
async def rate_session(req: RateRequest):
    """Record session feedback."""
    return {"rated": True, "direction": req.direction, "anchors": req.anchors}


@app.post("/api/session/end")
async def end_session(req: SessionEndRequest):
    """End a conversation session."""
    _orch.close_session(req.conversation_id)
    return {"ended": True, "conversation_id": req.conversation_id}


@app.post("/api/consolidate")
async def consolidate():
    """Run nightly consolidation."""
    result = _orch.handle_command("consolidate")
    return {"result": result, "completed": True}


@app.post("/api/command")
async def run_command(req: CommandRequest):
    """Execute a slash command."""
    result = _orch.handle_command(req.command)
    return {"command": req.command, "result": result}


# ================================================================
# QUERY API
# ================================================================

@app.get("/api/status")
async def get_status():
    return _orch.get_stats()


@app.get("/api/stats")
async def get_stats():
    return _orch.get_stats()


@app.get("/api/browse")
async def browse():
    """List all super contexts with hierarchy."""
    db = _orch.db
    scs = db.get_nodes_by_level("SC")
    result = []
    for sc in scs:
        children = db.get_children(sc["id"])
        ctx_list = []
        for child in children:
            anchors = db.get_children(child["id"])
            ctx_list.append({
                "id": child["id"],
                "title": child["content"][:100],
                "anchor_count": len(anchors),
                "precision": child.get("precision", 0.5),
            })
        result.append({
            "id": sc["id"],
            "title": sc["content"][:100],
            "context_count": len(ctx_list),
            "contexts": ctx_list,
        })
    return {"super_contexts": result, "total": len(result)}


@app.get("/api/orphans")
async def get_orphans():
    """List orphan anchors."""
    orphans = _orch.db.get_orphan_nodes()
    return {
        "orphans": [
            {"id": o["id"], "content": o["content"][:150], "access_count": o.get("access_count", 0)}
            for o in orphans
        ],
        "total": len(orphans),
    }


@app.get("/api/conversations")
async def get_conversations(
    date: Optional[str] = Query(None, description="Filter by date YYYY-MM-DD"),
    limit: int = Query(50, ge=1, le=200),
):
    """List recent conversations with summary."""
    conn = _orch.db._conn
    if date:
        rows = conn.execute("""
            SELECT conversation_id,
                   MIN(timestamp) as first_turn,
                   MAX(timestamp) as last_turn,
                   COUNT(*) as turn_count,
                   AVG(gamma) as avg_gamma,
                   mode
            FROM conversation_turns
            WHERE DATE(timestamp) = ?
            GROUP BY conversation_id
            ORDER BY first_turn DESC
            LIMIT ?
        """, (date, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT conversation_id,
                   MIN(timestamp) as first_turn,
                   MAX(timestamp) as last_turn,
                   COUNT(*) as turn_count,
                   AVG(gamma) as avg_gamma,
                   mode
            FROM conversation_turns
            GROUP BY conversation_id
            ORDER BY first_turn DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return {
        "conversations": [
            {
                "conversation_id": r["conversation_id"],
                "first_turn": r["first_turn"],
                "last_turn": r["last_turn"],
                "turn_count": r["turn_count"],
                "avg_gamma": round(r["avg_gamma"], 3) if r["avg_gamma"] else 0.5,
                "dominant_mode": r["mode"] or "BALANCED",
            }
            for r in rows
        ]
    }


@app.get("/api/conversation/{conv_id}")
async def get_conversation(conv_id: str):
    """Full turn log with rich detail (retrieved memories, epistemic questions, predictions)."""
    conn = _orch.db._conn

    # Get all turns with expanded columns
    turns = conn.execute("""
        SELECT * FROM conversation_turns
        WHERE conversation_id = ?
        ORDER BY turn_number
    """, (conv_id,)).fetchall()

    if not turns:
        return {"conversation_id": conv_id, "turns": []}

    # Collect turn IDs for batch detail queries
    turn_ids = [t["id"] for t in turns]
    placeholders = ",".join("?" * len(turn_ids))

    # Batch load retrieved memories
    retrieved_rows = conn.execute(f"""
        SELECT * FROM turn_retrieved_memories
        WHERE turn_id IN ({placeholders})
        ORDER BY turn_id, rank
    """, turn_ids).fetchall()

    # Batch load epistemic questions
    epistemic_rows = conn.execute(f"""
        SELECT * FROM turn_epistemic_questions
        WHERE turn_id IN ({placeholders})
        ORDER BY turn_id, information_gain DESC
    """, turn_ids).fetchall()

    # Batch load predictive memories
    predictive_rows = conn.execute(f"""
        SELECT * FROM turn_predictive_memories
        WHERE turn_id IN ({placeholders})
        ORDER BY turn_id
    """, turn_ids).fetchall()

    # Group details by turn_id
    from collections import defaultdict
    retrieved_by_turn = defaultdict(list)
    for r in retrieved_rows:
        retrieved_by_turn[r["turn_id"]].append({
            "rank": r["rank"],
            "node_id": r["memory_node_id"],
            "level": r["node_level"],
            "content_preview": r["content_preview"],
            "final_score": round(r["final_score"], 3) if r["final_score"] else 0,
            "semantic_score": round(r["semantic_score"], 3) if r["semantic_score"] else 0,
            "hierarchy_score": round(r["hierarchy_score"], 3) if r["hierarchy_score"] else 0,
            "temporal_score": round(r["temporal_score"], 3) if r["temporal_score"] else 0,
            "precision_score": round(r["precision_score"], 3) if r["precision_score"] else 0,
            "source": r["source"],
        })

    epistemic_by_turn = defaultdict(list)
    for q in epistemic_rows:
        epistemic_by_turn[q["turn_id"]].append({
            "question": q["question_text"],
            "information_gain": round(q["information_gain"], 3) if q["information_gain"] else 0,
            "parent_context": q["parent_context_name"],
            "anchor_content": q["anchor_content"],
        })

    predictive_by_turn = defaultdict(list)
    for p in predictive_rows:
        predictive_by_turn[p["turn_id"]].append({
            "content_preview": p["content_preview"],
            "depth": p["prediction_depth"],
            "frequency": p["prediction_frequency"],
        })

    # Build response
    turn_list = []
    for t in turns:
        tid = t["id"]
        turn_data = {
            "turn_number": t["turn_number"],
            "role": t["role"],
            "gamma": round(t["gamma"], 3) if t["gamma"] else 0.5,
            "surprise": round(t["effective_surprise"], 3) if t["effective_surprise"] else 0.0,
            "mode": t["mode"] or "BALANCED",
            "raw_surprise": round(t["raw_surprise"], 3) if t["raw_surprise"] else None,
            "cluster_precision": round(t["cluster_precision"], 3) if t["cluster_precision"] else None,
            "nearest_context": {
                "id": t["nearest_context_id"],
                "name": t["nearest_context_name"],
            } if t["nearest_context_id"] else None,
            "active_tree": t["active_tree"],
            "is_stale": bool(t["is_stale"]) if t["is_stale"] is not None else False,
            "storage_action": t["storage_action"],
            "stored_node_id": t["node_id"],
            "response_summary": t["response_summary"],
            "system_prompt": t["system_prompt"],
            "timestamp": t["timestamp"],
            # Detail arrays
            "retrieved_memories": retrieved_by_turn.get(tid, []),
            "epistemic_questions": epistemic_by_turn.get(tid, []),
            "predictive_memories": predictive_by_turn.get(tid, []),
        }

        # Stored node content
        if t["node_id"]:
            node = _orch.db.get_node(t["node_id"])
            if node:
                turn_data["stored_content"] = node["content"][:200]

        turn_list.append(turn_data)

    return {"conversation_id": conv_id, "turns": turn_list}


@app.get("/api/stats/daily")
async def daily_stats(date: Optional[str] = Query(None)):
    """Daily statistics for dashboard."""
    target_date = date or datetime.now().strftime("%Y-%m-%d")
    conn = _orch.db._conn

    stats = _orch.get_stats()

    total_turns = conn.execute(
        "SELECT COUNT(*) FROM conversation_turns WHERE DATE(timestamp)=?",
        (target_date,)
    ).fetchone()[0]

    new_nodes = conn.execute(
        "SELECT COUNT(*) FROM memory_nodes WHERE DATE(created_at)=? AND is_deleted=0",
        (target_date,)
    ).fetchone()[0]

    updated_nodes = conn.execute(
        "SELECT COUNT(DISTINCT node_id) FROM access_log WHERE DATE(accessed_at)=?",
        (target_date,)
    ).fetchone()[0]

    return {
        "date": target_date,
        **stats,
        "total_turns_today": total_turns,
        "new_nodes_today": new_nodes,
        "updated_nodes_today": updated_nodes,
    }


# ================================================================
# PLATFORM API
# ================================================================

@app.get("/api/platforms")
async def list_platforms():
    """List all registered platforms with stats."""
    if not _platform_store:
        return {"platforms": []}
    return {"platforms": _platform_store.get_all_platform_stats()}


@app.post("/api/platforms")
async def register_platform(req: PlatformRegister):
    """Register a new platform."""
    platform = _platform_store.register_platform(req.id, req.name)
    return {"platform": platform}


@app.get("/api/platforms/{platform_id}")
async def get_platform(platform_id: str):
    """Get platform detail + stats."""
    platform = _platform_store.get_platform(platform_id)
    if not platform:
        raise HTTPException(status_code=404, detail="Platform not found")
    stats = _platform_store.get_platform_stats(platform_id)
    return {**platform, "memory_stats": stats}


@app.get("/api/platforms/{platform_id}/memories")
async def get_platform_memories(platform_id: str, limit: int = 50, offset: int = 0):
    """Platform's memory audit log."""
    memories = _platform_store.get_platform_memories(platform_id, limit, offset)
    return {"memories": memories, "platform_id": platform_id}


@app.post("/api/platforms/{platform_id}/test")
async def test_platform(platform_id: str):
    """Test connection for a platform."""
    platform = _platform_store.get_platform(platform_id)
    return {
        "platform": platform_id,
        "exists": platform is not None,
        "status": platform["status"] if platform else "not_registered",
        "server_healthy": True,
    }


@app.post("/api/platforms/{platform_id}/generate-key")
async def generate_platform_key(platform_id: str):
    """Generate a new API key for a platform."""
    raw_key = create_key(platform_id)
    # Register platform if not exists
    if not _platform_store.get_platform(platform_id):
        _platform_store.register_platform(platform_id, platform_id.replace("-", " ").title())
    return {"api_key": raw_key, "platform": platform_id, "note": "Save this key — it won't be shown again."}


@app.delete("/api/platforms/{platform_id}/revoke-key")
async def revoke_platform_key(platform_id: str):
    """Revoke all API keys for a platform."""
    count = revoke_key(platform_id)
    return {"revoked": count, "platform": platform_id}


@app.get("/api/auth/keys")
async def get_api_keys():
    """List all API keys (hashed, not raw)."""
    return {"keys": list_keys()}


# ================================================================
# SSE — Server-Sent Events for live dashboard updates
# ================================================================

def _push_sse_event(data: dict):
    """Push an event to all connected SSE clients."""
    message = json.dumps(data)
    disconnected = []
    for q in _sse_clients:
        try:
            q.put_nowait(message)
        except Exception:
            disconnected.append(q)
    for q in disconnected:
        _sse_clients.remove(q)


async def _sse_generator(queue: asyncio.Queue):
    """Async generator for SSE stream."""
    try:
        while True:
            data = await asyncio.wait_for(queue.get(), timeout=30)
            yield f"data: {data}\n\n"
    except asyncio.TimeoutError:
        yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    except asyncio.CancelledError:
        return


@app.get("/api/platforms/stream")
async def platform_stream():
    """SSE stream of platform status changes and memory events."""
    queue = asyncio.Queue(maxsize=100)
    _sse_clients.append(queue)

    async def event_generator():
        try:
            # Send initial state
            if _platform_store:
                platforms = _platform_store.get_all_platform_stats()
                yield f"data: {json.dumps({'type': 'init', 'platforms': platforms})}\n\n"
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in _sse_clients:
                _sse_clients.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ================================================================
# AGENT STATUS
# ================================================================

def _check_ollama_status() -> bool:
    """Check if Ollama is running."""
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _agent_health_loop():
    """Background thread: check platform health every 60s."""
    while True:
        _time.sleep(60)
        try:
            if not _platform_store:
                continue

            platforms = _platform_store.list_platforms()
            for p in platforms:
                last_seen = p.get("last_seen")
                if not last_seen:
                    continue
                try:
                    seen_dt = datetime.fromisoformat(last_seen)
                except (ValueError, TypeError):
                    continue
                delta_minutes = (datetime.now() - seen_dt).total_seconds() / 60

                if delta_minutes < 5:
                    new_status = "active"
                elif delta_minutes < 60:
                    new_status = "idle"
                else:
                    new_status = "disconnected"

                if p["status"] != new_status:
                    _platform_store.update_platform_status(p["id"], new_status)
                    _push_sse_event({
                        "type": "status_change",
                        "platform_id": p["id"],
                        "old_status": p["status"],
                        "new_status": new_status,
                    })
        except Exception:
            pass


@app.post("/api/agent/connect/{platform_id}")
async def agent_connect(platform_id: str):
    """Run setup wizard for a platform via API (used by dashboard UI)."""
    sys.path.insert(0, str(PMIS_DIR))
    from agent.setup_wizard import get_setup_steps, get_platform_config
    config = get_platform_config(platform_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Unknown platform: {platform_id}")

    # Auto-register platform
    if _platform_store and not _platform_store.get_platform(platform_id):
        _platform_store.register_platform(platform_id, config["name"])

    return {
        "platform_id": platform_id,
        "name": config["name"],
        "transport": config["transport"],
        "needs_tunnel": config["needs_tunnel"],
        "steps": get_setup_steps(platform_id),
    }


@app.get("/api/agent/status")
async def agent_status():
    """Agent health + server uptime + dependency status."""
    uptime_seconds = (datetime.now() - _server_start_time).total_seconds() if _server_start_time else 0
    ollama_ok = _check_ollama_status()
    stats = _orch.get_stats() if _orch else {}
    platforms = _platform_store.get_all_platform_stats() if _platform_store else []

    return {
        "agent_running": _agent_thread is not None and _agent_thread.is_alive(),
        "server_uptime_seconds": int(uptime_seconds),
        "server_start_time": _server_start_time.isoformat() if _server_start_time else None,
        "ollama_status": "running" if ollama_ok else "stopped",
        "memory_stats": stats,
        "platforms": platforms,
        "total_platforms": len(platforms),
        "active_platforms": sum(1 for p in platforms if p.get("status") == "active"),
    }


# ================================================================
# OPENAPI ACTIONS SCHEMA (for ChatGPT Custom GPTs)
# ================================================================

@app.get("/openapi-actions.json")
async def openapi_actions(request: Request):
    """OpenAPI schema for ChatGPT Custom GPT Actions."""
    base_url = str(request.base_url).rstrip("/")
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "PMIS Memory System",
            "description": "Personal Memory Intelligence System — retrieve and store memories across conversations",
            "version": "2.0"
        },
        "servers": [{"url": base_url}],
        "paths": {
            "/api/webhook": {
                "post": {
                    "operationId": "processMemoryTurn",
                    "summary": "Process a conversation turn and retrieve relevant memories",
                    "description": "Send the user's message. Returns memory_context to inject into your system prompt.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["message"],
                                    "properties": {
                                        "message": {"type": "string", "description": "The user's message text"},
                                        "platform": {"type": "string", "default": "openai-gpt"},
                                        "conversation_id": {"type": "string", "description": "Optional conversation ID for session continuity"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Memory context and metadata",
                            "content": {"application/json": {"schema": {"type": "object"}}}
                        }
                    }
                }
            },
            "/api/browse": {
                "get": {
                    "operationId": "browseMemories",
                    "summary": "Browse all super contexts and their hierarchy",
                    "responses": {"200": {"description": "Memory hierarchy"}}
                }
            },
            "/api/stats": {
                "get": {
                    "operationId": "getMemoryStats",
                    "summary": "Get memory system statistics",
                    "responses": {"200": {"description": "System statistics"}}
                }
            },
        },
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
        },
        "security": [{"ApiKeyAuth": []}]
    }


# ================================================================
# HYPERPARAMETERS
# ================================================================

@app.get("/api/hyperparams")
async def get_hyperparams():
    """Return current hyperparameters."""
    hp_path = PMIS_DIR / "hyperparameters.yaml"
    if hp_path.exists():
        with open(hp_path) as f:
            return yaml.safe_load(f)
    return _orch.hp


@app.post("/api/hyperparams")
async def update_hyperparams(req: HyperparamUpdate):
    """Update hyperparameters and save to yaml."""
    hp_path = PMIS_DIR / "hyperparameters.yaml"

    # Load current
    if hp_path.exists():
        with open(hp_path) as f:
            current = yaml.safe_load(f) or {}
    else:
        current = dict(_orch.hp)

    # Apply updates
    for key, value in req.updates.items():
        if key in current:
            current[key] = value

    # Save
    with open(hp_path, "w") as f:
        yaml.dump(current, f, default_flow_style=False, sort_keys=False)

    # Reload in orchestrator
    _orch.hp = current

    return {"updated": list(req.updates.keys()), "saved": True}


# ================================================================
# DASHBOARD (server-rendered)
# ================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    template_path = templates_dir / "dashboard.html"
    if template_path.exists():
        return templates.TemplateResponse(request, "dashboard.html", {
            "title": "PMIS V2 Dashboard",
        })
    else:
        # Inline minimal dashboard until template is built
        stats = _orch.get_stats()
        return HTMLResponse(f"""
        <html>
        <head><title>PMIS V2 Dashboard</title>
        <style>
        body {{ font-family: monospace; background: #0a0a0f; color: #e0e0e0; padding: 40px; }}
        h1 {{ color: #fff; }}
        .stat {{ display: inline-block; background: #12121a; border: 1px solid #2a2a3a;
                 border-radius: 10px; padding: 16px 24px; margin: 8px; text-align: center; }}
        .stat .val {{ font-size: 2em; font-weight: bold; color: #4ade80; }}
        .stat .lbl {{ font-size: 0.8em; color: #888; margin-top: 4px; }}
        a {{ color: #60a5fa; }}
        </style></head>
        <body>
        <h1>PMIS V2 Dashboard</h1>
        <p>Server running on port 8100. Full dashboard template coming in Phase 4.</p>
        <div>
            <div class="stat"><div class="val">{stats['super_contexts']}</div><div class="lbl">Super Contexts</div></div>
            <div class="stat"><div class="val">{stats['contexts']}</div><div class="lbl">Contexts</div></div>
            <div class="stat"><div class="val">{stats['anchors']}</div><div class="lbl">Anchors</div></div>
            <div class="stat"><div class="val">{stats['total_nodes']}</div><div class="lbl">Total Nodes</div></div>
            <div class="stat"><div class="val">{stats['orphans']}</div><div class="lbl">Orphans</div></div>
            <div class="stat"><div class="val">{stats['trees']}</div><div class="lbl">Trees</div></div>
        </div>
        <h3>API Endpoints</h3>
        <ul>
            <li><a href="/health">/health</a></li>
            <li><a href="/api/stats">/api/stats</a></li>
            <li><a href="/api/browse">/api/browse</a></li>
            <li><a href="/api/orphans">/api/orphans</a></li>
            <li><a href="/api/conversations">/api/conversations</a></li>
            <li><a href="/api/stats/daily">/api/stats/daily</a></li>
            <li><a href="/api/hyperparams">/api/hyperparams</a></li>
            <li><a href="/docs">/docs</a> (Swagger UI)</li>
        </ul>
        </body></html>
        """)


# ================================================================
# INTEGRATIONS DASHBOARD
# ================================================================

@app.get("/integrations", response_class=HTMLResponse)
async def integrations_page(request: Request):
    """Integration dashboard with setup wizard and live platform status."""
    template_path = templates_dir / "integrations.html"
    if template_path.exists():
        return templates.TemplateResponse(request, "integrations.html", {
            "title": "PMIS V2 — Platform Integrations",
        })
    return HTMLResponse("<h1>Integration template not found. Run the build step.</h1>")


# ================================================================
# WIKI PAGES
# ================================================================

from wiki_renderer import WikiRenderer

def _get_wiki():
    return WikiRenderer(_orch.db)


@app.get("/wiki/", response_class=HTMLResponse)
async def wiki_index(request: Request):
    """Wiki index — all SCs with stats."""
    wiki = _get_wiki()
    data = wiki.render_index()
    return templates.TemplateResponse(request, "wiki_index.html", data)


@app.get("/wiki/node/{node_id}", response_class=HTMLResponse)
async def wiki_node(request: Request, node_id: str):
    """Wiki page for any node (SC, CTX, ANC). Generates LLM prose."""
    wiki = _get_wiki()
    data = wiki.render_node(node_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    # Check for regenerate flag
    regenerate = request.query_params.get("regenerate") == "1"
    if regenerate:
        import sqlite3
        conn = sqlite3.connect(_orch.db.db_path)
        conn.execute("DELETE FROM wiki_page_cache WHERE node_id=?", (node_id,))
        conn.commit()
        conn.close()

    # Generate or retrieve cached prose
    prose_md = wiki.generate_wiki_prose(node_id)
    if prose_md:
        # Convert markdown to HTML (simple conversion)
        prose_html = _markdown_to_html(prose_md)
        data["prose"] = prose_html
    else:
        data["prose"] = None

    return templates.TemplateResponse(request, "wiki_node.html", data)


def _markdown_to_html(md: str) -> str:
    """Simple markdown to HTML conversion."""
    import re
    html = md

    # Headers — H2 gets auto-generated id for anchor links
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)

    def h2_with_id(match):
        title = match.group(1).strip()
        # Strip any existing <span id="..."> wrapper
        clean = re.sub(r'<span[^>]*>([^<]*)</span>', r'\1', title)
        slug = clean.lower().replace(' ', '-').replace('&', 'and').replace('/', '-')
        slug = re.sub(r'[^a-z0-9-]', '', slug)
        return f'<h2 id="{slug}">{clean}</h2>'

    html = re.sub(r'^## (.+)$', h2_with_id, html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

    # Italic
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    # Middot separators (from TOC)
    html = html.replace('&middot;', '&middot;')

    # Links [text](url)
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

    # Paragraphs (double newline)
    paragraphs = html.split('\n\n')
    result = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith('<h') or p.startswith('<ul') or p.startswith('<ol'):
            result.append(p)
        else:
            # Wrap in <p> if not already a block element
            result.append(f'<p>{p}</p>')

    return '\n'.join(result)


@app.get("/wiki/node/{node_id}/backend", response_class=JSONResponse)
async def wiki_node_backend(node_id: str):
    """Backend data panel for a node (JSON)."""
    wiki = _get_wiki()
    data = wiki.render_backend(node_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    return data


@app.get("/wiki/goals", response_class=HTMLResponse)
async def wiki_goals(request: Request):
    """Goals page."""
    wiki = _get_wiki()
    data = wiki.render_goals()
    return templates.TemplateResponse(request, "wiki_goals.html", data)


@app.get("/wiki/feedback", response_class=HTMLResponse)
async def wiki_feedback(request: Request):
    """Feedback log page."""
    wiki = _get_wiki()
    data = wiki.render_feedback_log()
    return templates.TemplateResponse(request, "wiki_feedback.html", data)


@app.get("/wiki/health", response_class=HTMLResponse)
async def wiki_health(request: Request):
    """Health report page."""
    wiki = _get_wiki()
    data = wiki.render_health()
    return templates.TemplateResponse(request, "wiki_health.html", data)


@app.get("/wiki/productivity", response_class=HTMLResponse)
async def wiki_productivity(request: Request):
    """Productivity dashboard — live activity data from tracker."""
    wiki = _get_wiki()
    data = wiki.render_productivity()
    return templates.TemplateResponse(request, "wiki_productivity.html", data)


@app.get("/wiki/diagnostics", response_class=HTMLResponse)
async def wiki_diagnostics(request: Request):
    """Diagnostics page."""
    wiki = _get_wiki()
    data = wiki.render_diagnostics()
    return templates.TemplateResponse(request, "wiki_diagnostics.html", data)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8100,
        reload=False,
        log_level="info",
    )
