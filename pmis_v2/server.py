"""
ProMe HTTP Server

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
import sqlite3
import threading
import time as _time
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# Add pmis_v2 to path (internal modules) + repo root (so `import pmis` works)
PMIS_DIR = Path(__file__).parent
REPO_ROOT = PMIS_DIR.parent
sys.path.insert(0, str(PMIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Request, Query, Header, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
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

    # Robustness: close any work_sessions left hanging >6h (crashed tracker,
    # forgotten /api/work/end, server restart mid-session). Without this, a
    # stale session blocks /api/work/start from ever creating a new one.
    try:
        stale = _orch.db.auto_end_stale_work_sessions(max_age_hours=6.0)
        if stale:
            print(f"[PMIS V2] Auto-ended {len(stale)} stale work session(s): {stale}")
    except Exception as e:
        print(f"[PMIS V2] Stale-session sweep skipped: {e}")

    # Sync YAML PM board (goals.yaml / deliverables.yaml) into projects +
    # deliverables tables so live_matcher and /api/work/deliverables see them.
    # Embedder is passed so deliverables get auto-embedded into synthetic CTX
    # nodes (Fix 2 — without this, LiveMatcher has nothing to score against).
    try:
        from sync.yaml_to_db import sync_pm_yaml_to_db
        counts = sync_pm_yaml_to_db(_orch.db, embedder=_orch.embedder, hyperparams=_orch.hp)
        print(f"[PMIS V2] YAML→DB sync: {counts}")
    except Exception as e:
        print(f"[PMIS V2] YAML→DB sync skipped: {e}")

    # Start agent health checker background thread
    _agent_thread = threading.Thread(target=_agent_health_loop, daemon=True)
    _agent_thread.start()
    print("[PMIS V2] Agent health checker started.")

    # Phase 5 (2026-04-20): two-part in-process scheduler.
    # We inherit the Terminal's TCC permissions when the user launches
    # server.py, so there's no "Operation not permitted" on ~/Desktop like
    # a launchd agent would hit. All consolidation timing lives here.
    #
    #   1. Startup catchup — fires once immediately to consolidate any
    #      missed days since last_consolidation_date.
    #   2. Daily scheduler — polls every 30 min; when clock crosses 18:00
    #      local and today hasn't fired yet, triggers run_idempotent.
    def _runner_startup():
        try:
            from consolidation.runner import run_startup_catchup
            run_startup_catchup()
        except Exception as e:
            print(f"[PMIS V2] Startup catchup failed: {e}")

    def _runner_scheduler():
        try:
            from consolidation.runner import run_daily_scheduler
            run_daily_scheduler(evening_cutoff_hour=18)
        except Exception as e:
            print(f"[PMIS V2] Daily scheduler crashed: {e}")

    threading.Thread(target=_runner_startup, daemon=True, name="pmis-nightly-catchup").start()
    threading.Thread(target=_runner_scheduler, daemon=True, name="pmis-nightly-scheduler").start()
    print("[PMIS V2] Nightly catchup + 18:00 scheduler started (background).")

    yield

    # Cleanup: close any open sessions
    for sid in list(_orch._sessions.keys()):
        _orch.close_session(sid)
    _orch.db.close()
    print("[PMIS V2] Server shutdown complete.")


app = FastAPI(
    title="ProMe Memory System",
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

class IngestRequest(BaseModel):
    text: str
    conversation_id: Optional[str] = None
    role: str = "user"

class AttachRequest(BaseModel):
    node_id: Optional[str] = None
    project: Optional[str] = None

class RetrieveRequest(BaseModel):
    query: str
    mode: str = "auto"
    k: int = 8

class DeleteRequest(BaseModel):
    node_id: Optional[str] = None
    all: bool = False

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


@app.post("/api/match/{match_id}/thumbs")
async def match_thumbs(match_id: str, req: Request):
    """Record user correctness label on a project_work_match_log row.

    Body: {"polarity": "up" | "down"}
    Sets is_correct=1 for up, 0 for down. HGCN consumes these nightly via
    build_match_feedback_edges (Phase 3) to pull/push Poincaré positions.
    """
    body = await req.json()
    polarity = (body.get("polarity") or "").lower()
    if polarity not in ("up", "down"):
        raise HTTPException(status_code=400, detail="polarity must be 'up' or 'down'")
    is_correct = 1 if polarity == "up" else 0
    updated = _orch.db.set_match_correctness(match_id, is_correct)
    if not updated:
        raise HTTPException(status_code=404, detail=f"match {match_id} not found")
    return {"match_id": match_id, "polarity": polarity, "is_correct": is_correct}


@app.post("/api/consolidate")
async def consolidate():
    """Run nightly consolidation."""
    result = _orch.handle_command("consolidate")
    return {"result": result, "completed": True}


# ---------------- 5-verb public API (cognee-parity) ----------------
# These mirror the `pmis` package: ingest / attach / retrieve / delete.
# `consolidate` already exists above; these four complete the set.

@app.post("/api/ingest")
async def api_ingest(req: IngestRequest):
    """Embed + surprise-gate a new memory. Returns node_id or null."""
    import pmis
    node_id = await pmis.ingest(
        req.text,
        conversation_id=req.conversation_id or "web",
        role=req.role,
    )
    return {"node_id": node_id, "stored": node_id is not None}


@app.post("/api/attach")
async def api_attach(req: AttachRequest):
    """Attach an orphan to its nearest Context."""
    import pmis
    return await pmis.attach(req.node_id, project=req.project)


@app.post("/api/retrieve")
async def api_retrieve(req: RetrieveRequest):
    """γ-blended retrieval across semantic/hyperbolic/temporal/precision."""
    import pmis
    hits = await pmis.retrieve(req.query, mode=req.mode, k=req.k)
    return {"hits": hits, "count": len(hits)}


@app.post("/api/delete")
async def api_delete(req: DeleteRequest):
    """Soft-delete a node (node_id) or reset the store (all=true)."""
    import pmis
    return await pmis.delete(req.node_id, all=req.all)


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
            "title": "ProMe Memory System",
            "description": "ProMe — Personal Memory Intelligence System — retrieve and store memories across conversations",
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
# DASHBOARD — shifted to port 8200. See health_dashboard.py /dashboard
# ================================================================


# ================================================================
# INTEGRATIONS DASHBOARD
# ================================================================

@app.get("/integrations", response_class=HTMLResponse)
async def integrations_page(request: Request):
    """Integration dashboard with setup wizard and live platform status."""
    template_path = templates_dir / "integrations.html"
    if template_path.exists():
        return templates.TemplateResponse(request, "integrations.html", {
            "title": "ProMe — Platform Integrations",
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


@app.get("/api/work_pages")
async def api_list_work_pages(date: Optional[str] = None, state: str = "open"):
    """List work_pages filtered by date + state. state in {open,tagged,stale,archived}."""
    from datetime import date as _date
    target = date or _date.today().isoformat()
    pages = _orch.db.list_work_pages_by_state(state, date_local=target)
    for p in pages:
        p.pop("embedding_blob", None)
        p["segment_count"] = len(_orch.db.get_page_segments(p["id"]))
    return {"date": target, "state": state, "count": len(pages), "pages": pages}


@app.get("/api/work_pages/{page_id}")
async def api_get_work_page(page_id: str):
    page = _orch.db.get_work_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=f"work_page {page_id} not found")
    page.pop("embedding_blob", None)
    page["segments"] = _orch.db.get_page_segments(page_id)
    return page


@app.post("/api/work_pages/{page_id}/tag")
async def api_tag_work_page(page_id: str, req: Request):
    """User-tag a page to a project. Body: {project_id, deliverable_id?}."""
    body = await req.json()
    project_id = (body.get("project_id") or "").strip()
    deliverable_id = (body.get("deliverable_id") or "").strip()
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id required")
    page = _orch.db.get_work_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=f"work_page {page_id} not found")
    _orch.db.set_work_page_tag(
        page_id=page_id,
        project_id=project_id,
        deliverable_id=deliverable_id,
        tag_state="confirmed",
        tag_source="user",
    )
    return {
        "ok": True,
        "page_id": page_id,
        "project_id": project_id,
        "deliverable_id": deliverable_id,
    }


@app.post("/api/work_pages/{page_id}/reject")
async def api_reject_work_page(page_id: str):
    """User explicitly rejects a page → archived, not tagged."""
    page = _orch.db.get_work_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=f"work_page {page_id} not found")
    _orch.db.archive_work_page(page_id, tag_state="rejected")
    return {"ok": True, "page_id": page_id, "state": "archived"}


@app.post("/api/work_pages/{page_id}/revive")
async def api_revive_work_page(page_id: str):
    """User: "this isn't kachra" — flip salience back to 'salient',
    clear kachra_reason. Keeps state/tag_state untouched."""
    from sync.salience import revive_page
    result = revive_page(_orch.db, page_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"work_page {page_id} not found")
    return {
        "ok": True,
        "page_id": page_id,
        "salience": result.get("salience"),
        "kachra_reason": result.get("kachra_reason"),
    }


@app.post("/api/sync/rescan-salience")
async def api_rescan_salience(req: Request):
    """Re-score all work_pages. Body: {date?}. Idempotent."""
    from sync.salience import rescan_all
    body = {}
    try:
        body = await req.json()
    except Exception:
        pass
    return rescan_all(_orch.db, date_local=body.get("date"))


@app.post("/api/sync/humanize")
async def api_sync_humanize(req: Request):
    """Batch humanize. Body: {date?, force?, local?, model?}."""
    from sync.humanizer import humanize_all
    body = {}
    try:
        body = await req.json()
    except Exception:
        pass
    hp = dict(_orch.hp) if hasattr(_orch, "hp") else {}
    if body.get("model"):
        hp["humanize_model_cloud"] = body["model"]
    if body.get("local"):
        hp["humanize_use_cloud"] = False
    return humanize_all(
        _orch.db, hp,
        date_local=body.get("date"),
        force=bool(body.get("force", False)),
    )


@app.get("/api/narratives")
async def api_list_narratives(date: Optional[str] = None, limit: int = 50):
    """List narratives for a date (default: today). Newest first otherwise."""
    narratives = _orch.db.list_narratives(date_local=date, limit=limit)
    return {"date": date, "count": len(narratives), "narratives": narratives}


@app.post("/api/narratives/compose")
async def api_compose_narratives(req: Request):
    """Compose daily stories for a target date. Body: {date?}. Wipes
    existing narratives for that date and writes the fresh set."""
    from sync.narrator import compose_narratives_for_date
    body = {}
    try:
        body = await req.json()
    except Exception:
        pass
    hp = dict(_orch.hp) if hasattr(_orch, "hp") else {}
    return compose_narratives_for_date(
        _orch.db, hp,
        target_date=body.get("date"),
        generated_by="manual",
    )


@app.post("/api/work_pages/{page_id}/humanize")
async def api_work_page_humanize(page_id: str, req: Request):
    """Humanize a single page. Body: {force?, local?, model?}."""
    from sync.humanizer import humanize_page
    page = _orch.db.get_work_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=f"work_page {page_id} not found")
    body = {}
    try:
        body = await req.json()
    except Exception:
        pass
    hp = dict(_orch.hp) if hasattr(_orch, "hp") else {}
    if body.get("model"):
        hp["humanize_model_cloud"] = body["model"]
    if body.get("local"):
        hp["humanize_use_cloud"] = False
    return humanize_page(_orch.db, page, hp, force=bool(body.get("force", False)))


@app.post("/api/work_pages/{page_id}/confirm")
async def api_confirm_work_page(page_id: str):
    """Confirm a Dream-proposed tag. Flips tag_state='proposed' → 'confirmed'
    and state='open' → 'tagged'. Only confirmed tags feed HGCN feedback edges."""
    result = _orch.db.confirm_work_page_proposal(page_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"no proposed tag found on work_page {page_id}",
        )
    return {
        "ok": True,
        "page_id": page_id,
        "project_id": result.get("project_id"),
        "deliverable_id": result.get("deliverable_id"),
        "state": result.get("state"),
        "tag_state": result.get("tag_state"),
    }


@app.post("/api/dream/match-pages")
async def api_dream_match_pages(req: Request):
    """Manually trigger Dream's auto-match pass. Body: {force?, since?}."""
    from consolidation.work_page_matcher import run_work_page_matching
    body = {}
    try:
        body = await req.json()
    except Exception:
        pass
    hp = _orch.hp if hasattr(_orch, "hp") else {}
    return run_work_page_matching(
        _orch.db, hp,
        force=bool(body.get("force", False)),
        since_date=body.get("since"),
    )


# ─── Project digests ───────────────────────────────────────────────────

def _project_title_for(project_id: str) -> str:
    """Look up a project's display title from goals.yaml (best-effort)."""
    try:
        wiki = _get_wiki()
        pm = wiki._render_pm_projects()
        for g in pm.get("goals", []):
            for p in g.get("projects", []):
                if p.get("id") == project_id:
                    return p.get("title", project_id)
    except Exception:
        pass
    return project_id


@app.get("/api/project/{project_id}/digests")
async def api_list_project_digests(
    project_id: str, window_type: Optional[str] = None, limit: int = 20
):
    """Recent digests for a project. Newest first."""
    digests = _orch.db.list_project_digests(
        project_id=project_id, window_type=window_type, limit=limit
    )
    return {
        "project_id": project_id,
        "count": len(digests),
        "digests": digests,
    }


@app.post("/api/project/{project_id}/digest")
async def api_generate_project_digest(project_id: str, req: Request):
    """Compose a digest for a project over a given window.

    Body: {window_start: YYYY-MM-DD, window_end: YYYY-MM-DD, window_type: day|week|custom, model?}
    Returns the composed digest including the markdown body. Stores to DB
    (upserts on conflict with same user+project+type+start).
    """
    body = await req.json()
    window_start = (body.get("window_start") or "").strip()
    window_end = (body.get("window_end") or window_start).strip()
    window_type = (body.get("window_type") or "day").strip()
    model = (body.get("model") or "").strip() or None

    if not window_start:
        raise HTTPException(status_code=400, detail="window_start required (YYYY-MM-DD)")
    if window_type not in ("day", "week", "custom"):
        raise HTTPException(status_code=400, detail="window_type in {day, week, custom}")

    from reports.digest_composer import compose_digest

    hp = _orch.hp if hasattr(_orch, "hp") else {}
    llm_model = model or hp.get("consolidation_model_local", "qwen2.5:14b")

    result = compose_digest(
        db=_orch.db,
        project_id=project_id,
        window_start=window_start,
        window_end=window_end,
        window_type=window_type,
        project_title=_project_title_for(project_id),
        model=llm_model,
        generated_by="manual",
    )
    return result


@app.get("/wiki/goals", response_class=HTMLResponse)
async def wiki_goals(request: Request):
    """Goals page."""
    wiki = _get_wiki()
    data = wiki.render_goals()
    # Phase 3: surface pending-review count on the Goals tab as a badge
    try:
        data["review_pending_count"] = _review_pending_count()
    except Exception:
        data["review_pending_count"] = 0
    return templates.TemplateResponse(request, "wiki_goals.html", data)


# ─── Phase C: project shell view ─────────────────────────────────────
#
# /wiki/goals/p/{pid}            — opens first deliverable, or Overview
#                                   when the project has none yet.
# /wiki/goals/p/{pid}/d/{did}    — same shell with a specific deliverable
#                                   selected. 404 when did not under pid.

@app.get("/wiki/goals/p/{project_id}", response_class=HTMLResponse)
async def wiki_project_detail(
    request: Request, project_id: str,
    view: str = "overall", date: Optional[str] = None,
):
    wiki = _get_wiki()
    data = wiki.render_project_detail(project_id, view=view, date=date)
    if data is None:
        raise HTTPException(status_code=404, detail=f"project {project_id} not found")
    if data["deliverables"]:
        first_id = data["deliverables"][0].get("id") or ""
        if first_id:
            qs = f"?view={view}" + (f"&date={date}" if date else "")
            return RedirectResponse(
                url=f"/wiki/goals/p/{project_id}/d/{first_id}{qs}",
                status_code=302,
            )
    return templates.TemplateResponse(request, "wiki_project.html", data)


@app.get(
    "/wiki/goals/p/{project_id}/d/{deliverable_id}",
    response_class=HTMLResponse,
)
async def wiki_project_deliverable(
    request: Request, project_id: str, deliverable_id: str,
    view: str = "overall", date: Optional[str] = None,
):
    wiki = _get_wiki()
    data = wiki.render_project_detail(
        project_id, deliverable_id, view=view, date=date,
    )
    if data is None:
        raise HTTPException(status_code=404, detail=f"project {project_id} not found")
    if data["selected_deliverable"] is None:
        raise HTTPException(
            status_code=404,
            detail=f"deliverable {deliverable_id} not under project {project_id}",
        )
    return templates.TemplateResponse(request, "wiki_project.html", data)


# ─── Review tab ───────────────────────────────────────────────────────
#
# Two-stage flow: the user sees raw unconsolidated segments, clicks
# Consolidate, then confirms or rejects each proposed group. Confirming
# writes activity_time_log rows with match_source='user_review' — the
# highest-authority tag nightly + manual consolidation will then defer to.

def _review_proposals():
    """Build the ReviewProposals service — lazy so imports stay cheap."""
    from review.proposals import ReviewProposals
    from core import config as _cfg
    from db.manager import DBManager as _DB
    db = _DB(_cfg.get("db_path", "data/memory.db"))
    return ReviewProposals(db, _cfg.get_all()), db


REVIEW_DEFAULT_DAYS = 2  # window size for the Review page's unconsolidated list


def _review_pending_count() -> int:
    """Count unconsolidated segments + active drafts — drives the Goals badge."""
    try:
        rp, _ = _review_proposals()
        return (len(rp.list_unconsolidated(None, days=REVIEW_DEFAULT_DAYS))
                + len(rp.list_drafts(None)))
    except Exception:
        return 0


def _review_payload(days: int = REVIEW_DEFAULT_DAYS) -> Dict[str, Any]:
    """Assemble the wiki_review.html context dict for the two-state UI:
    raw unconsolidated segments (bounded window) + active draft proposals."""
    try:
        rp, _ = _review_proposals()
        unconsolidated = rp.list_unconsolidated(None, days=days)
        drafts = rp.list_drafts(None)
    except Exception as e:
        import logging
        logging.getLogger("pmis.server").warning(f"review_payload failed: {e}")
        unconsolidated = []
        drafts = []

    # Surface the active-projects catalog so per-group pickers can offer
    # "tag to something else" from the same flat list. Reuses the same
    # merger the scorer uses for symmetry.
    try:
        rp2, _ = _review_proposals()
        projects_catalog = rp2._list_active_projects()
    except Exception:
        projects_catalog = []

    return {
        "unconsolidated": unconsolidated,
        "drafts": drafts,
        "projects_catalog": [
            {"id": p["id"], "name": p.get("name") or p["id"]}
            for p in projects_catalog
        ],
        "unconsolidated_count": len(unconsolidated),
        "draft_count": len(drafts),
        "window_days": days,
    }


@app.get("/wiki/review", response_class=HTMLResponse)
async def wiki_review(request: Request, days: int = REVIEW_DEFAULT_DAYS):
    """Review page — two-state: raw unconsolidated segments + draft proposals.
    Defaults to a 2-day window (today + yesterday); ?days=N to widen."""
    return templates.TemplateResponse(request, "wiki_review.html", _review_payload(days=days))


# ─── Review_proposals flow ────────────────────────────────────────────


class ReviewConsolidatePayload(BaseModel):
    date: Optional[str] = None
    days: Optional[int] = REVIEW_DEFAULT_DAYS


class ReviewProposalConfirmPayload(BaseModel):
    project_id: str
    deliverable_id: Optional[str] = ""


class ReviewProposalAssignPayload(BaseModel):
    project_id: Optional[str] = ""
    deliverable_id: Optional[str] = ""


@app.get("/api/review/unconsolidated")
async def api_review_unconsolidated(
    date: Optional[str] = None,
    days: int = REVIEW_DEFAULT_DAYS,
):
    rp, _ = _review_proposals()
    return {"segments": rp.list_unconsolidated(date, days=days), "window_days": days}


@app.post("/api/review/consolidate")
async def api_review_consolidate(payload: ReviewConsolidatePayload):
    rp, _ = _review_proposals()
    proposals = rp.consolidate(payload.date, days=payload.days)
    return {"ok": True, "proposals": proposals}


@app.get("/api/review/proposals")
async def api_review_proposals(date: Optional[str] = None):
    rp, _ = _review_proposals()
    return {"proposals": rp.list_drafts(date)}


@app.post("/api/review/proposals/{proposal_id}/confirm")
async def api_review_proposal_confirm(proposal_id: str, payload: ReviewProposalConfirmPayload):
    rp, _ = _review_proposals()
    result = rp.confirm(proposal_id, payload.project_id, payload.deliverable_id or "")
    if not result.get("ok"):
        status = 503 if result.get("error") == "consolidation_locked" else 400
        raise HTTPException(status_code=status, detail=result)
    return result


@app.post("/api/review/proposals/{proposal_id}/reject")
async def api_review_proposal_reject(proposal_id: str):
    rp, _ = _review_proposals()
    result = rp.reject(proposal_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result)
    return result


@app.patch("/api/review/proposals/{proposal_id}")
async def api_review_proposal_assign(proposal_id: str, payload: ReviewProposalAssignPayload):
    rp, _ = _review_proposals()
    return rp.set_assignment(
        proposal_id, payload.project_id or "", payload.deliverable_id or ""
    )


# ─── Manual per-project daily consolidation ──────────────────────────

def _manual_consolidator():
    from consolidation.manual_project import ManualProjectConsolidator
    from core import config as _cfg
    from db.manager import DBManager as _DB
    db = _DB(_cfg.get("db_path", "data/memory.db"))
    return ManualProjectConsolidator(db, _cfg.get_all()), db


@app.get("/api/project/{project_id}/consolidate-preview")
async def api_project_consolidate_preview(project_id: str, date: str):
    """Return the LLM-generated markdown draft + segment list — user edits
    before confirming. Does NOT persist anything."""
    mc, _db = _manual_consolidator()
    segments = mc.collect_segments(project_id, date)
    draft = mc.draft_summary(project_id, date, segments) if segments else ""
    return {
        "project_id": project_id,
        "date": date,
        "segment_count": len(segments),
        "duration_mins": round(
            sum(s.get("duration_secs", 10) for s in segments) / 60.0, 1
        ),
        "draft_markdown": draft,
        "segment_ids": [s.get("id") for s in segments],
    }


class ManualConsolidatePayload(BaseModel):
    date: str
    edited_markdown: str
    segment_ids: Optional[List[str]] = None


@app.post("/api/project/{project_id}/consolidate-day")
async def api_project_consolidate_day(
    project_id: str, payload: ManualConsolidatePayload
):
    """Persist the manual consolidation.

    - Recollects segments to make sure we mark-as-consolidated exactly the
      ones the caller saw (optionally filtered by `segment_ids` if provided).
    - Marks those segments so nightly won't touch them.
    - Stores the final markdown as an ANC attached to the project's SC.
    """
    mc, _db = _manual_consolidator()
    segments = mc.collect_segments(project_id, payload.date)
    if payload.segment_ids:
        wanted = set(payload.segment_ids)
        segments = [s for s in segments if s.get("id") in wanted]
    if not segments:
        raise HTTPException(
            status_code=400,
            detail="No segments to consolidate (either already consolidated, or no tagged work_sessions found).",
        )
    result = mc.commit(project_id, payload.date, payload.edited_markdown, segments)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "commit failed"))
    return result


# ─── PM board CRUD (Goal → Project → Deliverable) ─────────────────────

class PMGoalsPayload(BaseModel):
    goals: List[Dict[str, Any]]


@app.get("/api/pm/goals")
async def api_pm_goals():
    """Enriched PM board state (for JS hydration)."""
    wiki = _get_wiki()
    return wiki.render_pm_projects()


@app.put("/api/pm/goals")
async def api_pm_goals_save(payload: PMGoalsPayload):
    """Persist goals.yaml from UI state; returns re-enriched view."""
    wiki = _get_wiki()
    try:
        return wiki.save_pm_goals(payload.goals)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/pm/deliverables")
async def api_pm_deliverables():
    """Deliverables catalog for the picker (reads deliverables.yaml)."""
    wiki = _get_wiki()
    return {"deliverables": wiki.list_pm_deliverables()}


class PMDeliverableCreate(BaseModel):
    name: str
    supercontext: Optional[str] = None
    project_id: Optional[str] = None
    deadline: Optional[str] = None
    description: Optional[str] = None


@app.post("/api/pm/deliverables")
async def api_pm_deliverable_create(payload: PMDeliverableCreate):
    """Create a new deliverable:
      1. Append to deliverables.yaml (auto D-00X id)
      2. Optionally link to project_id via goals.yaml deliverable_patterns
      3. Re-run YAML→DB sync so the new row appears in the picker and
         gets a synthetic CTX node auto-embedded (Fix 2). Immediately
         matchable by the LiveMatcher.
    """
    wiki = _get_wiki()
    try:
        created = wiki.create_pm_deliverable(payload.name, payload.supercontext)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Add optional fields straight into deliverables.yaml row
    try:
        import yaml as _yaml
        pt_root = Path.home() / "Desktop" / "memory" / "productivity-tracker"
        deliv_path = pt_root / "config" / "deliverables.yaml"
        if deliv_path.exists():
            raw = _yaml.safe_load(deliv_path.read_text()) or {}
            for d in raw.get("deliverables", []) or []:
                if d.get("id") == created.get("id"):
                    if payload.deadline: d["deadline"] = payload.deadline
                    if payload.description: d["description"] = payload.description
                    if payload.supercontext: d["supercontext"] = payload.supercontext
                    break
            deliv_path.write_text(_yaml.safe_dump(raw, sort_keys=False, default_flow_style=False))
    except Exception:
        pass

    # Link to a project: add this deliverable's id as a key under the project's
    # deliverable_patterns so the pattern miner + yaml_to_db.sync can see it.
    if payload.project_id:
        try:
            import yaml as _yaml
            pt_root = Path.home() / "Desktop" / "memory" / "productivity-tracker"
            goals_path = pt_root / "config" / "goals.yaml"
            if goals_path.exists():
                g = _yaml.safe_load(goals_path.read_text()) or {}
                for goal in g.get("goals", []) or []:
                    for proj in goal.get("projects", []) or []:
                        if proj.get("id") == payload.project_id:
                            dp = proj.setdefault("deliverable_patterns", {})
                            dp.setdefault(created["id"], [payload.name.split()[0] if payload.name else created["id"]])
                            break
                goals_path.write_text(_yaml.safe_dump(g, sort_keys=False, default_flow_style=False))
        except Exception:
            pass

    # Run sync so new rows land in projects+deliverables DB tables and
    # LiveMatcher gets a new candidate to score against.
    try:
        from sync.yaml_to_db import sync_pm_yaml_to_db
        sync_counts = sync_pm_yaml_to_db(_orch.db, embedder=_orch.embedder, hyperparams=_orch.hp)
        created["sync_counts"] = sync_counts
    except Exception as e:
        created["sync_error"] = str(e)

    return created


class PMProjectCreate(BaseModel):
    title: str
    goal_id: Optional[str] = None            # attach to an existing goal (fallback: first active)
    description: Optional[str] = None
    deadline: Optional[str] = None
    match_patterns: Optional[List[str]] = None


@app.post("/api/pm/projects")
async def api_pm_project_create(payload: PMProjectCreate):
    """Create a new project inside goals.yaml, under an existing goal (given
    by goal_id or the first active goal as fallback). Re-syncs YAML → DB."""
    import yaml as _yaml
    import uuid as _uuid

    title = (payload.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title required")

    pt_root = Path.home() / "Desktop" / "memory" / "productivity-tracker"
    goals_path = pt_root / "config" / "goals.yaml"
    goals_path.parent.mkdir(parents=True, exist_ok=True)
    raw: Dict[str, Any] = {}
    if goals_path.exists():
        try:
            raw = _yaml.safe_load(goals_path.read_text()) or {}
        except Exception:
            raw = {}
    goals_list = raw.get("goals", []) or []
    if not goals_list:
        # seed a generic container goal so there's somewhere to put the project
        goals_list = [{"id": f"G-{_uuid.uuid4().hex[:9]}",
                        "title": "General", "why": "", "status": "active",
                        "projects": []}]
        raw["goals"] = goals_list

    target_goal = None
    if payload.goal_id:
        for g in goals_list:
            if g.get("id") == payload.goal_id:
                target_goal = g
                break
    if target_goal is None:
        for g in goals_list:
            if g.get("status", "active") == "active":
                target_goal = g
                break
    if target_goal is None:
        target_goal = goals_list[0]

    # Generate a P-XXX id unique across the file
    used = set()
    for g in goals_list:
        for p in g.get("projects", []) or []:
            pid = p.get("id") or ""
            if pid:
                used.add(pid)
    new_id = None
    for n in range(1, 999):
        cand = f"P-{n:03d}"
        if cand not in used:
            new_id = cand
            break
    if new_id is None:
        new_id = f"P-{_uuid.uuid4().hex[:9]}"

    new_project = {
        "id": new_id,
        "title": title,
        "status": "active",
        "target_week": "",
        "match_patterns": payload.match_patterns or [],
        "deliverable_patterns": {},
        "lifecycle": [{
            "date": datetime.now().date().isoformat(),
            "event": "Inception",
        }],
    }
    target_goal.setdefault("projects", []).append(new_project)

    goals_path.write_text(_yaml.safe_dump(raw, sort_keys=False, default_flow_style=False))

    sync_counts: Dict[str, Any] = {}
    try:
        from sync.yaml_to_db import sync_pm_yaml_to_db
        sync_counts = sync_pm_yaml_to_db(_orch.db, embedder=_orch.embedder, hyperparams=_orch.hp)
    except Exception as e:
        sync_counts = {"error": str(e)}

    return {
        "id": new_id,
        "title": title,
        "goal_id": target_goal.get("id"),
        "sync_counts": sync_counts,
    }


class PMProjectUpdate(BaseModel):
    title: str


@app.put("/api/pm/projects/{project_id}")
async def api_pm_project_update(project_id: str, payload: PMProjectUpdate):
    """Inline rename a project. Mutates goals.yaml in place — finds the
    project by id across all goals, updates its title, writes back, and
    re-runs the YAML→DB sync so the new name surfaces immediately."""
    import yaml as _yaml

    title = (payload.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title required")

    pt_root = Path.home() / "Desktop" / "memory" / "productivity-tracker"
    goals_path = pt_root / "config" / "goals.yaml"
    if not goals_path.exists():
        raise HTTPException(status_code=404, detail="goals.yaml not found")

    try:
        raw = _yaml.safe_load(goals_path.read_text()) or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yaml parse: {e}")

    found_goal_id: Optional[str] = None
    for g in raw.get("goals", []) or []:
        for p in g.get("projects", []) or []:
            if p.get("id") == project_id:
                p["title"] = title
                found_goal_id = g.get("id")
                break
        if found_goal_id is not None:
            break

    if found_goal_id is None:
        raise HTTPException(status_code=404, detail=f"project {project_id} not found")

    goals_path.write_text(
        _yaml.safe_dump(raw, sort_keys=False, default_flow_style=False)
    )

    sync_counts: Dict[str, Any] = {}
    try:
        from sync.yaml_to_db import sync_pm_yaml_to_db
        sync_counts = sync_pm_yaml_to_db(
            _orch.db, embedder=_orch.embedder, hyperparams=_orch.hp,
        )
    except Exception as e:
        sync_counts = {"error": str(e)}

    return {
        "id": project_id,
        "title": title,
        "goal_id": found_goal_id,
        "sync_counts": sync_counts,
    }


@app.get("/api/pm/projects")
async def api_pm_projects():
    """List active projects (flat, for widget picker)."""
    return {"projects": _orch.db.list_projects(status="active")}


# ─── Phase D4: link binding contributed-toggle ───────────────────────

class LinkBindingToggle(BaseModel):
    link_id: str
    scope: str
    scope_id: str
    contributed: int  # 0 or 1


# ─── Phase E: work-match actions + daily feedback ───────────────────

class WorkMatchReassignPayload(BaseModel):
    project_id: str
    deliverable_id: str = ""


@app.post("/api/work_match/{match_id}/confirm")
async def api_work_match_confirm(match_id: str):
    """Flip is_correct=1 on a project_work_match_log row. Used by the
    Daily view's ✓ button."""
    conn = sqlite3.connect(_orch.db.db_path)
    try:
        cur = conn.execute(
            "UPDATE project_work_match_log SET is_correct = 1 "
            "WHERE id = ?",
            (match_id,),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"match {match_id} not found")
    finally:
        conn.close()
    return {"match_id": match_id, "is_correct": 1}


@app.post("/api/work_match/{match_id}/remove")
async def api_work_match_remove(match_id: str):
    """Strike a wrongly-tagged work item: is_correct=0 + clear the
    deliverable_id so it no longer counts toward this deliverable's
    daily view. The segment goes back to the unconsolidated pool on
    the next consolidate run."""
    conn = sqlite3.connect(_orch.db.db_path)
    try:
        cur = conn.execute(
            "UPDATE project_work_match_log "
            "SET is_correct = 0, deliverable_id = '' "
            "WHERE id = ?",
            (match_id,),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"match {match_id} not found")
    finally:
        conn.close()
    return {"match_id": match_id, "is_correct": 0, "deliverable_id": ""}


@app.post("/api/work_match/{match_id}/reassign")
async def api_work_match_reassign(
    match_id: str, payload: WorkMatchReassignPayload,
):
    """Move a tagged work item to a different project + deliverable.
    Keeps is_correct=1 (the user is making a positive assertion about
    the new target, not rejecting the segment)."""
    if not payload.project_id:
        raise HTTPException(status_code=400, detail="project_id required")
    conn = sqlite3.connect(_orch.db.db_path)
    try:
        cur = conn.execute(
            "UPDATE project_work_match_log "
            "SET project_id = ?, deliverable_id = ?, is_correct = 1 "
            "WHERE id = ?",
            (payload.project_id, payload.deliverable_id or "", match_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"match {match_id} not found")
    finally:
        conn.close()
    return {
        "match_id": match_id,
        "project_id": payload.project_id,
        "deliverable_id": payload.deliverable_id or "",
        "is_correct": 1,
    }


class DailyFeedbackPayload(BaseModel):
    feedback_text: str


@app.post("/api/daily/{daily_summary_id}/feedback")
async def api_daily_feedback(
    daily_summary_id: str, payload: DailyFeedbackPayload,
):
    """Free-text feedback on a past daily summary. Phase G's nightly
    pass picks up daily_feedback rows where applied=0, re-composes the
    target summary, and flips applied=1."""
    text = (payload.feedback_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="feedback_text required")

    conn = sqlite3.connect(_orch.db.db_path)
    try:
        # Confirm the daily exists — feeding back into a phantom row
        # would just create unprocessable noise for nightly.
        row = conn.execute(
            "SELECT id FROM daily_summaries WHERE id = ?",
            (daily_summary_id,),
        ).fetchone()
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"daily_summary {daily_summary_id} not found",
            )
        conn.execute(
            "INSERT INTO daily_feedback (daily_summary_id, feedback_text) "
            "VALUES (?, ?)",
            (daily_summary_id, text[:2000]),
        )
        conn.commit()
        fb_id = conn.execute(
            "SELECT last_insert_rowid()"
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        "id": fb_id,
        "daily_summary_id": daily_summary_id,
        "feedback_text": text[:2000],
        "applied": 0,
    }


@app.patch("/api/links/bindings")
async def api_link_binding_toggle(payload: LinkBindingToggle):
    """Flip link_bindings.contributed for a single binding.
    UI-driven ✓/✗ override on the deliverable page."""
    if payload.contributed not in (0, 1):
        raise HTTPException(
            status_code=400, detail="contributed must be 0 or 1",
        )
    if payload.scope not in ("project", "deliverable", "daily",
                             "work_match", "frame", "segment"):
        raise HTTPException(
            status_code=400, detail=f"invalid scope: {payload.scope}",
        )

    conn = sqlite3.connect(_orch.db.db_path)
    try:
        cur = conn.execute(
            "UPDATE link_bindings SET contributed = ? "
            "WHERE link_id = ? AND scope = ? AND scope_id = ?",
            (payload.contributed, payload.link_id,
             payload.scope, payload.scope_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail=(f"binding not found: link_id={payload.link_id} "
                        f"scope={payload.scope} scope_id={payload.scope_id}"),
            )
        row = conn.execute(
            """SELECT lb.link_id, l.url, l.kind, lb.contributed,
                      lb.dwell_frames
               FROM link_bindings lb JOIN links l ON l.id = lb.link_id
               WHERE lb.link_id = ? AND lb.scope = ? AND lb.scope_id = ?""",
            (payload.link_id, payload.scope, payload.scope_id),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        # The UPDATE succeeded but the JOIN returned nothing — orphaned
        # binding; rare, but bubble it up as 404 rather than crash.
        raise HTTPException(status_code=404, detail="link missing for binding")
    return {
        "link_id": row[0], "url": row[1], "kind": row[2] or "other",
        "contributed": int(row[3]), "dwell_frames": int(row[4] or 0),
    }


# ─── Phase C2: deliverable section scaffold ──────────────────────────

_DELIVERABLE_SLOTS = (
    "overview", "progress", "decisions", "questions", "risks", "links",
)
_DELIVERABLE_SOURCES = ("user", "auto", "llm")


class PMDeliverableSectionUpdate(BaseModel):
    body_md: str = ""
    source: str = "user"


@app.put("/api/pm/deliverables/{deliverable_id}/sections/{slot}")
async def api_pm_deliverable_section_update(
    deliverable_id: str, slot: str,
    payload: PMDeliverableSectionUpdate,
):
    """UPSERT one slot of a deliverable's 6-slot scaffold. Returns the
    rendered HTML so the UI can swap content without a full reload."""
    if slot not in _DELIVERABLE_SLOTS:
        raise HTTPException(
            status_code=400,
            detail=f"slot must be one of {_DELIVERABLE_SLOTS}",
        )
    if payload.source not in _DELIVERABLE_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"source must be one of {_DELIVERABLE_SOURCES}",
        )
    if not deliverable_id:
        raise HTTPException(status_code=400, detail="deliverable_id required")

    body_md = payload.body_md or ""
    # C3: empty save = "give me the auto-fill back". Saving empty
    # content with source='user' would land in a confusing state
    # (auto-fill suppressed, body shown empty), so we coerce to
    # 'auto' instead. The user can re-save real content to flip
    # source back to 'user'.
    effective_source = "auto" if not body_md.strip() else payload.source

    conn = sqlite3.connect(_orch.db.db_path)
    try:
        conn.execute(
            """INSERT INTO deliverable_sections
               (deliverable_id, slot, body_md, source, updated_at)
               VALUES (?, ?, ?, ?, datetime('now'))
               ON CONFLICT(deliverable_id, slot) DO UPDATE SET
                   body_md = excluded.body_md,
                   source = excluded.source,
                   updated_at = excluded.updated_at""",
            (deliverable_id, slot, body_md, effective_source),
        )
        conn.commit()
        row = conn.execute(
            "SELECT updated_at FROM deliverable_sections "
            "WHERE deliverable_id = ? AND slot = ?",
            (deliverable_id, slot),
        ).fetchone()
    finally:
        conn.close()

    body_html = _markdown_to_html(body_md) if body_md else ""
    return {
        "deliverable_id": deliverable_id,
        "slot": slot,
        "body_md": body_md,
        "body_html": body_html,
        "source": effective_source,
        "updated_at": row[0] if row else "",
    }


class PMQuickAddPayload(BaseModel):
    text: str
    threshold: Optional[float] = 0.55      # cosine threshold for linking to existing project
    force_new_project: Optional[bool] = False


def _derive_project_title(text: str, max_words: int = 5) -> str:
    """Heuristic: first few meaningful words, Title Cased. Good enough for
    auto-generated project names until an LLM-mediated path is added."""
    import re
    stop = {"the","a","an","for","to","of","on","in","and","or","about","with","my","our"}
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", text)]
    keep = [w for w in words if w.lower() not in stop][:max_words]
    if not keep:
        keep = words[:max_words]
    return " ".join(w.capitalize() for w in keep) or "Untitled"


@app.post("/api/pm/quick-add")
async def api_pm_quick_add(payload: PMQuickAddPayload):
    """Text-prompt quick-add. Embeds the user's input, cosine-matches it
    against every active project's context embedding (the synthetic CTX
    node created by Fix-2 sync). If best similarity ≥ threshold → add as
    deliverable under that project. Otherwise → spin up a new project
    from the text + add the deliverable under it.

    Returns {mode: 'linked' | 'new_project',
              project_id, project_title, deliverable_id,
              matched_score, candidates: [{project_id, score}...]}
    """
    import numpy as _np
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")

    # 1. Embed the user's text
    try:
        text_emb = _orch.embedder.embed_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embed failed: {e}")

    # 2. Collect active projects + their representative embeddings.
    # Representative = the project's sc_node_id embedding, else the
    # embedding of any deliverable's context_node under that project.
    projects = _orch.db.list_projects(status="active")
    candidates: List[Dict[str, Any]] = []

    def _cos(a, b):
        if a is None or b is None: return 0.0
        if getattr(a, "shape", None) != getattr(b, "shape", None): return 0.0
        na = float(_np.linalg.norm(a)); nb = float(_np.linalg.norm(b))
        if na == 0 or nb == 0: return 0.0
        return max(0.0, min(1.0, float(_np.dot(a, b) / (na * nb))))

    with _orch.db._connect() as conn:
        for p in projects:
            pid = p.get("id")
            rep_node_ids = []
            if p.get("sc_node_id"):
                rep_node_ids.append(p["sc_node_id"])
            deliv_rows = conn.execute(
                "SELECT context_node_id FROM deliverables WHERE project_id = ? AND status='active'",
                (pid,),
            ).fetchall()
            for r in deliv_rows:
                if r["context_node_id"]:
                    rep_node_ids.append(r["context_node_id"])

            best_score = 0.0
            for nid in rep_node_ids:
                try:
                    eu = _orch.db.get_embeddings(nid).get("euclidean")
                    if eu is None: continue
                    s = _cos(text_emb, eu)
                    if s > best_score:
                        best_score = s
                except Exception:
                    continue
            # Also blend a cheap name-level signal for projects with no embeddable descendants
            if best_score < 0.1 and p.get("name"):
                try:
                    name_emb = _orch.embedder.embed_text(p["name"])
                    best_score = _cos(text_emb, name_emb)
                except Exception:
                    pass
            candidates.append({
                "project_id": pid,
                "project_name": p.get("name") or pid,
                "score": best_score,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    threshold = float(payload.threshold or 0.55)
    best = candidates[0] if candidates else None
    link_to_existing = (
        (not payload.force_new_project)
        and best is not None
        and best["score"] >= threshold
    )

    # Build the deliverable payload that both branches will write
    deliv_name = text[:100]   # keep deliverable name ≈ the raw intent

    if link_to_existing:
        # Route through the regular /api/pm/deliverables handler for
        # consistent yaml+db+embed side-effects
        deliv_res = await api_pm_deliverable_create(PMDeliverableCreate(
            name=deliv_name,
            project_id=best["project_id"],
            description=text,
        ))
        return {
            "mode": "linked",
            "project_id": best["project_id"],
            "project_title": best["project_name"],
            "deliverable_id": deliv_res.get("id"),
            "deliverable_name": deliv_name,
            "matched_score": round(best["score"], 3),
            "candidates": [
                {"project_id": c["project_id"], "project_name": c["project_name"],
                 "score": round(c["score"], 3)}
                for c in candidates[:3]
            ],
        }

    # 3. No good match → spin up a new project, then add the deliverable
    title = _derive_project_title(text)
    proj_res = await api_pm_project_create(PMProjectCreate(title=title))
    deliv_res = await api_pm_deliverable_create(PMDeliverableCreate(
        name=deliv_name,
        project_id=proj_res["id"],
        description=text,
    ))
    return {
        "mode": "new_project",
        "project_id": proj_res["id"],
        "project_title": title,
        "deliverable_id": deliv_res.get("id"),
        "deliverable_name": deliv_name,
        "matched_score": round(best["score"], 3) if best else 0.0,
        "candidates": [
            {"project_id": c["project_id"], "project_name": c["project_name"],
             "score": round(c["score"], 3)}
            for c in candidates[:3]
        ],
    }


# Feedback / Health (lint) / Diagnostics shifted to port 8200.
# See health_dashboard.py: /feedback, /lint, /diagnostics.


# ─── Phase 1: Live work sessions (infinite-loop widget) ───────────────

_live_matcher = None


def _get_live_matcher():
    """Lazy construct — Orchestrator must be initialized first."""
    global _live_matcher
    if _live_matcher is None:
        from retrieval.live_matcher import LiveMatcher
        _live_matcher = LiveMatcher(_orch.db, _orch.embedder, hyperparams=_orch.hp)
    return _live_matcher


class WorkStartPayload(BaseModel):
    deliverable_id: Optional[str] = None
    project_id: Optional[str] = None
    auto_assigned: Optional[int] = 0
    note: Optional[str] = ""


@app.post("/api/work/start")
async def api_work_start(payload: WorkStartPayload):
    """Begin a work session. If deliverable_id is empty the session is
    'observing' — suggestions flow but nothing is hard-bound yet."""
    active = _orch.db.get_active_work_session()
    if active:
        return {"status": "already_active", "session": active}

    project_id = payload.project_id or ""
    if payload.deliverable_id and not project_id:
        deliv = _orch.db.get_deliverable(payload.deliverable_id)
        if deliv:
            project_id = deliv.get("project_id", "")

    sid = _orch.db.create_work_session(
        project_id=project_id,
        deliverable_id=payload.deliverable_id or "",
        auto_assigned=int(payload.auto_assigned or 0),
        confirmed_by_user=1 if payload.deliverable_id else 0,
        note=payload.note or "",
    )

    # Phase 6 — if user picked at session-start, log assignment training event
    if payload.deliverable_id:
        try:
            _orch.db.log_training_event({
                "event_type": "assignment",
                "deliverable_id": payload.deliverable_id,
                "features": {
                    "session_id": sid,
                    "auto_assigned": int(payload.auto_assigned or 0),
                    "hour_of_day": datetime.now().hour,
                    "source": "session_start",
                },
                "label": {
                    "deliverable_id": payload.deliverable_id,
                    "project_id": project_id,
                    "confirmed_by_user": 1,
                },
                "pmis_version": "phase-6",
            })
        except Exception:
            pass

    return {"status": "started", "session": _orch.db.get_work_session(sid)}


class WorkConfirmPayload(BaseModel):
    deliverable_id: str
    auto_assigned: Optional[int] = 0


@app.post("/api/work/confirm")
async def api_work_confirm(payload: WorkConfirmPayload):
    """Bind (or rebind) the active session to a specific deliverable.
    Also logs an 'assignment' training event so Phase 6 can train a
    deliverable-ranker from user-confirmed picks."""
    active = _orch.db.get_active_work_session()
    if not active:
        raise HTTPException(status_code=400, detail="no_active_session")

    project_id = ""
    deliv = _orch.db.get_deliverable(payload.deliverable_id)
    if deliv:
        project_id = deliv.get("project_id", "")

    _orch.db.update_work_session(active["id"], {
        "deliverable_id": payload.deliverable_id,
        "project_id": project_id,
        "auto_assigned": int(payload.auto_assigned or 0),
        "confirmed_by_user": 1,
    })

    # Phase 6 — labeled supervision signal for learning-to-rank over deliverables
    try:
        lm = _get_live_matcher()
        seg = lm.get_latest_segment(after_iso=active["started_at"])
        features: Dict[str, Any] = {
            "session_id": active["id"],
            "auto_assigned": int(payload.auto_assigned or 0),
            "hour_of_day": datetime.now().hour,
        }
        if seg:
            features.update({
                "platform": seg.platform,
                "window_name": seg.window_name,
                "segment_length_secs": seg.length_secs,
                "summary_prefix": (seg.summary or "")[:180],
            })
        _orch.db.log_training_event({
            "event_type": "assignment",
            "deliverable_id": payload.deliverable_id,
            "features": features,
            "label": {
                "deliverable_id": payload.deliverable_id,
                "project_id": project_id,
                "confirmed_by_user": 1,
            },
            "pmis_version": "phase-6",
        })
    except Exception as e:
        # Non-fatal — confirmation still succeeds even if logging misses
        pass

    return {"status": "bound", "session": _orch.db.get_work_session(active["id"])}


@app.post("/api/work/end")
async def api_work_end():
    """Close the active session and stamp segment_override_bindings for every
    segment whose start time falls within the session window. This is what
    nightly ProjectMatcher will short-circuit on."""
    active = _orch.db.get_active_work_session()
    if not active:
        return {"status": "no_active_session"}

    _orch.db.end_work_session(active["id"])
    ended = _orch.db.get_work_session(active["id"])

    override_count = 0
    if ended and ended.get("deliverable_id"):
        lm = _get_live_matcher()
        segs = lm.get_segments_in_window(
            start_iso=ended["started_at"],
            end_iso=ended["ended_at"],
        )
        for seg in segs:
            if not seg.segment_id:
                continue
            _orch.db.upsert_segment_override(
                segment_id=seg.segment_id,
                session_id=ended["id"],
                project_id=ended.get("project_id", ""),
                deliverable_id=ended.get("deliverable_id", ""),
                source="session",
            )
            override_count += 1

    return {
        "status": "ended",
        "session": ended,
        "segments_bound": override_count,
    }


@app.get("/api/work/current")
async def api_work_current():
    """Return the active session plus realtime drift / suggestion state.

    Widget polls this every 30s. Three shapes of response:
      - No active session → {"active": false, ...}
      - Active + unbound → suggestions (after 5-min segment gate)
      - Active + bound → drift status (after 5-min drift gate)
    """
    active = _orch.db.get_active_work_session()
    if not active:
        return {"active": False}

    lm = _get_live_matcher()
    resp: Dict[str, Any] = {
        "active": True,
        "session": active,
        "drift": None,
        "suggestions": None,
        "latest_segment": None,
    }

    if active.get("deliverable_id"):
        drift = lm.check_drift(
            bound_deliverable_id=active["deliverable_id"],
            session_start_iso=active["started_at"],
        )
        resp["drift"] = drift
        resp["latest_segment"] = drift.get("latest_segment")
        # Enrich with deliverable name so widget can show it
        deliv = _orch.db.get_deliverable(active["deliverable_id"])
        if deliv:
            resp["bound_deliverable"] = {
                "id": deliv["id"],
                "name": deliv["name"],
                "project_id": deliv.get("project_id", ""),
                "project_name": deliv.get("project_name", ""),
            }
    else:
        sug = lm.suggest(after_iso=active["started_at"])
        resp["suggestions"] = sug.get("suggestions", [])
        resp["latest_segment"] = sug.get("segment")
        resp["suggest_ready"] = sug.get("ready", False)
        resp["suggest_reason"] = sug.get("reason", "")

    return resp


@app.get("/api/work/deliverables")
async def api_work_deliverables():
    """Flat list of active deliverables with project names — for the picker."""
    cands = _orch.db.get_active_deliverable_candidates()
    # Dedupe by deliverable_id (get_active_deliverable_candidates expands anchors)
    seen: Dict[str, Dict[str, Any]] = {}
    for c in cands:
        did = c.get("deliverable_id", "")
        if not did or did in seen:
            continue
        seen[did] = {
            "deliverable_id": did,
            "project_id": c.get("project_id", ""),
            "name": c.get("name", ""),
            "deadline": c.get("deadline", ""),
        }
    # Attach project names
    projects = {p["id"]: p["name"] for p in _orch.db.list_projects()}
    for row in seen.values():
        row["project_name"] = projects.get(row["project_id"], "")
    return {"deliverables": list(seen.values())}


@app.get("/api/work/sessions")
async def api_work_sessions_list(limit: int = 20):
    """Recent sessions — for history panel."""
    return {"sessions": _orch.db.list_work_sessions(limit=limit)}


_value_calc = None


def _get_value_calc():
    global _value_calc
    if _value_calc is None:
        from core.value_score import ValueScoreCalculator
        _value_calc = ValueScoreCalculator(_orch.db, hyperparams=_orch.hp)
    return _value_calc


@app.post("/api/value/recompute")
async def api_value_recompute():
    """Phase 3 — rebuild value_score for every non-deleted node. Nightly
    consolidation will also call this, but the endpoint lets us trigger
    manually after schema changes or weight tuning."""
    return _get_value_calc().recompute_all()


@app.get("/api/node/{node_id}/value")
async def api_node_value(node_id: str, recompute: bool = False):
    """Return the value_score breakdown for a node. By default reads the
    materialized columns; pass ?recompute=true to compute fresh (for
    debugging weight tuning)."""
    with _orch.db._connect() as conn:
        row = conn.execute(
            """SELECT id, level, content, value_score, value_goal, value_feedback,
                      value_usage, value_recency, value_computed_at
               FROM memory_nodes WHERE id = ? AND is_deleted = 0""",
            (node_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="node_not_found")

    if recompute:
        fresh = _get_value_calc().compute_one(node_id)
        if fresh:
            return {
                "node_id": node_id,
                "level": row["level"],
                "value_score": fresh.score,
                "components": {
                    "G": fresh.G, "F": fresh.F, "U": fresh.U, "R": fresh.R,
                    "feedback_raw": fresh.feedback_raw_sum,
                },
                "redflag": fresh.redflag,
                "computed_at": "just_now",
                "source": "recompute",
            }

    F = float(row["value_feedback"] or 0.0)
    hp = _orch.hp
    return {
        "node_id": node_id,
        "level": row["level"],
        "value_score": float(row["value_score"] or 0.0),
        "components": {
            "G": float(row["value_goal"] or 0.0),
            "F": F,
            "U": float(row["value_usage"] or 0.0),
            "R": float(row["value_recency"] or 0.0),
        },
        "redflag": F <= float(hp.get("value_feedback_redflag", -0.3)),
        "computed_at": row["value_computed_at"] or "",
        "source": "materialized",
    }


_brief_composer = None


def _get_brief_composer():
    global _brief_composer
    if _brief_composer is None:
        from retrieval.brief_composer import BriefComposer
        _brief_composer = BriefComposer(_orch.db, _orch.embedder, hyperparams=_orch.hp)
    return _brief_composer


@app.get("/api/work/brief")
async def api_work_brief(deliverable_id: str):
    """Phase 2 pre-work brief: two buckets — 'Claude can do this' and
    'You've done this before'."""
    if not deliverable_id:
        raise HTTPException(status_code=400, detail="deliverable_id required")
    return _get_brief_composer().compose(deliverable_id)


_meta_composer = None


def _get_meta_composer():
    global _meta_composer
    if _meta_composer is None:
        from claude_integration.meta_composer import ProblemStatementComposer
        _meta_composer = ProblemStatementComposer(
            _orch.db, _orch.embedder,
            hyperparams=_orch.hp,
            brief_composer=_get_brief_composer(),
        )
    return _meta_composer


@app.post("/api/work/compose-problem")
async def api_work_compose_problem(
    deliverable_id: str,
    use_llm: bool = True,
    include_activity: bool = True,
):
    """Phase 3.5 — Meta-LLM composes problem_statement.md for a deliverable.
    Set use_llm=false for deterministic template fallback (fast + offline)."""
    if not deliverable_id:
        raise HTTPException(status_code=400, detail="deliverable_id required")
    try:
        bundle = _get_meta_composer().compose(
            deliverable_id, use_llm=use_llm, include_activity=include_activity,
        )
        return bundle.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_harness_builder = None


def _get_harness_builder():
    global _harness_builder
    if _harness_builder is None:
        from harness.harness_builder import HarnessBuilder
        _harness_builder = HarnessBuilder(
            _orch.db, _orch.embedder, hyperparams=_orch.hp,
            meta_composer=_get_meta_composer(),
        )
    return _harness_builder


class HarnessBuildPayload(BaseModel):
    deliverable_id: str
    use_llm: Optional[bool] = False
    title: Optional[str] = None
    trigger_source: Optional[str] = "manual"


@app.post("/api/harness/build")
async def api_harness_build(payload: HarnessBuildPayload):
    """Materialize a new harness bundle on disk and register it. Phase 3.5's
    Meta-LLM composes the problem statement; this pass adds CLAUDE.md,
    context/ files, and bundle.json."""
    try:
        rec = _get_harness_builder().build(
            deliverable_id=payload.deliverable_id,
            use_llm=bool(payload.use_llm),
            title_override=payload.title,
            trigger_source=payload.trigger_source or "manual",
        )
        return rec
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/harness")
async def api_harness_list(deliverable_id: Optional[str] = None):
    return {"harnesses": _orch.db.list_harnesses(deliverable_id=deliverable_id)}


@app.get("/api/harness/{harness_id}")
async def api_harness_get(harness_id: str):
    rec = _orch.db.get_harness(harness_id)
    if not rec:
        raise HTTPException(status_code=404, detail="harness_not_found")
    # enrich with bundle.json if available
    try:
        import json as _json
        bundle_json = Path(rec["bundle_path"]) / "bundle.json"
        if bundle_json.exists():
            rec["manifest"] = _json.loads(bundle_json.read_text())
    except Exception:
        pass
    return rec


class HarnessStepClassification(BaseModel):
    step: str                            # short description of the step
    tag: str                             # 'CLAUDE' | 'USER' | 'REVIEW'
    executed: Optional[bool] = None      # True if Claude ran it; False if skipped


class HarnessRunPayload(BaseModel):
    thumb: Optional[str] = None          # 'up' | 'down' | None
    outcome: Optional[str] = None        # 'goal_achieved' | 'goal_unchanged'
    notes: Optional[str] = ""
    plan_steps: Optional[List[HarnessStepClassification]] = None


@app.post("/api/harness/{harness_id}/record-run")
async def api_harness_record_run(harness_id: str, payload: HarnessRunPayload):
    """Close the feedback loop. Claude Code (or the widget) calls this after
    a harness run. Writes to training_events for Phase 6 corpus.

    Step 3 addition: if `plan_steps` is provided, write one `automation_class`
    training event per step capturing the CLAUDE/USER/REVIEW label plus which
    ones actually executed. Feeds the future automatability classifier
    described in the Phase 6 plan."""
    if payload.thumb not in (None, "up", "down"):
        raise HTTPException(status_code=400, detail="thumb must be 'up' or 'down'")

    try:
        _orch.db.record_harness_run(
            harness_id=harness_id,
            thumb=payload.thumb,
            outcome=payload.outcome,
            post_run_signals={
                "notes": payload.notes or "",
                "plan_step_count": len(payload.plan_steps or []),
            },
        )

        # Step 3 — per-step automation_class events
        steps_logged = 0
        if payload.plan_steps:
            harness = _orch.db.get_harness(harness_id)
            deliverable_id = (harness or {}).get("deliverable_id", "")
            valid_tags = {"CLAUDE", "USER", "REVIEW"}
            for idx, step in enumerate(payload.plan_steps):
                tag = (step.tag or "").upper().strip()
                if tag not in valid_tags:
                    continue
                _orch.db.log_training_event({
                    "event_type": "automation_class",
                    "harness_id": harness_id,
                    "deliverable_id": deliverable_id,
                    "features": {
                        "step_index": idx,
                        "step_text": (step.step or "")[:280],
                        "executed": bool(step.executed) if step.executed is not None else None,
                        "plan_length": len(payload.plan_steps),
                    },
                    "label": {
                        "tag": tag,                 # CLAUDE | USER | REVIEW
                        "is_automated": tag == "CLAUDE",
                    },
                    "pmis_version": "phase-6",
                })
                steps_logged += 1

        result = _orch.db.get_harness(harness_id) or {}
        result["steps_logged"] = steps_logged
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/training/counts")
async def api_training_counts():
    """How many labeled events we've accumulated. Phase 6 exports these."""
    return _orch.db.count_training_events()


_training_exporter = None


def _get_training_exporter():
    global _training_exporter
    if _training_exporter is None:
        from consolidation.training_export import TrainingCorpusExporter
        _training_exporter = TrainingCorpusExporter(_orch.db, hyperparams=_orch.hp)
    return _training_exporter


@app.post("/api/training/export")
async def api_training_export(backfill: bool = False):
    """Phase 6 — flush training_events to JSONL corpus. Incremental by default
    (flips exported_to_training=1); backfill=true re-exports everything (for
    model-upgrade backtests) without flipping the flag."""
    return _get_training_exporter().export(backfill=backfill)


@app.get("/api/training/stats")
async def api_training_stats():
    """Side-by-side view of DB training_events vs on-disk JSONL corpus."""
    return _get_training_exporter().stats()


_boilerplate_detector = None


def _get_boilerplate_detector():
    global _boilerplate_detector
    if _boilerplate_detector is None:
        from harness.boilerplate_detector import BoilerplateDetector
        _boilerplate_detector = BoilerplateDetector(_orch.db, hyperparams=_orch.hp)
    return _boilerplate_detector


@app.post("/api/boilerplate/extract")
async def api_boilerplate_extract(days: Optional[int] = None):
    """Heuristic extraction of artifacts from tracker segments into
    segment_artifacts table."""
    return _get_boilerplate_detector().extract(days=days)


@app.post("/api/boilerplate/mine")
async def api_boilerplate_mine(min_reps: Optional[int] = None):
    """Cluster segment_artifacts by content_hash and write boilerplate
    training_events for every cluster crossing min_reps."""
    return _get_boilerplate_detector().mine(min_reps=min_reps)


@app.get("/api/boilerplate/clusters")
async def api_boilerplate_clusters(min_reps: int = 3, limit: int = 50):
    return {
        "min_reps": min_reps,
        "clusters": _orch.db.list_artifact_clusters(min_repetitions=min_reps, limit=limit),
        "artifact_counts": _orch.db.count_segment_artifacts(),
    }


_pattern_miner = None


def _get_pattern_miner():
    global _pattern_miner
    if _pattern_miner is None:
        from harness.pattern_miner import PatternMiner
        _pattern_miner = PatternMiner(_orch.db, hyperparams=_orch.hp)
    return _pattern_miner


@app.get("/api/harness/candidates")
async def api_harness_candidates(deliverable_id: Optional[str] = None):
    """Phase 4b — pattern-mined harness suggestions. Recurring (platform,
    window_root) task shapes from tracker segments, filtered to ≥5 reps +
    sufficient avg value_score."""
    cands = _get_pattern_miner().find_candidates(deliverable_id=deliverable_id)
    return {"candidates": [c.to_dict() for c in cands]}


class GoalAchievePayload(BaseModel):
    note: Optional[str] = ""


@app.post("/api/goals/{goal_id}/achieve")
async def api_goals_achieve(goal_id: str, payload: Optional[GoalAchievePayload] = None):
    """Mark a goal achieved AND propagate positive feedback to every anchor
    linked to that goal (source='outcome', strength=0.6 per locked decisions).
    Also touches harnesses that used those anchors — bumps their thumbs_up via
    training_events of type harness_outcome."""
    note = payload.note if payload and payload.note else ""
    result = _orch.db.propagate_goal_achievement(goal_id=goal_id, note=note)
    # Recompute value_scores so UI reflects the new G(n) + F(n) immediately
    try:
        _get_value_calc().recompute_all()
    except Exception as e:
        result["recompute_error"] = str(e)
    return result


@app.get("/api/work/compose-prompt-preview")
async def api_work_compose_prompt_preview(deliverable_id: str):
    """Return the exact user-prompt Markdown the meta-LLM will see. Useful
    for debugging weight tuning without spending API tokens."""
    if not deliverable_id:
        raise HTTPException(status_code=400, detail="deliverable_id required")
    try:
        return {
            "deliverable_id": deliverable_id,
            "prompt": _get_meta_composer().render_prompt(deliverable_id),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/work/sync-yaml")
async def api_work_sync_yaml():
    """Re-sync goals.yaml + deliverables.yaml into projects + deliverables
    tables. Call after editing the Goals UI to refresh the picker."""
    try:
        from sync.yaml_to_db import sync_pm_yaml_to_db
        return {
            "status": "ok",
            "counts": sync_pm_yaml_to_db(
                _orch.db, embedder=_orch.embedder, hyperparams=_orch.hp,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/work/recent-work")
async def api_work_recent(lookback_minutes: int = 120, limit: int = 3):
    """Aggregated task clusters from the last N minutes of tracker segments.
    Fuels the widget's 'Resume where you left off' section."""
    from retrieval.recent_work import get_recent_work
    try:
        return {"lookback_minutes": lookback_minutes,
                 "items": get_recent_work(lookback_minutes=lookback_minutes, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pm/deadlines")
async def api_pm_deadlines(limit: int = 3, include_overdue: bool = True):
    """Upcoming deliverables ordered by deadline ASC with days_until."""
    from datetime import date as _date
    with _orch.db._connect() as conn:
        rows = conn.execute(
            """SELECT d.id, d.name, d.deadline, d.project_id, d.status,
                      p.name AS project_name
               FROM deliverables d
               LEFT JOIN projects p ON p.id = d.project_id
               WHERE d.status = 'active' AND d.deadline != ''"""
        ).fetchall()

    today = _date.today()
    items: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            dl = datetime.strptime(d["deadline"][:10], "%Y-%m-%d").date()
        except Exception:
            continue
        days_until = (dl - today).days
        if not include_overdue and days_until < 0:
            continue
        d["days_until"] = days_until
        d["overdue"] = days_until < 0
        items.append(d)

    items.sort(key=lambda x: x["days_until"])
    return {"deadlines": items[:limit], "total": len(items)}


_TRACKER_LABEL = "com.yantra.productivity-tracker"
_TRACKER_PLIST = str(Path.home() / "Library" / "LaunchAgents" / f"{_TRACKER_LABEL}.plist")
_TRACKER_ROOT = Path.home() / "Desktop" / "memory" / "productivity-tracker"
_TRACKER_VENV_PY = _TRACKER_ROOT / ".venv" / "bin" / "python"


def _tracker_pgrep_pid() -> Optional[int]:
    import subprocess
    try:
        out = subprocess.run(
            ["pgrep", "-f", "src.agent.tracker"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None
    return None


def _tracker_launchd_loaded() -> bool:
    import subprocess
    try:
        out = subprocess.run(
            ["launchctl", "list", _TRACKER_LABEL],
            capture_output=True, text=True, timeout=3,
        )
        return out.returncode == 0
    except Exception:
        return False


@app.get("/api/tracker/status")
async def api_tracker_status():
    """Return whether the productivity-tracker daemon is running."""
    pid = _tracker_pgrep_pid()
    return {
        "label": _TRACKER_LABEL,
        "loaded": _tracker_launchd_loaded(),
        "running": pid is not None,
        "pid": pid,
        "plist": _TRACKER_PLIST,
    }


class TrackerTogglePayload(BaseModel):
    desired_state: Optional[str] = None  # 'on' | 'off' | None → toggle


@app.post("/api/tracker/toggle")
async def api_tracker_toggle(payload: Optional[TrackerTogglePayload] = None):
    """Toggle (or explicitly set) the productivity-tracker daemon.

    Strategy (macOS 10.11+):
      1. Primary: `launchctl bootstrap/bootout gui/<uid> <plist>` — the modern
         API. `load/unload` is deprecated and often silently no-ops.
      2. Fallback: direct process control via `pkill -f src.agent.tracker`
         for stop, and `subprocess.Popen` of the venv python for start.
    Returns the attempt log so the widget can show *why* a toggle failed.
    """
    import os, subprocess, time as _t

    current = await api_tracker_status()
    desired = (payload.desired_state or "").lower() if payload else ""
    if desired not in ("on", "off"):
        desired = "off" if current["running"] else "on"

    uid = os.getuid()
    service_target = f"gui/{uid}/{_TRACKER_LABEL}"
    domain_target = f"gui/{uid}"
    attempts: List[Dict[str, Any]] = []

    def _run(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            return {"cmd": " ".join(cmd), "rc": r.returncode,
                     "stdout": r.stdout.strip()[-400:], "stderr": r.stderr.strip()[-400:]}
        except Exception as e:
            return {"cmd": " ".join(cmd), "rc": -1, "error": str(e)}

    if desired == "off":
        # Primary: launchctl bootout
        attempts.append(_run(["launchctl", "bootout", service_target]))
        # Also try deprecated unload for older setups
        attempts.append(_run(["launchctl", "unload", _TRACKER_PLIST]))
        # Fallback: if a tracker process is still alive, kill it directly.
        _t.sleep(0.4)
        pid_after = _tracker_pgrep_pid()
        if pid_after is not None:
            attempts.append(_run(["kill", "-TERM", str(pid_after)]))
            _t.sleep(0.6)
            pid_after = _tracker_pgrep_pid()
            if pid_after is not None:
                attempts.append(_run(["kill", "-KILL", str(pid_after)]))
    else:
        # Primary: launchctl bootstrap
        attempts.append(_run(["launchctl", "bootstrap", domain_target, _TRACKER_PLIST]))
        # Also try deprecated load for older setups
        attempts.append(_run(["launchctl", "load", _TRACKER_PLIST]))
        # Kickstart so it starts right now even if the plist is already loaded
        attempts.append(_run(["launchctl", "kickstart", "-k", service_target]))
        _t.sleep(0.6)
        # Fallback: if still not running, spawn the daemon directly.
        if _tracker_pgrep_pid() is None and _TRACKER_VENV_PY.exists():
            try:
                proc = subprocess.Popen(
                    [str(_TRACKER_VENV_PY), "-m", "src.agent.tracker"],
                    cwd=str(_TRACKER_ROOT),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                attempts.append({"cmd": "Popen src.agent.tracker", "pid": proc.pid})
            except Exception as e:
                attempts.append({"cmd": "Popen src.agent.tracker", "error": str(e)})

    _t.sleep(0.5)
    new_status = await api_tracker_status()
    return {"requested": desired, "status": new_status, "attempts": attempts}


# ================================================================
# WIKI FEEDBACK + EDIT  (audit-fix additions)
# ================================================================

class WikiFeedbackBody(BaseModel):
    node_id: str
    polarity: str                 # 'positive' | 'negative' | 'correction' OR 'up'|'down'
    content: Optional[str] = ""   # optional free-text note
    goal_id: Optional[str] = None
    source: Optional[str] = "explicit"
    strength: Optional[float] = 1.0


class WikiContentPatchBody(BaseModel):
    node_id: str
    new_content: str              # full-replace patch (V1)


@app.post("/api/wiki/feedback")
async def api_wiki_feedback(body: WikiFeedbackBody):
    """HTTP wrapper around db.add_feedback. Writes to the production
    feedback table. Accepts 'up'/'down' as aliases for 'positive'/'negative'."""
    polarity = (body.polarity or "").lower().strip()
    alias = {"up": "positive", "down": "negative"}
    polarity = alias.get(polarity, polarity)
    if polarity not in ("positive", "negative", "correction"):
        raise HTTPException(
            status_code=400,
            detail=f"polarity must be positive|negative|correction (got {body.polarity!r})",
        )

    db = _orch.db
    if not db.get_node(body.node_id):
        raise HTTPException(status_code=404, detail=f"node not found: {body.node_id}")

    fb_id = db.add_feedback(
        node_id=body.node_id,
        polarity=polarity,
        content=body.content or "",
        goal_id=body.goal_id,
        source=(body.source or "explicit"),
        strength=float(body.strength or 1.0),
    )
    return {
        "ok": True,
        "feedback_id": fb_id,
        "node_id": body.node_id,
        "polarity": polarity,
    }


@app.post("/api/wiki/content-patch")
async def api_wiki_content_patch(body: WikiContentPatchBody):
    """Apply a user-authored full-content replacement to a node.

      1. Updates memory_nodes.content + sets is_user_edited=1 + last_modified=now
      2. Re-embeds (Euclidean) via embedder + syncs ChromaDB via refresh_node_embedding
      3. Writes an audit row into consolidation_log with before/after
    """
    db = _orch.db
    row = db.get_node(body.node_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"node not found: {body.node_id}")

    before = row.get("content") or ""
    after = (body.new_content or "").strip()
    if not after:
        raise HTTPException(status_code=400, detail="new_content cannot be empty")
    if after == before:
        return {"ok": True, "changed": False, "reason": "no change"}

    # Persist the content change + is_user_edited flag
    with db._connect() as conn:
        conn.execute("""
            UPDATE memory_nodes
            SET content = ?, is_user_edited = 1, last_modified = datetime('now')
            WHERE id = ?
        """, (after, body.node_id))

    # Re-embed (best-effort, never fails the request)
    embedding_refreshed = False
    try:
        new_euc = _orch.ingestion.embedder.embed_text(after)
        embedding_refreshed = db.refresh_node_embedding(body.node_id, new_euc)
    except Exception as e:
        print(f"[content-patch] re-embed failed for {body.node_id}: {e}")

    # Audit into consolidation_log (no new table required)
    try:
        db.log_consolidation({
            "action": "user_content_patch",
            "source_node_ids": [body.node_id],
            "target_node_id": body.node_id,
            "reason": "user edit via /api/wiki/content-patch",
            "details": {
                "before_preview": before[:400],
                "after_preview": after[:400],
                "embedding_refreshed": embedding_refreshed,
            },
        })
    except Exception as e:
        print(f"[content-patch] audit log failed for {body.node_id}: {e}")

    return {
        "ok": True,
        "changed": True,
        "node_id": body.node_id,
        "before_preview": before[:200],
        "after_preview": after[:200],
        "is_user_edited": True,
        "embedding_refreshed": embedding_refreshed,
    }


class WikiRegenNowBody(BaseModel):
    node_id: str
    scope: Optional[str] = None      # 'anchor' | 'context' | None (auto)
    force: Optional[bool] = False    # override is_user_edited skip
    reason: Optional[str] = "manual_override"


@app.post("/api/wiki/regen-now")
async def api_wiki_regen_now(body: WikiRegenNowBody):
    """Manual trigger: regen a single node's content via LLM. Bypasses queue.

    Respects is_user_edited unless force=true. Re-embeds on success."""
    try:
        from consolidation.restructure import Restructurer
        from core import config as _cfg
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"restructurer unavailable: {e}")

    rs = Restructurer(_orch.db, _cfg.get_all(),
                      embedder=_orch.ingestion.embedder)
    result = rs.regen_now(body.node_id, scope=body.scope,
                          reason=body.reason or "manual_override",
                          force=bool(body.force))
    # Also drop it from queue if it was enqueued
    with _orch.db._connect() as conn:
        conn.execute("""
            UPDATE restructure_queue SET status = 'done',
                processed_at = datetime('now')
            WHERE node_id = ? AND status IN ('queued', 'processing')
        """, (body.node_id,))
    return {"ok": True, "result": result}


@app.get("/api/wiki/regen-queue")
async def api_wiki_regen_queue(status: str = "queued", limit: int = 50):
    """List restructure queue entries. status: queued|processing|done|skipped|all."""
    with _orch.db._connect() as conn:
        if status == "all":
            rows = conn.execute("""
                SELECT rq.id, rq.node_id, rq.scope, rq.reason, rq.status,
                       rq.queued_at, rq.processed_at,
                       mn.level, mn.value_feedback, mn.is_user_edited,
                       SUBSTR(mn.content, 1, 100) AS content_preview
                FROM restructure_queue rq
                LEFT JOIN memory_nodes mn ON mn.id = rq.node_id
                ORDER BY rq.queued_at DESC LIMIT ?
            """, (limit,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT rq.id, rq.node_id, rq.scope, rq.reason, rq.status,
                       rq.queued_at, rq.processed_at,
                       mn.level, mn.value_feedback, mn.is_user_edited,
                       SUBSTR(mn.content, 1, 100) AS content_preview
                FROM restructure_queue rq
                LEFT JOIN memory_nodes mn ON mn.id = rq.node_id
                WHERE rq.status = ?
                ORDER BY rq.queued_at DESC LIMIT ?
            """, (status, limit)).fetchall()
        return {"status_filter": status, "count": len(rows),
                "jobs": [dict(r) for r in rows]}


@app.get("/api/wiki/stale")
async def api_wiki_stale(drift_threshold: float = 0.2, limit: int = 50):
    """Phase 5 — pages whose cached prose is out of sync with current
    value_score distribution (drift > threshold) or pitfall counts have
    changed. Nightly consolidation should prioritize regenerating these."""
    wiki = _get_wiki()
    try:
        return {
            "drift_threshold": drift_threshold,
            "stale_pages": wiki.list_stale_pages(drift_threshold=drift_threshold, limit=limit),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/widget/floating", response_class=HTMLResponse)
async def widget_floating(request: Request):
    """Naked HTML (no wiki chrome) for the menu-bar popover WKWebView."""
    return templates.TemplateResponse(request, "floating_widget.html", {})


@app.get("/wiki/productivity", response_class=HTMLResponse)
async def wiki_productivity(request: Request):
    """Productivity dashboard — live activity data from tracker."""
    wiki = _get_wiki()
    data = wiki.render_productivity()
    return templates.TemplateResponse(request, "wiki_productivity.html", data)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import os as _os
    _port = int(_os.environ.get("PORT", "8100"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=_port,
        reload=False,
        log_level="info",
    )
