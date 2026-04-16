"""
PMIS System Health Dashboard — port 8200

Monitors every organ of the PMIS engine:
  Heart    = Overall system vitals
  Blood    = Ingestion (Claude + Tracker streams)
  Brain    = HGCN + Poincare learning
  Lungs    = Retrieval engine
  Immune   = Consolidation pipeline
  Skeleton = Knowledge graph structure
  Nervous  = Wiki + Goals + Feedback

Manual override buttons for all automated operations.
Auto-refreshes every 30 seconds.

Usage: python3 pmis_v2/health_dashboard.py
"""

import sys
import os
import sqlite3
import json
import time
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

PMIS_DIR = Path(__file__).parent
sys.path.insert(0, str(PMIS_DIR))

PMIS_DB = str(PMIS_DIR / "data" / "memory.db")
TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")

app = FastAPI(title="PMIS Health Dashboard")
templates = Jinja2Templates(directory=str(PMIS_DIR / "templates"))

logger = logging.getLogger("pmis.health")
_start_time = time.time()


# ═══════════════════════════════════════════════════════════
# DATABASE HELPERS
# ═══════════════════════════════════════════════════════════

def _pmis_conn():
    conn = sqlite3.connect(PMIS_DB, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn

def _tracker_conn():
    if not os.path.exists(TRACKER_DB):
        return None
    conn = sqlite3.connect(TRACKER_DB, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn

def _safe_query(conn, sql, params=()):
    try:
        return conn.execute(sql, params).fetchall()
    except Exception:
        return []

def _safe_one(conn, sql, params=()):
    try:
        r = conn.execute(sql, params).fetchone()
        return dict(r) if r else {}
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════
# ORGAN COLLECTORS
# ═══════════════════════════════════════════════════════════

def collect_heart() -> Dict:
    """Overall system vitals."""
    conn = _pmis_conn()
    alerts = []

    uptime_secs = time.time() - _start_time
    uptime_str = f"{int(uptime_secs // 3600)}h {int((uptime_secs % 3600) // 60)}m"

    last_turn = _safe_one(conn, "SELECT MAX(timestamp) as ts FROM turn_diagnostics")
    last_consol = _safe_one(conn, "SELECT MAX(run_date) as ts FROM consolidation_log")
    total_nodes = _safe_one(conn, "SELECT COUNT(*) as c FROM memory_nodes WHERE is_deleted=0")
    error_count = len(_safe_query(conn, "SELECT 1 FROM turn_diagnostics WHERE storage_action='skip' AND timestamp > datetime('now','-24 hours')"))

    # Determine overall status
    status = "green"
    last_ts = last_turn.get("ts", "")
    if last_ts:
        try:
            hours_ago = (datetime.now() - datetime.fromisoformat(last_ts)).total_seconds() / 3600
            if hours_ago > 24:
                status = "red"
                alerts.append("No ingestion activity in 24+ hours")
            elif hours_ago > 6:
                status = "yellow"
                alerts.append(f"No activity in {hours_ago:.0f} hours")
        except (ValueError, TypeError):
            pass

    conn.close()
    return {
        "status": status,
        "metrics": {
            "uptime": uptime_str,
            "total_nodes": total_nodes.get("c", 0),
            "last_activity": last_ts[:16] if last_ts else "never",
            "last_consolidation": (last_consol.get("ts") or "never")[:16],
            "errors_24h": error_count,
        },
        "alerts": alerts,
    }


def collect_blood() -> Dict:
    """Ingestion — two streams: Claude + Tracker."""
    conn = _pmis_conn()
    alerts = []

    # Stream 1: Claude Desktop
    claude_today = _safe_one(conn, "SELECT COUNT(*) as c FROM turn_diagnostics WHERE DATE(timestamp)=DATE('now')")
    claude_7d = _safe_one(conn, "SELECT COUNT(*) as c FROM turn_diagnostics WHERE timestamp > datetime('now','-7 days')")
    gamma_avg = _safe_one(conn, "SELECT AVG(gamma_final) as v FROM turn_diagnostics WHERE timestamp > datetime('now','-7 days')")
    modes = _safe_query(conn, "SELECT gamma_mode, COUNT(*) as c FROM turn_diagnostics WHERE timestamp > datetime('now','-7 days') GROUP BY gamma_mode")
    storage = _safe_query(conn, "SELECT storage_action, COUNT(*) as c FROM turn_diagnostics WHERE timestamp > datetime('now','-7 days') GROUP BY storage_action")
    last_turn = _safe_one(conn, "SELECT MAX(timestamp) as ts FROM turn_diagnostics")
    conn.close()

    # Stream 2: Tracker
    tracker = {"segments_today": 0, "frames_today": 0, "last_segment": "unavailable", "ollama": "unknown"}
    tconn = _tracker_conn()
    if tconn:
        t_segs = _safe_one(tconn, "SELECT COUNT(*) as c FROM context_1 WHERE DATE(timestamp_start)=DATE('now')")
        t_frames = _safe_one(tconn, "SELECT COUNT(*) as c FROM context_2 WHERE DATE(frame_timestamp)=DATE('now')")
        t_last = _safe_one(tconn, "SELECT MAX(timestamp_start) as ts FROM context_1")
        tracker = {
            "segments_today": t_segs.get("c", 0),
            "frames_today": t_frames.get("c", 0),
            "last_segment": (t_last.get("ts") or "never")[:16],
        }
        tconn.close()

    # Ollama check
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        tracker["ollama"] = "online" if r.status_code == 200 else "offline"
    except Exception:
        tracker["ollama"] = "offline"

    status = "green"
    if claude_today.get("c", 0) == 0 and tracker["segments_today"] == 0:
        status = "yellow"
        alerts.append("No ingestion from either stream today")
    if tracker["ollama"] == "offline":
        alerts.append("Ollama offline — tracker frame analysis halted")
        status = "yellow"

    return {
        "status": status,
        "metrics": {
            "claude_turns_today": claude_today.get("c", 0),
            "claude_turns_7d": claude_7d.get("c", 0),
            "gamma_avg": round(gamma_avg.get("v") or 0, 3),
            "modes": {r["gamma_mode"]: r["c"] for r in modes},
            "storage": {r["storage_action"]: r["c"] for r in storage},
            "last_turn": (last_turn.get("ts") or "never")[:16],
            "tracker_segments_today": tracker["segments_today"],
            "tracker_frames_today": tracker["frames_today"],
            "tracker_last": tracker["last_segment"],
            "ollama": tracker["ollama"],
        },
        "alerts": alerts,
    }


def collect_brain() -> Dict:
    """HGCN + Poincare learning."""
    conn = _pmis_conn()
    alerts = []

    # Latest HGCN from consolidation_log
    hgcn_rows = _safe_query(conn, "SELECT run_date, details FROM consolidation_log WHERE action='hgcn_train' ORDER BY id DESC LIMIT 1")
    hgcn = {}
    if hgcn_rows:
        hgcn["last_run"] = hgcn_rows[0]["run_date"][:16] if hgcn_rows[0]["run_date"] else "never"
        try:
            details = json.loads(hgcn_rows[0]["details"] or "{}")
            hgcn.update(details)
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: check rsgd_runs for latest training info
    if not hgcn.get("final_loss"):
        rsgd = _safe_one(conn, "SELECT final_loss, wall_time_seconds, epochs, nodes_updated, run_at FROM rsgd_runs ORDER BY id DESC LIMIT 1")
        if rsgd:
            hgcn["final_loss"] = rsgd.get("final_loss")
            hgcn["wall_time"] = rsgd.get("wall_time_seconds")
            hgcn["epochs"] = rsgd.get("epochs")
            hgcn["last_run"] = (rsgd.get("run_at") or "never")[:16]

    # Norm ordering
    norms = {}
    for level in ["SC", "CTX", "ANC"]:
        r = _safe_one(conn, "SELECT AVG(e.hyperbolic_norm) as v FROM embeddings e JOIN memory_nodes n ON e.node_id=n.id WHERE n.level=? AND n.is_deleted=0 AND e.is_learned=1", (level,))
        norms[level] = round(r.get("v") or 0, 4)

    conn.close()

    # Status
    status = "green"
    ordering_ok = norms.get("SC", 0) < norms.get("CTX", 0) < norms.get("ANC", 1)
    if not ordering_ok:
        status = "yellow"
        alerts.append("Norm ordering violation: SC should < CTX < ANC")

    return {
        "status": status,
        "metrics": {
            "last_trained": hgcn.get("last_run", "never"),
            "loss": round(hgcn.get("final_loss") or 0, 4),
            "epochs": hgcn.get("epochs", 0),
            "wall_time": f"{hgcn.get('wall_time', hgcn.get('wall_time_seconds', 0)):.1f}s",
            "sc_norm": norms.get("SC", 0),
            "ctx_norm": norms.get("CTX", 0),
            "anc_norm": norms.get("ANC", 0),
            "ordering": "SC < CTX < ANC" if ordering_ok else "VIOLATION",
        },
        "alerts": alerts,
    }


def collect_lungs() -> Dict:
    """Retrieval engine health."""
    conn = _pmis_conn()
    alerts = []

    r7d = "timestamp > datetime('now','-7 days')"
    avg_range = _safe_one(conn, f"SELECT AVG(score_range) as v FROM turn_diagnostics WHERE {r7d} AND score_range IS NOT NULL")
    avg_disc = _safe_one(conn, f"SELECT AVG(hierarchy_score_discriminative) as v FROM turn_diagnostics WHERE {r7d}")
    avg_sem = _safe_one(conn, f"SELECT AVG(avg_semantic) as v FROM turn_diagnostics WHERE {r7d}")
    avg_hier = _safe_one(conn, f"SELECT AVG(avg_hierarchy) as v FROM turn_diagnostics WHERE {r7d}")
    narrow_hit = _safe_one(conn, f"SELECT AVG(CASE WHEN narrow_candidates_found > 0 THEN 1.0 ELSE 0.0 END) as v FROM turn_diagnostics WHERE {r7d}")
    conn.close()

    disc = avg_disc.get("v") or 0
    sr = avg_range.get("v") or 0
    status = "green"
    if disc < 0.03 or sr < 0.03:
        status = "red"
        alerts.append("Retrieval not discriminating — all results score similarly")
    elif disc < 0.05 or sr < 0.05:
        status = "yellow"

    return {
        "status": status,
        "metrics": {
            "score_range": round(sr, 4),
            "hierarchy_disc": round(disc, 4),
            "avg_semantic": round(avg_sem.get("v") or 0, 3),
            "avg_hierarchy": round(avg_hier.get("v") or 0, 3),
            "narrow_hit_rate": f"{(narrow_hit.get('v') or 0)*100:.0f}%",
        },
        "alerts": alerts,
    }


def collect_immune() -> Dict:
    """Consolidation pipeline health."""
    conn = _pmis_conn()
    alerts = []

    last = _safe_one(conn, "SELECT MAX(run_date) as ts FROM consolidation_log")
    total = _safe_one(conn, "SELECT COUNT(*) as c FROM consolidation_log")
    by_action = _safe_query(conn, "SELECT action, COUNT(*) as c FROM consolidation_log GROUP BY action ORDER BY c DESC")

    # Orphans, stale, oversized
    orphans = _safe_one(conn, "SELECT COUNT(*) as c FROM memory_nodes WHERE is_orphan=1 AND is_deleted=0")
    stale = _safe_one(conn, "SELECT COUNT(*) as c FROM memory_nodes WHERE level IN ('SC','CTX') AND is_deleted=0 AND (last_accessed IS NULL OR last_accessed < datetime('now','-14 days'))")
    oversized = _safe_one(conn, "SELECT COUNT(*) as c FROM (SELECT target_id FROM relations WHERE relation_type='child_of' GROUP BY target_id HAVING COUNT(*)>25)")
    conn.close()

    status = "green"
    if orphans.get("c", 0) > 50:
        status = "yellow"
        alerts.append(f"{orphans['c']} orphan nodes need adoption")
    if oversized.get("c", 0) > 0:
        alerts.append(f"{oversized['c']} oversized nodes need cell division")

    return {
        "status": status,
        "metrics": {
            "last_run": (last.get("ts") or "never")[:16],
            "total_actions": total.get("c", 0),
            "by_action": {r["action"]: r["c"] for r in by_action},
            "orphans": orphans.get("c", 0),
            "stale_contexts": stale.get("c", 0),
            "oversized": oversized.get("c", 0),
        },
        "alerts": alerts,
    }


def collect_skeleton() -> Dict:
    """Knowledge graph structure."""
    conn = _pmis_conn()
    alerts = []

    counts = {}
    for level in ["SC", "CTX", "ANC"]:
        r = _safe_one(conn, "SELECT COUNT(*) as c FROM memory_nodes WHERE level=? AND is_deleted=0", (level,))
        counts[level] = r.get("c", 0)

    total = sum(counts.values())
    edges = _safe_one(conn, "SELECT COUNT(*) as c FROM relations")
    trees = _safe_one(conn, "SELECT COUNT(*) as c FROM trees")
    orphans = _safe_one(conn, "SELECT COUNT(*) as c FROM memory_nodes WHERE is_orphan=1 AND is_deleted=0")
    deleted = _safe_one(conn, "SELECT COUNT(*) as c FROM memory_nodes WHERE is_deleted=1")
    conn.close()

    orphan_ratio = (orphans.get("c", 0) / max(total, 1)) * 100
    status = "green"
    if orphan_ratio > 15:
        status = "red"
        alerts.append(f"Orphan ratio {orphan_ratio:.1f}% — graph fragmenting")
    elif orphan_ratio > 5:
        status = "yellow"

    return {
        "status": status,
        "metrics": {
            "sc": counts.get("SC", 0),
            "ctx": counts.get("CTX", 0),
            "anc": counts.get("ANC", 0),
            "total": total,
            "edges": edges.get("c", 0),
            "trees": trees.get("c", 0),
            "orphan_ratio": f"{orphan_ratio:.1f}%",
            "archived": deleted.get("c", 0),
        },
        "alerts": alerts,
    }


def collect_nervous() -> Dict:
    """Wiki + Goals + Feedback."""
    conn = _pmis_conn()
    alerts = []

    cached = _safe_one(conn, "SELECT COUNT(*) as c FROM wiki_page_cache")
    goals_active = _safe_one(conn, "SELECT COUNT(*) as c FROM goals WHERE status='active'")
    goals_total = _safe_one(conn, "SELECT COUNT(*) as c FROM goals")
    feedback = _safe_one(conn, "SELECT COUNT(*) as c FROM feedback")
    fb_7d = _safe_one(conn, "SELECT COUNT(*) as c FROM feedback WHERE timestamp > datetime('now','-7 days')")
    goal_links = _safe_one(conn, "SELECT COUNT(*) as c FROM goal_links")
    conn.close()

    status = "green"
    if goals_active.get("c", 0) == 0:
        status = "yellow"
        alerts.append("No active goals defined")
    if feedback.get("c", 0) == 0:
        alerts.append("No feedback recorded yet")

    return {
        "status": status,
        "metrics": {
            "cached_pages": cached.get("c", 0),
            "active_goals": goals_active.get("c", 0),
            "total_goals": goals_total.get("c", 0),
            "goal_links": goal_links.get("c", 0),
            "feedback_total": feedback.get("c", 0),
            "feedback_7d": fb_7d.get("c", 0),
        },
        "alerts": alerts,
    }


def collect_timeline() -> List[Dict]:
    """Recent events across all organs."""
    events = []

    conn = _pmis_conn()
    # Recent turns
    for r in _safe_query(conn, "SELECT conversation_id, turn_number, gamma_final, gamma_mode, storage_action, timestamp FROM turn_diagnostics ORDER BY id DESC LIMIT 8"):
        events.append({
            "time": r["timestamp"][:16] if r["timestamp"] else "",
            "source": "blood/claude",
            "detail": f"turn #{r['turn_number']} γ={r['gamma_final']:.2f} {r['gamma_mode']} {r['storage_action'] or ''}",
        })

    # Recent consolidation
    for r in _safe_query(conn, "SELECT action, run_date, reason FROM consolidation_log ORDER BY id DESC LIMIT 8"):
        events.append({
            "time": r["run_date"][:16] if r["run_date"] else "",
            "source": "immune",
            "detail": f"{r['action']} {(r['reason'] or '')[:40]}",
        })
    conn.close()

    # Recent tracker segments
    tconn = _tracker_conn()
    if tconn:
        for r in _safe_query(tconn, "SELECT target_segment_id, worker, target_segment_length_secs, timestamp_start FROM context_1 ORDER BY timestamp_start DESC LIMIT 8"):
            events.append({
                "time": r["timestamp_start"][:16] if r["timestamp_start"] else "",
                "source": "blood/tracker",
                "detail": f"{r['target_segment_id']} {r['worker']} {r['target_segment_length_secs'] or 0}s",
            })
        tconn.close()

    # Sort by time descending
    events.sort(key=lambda e: e.get("time", ""), reverse=True)
    return events[:20]


# ═══════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "health_dashboard.html", {})


@app.get("/api/health")
async def health_api():
    """Full health payload for all 7 organs."""
    heart = collect_heart()
    blood = collect_blood()
    brain = collect_brain()
    lungs = collect_lungs()
    immune = collect_immune()
    skeleton = collect_skeleton()
    nervous = collect_nervous()
    timeline = collect_timeline()

    # Aggregate alerts
    all_alerts = []
    for name, organ in [("Heart", heart), ("Blood", blood), ("Brain", brain),
                        ("Lungs", lungs), ("Immune", immune), ("Skeleton", skeleton),
                        ("Nervous", nervous)]:
        for a in organ.get("alerts", []):
            all_alerts.append({"source": name, "message": a, "severity": organ["status"]})

    return {
        "timestamp": datetime.now().isoformat()[:19],
        "heart": heart,
        "blood": blood,
        "brain": brain,
        "lungs": lungs,
        "immune": immune,
        "skeleton": skeleton,
        "nervous": nervous,
        "timeline": timeline,
        "alerts": all_alerts,
    }


# ═══════════════════════════════════════════════════════════
# ACTION ENDPOINTS (Manual Overrides)
# ═══════════════════════════════════════════════════════════

def _run_in_thread(fn, *args):
    """Run a blocking function in a background thread."""
    result = {"status": "running", "error": None, "result": None}
    def worker():
        try:
            result["result"] = fn(*args)
            result["status"] = "done"
        except Exception as e:
            result["error"] = str(e)
            result["status"] = "failed"
    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=300)
    return result


@app.post("/api/action/train-hgcn")
async def action_train_hgcn():
    """Trigger HGCN training."""
    def do_train():
        os.chdir(str(PMIS_DIR))
        from db.manager import DBManager
        from core.hgcn import HGCNTrainer
        from core import config
        hp = config.get_all()
        hp['hgcn_epochs'] = 100
        hp['hgcn_patience'] = 15
        hp['hgcn_lr'] = 0.005
        db = DBManager("data/memory.db")
        trainer = HGCNTrainer(db, hp)
        return trainer.train()
    return _run_in_thread(do_train)


@app.post("/api/action/consolidate")
async def action_consolidate():
    """Trigger full consolidation."""
    def do_consolidate():
        os.chdir(str(PMIS_DIR))
        from db.manager import DBManager
        from db.chroma_store import ChromaStore
        from consolidation.nightly import NightlyConsolidation
        from core import config
        hp = config.get_all()
        hp['hgcn_epochs'] = 50
        hp['hgcn_patience'] = 10
        hp['hgcn_lr'] = 0.005
        chroma = ChromaStore(persist_dir=str(PMIS_DIR / "data" / "chroma"))
        db = DBManager("data/memory.db", chroma_store=chroma)
        consol = NightlyConsolidation(db, hp)
        return consol.run()
    r = _run_in_thread(do_consolidate)
    return {"status": r["status"], "error": r.get("error")}


@app.post("/api/action/regen-wiki")
async def action_regen_wiki():
    """Clear wiki cache and regenerate all SC pages."""
    def do_regen():
        os.chdir(str(PMIS_DIR))
        conn = sqlite3.connect(PMIS_DB)
        conn.execute("DELETE FROM wiki_page_cache")
        conn.commit()
        scs = conn.execute("SELECT id FROM memory_nodes WHERE level='SC' AND is_deleted=0").fetchall()
        conn.close()
        from db.manager import DBManager
        from wiki_renderer import WikiRenderer
        db = DBManager("data/memory.db")
        wiki = WikiRenderer(db)
        for sc in scs:
            wiki.generate_wiki_prose(sc[0])
        return {"pages_regenerated": len(scs)}
    return _run_in_thread(do_regen)


@app.post("/api/action/activity-merge")
async def action_activity_merge():
    """Run daily activity merge."""
    def do_merge():
        os.chdir(str(PMIS_DIR))
        from db.manager import DBManager
        from daily_activity_merge import DailyActivityMerger
        from core import config
        db = DBManager("data/memory.db")
        merger = DailyActivityMerger(db, config.get_all())
        return merger.run()
    return _run_in_thread(do_merge)


@app.post("/api/action/assign-time")
async def action_assign_time():
    """Run time assignment."""
    def do_assign():
        os.chdir(str(PMIS_DIR))
        from db.manager import DBManager
        from time_assignment import TimeAssignment
        from core import config
        db = DBManager("data/memory.db")
        assigner = TimeAssignment(db, config.get_all())
        return assigner.run()
    return _run_in_thread(do_assign)


@app.post("/api/action/cell-divide")
async def action_cell_divide():
    """Run cell division only."""
    def do_divide():
        os.chdir(str(PMIS_DIR))
        from db.manager import DBManager
        from consolidation.cell_division import CellDivision
        from core.poincare import ProjectionManager
        from core import config
        hp = config.get_all()
        db = DBManager("data/memory.db")
        pm = ProjectionManager(input_dim=hp.get("local_embedding_dimensions", 768), output_dim=hp.get("poincare_dimensions", 16))
        divider = CellDivision(db, hp, pm)
        return divider.run()
    return _run_in_thread(do_divide)


@app.post("/api/action/reindex")
async def action_reindex():
    """Rebuild ChromaDB ANN index."""
    def do_reindex():
        os.chdir(str(PMIS_DIR))
        from db.manager import DBManager
        from db.chroma_store import ChromaStore
        chroma = ChromaStore(persist_dir=str(PMIS_DIR / "data" / "chroma"))
        db = DBManager("data/memory.db", chroma_store=chroma)
        if db.has_ann_index and db._chroma:
            db._chroma.rebuild_from_db(db)
            return {"status": "rebuilt"}
        return {"status": "no chroma index"}
    return _run_in_thread(do_reindex)


if __name__ == "__main__":
    os.chdir(str(PMIS_DIR))
    uvicorn.run(app, host="0.0.0.0", port=8200, log_level="info")
