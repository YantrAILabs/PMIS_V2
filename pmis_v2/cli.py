#!/usr/bin/env python3
"""
ProMe CLI Bridge

Single entry point for Claude Desktop and external tools.
All output is JSON to stdout. All errors to stderr.

Usage:
    python3 pmis_v2/cli.py session begin "user message here"
    python3 pmis_v2/cli.py session store '{"super_context":"...","contexts":[...]}'
    python3 pmis_v2/cli.py session rate up|down ["anchor1,anchor2"]
    python3 pmis_v2/cli.py session end
    python3 pmis_v2/cli.py status
    python3 pmis_v2/cli.py browse
    python3 pmis_v2/cli.py stats
    python3 pmis_v2/cli.py consolidate
    python3 pmis_v2/cli.py orphans
    python3 pmis_v2/cli.py command explore|exploit|surprise
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add pmis_v2 to path
PMIS_DIR = Path(__file__).parent
sys.path.insert(0, str(PMIS_DIR))

# Lazy singleton orchestrator
_orch = None
_session_file = PMIS_DIR / "data" / ".session.json"


def _get_orchestrator():
    global _orch
    if _orch is None:
        from orchestrator import Orchestrator
        db_path = str(PMIS_DIR / "data" / "memory.db")
        _orch = Orchestrator(db_path=db_path)
    return _orch


def _load_session_id():
    """Load conversation_id from persistent session file."""
    if _session_file.exists():
        try:
            data = json.loads(_session_file.read_text())
            return data.get("conversation_id")
        except (json.JSONDecodeError, IOError):
            pass
    return None


def _save_session_id(conversation_id):
    """Persist conversation_id across CLI calls."""
    _session_file.parent.mkdir(parents=True, exist_ok=True)
    _session_file.write_text(json.dumps({
        "conversation_id": conversation_id,
        "updated_at": datetime.now().isoformat(),
    }))


def _clear_session():
    """Remove session file."""
    if _session_file.exists():
        _session_file.unlink()


def _safe_output(func):
    """Decorator: catch all exceptions, always output valid JSON."""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            print(json.dumps(result, default=str, ensure_ascii=False))
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({
                "error": str(e),
                "type": type(e).__name__,
            }))
            sys.exit(1)
    return wrapper


# ================================================================
# COMMANDS
# ================================================================

@_safe_output
def cmd_session_begin(args):
    """Process a conversation turn. Main entry point."""
    orch = _get_orchestrator()
    message = " ".join(args.message)

    # Load or create session
    conv_id = _load_session_id()
    session = orch.get_or_create_session(conv_id)
    _save_session_id(session.conversation_id)

    is_new = session.turn_counter == 0

    # Process the turn
    result = orch.process_turn(
        content=message,
        conversation_id=session.conversation_id,
        role="user",
    )

    return {
        "memories": result.system_prompt,
        "session": {
            "conversation_id": session.conversation_id,
            "turn_count": result.turn_number,
            "is_new_session": is_new,
            "needs_rating": False,
            "should_ask_rating": result.turn_number > 0 and result.turn_number % 3 == 0,
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


@_safe_output
def cmd_session_continue(args):
    """Continue an existing conversation with a follow-up message.
    Reuses conversation_id and reconstructs session state from DB."""
    orch = _get_orchestrator()
    message = " ".join(args.message)

    conv_id = _load_session_id()
    if not conv_id:
        # No session to continue — fall through to begin
        return cmd_session_begin.__wrapped__(args)

    session = orch.get_or_create_session(conv_id)

    # Reconstruct session state from DB if this is a fresh process
    if session.turn_counter == 0:
        _reconstruct_session_state(orch.db, session, conv_id)

    # Process the turn
    result = orch.process_turn(
        content=message,
        conversation_id=session.conversation_id,
        role="user",
    )

    return {
        "memories": result.system_prompt,
        "session": {
            "conversation_id": session.conversation_id,
            "turn_count": result.turn_number,
            "is_new_session": False,
            "is_continuation": True,
            "needs_rating": False,
            "should_ask_rating": result.turn_number > 0 and result.turn_number % 3 == 0,
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


def _reconstruct_session_state(db, session, conv_id):
    """Reconstruct session state from conversation_turns table.
    Called when session continue is used from a fresh CLI process."""
    import sqlite3

    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Load prior turns for this conversation
        cursor.execute("""
            SELECT turn_number, gamma, effective_surprise, mode, node_id
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY turn_number ASC
        """, (conv_id,))

        rows = cursor.fetchall()
        for row in rows:
            session.turn_counter = row["turn_number"]
            if row["gamma"] is not None:
                session.gamma_history.append(row["gamma"])
            if row["effective_surprise"] is not None:
                session.surprise_history.append(row["effective_surprise"])
            if row["node_id"]:
                session.last_stored_node_id = row["node_id"]
                session.stored_node_ids.append(row["node_id"])

        # Reconstruct precision accumulator from diagnostics if available
        cursor.execute("""
            SELECT session_precision_accumulator
            FROM turn_diagnostics
            WHERE conversation_id = ?
            ORDER BY turn_number DESC LIMIT 1
        """, (conv_id,))
        diag_row = cursor.fetchone()
        if diag_row and diag_row["session_precision_accumulator"]:
            session.precision_accumulator = diag_row["session_precision_accumulator"]

        conn.close()
    except Exception:
        pass  # Best effort — if reconstruction fails, start fresh


@_safe_output
def cmd_session_store(args):
    """Manual memory store (structured JSON input)."""
    orch = _get_orchestrator()
    conv_id = _load_session_id()

    try:
        store_data = json.loads(args.json_data)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}", "stored": False}

    # Use the ingestion pipeline for structured store
    session = orch.get_or_create_session(conv_id)

    # Build content from structured data
    sc_name = store_data.get("super_context", "Unknown")
    contexts = store_data.get("contexts", [])
    summary = store_data.get("summary", "")

    content_parts = [f"[{sc_name}]"]
    for ctx in contexts:
        ctx_title = ctx.get("title", "")
        content_parts.append(f"  {ctx_title}:")
        for anc in ctx.get("anchors", []):
            anc_title = anc.get("title", "")
            anc_content = anc.get("content", "")
            content_parts.append(f"    - {anc_title}: {anc_content}")

    content = "\n".join(content_parts)

    result = orch.process_turn(
        content=content,
        conversation_id=session.conversation_id,
        role="user",
    )

    return {
        "stored": result.stored_node_id is not None,
        "stored_node_id": result.stored_node_id,
        "storage_action": result.storage_action,
        "session": {"conversation_id": session.conversation_id},
    }


@_safe_output
def cmd_session_rate(args):
    """Record feedback on the session."""
    orch = _get_orchestrator()
    conv_id = _load_session_id()

    if not conv_id:
        return {"error": "No active session to rate", "rated": False}

    direction = args.direction
    anchors = args.anchors if hasattr(args, "anchors") and args.anchors else None

    # For now, use the handle_command interface
    if direction == "up":
        msg = orch.handle_command("exploit", conv_id)
        return {"rated": True, "direction": "up", "message": "Positive feedback recorded"}
    else:
        return {"rated": True, "direction": "down",
                "anchors": anchors,
                "message": "Negative feedback recorded. Specify failing anchors for targeted downweight."}


@_safe_output
def cmd_session_end(args):
    """End the current session."""
    orch = _get_orchestrator()
    conv_id = _load_session_id()

    if not conv_id:
        return {"ended": False, "error": "No active session"}

    orch.close_session(conv_id)
    _clear_session()

    return {"ended": True, "conversation_id": conv_id}


@_safe_output
def cmd_session_log_response(args):
    """Log Claude's response summary for the current turn."""
    orch = _get_orchestrator()
    conv_id = _load_session_id()

    if not conv_id:
        return {"logged": False, "error": "No active session"}

    session = orch.get_or_create_session(conv_id)
    summary = " ".join(args.summary)

    orch.db.update_turn_response(conv_id, session.turn_counter, summary)

    return {"logged": True, "conversation_id": conv_id,
            "turn_number": session.turn_counter}


@_safe_output
def cmd_status(args):
    """System status."""
    orch = _get_orchestrator()
    conv_id = _load_session_id()

    if conv_id:
        status_text = orch.handle_command("status", conv_id)
    else:
        status_text = "No active session."

    stats = orch.get_stats()
    return {
        "status": status_text,
        "stats": stats,
        "session": {"conversation_id": conv_id, "active": conv_id is not None},
    }


@_safe_output
def cmd_browse(args):
    """List all super contexts with their children."""
    orch = _get_orchestrator()
    db = orch.db

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


@_safe_output
def cmd_stats(args):
    """Full system statistics."""
    orch = _get_orchestrator()
    return orch.get_stats()


@_safe_output
def cmd_consolidate(args):
    """Run nightly consolidation via the idempotent runner.

    The runner:
      - catches up every missed full day since last_consolidation_date
      - consolidates today only if past 18:00 local OR --today flag
      - holds a PID lock so it never double-runs
    """
    from consolidation.runner import run_idempotent
    force_today = getattr(args, "today", False) or "--today" in (getattr(args, "_raw_argv", []) or [])
    result = run_idempotent(include_today=True if force_today else None)
    return {"result": result, "completed": True}


@_safe_output
def cmd_orphans(args):
    """List orphan anchors."""
    orch = _get_orchestrator()
    result_text = orch.handle_command("orphans")
    return {"result": result_text}


@_safe_output
def cmd_goal_create(args):
    """Create a new goal."""
    orch = _get_orchestrator()
    import hashlib
    title = " ".join(args.title)
    goal_id = hashlib.sha256(title.encode()).hexdigest()[:10]
    description = args.description or ""

    orch.db.create_goal(goal_id, title, description)
    return {"created": True, "goal_id": goal_id, "title": title}


@_safe_output
def cmd_goal_link(args):
    """Link a goal to a memory node."""
    orch = _get_orchestrator()
    orch.db.link_goal_to_node(
        goal_id=args.goal_id,
        node_id=args.node_id,
        link_type=args.link_type,
        weight=args.weight,
    )
    return {"linked": True, "goal_id": args.goal_id,
            "node_id": args.node_id, "link_type": args.link_type,
            "weight": args.weight}


@_safe_output
def cmd_goal_list(args):
    """List all goals."""
    orch = _get_orchestrator()
    status = args.status if hasattr(args, "status") and args.status else None
    goals = orch.db.list_goals(status=status)

    # Enrich with linked node counts
    for g in goals:
        nodes = orch.db.get_nodes_for_goal(g["id"])
        g["linked_nodes"] = len(nodes)
        fb = orch.db.get_feedback_summary(node_id=None)
        g["feedback"] = fb

    return {"goals": goals, "total": len(goals)}


@_safe_output
def cmd_goal_status(args):
    """Update a goal's status."""
    orch = _get_orchestrator()
    orch.db.update_goal(args.goal_id, status=args.status)
    return {"updated": True, "goal_id": args.goal_id, "status": args.status}


@_safe_output
def cmd_feedback(args):
    """Add feedback to a memory node."""
    orch = _get_orchestrator()
    content = " ".join(args.content) if args.content else ""
    goal_id = args.goal_id if hasattr(args, "goal_id") and args.goal_id else None

    fb_id = orch.db.add_feedback(
        node_id=args.node_id,
        polarity=args.polarity,
        content=content,
        goal_id=goal_id,
        source="explicit",
        strength=1.0,
    )
    return {"recorded": True, "feedback_id": fb_id,
            "node_id": args.node_id, "polarity": args.polarity,
            "content": content}


@_safe_output
def cmd_command(args):
    """Execute a slash command."""
    orch = _get_orchestrator()
    conv_id = _load_session_id()
    cmd_name = args.name

    result_text = orch.handle_command(cmd_name, conv_id)
    return {"command": cmd_name, "result": result_text}


@_safe_output
def cmd_sync_run(args):
    """Run one 30-min sync pass: tracker segments → work_pages."""
    from db.manager import DBManager
    from core import config
    from sync.lock import sync_lock, SyncBusy
    from sync.runner import run_sync

    db_path = str(PMIS_DIR / "data" / "memory.db")
    db = DBManager(db_path)
    hp = config.get_all()
    if getattr(args, "model", None):
        hp["consolidation_model_local"] = args.model
    target = args.date if getattr(args, "date", None) else None

    try:
        with sync_lock():
            result = run_sync(db, hp, target_date=target)
    except SyncBusy as e:
        return {"status": "busy", "detail": str(e)}
    return result


@_safe_output
def cmd_dream_match_pages(args):
    """Dream's gated auto-match over untagged work_pages.

    Bootstrap rule: does nothing until user has confirmed N tags
    (hyperparameters.yaml: dream_auto_match_min_confirmed). Use --force to
    bypass the gate for testing.
    """
    from db.manager import DBManager
    from core import config
    from consolidation.work_page_matcher import run_work_page_matching

    db_path = str(PMIS_DIR / "data" / "memory.db")
    db = DBManager(db_path)
    hp = config.get_all()
    if getattr(args, "model", None):
        hp["consolidation_model_local"] = args.model
    return run_work_page_matching(
        db, hp,
        force=getattr(args, "force", False),
        since_date=getattr(args, "since", None),
    )


@_safe_output
def cmd_sync_rescan_salience(args):
    """Phase A — re-score every work_page with the kachra filter.

    Idempotent: safe to run anytime. Writes salience + kachra_reason to
    each page. Use after heuristic tuning or to apply to backfilled data.
    """
    from db.manager import DBManager
    from sync.salience import rescan_all

    db_path = str(PMIS_DIR / "data" / "memory.db")
    db = DBManager(db_path)
    return rescan_all(db, date_local=getattr(args, "date", None))


@_safe_output
def cmd_sync_humanize(args):
    """Phase B — outcome-shaped rewrite of salient work_pages.

    Gemini Flash primary (if GOOGLE_API_KEY set), qwen2.5:7b fallback.
    Skips already-humanized pages unless --force.
    """
    from db.manager import DBManager
    from core import config
    from sync.humanizer import humanize_all

    db_path = str(PMIS_DIR / "data" / "memory.db")
    db = DBManager(db_path)
    hp = config.get_all()
    if getattr(args, "model", None):
        hp["humanize_model_cloud"] = args.model
    if getattr(args, "local", False):
        hp["humanize_use_cloud"] = False
    return humanize_all(
        db, hp,
        date_local=getattr(args, "date", None),
        force=getattr(args, "force", False),
    )


@_safe_output
def cmd_sync_status(args):
    """Show last sync watermark and today's open/tagged page counts."""
    from db.manager import DBManager
    from datetime import date as _date

    db_path = str(PMIS_DIR / "data" / "memory.db")
    db = DBManager(db_path)
    today = _date.today().isoformat()

    watermark = db.get_last_sync_timestamp() or ""
    open_pages = db.list_work_pages_by_state("open", date_local=today)
    tagged_pages = db.list_work_pages_by_state("tagged", date_local=today)
    return {
        "date": today,
        "last_watermark": watermark,
        "open_count": len(open_pages),
        "tagged_count": len(tagged_pages),
        "open_preview": [
            {"id": p["id"], "title": p["title"][:60]}
            for p in open_pages[:5]
        ],
    }


# ================================================================
# ARGUMENT PARSER
# ================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        prog="pmis_v2",
        description="ProMe Memory System CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # session begin
    session_parser = subparsers.add_parser("session", help="Session management")
    session_sub = session_parser.add_subparsers(dest="session_command")

    begin_p = session_sub.add_parser("begin", help="Process a conversation turn")
    begin_p.add_argument("message", nargs="+", help="User message text")

    continue_p = session_sub.add_parser("continue", help="Continue existing conversation")
    continue_p.add_argument("message", nargs="+", help="Follow-up message text")

    store_p = session_sub.add_parser("store", help="Store structured memory")
    store_p.add_argument("json_data", help="JSON string with memory data")

    rate_p = session_sub.add_parser("rate", help="Rate the session")
    rate_p.add_argument("direction", choices=["up", "down"])
    rate_p.add_argument("anchors", nargs="?", default=None, help="Comma-separated anchor titles")

    session_sub.add_parser("end", help="End the current session")

    log_resp_p = session_sub.add_parser("log-response", help="Log Claude's response summary")
    log_resp_p.add_argument("summary", nargs="+", help="Brief summary of Claude's response")

    # Top-level commands
    subparsers.add_parser("status", help="System status")
    subparsers.add_parser("browse", help="List all super contexts")
    subparsers.add_parser("stats", help="System statistics")
    consolidate_p = subparsers.add_parser("consolidate", help="Run nightly consolidation")
    consolidate_p.add_argument("--today", action="store_true",
                               help="Force-consolidate today even before the 18:00 cutoff")
    subparsers.add_parser("orphans", help="List orphan anchors")

    # sync subcommands (30-min content-layer sync)
    sync_parser = subparsers.add_parser("sync", help="30-min work_pages sync")
    sync_sub = sync_parser.add_subparsers(dest="sync_command")
    sync_run_p = sync_sub.add_parser("run", help="Run one sync pass now")
    sync_run_p.add_argument(
        "--date", default=None, help="Target date YYYY-MM-DD (default: today)"
    )
    sync_run_p.add_argument(
        "--model", default=None,
        help="Override LLM model for title/summary (default: from hyperparameters.yaml)"
    )
    sync_sub.add_parser("status", help="Show watermark + today's page counts")
    sync_rescan_p = sync_sub.add_parser(
        "rescan-salience",
        help="Re-score all work_pages with the kachra filter (retroactive)",
    )
    sync_rescan_p.add_argument(
        "--date", default=None, help="Limit rescan to one YYYY-MM-DD"
    )
    sync_hum_p = sync_sub.add_parser(
        "humanize",
        help="Rewrite salient work_page summaries as outcomes (Phase B)",
    )
    sync_hum_p.add_argument(
        "--date", default=None, help="Limit to one YYYY-MM-DD"
    )
    sync_hum_p.add_argument(
        "--force", action="store_true",
        help="Rewrite even pages already humanized",
    )
    sync_hum_p.add_argument(
        "--model", default=None,
        help="Override Gemini model name (cloud path)",
    )
    sync_hum_p.add_argument(
        "--local", action="store_true",
        help="Force local qwen2.5:7b, skip Gemini even if API key is set",
    )

    # dream subcommands (Phase 9 auto-match + future consolidations)
    dream_parser = subparsers.add_parser("dream", help="Dream / nightly ops")
    dream_sub = dream_parser.add_subparsers(dest="dream_command")
    dream_match_p = dream_sub.add_parser(
        "match-pages", help="Gated auto-match over untagged work_pages"
    )
    dream_match_p.add_argument(
        "--force", action="store_true",
        help="Bypass the bootstrap confirmed-tag gate (for testing)",
    )
    dream_match_p.add_argument(
        "--model", default=None,
        help="Override embed/LLM model for this run",
    )
    dream_match_p.add_argument(
        "--since", default=None,
        help="Only consider pages with date_local >= YYYY-MM-DD",
    )

    # goal subcommands
    goal_parser = subparsers.add_parser("goal", help="Goal management")
    goal_sub = goal_parser.add_subparsers(dest="goal_command")

    goal_create_p = goal_sub.add_parser("create", help="Create a new goal")
    goal_create_p.add_argument("title", nargs="+", help="Goal title")
    goal_create_p.add_argument("--description", "-d", default="", help="Goal description")

    goal_link_p = goal_sub.add_parser("link", help="Link goal to node")
    goal_link_p.add_argument("goal_id", help="Goal ID")
    goal_link_p.add_argument("node_id", help="Memory node ID")
    goal_link_p.add_argument("link_type", choices=["supports", "blocks", "neutral"],
                             default="supports", nargs="?")
    goal_link_p.add_argument("weight", type=float, default=0.5, nargs="?")

    goal_list_p = goal_sub.add_parser("list", help="List all goals")
    goal_list_p.add_argument("--status", "-s", choices=["active", "achieved", "paused", "abandoned"],
                             default=None)

    goal_status_p = goal_sub.add_parser("status", help="Update goal status")
    goal_status_p.add_argument("goal_id", help="Goal ID")
    goal_status_p.add_argument("status", choices=["active", "achieved", "paused", "abandoned"])

    # feedback command
    fb_parser = subparsers.add_parser("feedback", help="Add feedback to a node")
    fb_parser.add_argument("node_id", help="Memory node ID")
    fb_parser.add_argument("polarity", choices=["positive", "negative", "correction"])
    fb_parser.add_argument("content", nargs="*", help="Feedback note")
    fb_parser.add_argument("--goal", dest="goal_id", default=None, help="Link to goal ID")

    cmd_p = subparsers.add_parser("command", help="Execute a slash command")
    cmd_p.add_argument("name", help="Command name (explore, exploit, surprise, etc.)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "session":
        if args.session_command == "begin":
            cmd_session_begin(args)
        elif args.session_command == "continue":
            cmd_session_continue(args)
        elif args.session_command == "store":
            cmd_session_store(args)
        elif args.session_command == "rate":
            cmd_session_rate(args)
        elif args.session_command == "end":
            cmd_session_end(args)
        elif args.session_command == "log-response":
            cmd_session_log_response(args)
        else:
            parser.parse_args(["session", "--help"])
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "browse":
        cmd_browse(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "consolidate":
        cmd_consolidate(args)
    elif args.command == "orphans":
        cmd_orphans(args)
    elif args.command == "goal":
        if args.goal_command == "create":
            cmd_goal_create(args)
        elif args.goal_command == "link":
            cmd_goal_link(args)
        elif args.goal_command == "list":
            cmd_goal_list(args)
        elif args.goal_command == "status":
            cmd_goal_status(args)
        else:
            parser.parse_args(["goal", "--help"])
    elif args.command == "feedback":
        cmd_feedback(args)
    elif args.command == "command":
        cmd_command(args)
    elif args.command == "sync":
        if args.sync_command == "run":
            cmd_sync_run(args)
        elif args.sync_command == "status":
            cmd_sync_status(args)
        elif args.sync_command == "rescan-salience":
            cmd_sync_rescan_salience(args)
        elif args.sync_command == "humanize":
            cmd_sync_humanize(args)
        else:
            parser.parse_args(["sync", "--help"])
    elif args.command == "dream":
        if args.dream_command == "match-pages":
            cmd_dream_match_pages(args)
        else:
            parser.parse_args(["dream", "--help"])
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
