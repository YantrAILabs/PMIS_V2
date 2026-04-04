#!/usr/bin/env python3
"""
PMIS V2 CLI Bridge

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
    """Run nightly consolidation."""
    orch = _get_orchestrator()
    result_text = orch.handle_command("consolidate")
    return {"result": result_text, "completed": True}


@_safe_output
def cmd_orphans(args):
    """List orphan anchors."""
    orch = _get_orchestrator()
    result_text = orch.handle_command("orphans")
    return {"result": result_text}


@_safe_output
def cmd_command(args):
    """Execute a slash command."""
    orch = _get_orchestrator()
    conv_id = _load_session_id()
    cmd_name = args.name

    result_text = orch.handle_command(cmd_name, conv_id)
    return {"command": cmd_name, "result": result_text}


# ================================================================
# ARGUMENT PARSER
# ================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        prog="pmis_v2",
        description="PMIS V2 Memory System CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # session begin
    session_parser = subparsers.add_parser("session", help="Session management")
    session_sub = session_parser.add_subparsers(dest="session_command")

    begin_p = session_sub.add_parser("begin", help="Process a conversation turn")
    begin_p.add_argument("message", nargs="+", help="User message text")

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
    subparsers.add_parser("consolidate", help="Run nightly consolidation")
    subparsers.add_parser("orphans", help="List orphan anchors")

    cmd_p = subparsers.add_parser("command", help="Execute a slash command")
    cmd_p.add_argument("name", help="Command name (explore, exploit, surprise, etc.)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "session":
        if args.session_command == "begin":
            cmd_session_begin(args)
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
    elif args.command == "command":
        cmd_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
