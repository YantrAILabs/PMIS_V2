#!/usr/bin/env python3
"""
Telegram Approval Bot for Claude Code Projects

Send approval requests to Telegram, tap Approve/Reject on your phone,
get the result back as JSON. Zero new dependencies — uses httpx + stdlib.

Usage:
    python3 pmis_v2/approve.py request "Deploy vision-ai v2.1?"
    python3 pmis_v2/approve.py request "Push to prod?" --timeout 120 --source vision-ai
    python3 pmis_v2/approve.py history
    python3 pmis_v2/approve.py history --status approved
    python3 pmis_v2/approve.py test
    python3 pmis_v2/approve.py setup
"""

import sys
import os
import json
import argparse
import asyncio
import sqlite3
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone

try:
    import httpx
except ImportError:
    print(json.dumps({"error": "httpx not installed. Run: pip install httpx"}), file=sys.stdout)
    sys.exit(1)

try:
    import yaml
except ImportError:
    print(json.dumps({"error": "pyyaml not installed. Run: pip install pyyaml"}), file=sys.stdout)
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "approval_config.yaml"
DB_PATH = SCRIPT_DIR / "data" / "approval.db"
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


# --- Config ---

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(json.dumps({
            "error": "Config not found. Run: python3 pmis_v2/approve.py setup",
            "path": str(CONFIG_PATH)
        }))
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("bot_token") or cfg["bot_token"] == "YOUR_BOT_TOKEN":
        print(json.dumps({"error": "bot_token not configured. Run: python3 pmis_v2/approve.py setup"}))
        sys.exit(1)
    if not cfg.get("chat_id") or cfg["chat_id"] == 0:
        print(json.dumps({"error": "chat_id not configured. Run: python3 pmis_v2/approve.py setup"}))
        sys.exit(1)
    cfg.setdefault("timeout", 300)
    return cfg


# --- SQLite ---

def _init_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS approvals (
            id TEXT PRIMARY KEY,
            message TEXT NOT NULL,
            telegram_message_id INTEGER,
            status TEXT DEFAULT 'pending',
            requested_at TEXT NOT NULL,
            responded_at TEXT,
            responded_by TEXT,
            source TEXT
        )
    """)
    conn.commit()
    return conn


def _log_request(conn: sqlite3.Connection, request_id: str, message: str, source: str = None):
    conn.execute(
        "INSERT INTO approvals (id, message, status, requested_at, source) VALUES (?, ?, 'pending', ?, ?)",
        (request_id, message, datetime.now(timezone.utc).isoformat(), source)
    )
    conn.commit()


def _log_response(conn: sqlite3.Connection, request_id: str, status: str,
                   telegram_message_id: int = None, responded_by: str = None):
    conn.execute(
        "UPDATE approvals SET status=?, responded_at=?, responded_by=?, telegram_message_id=? WHERE id=?",
        (status, datetime.now(timezone.utc).isoformat(), responded_by, telegram_message_id, request_id)
    )
    conn.commit()


# --- Telegram API ---

def _api_url(token: str, method: str) -> str:
    return TELEGRAM_API.format(token=token, method=method)


async def _send_approval(client: httpx.AsyncClient, token: str, chat_id: int,
                          request_id: str, message: str) -> dict:
    short_id = request_id[:8]
    text = (
        f"🔔 *Approval Request*\n\n"
        f"{message}\n\n"
        f"_Requested: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n"
        f"_ID: `{short_id}`_"
    )
    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Approve", "callback_data": f"approve:{request_id}"},
            {"text": "❌ Reject", "callback_data": f"reject:{request_id}"}
        ]]
    }
    resp = await client.post(_api_url(token, "sendMessage"), json={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": keyboard
    })
    data = resp.json()
    if not data.get("ok"):
        err = data.get("description", "Unknown error")
        if "chat not found" in err.lower() or "403" in str(resp.status_code):
            raise RuntimeError(f"Bot can't reach chat_id {chat_id}. Make sure you sent /start to the bot first.")
        raise RuntimeError(f"Telegram API error: {err}")
    return data["result"]


async def _flush_old_updates(client: httpx.AsyncClient, token: str) -> int:
    """Flush old updates and return the next offset."""
    resp = await client.post(_api_url(token, "getUpdates"), json={"offset": -1, "timeout": 0})
    data = resp.json()
    if data.get("ok") and data.get("result"):
        return data["result"][-1]["update_id"] + 1
    return 0


async def _poll_for_response(client: httpx.AsyncClient, token: str,
                              request_id: str, timeout: int, offset: int) -> dict | None:
    deadline = time.monotonic() + timeout
    current_offset = offset

    while time.monotonic() < deadline:
        remaining = max(1, int(deadline - time.monotonic()))
        poll_timeout = min(30, remaining)

        try:
            resp = await client.post(
                _api_url(token, "getUpdates"),
                json={"offset": current_offset, "timeout": poll_timeout},
                timeout=poll_timeout + 10
            )
        except httpx.TimeoutException:
            continue

        data = resp.json()
        if not data.get("ok"):
            continue

        for update in data.get("result", []):
            current_offset = update["update_id"] + 1
            cb = update.get("callback_query")
            if not cb:
                continue
            cb_data = cb.get("data", "")
            if cb_data.endswith(f":{request_id}"):
                action = cb_data.split(":")[0]
                user = cb.get("from", {})
                username = user.get("username") or user.get("first_name", "unknown")
                # Answer the callback
                await client.post(_api_url(token, "answerCallbackQuery"), json={
                    "callback_query_id": cb["id"],
                    "text": "Approved ✅" if action == "approve" else "Rejected ❌"
                })
                return {
                    "action": action,
                    "username": username,
                    "message_id": cb.get("message", {}).get("message_id")
                }

    return None


async def _edit_message(client: httpx.AsyncClient, token: str, chat_id: int,
                         message_id: int, original: str, status: str, username: str):
    icon = "✅" if status == "approved" else "❌"
    label = "APPROVED" if status == "approved" else "REJECTED"
    text = (
        f"{icon} *{label}*\n\n"
        f"{original}\n\n"
        f"_{label} by @{username}_\n"
        f"_At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
    )
    await client.post(_api_url(token, "editMessageText"), json={
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "Markdown"
    })


# --- Main Request Flow ---

async def _request(message: str, source: str = None, timeout: int = None) -> dict:
    cfg = _load_config()
    token = cfg["bot_token"]
    chat_id = cfg["chat_id"]
    if timeout is None:
        timeout = cfg["timeout"]

    request_id = uuid.uuid4().hex[:12]
    conn = _init_db()
    _log_request(conn, request_id, message, source)

    async with httpx.AsyncClient() as client:
        # Flush old updates first
        offset = await _flush_old_updates(client, token)

        # Send approval message
        msg_result = await _send_approval(client, token, chat_id, request_id, message)
        msg_id = msg_result["message_id"]

        # Update DB with telegram message ID
        conn.execute("UPDATE approvals SET telegram_message_id=? WHERE id=?", (msg_id, request_id))
        conn.commit()

        print(json.dumps({"status": "waiting", "request_id": request_id, "message": message}),
              file=sys.stderr)

        # Poll for response
        result = await _poll_for_response(client, token, request_id, timeout, offset)

        if result is None:
            _log_response(conn, request_id, "timeout", msg_id)
            await _edit_message(client, token, chat_id, msg_id, message, "timeout", "system")
            conn.close()
            return {"approved": False, "status": "timeout", "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()}

        status = "approved" if result["action"] == "approve" else "rejected"
        _log_response(conn, request_id, status, msg_id, result["username"])

        # Edit the original message to show result
        await _edit_message(client, token, chat_id, msg_id, message, status, result["username"])

        conn.close()
        return {
            "approved": status == "approved",
            "status": status,
            "by": result["username"],
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# --- CLI Commands ---

def cmd_request(args):
    try:
        result = asyncio.run(_request(args.message, source=args.source, timeout=args.timeout))
    except RuntimeError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    print(json.dumps(result))
    if result.get("status") == "timeout":
        sys.exit(2)
    elif not result.get("approved"):
        sys.exit(1)
    sys.exit(0)


def cmd_history(args):
    conn = _init_db()
    query = "SELECT * FROM approvals"
    params = []
    if args.status:
        query += " WHERE status = ?"
        params.append(args.status)
    query += " ORDER BY requested_at DESC LIMIT ?"
    params.append(args.limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    result = [dict(r) for r in rows]
    print(json.dumps({"approvals": result, "count": len(result)}, indent=2))


def cmd_test(args):
    cfg = _load_config()

    async def _run_test():
        async with httpx.AsyncClient() as client:
            # Test getMe
            resp = await client.post(_api_url(cfg["bot_token"], "getMe"))
            me = resp.json()
            if not me.get("ok"):
                return {"error": f"Bot token invalid: {me.get('description')}"}

            bot_name = me["result"].get("username", "unknown")

            # Test sendMessage
            resp = await client.post(_api_url(cfg["bot_token"], "sendMessage"), json={
                "chat_id": cfg["chat_id"],
                "text": f"🧪 *Test from approve.py*\n\nBot `@{bot_name}` is working.\nConfig is valid.",
                "parse_mode": "Markdown"
            })
            send = resp.json()
            if not send.get("ok"):
                return {"error": f"Can't send to chat_id {cfg['chat_id']}: {send.get('description')}. "
                                 f"Did you /start the bot?"}

            return {"ok": True, "bot": f"@{bot_name}", "chat_id": cfg["chat_id"],
                    "message": "Test message sent successfully"}

    try:
        result = asyncio.run(_run_test())
    except Exception as e:
        result = {"error": str(e)}

    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("ok") else 1)


def cmd_setup(args):
    print("=" * 50)
    print("  Telegram Approval Bot — Setup")
    print("=" * 50)
    print()
    print("Step 1: Create a bot")
    print("  → Open Telegram, search for @BotFather")
    print("  → Send /newbot, follow prompts")
    print("  → Copy the bot token")
    print()

    token = input("Paste your bot token: ").strip()
    if not token:
        print("Aborted.")
        sys.exit(1)

    print()
    print("Step 2: Get your chat ID")
    print("  → Send any message to your new bot on Telegram")
    print("  → Press Enter here after you've sent a message...")
    input()

    # Fetch chat_id from recent messages
    async def _get_chat_id():
        async with httpx.AsyncClient() as client:
            resp = await client.post(_api_url(token, "getUpdates"), json={"timeout": 5})
            data = resp.json()
            if not data.get("ok") or not data.get("result"):
                return None
            for update in reversed(data["result"]):
                msg = update.get("message", {})
                chat = msg.get("chat", {})
                if chat.get("id"):
                    return chat["id"]
            return None

    chat_id = asyncio.run(_get_chat_id())
    if not chat_id:
        print("Could not detect your chat ID. Make sure you messaged the bot.")
        print("You can also find it manually:")
        print(f"  Visit: https://api.telegram.org/bot{token}/getUpdates")
        chat_id = input("Enter chat_id manually: ").strip()
        if not chat_id:
            sys.exit(1)
        chat_id = int(chat_id)

    print(f"\n  Detected chat_id: {chat_id}")

    config = {
        "bot_token": token,
        "chat_id": chat_id,
        "timeout": 300
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        f.write("# Telegram Approval Bot Config\n")
        f.write(f"bot_token: \"{token}\"\n")
        f.write(f"chat_id: {chat_id}\n")
        f.write("timeout: 300  # seconds (5 minutes)\n")

    print(f"\n  Config saved to: {CONFIG_PATH}")
    print("\nStep 3: Test it")
    print("  python3 pmis_v2/approve.py test")
    print()


# --- Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Telegram Approval Bot")
    sub = parser.add_subparsers(dest="command")

    # request
    p_req = sub.add_parser("request", help="Send an approval request")
    p_req.add_argument("message", help="The approval message")
    p_req.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    p_req.add_argument("--source", type=str, default=None, help="Source project tag")

    # history
    p_hist = sub.add_parser("history", help="List past approvals")
    p_hist.add_argument("--status", choices=["pending", "approved", "rejected", "timeout"])
    p_hist.add_argument("--limit", type=int, default=20)

    # test
    sub.add_parser("test", help="Test bot configuration")

    # setup
    sub.add_parser("setup", help="Interactive setup wizard")

    args = parser.parse_args()

    if args.command == "request":
        cmd_request(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "setup":
        cmd_setup(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
