"""
Claude.ai Conversation JSON Parser for PMIS v2 Migration.

Claude's export format (from Settings → Export Data):
  - JSON file with array of conversations
  - Each conversation has: uuid, name, created_at, updated_at, chat_messages
  - Each message has: uuid, text, sender (human/assistant), created_at

This parser extracts meaningful content from conversations and converts
each significant turn into a candidate MemoryNode.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


def parse_claude_export(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a Claude.ai exported JSON file.
    Returns a flat list of memory candidates, one per significant message.
    """
    path = Path(file_path)

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Expected .json file, got {path.suffix}")

    # Handle different export structures
    conversations = []
    if isinstance(raw, list):
        conversations = raw
    elif isinstance(raw, dict):
        # Some exports wrap in a top-level key
        for key in ["conversations", "chats", "data"]:
            if key in raw and isinstance(raw[key], list):
                conversations = raw[key]
                break
        if not conversations and "chat_messages" in raw:
            # Single conversation export
            conversations = [raw]

    results = []
    for conv in conversations:
        conv_results = _parse_single_conversation(conv)
        results.extend(conv_results)

    return results


def _parse_single_conversation(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a single Claude conversation into memory candidates."""
    results = []

    conv_id = conv.get("uuid", conv.get("id", ""))
    conv_name = conv.get("name", conv.get("title", "Untitled"))
    conv_created = _parse_timestamp(conv.get("created_at"))

    # Get messages — Claude uses different keys in different export versions
    messages = (
        conv.get("chat_messages") or
        conv.get("messages") or
        conv.get("mapping") or
        []
    )

    # If mapping (tree structure), flatten to ordered list
    if isinstance(messages, dict):
        messages = _flatten_mapping(messages)

    prev_id = None
    turn_number = 0

    for msg in messages:
        # Extract text content
        text = _extract_text(msg)
        if not text or len(text.strip()) < 30:
            continue

        sender = (
            msg.get("sender") or
            msg.get("role") or
            msg.get("author", {}).get("role", "unknown")
        )
        # Normalize sender
        if sender in ("human", "user"):
            role = "user"
        elif sender in ("assistant", "ai", "claude"):
            role = "assistant"
        else:
            continue  # Skip system messages

        msg_created = _parse_timestamp(
            msg.get("created_at") or msg.get("create_time")
        ) or conv_created

        turn_number += 1

        candidate = {
            "content": text[:2000],  # Cap at 2000 chars per memory
            "source": "claude",
            "conversation_id": conv_id,
            "conversation_name": conv_name,
            "role": role,
            "turn_number": turn_number,
            "created_at": msg_created,
            "message_id": msg.get("uuid", msg.get("id", "")),
            "prev_message_id": prev_id,
            # Level hint: long assistant messages with structure → likely Context-worthy
            "level_hint": _infer_level(text, role),
        }

        prev_id = candidate["message_id"]
        results.append(candidate)

    return results


def _extract_text(msg: Dict[str, Any]) -> Optional[str]:
    """Extract text content from various message formats."""
    # Direct text field
    if isinstance(msg.get("text"), str):
        return msg["text"]

    # Content field (string or structured)
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts = content.get("parts", [])
        if parts:
            return "\n".join(str(p) for p in parts if isinstance(p, str))
        return content.get("text", "")
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)

    # Message field wrapping
    if "message" in msg and isinstance(msg["message"], dict):
        return _extract_text(msg["message"])

    return None


def _flatten_mapping(mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten a tree-structured mapping (like ChatGPT's format) into
    an ordered list of messages by following parent → children chain.
    """
    # Find root (node with no parent or parent not in mapping)
    nodes = {}
    for node_id, node_data in mapping.items():
        nodes[node_id] = node_data

    # Find root
    root_id = None
    for node_id, node_data in nodes.items():
        parent = node_data.get("parent")
        if parent is None or parent not in nodes:
            root_id = node_id
            break

    if root_id is None:
        # Fallback: just return all nodes with messages
        return [n for n in nodes.values() if n.get("message")]

    # BFS from root
    ordered = []
    queue = [root_id]
    visited = set()

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        node = nodes.get(current, {})
        if node.get("message"):
            ordered.append(node)

        children = node.get("children", [])
        queue.extend(children)

    return ordered


def _parse_timestamp(ts) -> Optional[datetime]:
    """Parse various timestamp formats."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts)
        except (ValueError, OSError):
            return None
    if isinstance(ts, str):
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]:
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", ""))
        except ValueError:
            return None
    return None


def _infer_level(text: str, role: str) -> str:
    """
    Heuristic to guess hierarchy level from content.
    User questions → likely Anchors
    Long structured assistant responses → likely Context-worthy
    Short responses → Anchors
    """
    if role == "user":
        return "ANC"

    # Assistant responses: longer + structured = more abstract
    if len(text) > 1000 and any(marker in text for marker in ["##", "**", "1.", "- "]):
        return "CTX"  # Structured analysis → Context candidate
    return "ANC"
