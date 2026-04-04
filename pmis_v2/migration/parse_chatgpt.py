"""
ChatGPT Conversation Parser for PMIS v2 Migration.

ChatGPT's export format (from Settings → Data Controls → Export):
  - conversations.json: Array of conversation objects
  - Each conversation has:
      title, create_time, update_time, mapping (tree of message nodes)
  - mapping is a dict of node_id → {id, message, parent, children}
  - message has: author.role, content.parts, create_time
  - The tree structure supports conversation branches (edits/regenerations)

This parser follows the main branch (first child) through the tree
and extracts meaningful turns as memory candidates.
"""

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


def parse_chatgpt_export(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse ChatGPT exported data.
    Accepts:
      - conversations.json file directly
      - .zip archive from OpenAI export (contains conversations.json)
    """
    path = Path(file_path)
    raw_data = None

    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            # Look for conversations.json inside the zip
            for name in z.namelist():
                if "conversations" in name.lower() and name.endswith(".json"):
                    with z.open(name) as f:
                        raw_data = json.loads(f.read().decode("utf-8"))
                    break
        if raw_data is None:
            raise ValueError("No conversations.json found in ZIP archive")
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        raise ValueError(f"Expected .json or .zip, got {path.suffix}")

    # Handle both list and dict formats
    if isinstance(raw_data, dict):
        if "conversations" in raw_data:
            conversations = raw_data["conversations"]
        else:
            conversations = [raw_data]
    elif isinstance(raw_data, list):
        conversations = raw_data
    else:
        return []

    results = []
    for conv in conversations:
        try:
            conv_results = _parse_single_conversation(conv)
            results.extend(conv_results)
        except Exception as e:
            title = conv.get("title", "?")
            print(f"[ChatGPT Parser] Skipping conversation '{title}': {e}")

    return results


def _parse_single_conversation(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a single ChatGPT conversation."""
    results = []

    conv_title = conv.get("title", "Untitled")
    conv_id = conv.get("id", conv.get("conversation_id", ""))
    conv_created = _ts_to_dt(conv.get("create_time"))
    model_slug = conv.get("default_model_slug", "unknown")

    mapping = conv.get("mapping", {})
    if not mapping:
        # Flat message list fallback (some export formats)
        messages = conv.get("messages", [])
        return _parse_flat_messages(messages, conv_id, conv_title, conv_created)

    # Traverse the tree following main branch (first child path)
    ordered_messages = _traverse_main_branch(mapping)

    prev_id = None
    turn_number = 0

    for node in ordered_messages:
        msg = node.get("message")
        if not msg:
            continue

        author = msg.get("author", {})
        role = author.get("role", "unknown")

        # Skip system and tool messages
        if role not in ("user", "assistant"):
            continue

        text = _extract_text_from_message(msg)
        if not text or len(text.strip()) < 20:
            continue

        msg_time = _ts_to_dt(msg.get("create_time")) or conv_created

        turn_number += 1
        node_id = node.get("id", msg.get("id", ""))

        candidate = {
            "content": text[:2000],
            "source": "chatgpt",
            "conversation_id": conv_id,
            "conversation_name": conv_title,
            "role": role,
            "turn_number": turn_number,
            "created_at": msg_time,
            "message_id": node_id,
            "prev_message_id": prev_id,
            "model": model_slug,
            "level_hint": _infer_level(text, role),
        }

        prev_id = node_id
        results.append(candidate)

    return results


def _traverse_main_branch(mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Traverse the ChatGPT conversation tree following the main branch.
    ChatGPT stores messages as a tree (supporting edits/regenerations).
    We follow the first child at each node to get the canonical conversation.
    """
    # Find root: node whose parent is None or not in mapping
    root_id = None
    for node_id, node_data in mapping.items():
        parent = node_data.get("parent")
        if parent is None or parent not in mapping:
            root_id = node_id
            break

    if root_id is None:
        return []

    # Follow first-child chain from root
    ordered = []
    current_id = root_id
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = mapping.get(current_id)
        if not node:
            break

        if node.get("message"):
            ordered.append(node)

        children = node.get("children", [])
        if children:
            # Follow LAST child (ChatGPT puts the most recent edit last)
            current_id = children[-1]
        else:
            break

    return ordered


def _extract_text_from_message(msg: Dict[str, Any]) -> Optional[str]:
    """Extract text from ChatGPT message content structure."""
    content = msg.get("content", {})

    if isinstance(content, str):
        return content

    content_type = content.get("content_type", "text")

    if content_type == "text":
        parts = content.get("parts", [])
        text_parts = []
        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                # Might be an image or other media — skip
                continue
        return "\n".join(text_parts)

    elif content_type == "code":
        text = content.get("text", "")
        return f"[Code]\n{text}" if text else None

    elif content_type == "execution_output":
        text = content.get("text", "")
        return f"[Output]\n{text}" if text else None

    elif content_type == "multimodal_text":
        parts = content.get("parts", [])
        text_parts = []
        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("content_type") == "text":
                text_parts.append(part.get("text", ""))
        return "\n".join(text_parts)

    return None


def _parse_flat_messages(
    messages: List[Dict[str, Any]],
    conv_id: str,
    conv_title: str,
    conv_created: Optional[datetime],
) -> List[Dict[str, Any]]:
    """Fallback parser for flat message lists (non-tree format)."""
    results = []
    prev_id = None

    for i, msg in enumerate(messages):
        role = msg.get("role", msg.get("author", {}).get("role", "unknown"))
        if role not in ("user", "assistant"):
            continue

        text = _extract_text_from_message(msg) or msg.get("text", "")
        if not text or len(text.strip()) < 20:
            continue

        msg_time = _ts_to_dt(msg.get("create_time") or msg.get("timestamp")) or conv_created
        msg_id = msg.get("id", f"msg_{i}")

        results.append({
            "content": text[:2000],
            "source": "chatgpt",
            "conversation_id": conv_id,
            "conversation_name": conv_title,
            "role": role,
            "turn_number": i + 1,
            "created_at": msg_time,
            "message_id": msg_id,
            "prev_message_id": prev_id,
            "level_hint": _infer_level(text, role),
        })
        prev_id = msg_id

    return results


def _ts_to_dt(ts) -> Optional[datetime]:
    """Convert timestamp (unix float or ISO string) to datetime."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts)
        except (ValueError, OSError):
            return None
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", ""))
        except ValueError:
            return None
    return None


def _infer_level(text: str, role: str) -> str:
    """Heuristic level assignment."""
    if role == "user":
        if len(text) > 500:
            return "CTX"  # Long user prompts often set context
        return "ANC"
    # Assistant
    if len(text) > 1500 and any(m in text for m in ["##", "**", "1.", "Step "]):
        return "CTX"
    return "ANC"
