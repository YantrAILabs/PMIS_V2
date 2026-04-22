"""
ProMe API Key Authentication

Simple API key management for platform authentication.
Keys stored in data/.api_keys.json (gitignored).
"""

import json
import secrets
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"
KEYS_FILE = DATA_DIR / ".api_keys.json"


def _load_keys() -> dict:
    if KEYS_FILE.exists():
        with open(KEYS_FILE) as f:
            return json.load(f)
    return {"keys": {}}


def _save_keys(data: dict):
    DATA_DIR.mkdir(exist_ok=True)
    with open(KEYS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(platform: str) -> str:
    """Generate a new API key for a platform. Returns the raw key (only shown once)."""
    raw_key = f"pmis-{platform}-{secrets.token_hex(16)}"
    hashed = _hash_key(raw_key)

    data = _load_keys()
    data["keys"][hashed] = {
        "platform": platform,
        "created_at": datetime.now().isoformat(),
        "last_used": None,
        "request_count": 0,
    }
    _save_keys(data)
    return raw_key


def validate_key(raw_key: str) -> Optional[str]:
    """Validate an API key. Returns platform name if valid, None if invalid."""
    if not raw_key:
        return None

    hashed = _hash_key(raw_key)
    data = _load_keys()
    entry = data["keys"].get(hashed)

    if not entry:
        return None

    # Update usage stats
    entry["last_used"] = datetime.now().isoformat()
    entry["request_count"] = entry.get("request_count", 0) + 1
    _save_keys(data)

    return entry["platform"]


def list_keys() -> list:
    """List all registered API keys (without exposing the actual keys)."""
    data = _load_keys()
    result = []
    for hashed, entry in data["keys"].items():
        result.append({
            "hash_prefix": hashed[:12] + "...",
            "platform": entry["platform"],
            "created_at": entry["created_at"],
            "last_used": entry.get("last_used"),
            "request_count": entry.get("request_count", 0),
        })
    return result


def revoke_key(platform: str) -> int:
    """Revoke all keys for a platform. Returns count of revoked keys."""
    data = _load_keys()
    to_remove = [h for h, e in data["keys"].items() if e["platform"] == platform]
    for h in to_remove:
        del data["keys"][h]
    _save_keys(data)
    return len(to_remove)


def get_platform_for_key(raw_key: str) -> Optional[str]:
    """Get the platform associated with a key without updating usage stats."""
    if not raw_key:
        return None
    hashed = _hash_key(raw_key)
    data = _load_keys()
    entry = data["keys"].get(hashed)
    return entry["platform"] if entry else None


# CLI interface
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 auth.py create <platform> | list | revoke <platform>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "create" and len(sys.argv) >= 3:
        platform = sys.argv[2]
        key = create_key(platform)
        print(f"API Key for {platform}: {key}")
        print("Save this key — it won't be shown again.")
    elif cmd == "list":
        keys = list_keys()
        if not keys:
            print("No API keys registered.")
        for k in keys:
            print(f"  {k['platform']:20s} created={k['created_at'][:10]}  used={k.get('request_count', 0)} times  last={k.get('last_used', 'never')}")
    elif cmd == "revoke" and len(sys.argv) >= 3:
        n = revoke_key(sys.argv[2])
        print(f"Revoked {n} key(s) for {sys.argv[2]}")
    else:
        print("Unknown command. Use: create <platform> | list | revoke <platform>")
