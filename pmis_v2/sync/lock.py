"""Sync scope lock — prevents two 30-min sync runs from racing.

Intentionally separate from `consolidation.lock`: sync and Dream operate on
disjoint data (sync writes work_pages, Dream reads confirmed ones into the
tree), so they don't block each other. Only sync-vs-sync needs mutual
exclusion.

Stale locks (mtime > LOCK_STALE_SECONDS) are reaped on next acquire so a
crashed run never deadlocks the next scheduler tick.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("pmis.sync.lock")

PMIS_DIR = Path(__file__).resolve().parent.parent
LOCK_DIR = PMIS_DIR / "data"
LOCK_STALE_SECONDS = 10 * 60  # sync should finish in a few minutes
LOCK_FILE = LOCK_DIR / "sync.lock"


class SyncBusy(RuntimeError):
    def __init__(self, holder: dict):
        self.holder = holder
        pid = holder.get("pid", "?")
        super().__init__(f"sync lock held by pid={pid}")


def _is_stale(p: Path) -> bool:
    try:
        return (time.time() - p.stat().st_mtime) > LOCK_STALE_SECONDS
    except OSError:
        return True


def _read_holder(p: Path) -> dict:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@contextmanager
def sync_lock(user_id: str = "local"):
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    if LOCK_FILE.exists():
        if _is_stale(LOCK_FILE):
            try:
                LOCK_FILE.unlink()
            except OSError:
                pass
        else:
            holder = _read_holder(LOCK_FILE)
            if holder.get("pid") != os.getpid():
                raise SyncBusy(holder)

    payload = {
        "pid": os.getpid(),
        "user_id": user_id,
        "acquired_at": datetime.now().isoformat(timespec="seconds"),
    }
    LOCK_FILE.write_text(json.dumps(payload), encoding="utf-8")

    released = [False]

    def _release() -> None:
        if released[0]:
            return
        released[0] = True
        try:
            if LOCK_FILE.exists():
                holder = _read_holder(LOCK_FILE)
                if holder.get("pid") == os.getpid():
                    LOCK_FILE.unlink()
        except Exception:
            pass

    atexit.register(_release)
    try:
        yield payload
    finally:
        _release()
