"""Shared scoped lock for consolidation runs (Phase 1 sync protocol).

Nightly acquires scope='global' -> blocks everything.
Manual acquires scope='date:YYYY-MM-DD' -> blocks only same-date manual runs
and anything global. Different dates proceed in parallel.

Stale locks (mtime > LOCK_STALE_SECONDS) are reaped automatically, so a crashed
runner never deadlocks the system. Release is best-effort on context exit and
as an atexit fallback.

    from consolidation.lock import consolidation_lock, LockBusy

    try:
        with consolidation_lock('global'):
            run_nightly()
    except LockBusy as e:
        logger.info("skip: %s (retry in %ds)", e, e.retry_after_secs)

    try:
        with consolidation_lock(f'date:{target_date}'):
            run_manual(target_date)
    except LockBusy as e:
        raise HTTPException(503, detail=str(e), headers={'Retry-After': str(e.retry_after_secs)})
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
from typing import Dict, List, Optional

logger = logging.getLogger("pmis.lock")

PMIS_DIR = Path(__file__).resolve().parent.parent
LOCK_DIR = PMIS_DIR / "data"
LOCK_STALE_SECONDS = 30 * 60  # matches legacy runner.py


class LockBusy(RuntimeError):
    """Raised when a conflicting active lock is held by another process."""

    def __init__(self, holder: Dict, retry_after_secs: int):
        self.holder = holder
        self.retry_after_secs = retry_after_secs
        msg = (
            f"consolidation lock held by {holder.get('kind', '?')}/"
            f"{holder.get('scope', '?')} (pid={holder.get('pid', '?')}); "
            f"retry in ~{retry_after_secs}s"
        )
        super().__init__(msg)


def _scope_to_filename(scope: str) -> str:
    safe = scope.replace(":", ".").replace("/", "_")
    return f"consol.{safe}.lock"


def _lock_path(scope: str) -> Path:
    return LOCK_DIR / _scope_to_filename(scope)


def _is_stale(p: Path) -> bool:
    try:
        return (time.time() - p.stat().st_mtime) > LOCK_STALE_SECONDS
    except OSError:
        return True


def _read_holder(p: Path) -> Dict:
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


def _retry_after(p: Path) -> int:
    try:
        age = time.time() - p.stat().st_mtime
    except OSError:
        return 60
    return max(60, int(LOCK_STALE_SECONDS - age))


def _owned_by_us(p: Path) -> bool:
    """True if our PID authored the lock file (re-entry is allowed)."""
    return _read_holder(p).get("pid") == os.getpid()


def _iter_active_date_locks() -> List[Path]:
    """Live date-scope lock files we do NOT own. Reaps stale entries."""
    live: List[Path] = []
    for p in LOCK_DIR.glob("consol.date.*.lock"):
        if _is_stale(p):
            _safe_unlink(p)
        elif not _owned_by_us(p):
            live.append(p)
    return live


@contextmanager
def consolidation_lock(scope: str, kind: Optional[str] = None):
    """Acquire a scope-specific consolidation lock.

    scope: 'global' OR 'date:YYYY-MM-DD'
    kind:  optional author tag; defaults to 'nightly' for global, 'manual' otherwise.

    Raises LockBusy if a conflicting active lock exists.
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    kind = kind or ("nightly" if scope == "global" else "manual")

    global_path = _lock_path("global")
    our_path = _lock_path(scope)

    conflicts: List[Path] = []
    # Global nightly reaps and conflicts with EVERY live lock we don't own.
    # Date-scoped manual only conflicts with global and same-scope siblings.
    if global_path.exists():
        if _is_stale(global_path):
            _safe_unlink(global_path)
        elif not _owned_by_us(global_path):
            conflicts.append(global_path)

    if scope == "global":
        conflicts.extend(_iter_active_date_locks())
    elif our_path.exists():
        if _is_stale(our_path):
            _safe_unlink(our_path)
        elif not _owned_by_us(our_path):
            conflicts.append(our_path)

    if conflicts:
        raise LockBusy(_read_holder(conflicts[0]), _retry_after(conflicts[0]))

    payload = {
        "kind": kind,
        "scope": scope,
        "pid": os.getpid(),
        "acquired_at": datetime.now().isoformat(timespec="seconds"),
    }
    our_path.write_text(json.dumps(payload))

    released = [False]

    def _release() -> None:
        if released[0]:
            return
        released[0] = True
        try:
            if our_path.exists():
                holder = _read_holder(our_path)
                if holder.get("pid") == os.getpid():
                    our_path.unlink()
        except Exception:
            pass

    atexit.register(_release)
    try:
        yield payload
    finally:
        _release()
