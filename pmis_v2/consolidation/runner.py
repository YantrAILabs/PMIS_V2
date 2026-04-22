"""
Idempotent nightly consolidation runner (Phase 5, 2026-04-20).

One function, three entry points:
  - server.py startup hook       (non-blocking background thread)
  - launchd agent at 18:00 local (backstop if server runs all day)
  - CLI: `python3 pmis_v2/cli.py consolidate` (manual)

Design:
  - `last_consolidation_date` is stored in `system_meta`. On every invocation
    we compute `[last_date+1 ... today-1]` and consolidate each missed day
    before optionally consolidating today (only if after 18:00 local OR
    explicitly forced).
  - A PID-file lock at `pmis_v2/data/nightly.lock` prevents two runners from
    colliding. The lock is removed on exit (atexit) and is also age-checked
    on acquire — a stale lock (>30 min old) is considered abandoned and
    overwritten, so a killed server doesn't deadlock forever.
  - Every catch-up day gets a structured log entry in consolidation_log
    via NightlyConsolidation.run() as normal.
"""

from __future__ import annotations

import os
import sys
import time
import atexit
import logging
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger("pmis.runner")


PMIS_DIR = Path(__file__).resolve().parent.parent
LOCK_PATH = PMIS_DIR / "data" / "nightly.lock"
LOCK_STALE_SECONDS = 30 * 60  # 30 min — longer than any sane run


# ---------------------------------------------------------------------
# Lock file
# ---------------------------------------------------------------------


def _acquire_lock() -> bool:
    """Try to acquire the runner lock. Returns False if another runner is
    active. A lock older than LOCK_STALE_SECONDS is overridden."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            age = time.time() - LOCK_PATH.stat().st_mtime
        except OSError:
            age = 0
        if age < LOCK_STALE_SECONDS:
            try:
                existing = LOCK_PATH.read_text().strip()
            except OSError:
                existing = "?"
            logger.info("Runner lock held by pid=%s (age %.0fs); skipping.", existing, age)
            return False
        logger.warning("Stale runner lock (age %.0fs); overriding.", age)
    LOCK_PATH.write_text(str(os.getpid()))
    atexit.register(_release_lock)
    return True


def _release_lock() -> None:
    try:
        if LOCK_PATH.exists() and LOCK_PATH.read_text().strip() == str(os.getpid()):
            LOCK_PATH.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------
# Date bookkeeping (system_meta)
# ---------------------------------------------------------------------


def _ensure_system_meta(db_path: str) -> None:
    """Schema-safety — system_meta is created elsewhere (without updated_at)
    but we want the column for telemetry. Create if missing, add column if
    the table exists but predates it."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS system_meta (
               key TEXT PRIMARY KEY, value TEXT,
               updated_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    # If the table existed without updated_at, add it.
    try:
        conn.execute("SELECT updated_at FROM system_meta LIMIT 1")
    except sqlite3.OperationalError:
        try:
            conn.execute(
                "ALTER TABLE system_meta ADD COLUMN updated_at TEXT DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def _get_last_date(db_path: str) -> Optional[date]:
    _ensure_system_meta(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT value FROM system_meta WHERE key = 'last_consolidation_date'"
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    try:
        return date.fromisoformat(row[0])
    except ValueError:
        return None


def _set_last_date(db_path: str, d: date) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO system_meta (key, value, updated_at)
           VALUES ('last_consolidation_date', ?, datetime('now'))
           ON CONFLICT(key) DO UPDATE SET
             value = excluded.value, updated_at = datetime('now')""",
        (d.isoformat(),),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------


def run_idempotent(
    db_path: Optional[str] = None,
    include_today: Optional[bool] = None,
    evening_cutoff_hour: int = 18,
) -> Dict[str, Any]:
    """
    Consolidate every day from (last_consolidation_date + 1) to yesterday,
    then optionally consolidate today if we've crossed `evening_cutoff_hour`
    (default 18:00 local) or include_today is forced True.

    Returns a summary dict of what ran.
    """
    from core import config as _cfg
    from db.manager import DBManager
    from consolidation.nightly import NightlyConsolidation

    if db_path is None:
        # Anchor to PMIS_DIR so running from any cwd (launchd, systemd, cron,
        # arbitrary shell) always hits the real DB. Previously the relative
        # default `data/memory.db` silently created an empty DB next to cwd.
        db_path = _cfg.get("db_path", "data/memory.db")
        if not os.path.isabs(db_path):
            db_path = str(PMIS_DIR / db_path)

    if not _acquire_lock():
        return {"ok": False, "reason": "another_runner_active"}

    summary: Dict[str, Any] = {
        "ok": True,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "days_run": [],
        "skipped_reason": None,
    }

    try:
        hp = _cfg.get_all()
        db = DBManager(db_path)

        today = date.today()
        last = _get_last_date(db_path)
        if last is None:
            # First ever run — only do today (or only past day if before
            # evening cutoff). Don't try to consolidate the entire history.
            last = today - timedelta(days=1)

        # Catch up missed FULL days: (last+1) .. (today-1)
        days: List[date] = []
        cursor = last + timedelta(days=1)
        while cursor < today:
            days.append(cursor)
            cursor += timedelta(days=1)

        # Should today also run?
        now = datetime.now()
        should_include_today = (
            include_today is True
            or (include_today is None and now.hour >= evening_cutoff_hour and last < today)
        )
        if should_include_today:
            days.append(today)

        if not days:
            summary["skipped_reason"] = "nothing_to_do"
            return summary

        for d in days:
            logger.info("Runner: consolidating %s", d.isoformat())
            # NightlyConsolidation is date-agnostic — its sub-passes use
            # today() internally. For catch-up days we set an environment
            # flag so date-aware passes (activity_merge, project_matcher)
            # use the target date instead of today.
            os.environ["PMIS_CONSOLIDATION_DATE"] = d.isoformat()
            try:
                engine = NightlyConsolidation(db, hp)
                results = engine.run()
            finally:
                os.environ.pop("PMIS_CONSOLIDATION_DATE", None)

            summary["days_run"].append({
                "date": d.isoformat(),
                "total_actions": sum(len(v) for v in results.values() if isinstance(v, list)),
            })
            _set_last_date(db_path, d)

        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        return summary

    except Exception as e:
        logger.exception("Runner failed: %s", e)
        summary["ok"] = False
        summary["error"] = str(e)
        return summary


def run_startup_catchup() -> None:
    """Entry point for server.py — run in a background thread so login
    isn't blocked. Silently no-ops if already up-to-date or locked."""
    try:
        result = run_idempotent()
        logger.info("Startup catchup result: %s", result)
    except Exception as e:
        logger.exception("Startup catchup crashed: %s", e)


def run_daily_scheduler(evening_cutoff_hour: int = 18) -> None:
    """In-process daily scheduler — replaces the launchd plist because macOS
    TCC Full Disk Access restrictions make it painful to run launchd agents
    that touch files under ~/Desktop.

    Sleeps until the next `evening_cutoff_hour` (default 18:00 local), fires
    the idempotent runner, then loops. Also fires catchup every 30 min as a
    defensive backstop for long-running servers that might miss the exact
    18:00 edge (sleep deferral, clock skew, etc.).

    Intended to run in a daemon thread from server.py lifespan.
    """
    import time
    logger.info("Daily scheduler started (cutoff hour %d).", evening_cutoff_hour)
    last_fired_date: Optional[date] = None

    while True:
        try:
            now = datetime.now()
            # Fire if past cutoff AND not yet fired today
            if now.hour >= evening_cutoff_hour and last_fired_date != now.date():
                logger.info("Scheduler firing nightly at %s", now.isoformat(timespec="seconds"))
                result = run_idempotent()
                logger.info("Scheduler result: %s", result)
                last_fired_date = now.date()
        except Exception as e:
            logger.exception("Scheduler tick failed: %s", e)

        # Sleep 30 min between checks — cheap, and bounds any clock drift
        # or laptop-wake latency to within a half hour.
        time.sleep(1800)


# ---------------------------------------------------------------------
# CLI entry (launchd and manual)
# ---------------------------------------------------------------------


if __name__ == "__main__":
    # Ensure we can import pmis_v2 modules
    sys.path.insert(0, str(PMIS_DIR))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    force_today = "--today" in sys.argv
    result = run_idempotent(include_today=True if force_today else None)
    print(result)
