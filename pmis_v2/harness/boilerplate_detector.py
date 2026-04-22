"""
Step 2 — Boilerplate detector.

Two passes:

  1. extract(days=N)
     Walks recent tracker segments, pulls low-cost artifact signals (file
     paths from IDE window titles, commands from Terminal titles, URLs from
     browser titles) and upserts them into segment_artifacts. Heuristic
     source — LLM extraction can overwrite rows later.

  2. mine(min_reps=N)
     Clusters segment_artifacts by (content_hash, artifact_type). Rows with
     ≥ min_reps instances become boilerplate candidates — the signal we've
     been designing since the Asana-gap analysis. For each candidate, logs
     a `training_events` row of type='boilerplate' so the Phase 6 corpus
     accumulates supervision for a future classifier ("this snippet/command
     is boilerplate, regardless of the segment context").

No LLM, no embeddings. Pure pattern extraction — fast enough to run as a
nightly pass or on-demand via API.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pmis.harness.boilerplate")

TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")

# Heuristic extractors — cheap regex on window titles
_IDE_PLATFORMS = {"Visual Studio Code", "VS Code", "Code", "Cursor", "Sublime Text",
                   "PyCharm", "IntelliJ IDEA", "Xcode", "Vim", "Neovim", "Emacs"}
_FILE_IN_TITLE = re.compile(r"([\w\-\.\/]+\.(?:py|js|ts|tsx|jsx|go|rs|java|cpp|c|h|rb|md|yaml|yml|json|sh|sql|html|css))")
_TERMINAL_CMD = re.compile(r"([a-zA-Z_][\w\-]*(?:\s+[\w\-\./=@:]+){0,6})")
_URL_FROM_TITLE = re.compile(r"https?://[\w\-\./?=&%#]+")


class BoilerplateDetector:
    def __init__(self, db, hyperparams: Optional[Dict[str, Any]] = None,
                 tracker_db_path: str = TRACKER_DB):
        self.db = db
        self.hp = hyperparams or {}
        self.tracker_db_path = tracker_db_path
        self.lookback_days = int(self.hp.get("boilerplate_lookback_days", 14))
        self.min_reps = int(self.hp.get("boilerplate_min_reps", 3))
        self.min_preview_len = int(self.hp.get("boilerplate_min_preview_len", 6))

    # ------------------------------------------------------------------

    def extract(self, days: Optional[int] = None) -> Dict[str, int]:
        """Walk recent segments, write heuristic artifact rows. Idempotent via
        (segment_id, artifact_type, content_hash) dedup — upsert."""
        if not Path(self.tracker_db_path).exists():
            return {"files": 0, "commands": 0, "urls": 0, "segments_scanned": 0}

        days = days if days is not None else self.lookback_days
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.tracker_db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """SELECT target_segment_id, window_name, platform, detailed_summary
                   FROM context_1
                   WHERE timestamp_start >= ? AND target_segment_id != ''""",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()

        counts = {"files": 0, "commands": 0, "urls": 0,
                  "decisions": 0, "segments_scanned": len(rows)}
        if not rows:
            return counts

        for r in rows:
            sid = r["target_segment_id"]
            platform = (r["platform"] or "").strip()
            title = r["window_name"] or ""
            summary = r["detailed_summary"] or ""

            # Files from IDE titles
            if platform in _IDE_PLATFORMS or any(p.lower() in platform.lower()
                                                  for p in _IDE_PLATFORMS):
                for match in _FILE_IN_TITLE.findall(title):
                    if len(match) < self.min_preview_len:
                        continue
                    chash = _hash(match)
                    self.db.upsert_segment_artifact({
                        "id": f"{sid}:file:{chash[:8]}",
                        "segment_id": sid,
                        "artifact_type": "file",
                        "path_or_uri": match,
                        "content_hash": chash,
                        "preview": match,
                    })
                    counts["files"] += 1

            # Commands from Terminal titles (title usually contains the most
            # recent command as prefix/suffix)
            if platform == "Terminal" or "terminal" in platform.lower() or "iterm" in platform.lower():
                for match in _TERMINAL_CMD.findall(title):
                    cmd = match.strip()
                    if len(cmd) < self.min_preview_len or " " not in cmd:
                        continue  # skip bare single-word titles (prompt names, etc)
                    chash = _hash(cmd)
                    self.db.upsert_segment_artifact({
                        "id": f"{sid}:cmd:{chash[:8]}",
                        "segment_id": sid,
                        "artifact_type": "command",
                        "preview": cmd[:200],
                        "content_hash": chash,
                    })
                    counts["commands"] += 1
                    break  # one command per segment — titles often echo the last

            # URLs from browser titles
            for match in _URL_FROM_TITLE.findall(title + " " + summary):
                chash = _hash(match)
                self.db.upsert_segment_artifact({
                    "id": f"{sid}:url:{chash[:8]}",
                    "segment_id": sid,
                    "artifact_type": "url",
                    "path_or_uri": match,
                    "content_hash": chash,
                    "preview": match[:200],
                })
                counts["urls"] += 1

            # Decisions — heuristic on summary
            if summary and re.search(r"\b(decided|chose|rejected|agreed|resolved)\b",
                                       summary, re.IGNORECASE):
                snippet = summary[:240]
                chash = _hash(snippet)
                self.db.upsert_segment_artifact({
                    "id": f"{sid}:dec:{chash[:8]}",
                    "segment_id": sid,
                    "artifact_type": "decision",
                    "preview": snippet,
                    "content_hash": chash,
                })
                counts["decisions"] += 1

        return counts

    # ------------------------------------------------------------------

    def mine(self, min_reps: Optional[int] = None) -> Dict[str, Any]:
        """Cluster artifacts by content_hash; log boilerplate training_events
        for every cluster that crosses min_reps."""
        min_reps = min_reps if min_reps is not None else self.min_reps
        clusters = self.db.list_artifact_clusters(min_repetitions=min_reps)

        events_written = 0
        for c in clusters:
            # Don't re-log a boilerplate event for the same content_hash within
            # the same day — idempotent check
            existing = self._already_logged_today(c["content_hash"])
            if existing:
                continue
            self.db.log_training_event({
                "event_type": "boilerplate",
                "features": {
                    "artifact_type": c["artifact_type"],
                    "content_hash": c["content_hash"],
                    "reps": c["reps"],
                    "distinct_segments": c["distinct_segments"],
                    "preview": (c.get("sample_preview") or "")[:180],
                },
                "label": {
                    "is_boilerplate": 1,
                    "threshold_reps": min_reps,
                },
                "pmis_version": "phase-6",
            })
            events_written += 1

        return {
            "clusters": len(clusters),
            "events_written": events_written,
            "min_reps": min_reps,
        }

    # ------------------------------------------------------------------

    def _already_logged_today(self, content_hash: str) -> bool:
        today = datetime.utcnow().date().isoformat()
        with self.db._connect() as conn:
            row = conn.execute(
                """SELECT id FROM training_events
                   WHERE event_type = 'boilerplate'
                     AND DATE(captured_at) = ?
                     AND features LIKE ?
                   LIMIT 1""",
                (today, f'%"content_hash": "{content_hash}"%'),
            ).fetchone()
            return row is not None


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]
