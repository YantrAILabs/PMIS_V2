"""
Phase 6 — Training corpus export.

Every labeled event accumulated by PMIS (assignment picks, harness outcomes,
boilerplate hash clusters, automation-class tags) sits in the `training_events`
table with `exported_to_training=0`. This pass writes them to disk as JSONL,
partitioned by event_type and capture date, then flips the flag.

Output layout:

  <corpus_root>/
    ├── manifest.json                  (version info, last export)
    ├── assignment/
    │    └── 2026-04-18.jsonl
    ├── harness_outcome/
    │    └── 2026-04-18.jsonl
    ├── automation_class/
    └── boilerplate/

Each JSONL line = one event with fully denormalized features + label:

  {
    "id": "...",
    "event_type": "assignment",
    "captured_at": "2026-04-18T12:34:56Z",
    "pmis_version": "phase-6",
    "model_version": "",
    "features": {...},
    "label": {...},
    "deliverable_id": "D-001",
    "harness_id": "",
    "node_id": "",
    "segment_id": ""
  }

Idempotent — runs skip events whose exported_to_training=1. Also supports
`--backfill` to re-export everything (ignoring the flag) for model-upgrade
backtests per the locked design.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pmis.consolidation.training_export")


EVENT_TYPES = ("assignment", "harness_outcome", "automation_class", "boilerplate")


class TrainingCorpusExporter:
    def __init__(self, db, hyperparams: Optional[Dict[str, Any]] = None,
                 corpus_root: Optional[str] = None):
        self.db = db
        self.hp = hyperparams or {}
        default_root = str(Path(__file__).resolve().parents[1] / "data" / "training_corpus")
        self.corpus_root = Path(self.hp.get("training_corpus_root", corpus_root or default_root))
        self.corpus_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def export(self, backfill: bool = False) -> Dict[str, Any]:
        """Export unflagged training_events (or all if backfill=True) to
        JSONL partitions. Returns counts per event_type."""
        counts: Dict[str, int] = defaultdict(int)
        files_written: Dict[str, List[str]] = defaultdict(list)

        rows = self._fetch_events(include_exported=backfill)
        if not rows:
            return {
                "counts": dict(counts),
                "files": dict(files_written),
                "message": "no events to export",
            }

        # Group by (event_type, date)
        buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            etype = (r.get("event_type") or "").strip()
            if etype not in EVENT_TYPES:
                continue
            day = (r.get("captured_at") or "")[:10] or datetime.utcnow().date().isoformat()
            buckets[(etype, day)].append(r)

        ids_to_flag: List[str] = []
        for (etype, day), events in buckets.items():
            etype_dir = self.corpus_root / etype
            etype_dir.mkdir(parents=True, exist_ok=True)
            path = etype_dir / f"{day}.jsonl"

            # Append mode — existing export for same day doesn't clobber.
            # We dedupe by `id` when reading back, and the exported_to_training
            # flag prevents us from writing the same event twice in non-backfill
            # runs anyway.
            with open(path, "a", encoding="utf-8") as f:
                for e in events:
                    line = self._format_event(e)
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    ids_to_flag.append(e["id"])
            counts[etype] += len(events)
            rel = str(path.relative_to(self.corpus_root))
            if rel not in files_written[etype]:
                files_written[etype].append(rel)

        if not backfill and ids_to_flag:
            self._flag_exported(ids_to_flag)

        self._write_manifest(counts)

        return {
            "counts": dict(counts),
            "files": dict(files_written),
            "total_events": sum(counts.values()),
            "flagged": len(ids_to_flag) if not backfill else 0,
            "mode": "backfill" if backfill else "incremental",
        }

    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """What's on disk vs what's pending in DB."""
        # DB side
        db_counts = self.db.count_training_events()
        pending: Dict[str, int] = defaultdict(int)
        with self.db._connect() as conn:
            rows = conn.execute(
                """SELECT event_type, COUNT(*) AS n FROM training_events
                   WHERE exported_to_training = 0 GROUP BY event_type"""
            ).fetchall()
            for r in rows:
                pending[r["event_type"]] = r["n"]

        # Disk side
        disk: Dict[str, Dict[str, Any]] = {}
        for etype in EVENT_TYPES:
            etype_dir = self.corpus_root / etype
            if not etype_dir.exists():
                disk[etype] = {"files": 0, "lines": 0}
                continue
            files = sorted(etype_dir.glob("*.jsonl"))
            line_count = 0
            for f in files:
                with open(f, encoding="utf-8") as fh:
                    line_count += sum(1 for _ in fh)
            disk[etype] = {
                "files": len(files),
                "lines": line_count,
                "latest": files[-1].name if files else "",
            }

        return {
            "corpus_root": str(self.corpus_root),
            "db_totals": db_counts,
            "db_pending_export": dict(pending),
            "disk": disk,
        }

    # ------------------------------------------------------------------

    def _fetch_events(self, include_exported: bool) -> List[Dict[str, Any]]:
        with self.db._connect() as conn:
            if include_exported:
                rows = conn.execute(
                    "SELECT * FROM training_events ORDER BY captured_at ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM training_events
                       WHERE exported_to_training = 0
                       ORDER BY captured_at ASC"""
                ).fetchall()
        return [dict(r) for r in rows]

    def _format_event(self, e: Dict[str, Any]) -> Dict[str, Any]:
        try:
            features = json.loads(e.get("features") or "{}")
        except Exception:
            features = {}
        try:
            label = json.loads(e.get("label") or "{}")
        except Exception:
            label = {}
        return {
            "id": e["id"],
            "event_type": e["event_type"],
            "captured_at": e.get("captured_at"),
            "pmis_version": e.get("pmis_version") or "",
            "model_version": e.get("model_version") or "",
            "segment_id": e.get("segment_id") or "",
            "node_id": e.get("node_id") or "",
            "deliverable_id": e.get("deliverable_id") or "",
            "harness_id": e.get("harness_id") or "",
            "features": features,
            "label": label,
        }

    def _flag_exported(self, ids: List[str]) -> None:
        if not ids:
            return
        CHUNK = 500
        with self.db._connect() as conn:
            for i in range(0, len(ids), CHUNK):
                chunk = ids[i:i + CHUNK]
                ph = ",".join("?" * len(chunk))
                conn.execute(
                    f"UPDATE training_events SET exported_to_training = 1 "
                    f"WHERE id IN ({ph})",
                    chunk,
                )
            conn.commit()

    def _write_manifest(self, counts: Dict[str, int]) -> None:
        manifest_path = self.corpus_root / "manifest.json"
        existing: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        runs = existing.get("runs", [])
        runs.append({
            "at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "counts": dict(counts),
            "total": sum(counts.values()),
        })
        existing["runs"] = runs[-50:]
        existing["last_run_at"] = runs[-1]["at"]
        existing["corpus_root"] = str(self.corpus_root)
        manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
