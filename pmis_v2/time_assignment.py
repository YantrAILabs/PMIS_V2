"""
Time Assignment — Consolidation Pass 8.

After HGCN training (Pass 7), uses the trained tree to:
1. Aggregate time from activity_time_log per SC/CTX branch
2. Match projects/deliverables to tree branches via semantic search
3. Update node_time_log with daily rollups
4. Update productivity_time_mins on matched memory_nodes

Runs AFTER HGCN so semantic matching uses trained Poincaré positions.
"""

import sqlite3
import os
import logging
import numpy as np
from datetime import date
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger("pmis.time_assignment")

TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")


class TimeAssignment:
    """Assigns accumulated activity time to knowledge tree branches."""

    def __init__(self, db, hyperparams: Dict):
        self.db = db
        self.hp = hyperparams
        self.actions: List[Dict] = []

    def run(self, target_date: str = None) -> List[Dict]:
        """Run time assignment for a given date."""
        if not target_date:
            target_date = date.today().isoformat()

        self.actions = []

        # Step 8a: Aggregate time from activity_time_log
        aggregates = self._aggregate_time(target_date)
        if not aggregates:
            logger.info(f"No activity time data for {target_date}")
            return self.actions

        logger.info(f"Aggregated time for {len(aggregates)} branches on {target_date}")

        # Step 8b: Match projects to branches
        project_matches = self._match_projects()

        # Step 8c: Write node_time_log entries
        for node_id, data in aggregates.items():
            project_id = project_matches.get(node_id)

            conn = sqlite3.connect(self.db.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO node_time_log
                (node_id, date, total_duration_mins, segment_count, project_id)
                VALUES (?, ?, ?, ?, ?)
            """, (node_id, target_date, data["duration_mins"],
                  data["segment_count"], project_id))
            conn.commit()
            conn.close()

            # Step 8d: Update productivity columns on memory_nodes
            conn = sqlite3.connect(self.db.db_path)
            conn.execute("""
                UPDATE memory_nodes
                SET productivity_time_mins = COALESCE(productivity_time_mins, 0) + ?,
                    last_productivity_sync = datetime('now')
                WHERE id = ?
            """, (data["duration_mins"], node_id))
            conn.commit()
            conn.close()

            self.actions.append({
                "action": "time_assignment",
                "node_id": node_id,
                "date": target_date,
                "duration_mins": data["duration_mins"],
                "segment_count": data["segment_count"],
                "project_id": project_id,
            })

        return self.actions

    def _aggregate_time(self, target_date: str) -> Dict[str, Dict]:
        """Aggregate activity_time_log by SC and CTX for the target date."""
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row

        # Aggregate by matched_ctx_id
        rows = conn.execute("""
            SELECT matched_ctx_id, matched_sc_id,
                   SUM(duration_seconds) / 60.0 as duration_mins,
                   COUNT(*) as segment_count
            FROM activity_time_log
            WHERE date = ?
            GROUP BY matched_ctx_id
        """, (target_date,)).fetchall()

        aggregates = {}
        for r in rows:
            if r["matched_ctx_id"]:
                aggregates[r["matched_ctx_id"]] = {
                    "duration_mins": r["duration_mins"],
                    "segment_count": r["segment_count"],
                    "sc_id": r["matched_sc_id"],
                }
            # Also aggregate at SC level
            if r["matched_sc_id"]:
                if r["matched_sc_id"] not in aggregates:
                    aggregates[r["matched_sc_id"]] = {
                        "duration_mins": 0, "segment_count": 0, "sc_id": None,
                    }
                aggregates[r["matched_sc_id"]]["duration_mins"] += r["duration_mins"]
                aggregates[r["matched_sc_id"]]["segment_count"] += r["segment_count"]

        conn.close()
        return aggregates

    def _match_projects(self) -> Dict[str, str]:
        """Match deliverables from tracker to knowledge tree branches."""
        if not os.path.exists(TRACKER_DB):
            return {}

        conn = sqlite3.connect(TRACKER_DB)
        conn.row_factory = sqlite3.Row
        deliverables = conn.execute(
            "SELECT id, name, supercontext FROM deliverables WHERE status = 'active'"
        ).fetchall()
        conn.close()

        if not deliverables:
            return {}

        # Semantic match each deliverable to knowledge tree nodes
        matches = {}
        from ingestion.embedder import Embedder
        embedder = Embedder(hyperparams=self.hp)

        for d in deliverables:
            search_text = f"{d['name']} {d['supercontext'] or ''}"
            try:
                query_emb = embedder.embed_text(search_text)
            except Exception:
                continue

            if query_emb is None:
                continue

            # Find best matching SC
            scs = self.db.get_nodes_by_level("SC")
            best_id = None
            best_sim = -1

            for sc in scs:
                embs = self.db.get_embeddings(sc["id"])
                sc_emb = embs.get("euclidean")
                if sc_emb is None:
                    continue
                sim = np.dot(query_emb, sc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(sc_emb) + 1e-8
                )
                if sim > best_sim:
                    best_sim = sim
                    best_id = sc["id"]

            if best_id and best_sim > 0.3:
                matches[best_id] = d["id"]

        return matches
