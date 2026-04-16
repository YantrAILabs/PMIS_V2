"""
Daily Activity Merge — Consolidation Pass 6.

Reads today's raw activity segments from the productivity tracker,
clusters them by similarity, extracts knowledge via LLM, and attaches
the resulting anchors to the correct branch of the knowledge tree.

No pre-labeling — SC/CTX assignment happens HERE via semantic search
against the TRAINED knowledge tree (after Passes 1-5 have cleaned it).

Usage:
    # As consolidation pass (called from nightly.py):
    merger = DailyActivityMerger(db, hp)
    results = merger.run(date="2026-04-14")

    # Standalone:
    python3 pmis_v2/daily_activity_merge.py              # today
    python3 pmis_v2/daily_activity_merge.py 2026-04-14   # specific date
"""

import sqlite3
import os
import sys
import json
import hashlib
import logging
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

PMIS_DIR = Path(__file__).parent
sys.path.insert(0, str(PMIS_DIR))

logger = logging.getLogger("pmis.activity_merge")

TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")


class DailyActivityMerger:
    """Merges daily activity into knowledge tree as Pass 6 of consolidation."""

    def __init__(self, db, hyperparams: Dict):
        self.db = db
        self.hp = hyperparams
        self.actions: List[Dict] = []

    def run(self, target_date: str = None) -> List[Dict]:
        """Run the full merge for a given date."""
        if not target_date:
            target_date = date.today().isoformat()

        self.actions = []

        if not os.path.exists(TRACKER_DB):
            logger.warning(f"Tracker DB not found at {TRACKER_DB}")
            return self.actions

        # Step 6a: Read today's segments
        segments = self._read_segments(target_date)
        if not segments:
            logger.info(f"No segments for {target_date}")
            return self.actions

        logger.info(f"Read {len(segments)} segments for {target_date}")

        # Step 6b: Cluster by embedding similarity
        clusters = self._cluster_segments(segments)
        logger.info(f"Formed {len(clusters)} clusters")

        # Step 6c + 6d + 6e: For each cluster, extract and attach
        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # Generate summary
            summary = self._extract_pattern(cluster)
            if not summary:
                continue

            # Semantic match to knowledge tree
            matched_ctx_id, matched_sc_id = self._semantic_match(summary)
            if not matched_ctx_id:
                logger.info(f"No match for cluster: {summary[:50]}")
                continue

            # Create ANC node
            node_id = self._create_anchor(summary, matched_ctx_id, matched_sc_id, target_date)

            # Record time data (Step 6f)
            total_duration = sum(s.get("duration_secs", 10) for s in cluster)
            for seg in cluster:
                self._log_activity_time(
                    segment_id=seg.get("id", ""),
                    memory_node_id=node_id,
                    matched_ctx_id=matched_ctx_id,
                    matched_sc_id=matched_sc_id,
                    duration=seg.get("duration_secs", 10),
                    target_date=target_date,
                )

            self.actions.append({
                "action": "activity_merge",
                "date": target_date,
                "cluster_size": len(cluster),
                "summary": summary[:200],
                "matched_ctx_id": matched_ctx_id,
                "matched_sc_id": matched_sc_id,
                "node_id": node_id,
                "total_duration_secs": total_duration,
            })

        return self.actions

    def _read_segments(self, target_date: str) -> List[Dict]:
        """Read today's context_1 segments from tracker DB."""
        conn = sqlite3.connect(TRACKER_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, detailed_summary, window_name, platform,
                   target_segment_length_secs, worker,
                   human_frame_count, ai_frame_count
            FROM context_1
            WHERE DATE(timestamp_start) = ?
        """, (target_date,)).fetchall()
        conn.close()

        return [{
            "id": r["id"],
            "summary": r["detailed_summary"] or "",
            "window": r["window_name"] or "",
            "platform": r["platform"] or "",
            "duration_secs": r["target_segment_length_secs"] or 10,
            "worker": r["worker"] or "human",
        } for r in rows if r["detailed_summary"]]

    def _cluster_segments(self, segments: List[Dict]) -> List[List[Dict]]:
        """Cluster segments by summary text similarity."""
        if not segments:
            return []

        # Embed all summaries
        from ingestion.embedder import Embedder
        embedder = Embedder(hyperparams=self.hp)

        embeddings = []
        valid_segments = []
        for seg in segments:
            try:
                emb = embedder.embed_text(seg["summary"])
                if emb is not None:
                    embeddings.append(emb)
                    valid_segments.append(seg)
            except Exception:
                continue

        if len(embeddings) < 2:
            return [valid_segments] if valid_segments else []

        emb_array = np.array(embeddings)

        # Simple greedy clustering (same as consolidation/nightly.py)
        threshold = self.hp.get("birth_cluster_threshold", 0.35)
        assigned = set()
        clusters = []

        for i in range(len(valid_segments)):
            if i in assigned:
                continue
            cluster = [valid_segments[i]]
            assigned.add(i)

            for j in range(i + 1, len(valid_segments)):
                if j in assigned:
                    continue
                # Cosine distance
                dot = np.dot(emb_array[i], emb_array[j])
                norm_i = np.linalg.norm(emb_array[i])
                norm_j = np.linalg.norm(emb_array[j])
                dist = 1.0 - (dot / (norm_i * norm_j + 1e-8))
                if dist < threshold:
                    cluster.append(valid_segments[j])
                    assigned.add(j)

            clusters.append(cluster)

        return clusters

    def _extract_pattern(self, cluster: List[Dict]) -> Optional[str]:
        """LLM extracts a knowledge summary from cluster of activity segments."""
        total_duration = sum(s.get("duration_secs", 10) for s in cluster)
        duration_mins = total_duration / 60.0

        # Build description for LLM
        descriptions = [s["summary"][:150] for s in cluster[:10]]
        windows = list(set(s["window"] for s in cluster if s.get("window")))[:5]

        prompt = (
            f"These {len(cluster)} activity segments total {duration_mins:.1f} minutes.\n"
            f"Applications used: {', '.join(windows[:3])}\n\n"
            f"Activity descriptions:\n"
            + "\n".join(f"- {d}" for d in descriptions)
            + "\n\nWrite ONE concise knowledge anchor (1-2 sentences) "
            "summarizing what was accomplished. Focus on the OUTCOME or LEARNING, "
            "not the activity itself. No reference codes."
        )

        # Try local LLM first
        try:
            import httpx
            model = self.hp.get("consolidation_model_local", "qwen2.5:14b")
            resp = httpx.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()[:500]
        except Exception:
            pass

        # Fallback: simple concatenation
        top3 = [s["summary"][:80] for s in cluster[:3]]
        return f"Activity pattern ({len(cluster)} segments, {duration_mins:.0f} min): {'. '.join(top3)}"

    def _semantic_match(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Find the best matching CTX in the knowledge tree via semantic search."""
        from ingestion.embedder import Embedder
        embedder = Embedder(hyperparams=self.hp)

        try:
            query_emb = embedder.embed_text(text)
        except Exception:
            return None, None

        if query_emb is None:
            return None, None

        # Search against all CTX nodes
        ctxs = self.db.get_nodes_by_level("CTX")
        best_ctx_id = None
        best_sc_id = None
        best_score = -1

        for ctx in ctxs:
            embs = self.db.get_embeddings(ctx["id"])
            ctx_emb = embs.get("euclidean")
            if ctx_emb is None:
                continue

            # Cosine similarity
            dot = np.dot(query_emb, ctx_emb)
            norm_q = np.linalg.norm(query_emb)
            norm_c = np.linalg.norm(ctx_emb)
            sim = dot / (norm_q * norm_c + 1e-8)

            if sim > best_score:
                best_score = sim
                best_ctx_id = ctx["id"]

        if best_ctx_id and best_score > 0.3:
            # Find parent SC
            conn = sqlite3.connect(self.db.db_path)
            row = conn.execute("""
                SELECT target_id FROM relations
                WHERE source_id = ? AND relation_type = 'child_of'
                LIMIT 1
            """, (best_ctx_id,)).fetchone()
            conn.close()
            if row:
                best_sc_id = row[0]

            return best_ctx_id, best_sc_id

        return None, None

    def _create_anchor(self, content: str, ctx_id: str, sc_id: str,
                       target_date: str) -> str:
        """Create an ANC node under the matched CTX."""
        from core.memory_node import MemoryNode, MemoryLevel
        from core.temporal import temporal_encode, compute_era

        hyp_dim = self.hp.get("poincare_dimensions", 16)
        from ingestion.embedder import Embedder
        embedder = Embedder(hyperparams=self.hp)

        try:
            euclidean = embedder.embed_text(content)
        except Exception:
            euclidean = np.zeros(self.hp.get("local_embedding_dimensions", 768))

        temporal = temporal_encode(datetime.now(), self.hp.get("temporal_embedding_dim", 16))
        era = compute_era(datetime.now(), self.hp.get("era_boundaries", {}))

        node = MemoryNode.create(
            content=content,
            level=MemoryLevel.ANCHOR,
            euclidean_embedding=euclidean,
            hyperbolic_coords=np.zeros(hyp_dim, dtype=np.float32),
            temporal_embedding=temporal,
            source_conversation_id=f"activity_merge_{target_date}",
            surprise=0.0,
            precision=0.3,
            era=era,
        )
        node.is_orphan = False
        node.is_tentative = False
        self.db.create_node(node)

        # Attach to parent CTX
        tree_id = self._get_tree_id(ctx_id) or "default"
        self.db.attach_to_parent(node.id, ctx_id, tree_id)

        return node.id

    def _log_activity_time(self, segment_id, memory_node_id, matched_ctx_id,
                           matched_sc_id, duration, target_date):
        """Record segment → node mapping in activity_time_log."""
        conn = sqlite3.connect(self.db.db_path)
        conn.execute("""
            INSERT INTO activity_time_log
            (segment_id, memory_node_id, matched_ctx_id, matched_sc_id, duration_seconds, date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (segment_id, memory_node_id, matched_ctx_id, matched_sc_id, duration, target_date))
        conn.commit()
        conn.close()

    def _get_tree_id(self, node_id: str) -> Optional[str]:
        conn = sqlite3.connect(self.db.db_path)
        row = conn.execute("""
            SELECT tree_id FROM relations
            WHERE (source_id = ? OR target_id = ?) AND tree_id != 'default'
            LIMIT 1
        """, (node_id, node_id)).fetchone()
        conn.close()
        return row[0] if row else None


if __name__ == "__main__":
    from db.manager import DBManager
    from core import config

    os.chdir(str(PMIS_DIR))
    hp = config.get_all()
    db = DBManager("data/memory.db")

    target = sys.argv[1] if len(sys.argv) > 1 else None
    merger = DailyActivityMerger(db, hp)
    results = merger.run(target)

    print(f"Activity merge: {len(results)} clusters merged")
    for r in results:
        print(f"  {r['cluster_size']} segments → {r['summary'][:60]}")
        print(f"    matched to: ctx={r['matched_ctx_id'][:10]} sc={r['matched_sc_id'][:10] if r['matched_sc_id'] else 'none'}")
        print(f"    duration: {r['total_duration_secs']/60:.1f} min")
