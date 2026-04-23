"""30-min sync runner — entry point.

Reads fresh segments from tracker.db.context_1 (newer than the last
watermark), clusters them, and either appends to an open work_page or
creates a new one. Advances the watermark on success.

Does NOT run project matching or touch memory_nodes. Dream owns those.

    # one-shot standalone:
    python3 -m pmis_v2.sync.runner [YYYY-MM-DD]

    # via CLI:
    python3 pmis_v2/cli.py sync run
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PMIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PMIS_DIR))

from sync.lock import SyncBusy, sync_lock  # noqa: E402
from sync.page_builder import (  # noqa: E402
    cluster_segments,
    compute_cluster_centroid,
    find_matching_page,
    llm_generate_title_and_summary,
    llm_restitch_page,
)

logger = logging.getLogger("pmis.sync.runner")

TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")


def run_sync(
    db,
    hyperparams: Dict,
    user_id: str = "local",
    target_date: Optional[str] = None,
) -> Dict:
    """Run one sync pass. Returns a summary dict describing the actions."""
    target_date = target_date or date.today().isoformat()

    if not os.path.exists(TRACKER_DB):
        return {"status": "skipped", "reason": "no_tracker_db"}

    last_ts = db.get_last_sync_timestamp(user_id=user_id) or ""
    new_segments = _read_new_segments(target_date, last_ts)
    if not new_segments:
        return {
            "status": "ok",
            "date": target_date,
            "segments": 0,
            "pages_created": 0,
            "pages_appended": 0,
        }

    from ingestion.embedder import Embedder

    embedder = Embedder(hyperparams=hyperparams)
    cluster_threshold = hyperparams.get("sync_cluster_threshold", 0.20)
    append_distance = hyperparams.get("sync_append_distance", 0.12)
    clusters = cluster_segments(
        new_segments, embedder, threshold=cluster_threshold
    )

    open_pages = db.list_open_work_pages(target_date, user_id=user_id)
    sync_turn = db.get_next_sync_turn(target_date, user_id=user_id)
    model = hyperparams.get("consolidation_model_local", "qwen2.5:14b")

    pages_created = 0
    pages_appended = 0

    for cluster in clusters:
        centroid = compute_cluster_centroid(cluster, embedder)
        if centroid is None:
            continue

        match_id, match_dist = find_matching_page(centroid, open_pages)

        if match_id and match_dist < append_distance:
            page = db.get_work_page(match_id)
            new_title, new_summary = llm_restitch_page(
                page["title"], page["summary"], cluster, model=model
            )
            new_emb = _blend_embeddings(
                _read_page_embedding(page), centroid, page_weight=0.7
            )
            db.append_to_work_page(
                match_id, new_summary, new_emb, new_title=new_title
            )
            for seg in cluster:
                db.add_page_segment(match_id, str(seg["id"]), sync_turn)
            for p in open_pages:
                if p["id"] == match_id:
                    p["title"] = new_title
                    p["summary"] = new_summary
                    p["embedding"] = new_emb
                    break
            pages_appended += 1
            logger.info(
                "appended %d segs to page %s (dist=%.3f)",
                len(cluster), match_id, match_dist,
            )
        else:
            title, summary = llm_generate_title_and_summary(cluster, model=model)
            page_id = db.create_work_page(
                title=title,
                summary=summary,
                date_local=target_date,
                embedding=centroid,
                user_id=user_id,
            )
            for seg in cluster:
                db.add_page_segment(page_id, str(seg["id"]), sync_turn)
            open_pages.append(
                {
                    "id": page_id,
                    "title": title,
                    "summary": summary,
                    "embedding": centroid,
                }
            )
            pages_created += 1
            logger.info("new page %s from %d segs", page_id, len(cluster))

    latest_ts = max(
        (s.get("timestamp_start", "") for s in new_segments), default=""
    )
    if not latest_ts:
        latest_ts = datetime.now().isoformat(timespec="seconds")
    db.set_last_sync_timestamp(latest_ts, user_id=user_id)

    # Phase A kachra filter — score every page we just touched so the
    # Unassigned lane hides noise by default. Reversible via Revive.
    kachra_counts = {"salient": 0, "kachra": 0}
    touched_ids = {p["id"] for p in open_pages}
    try:
        from sync.salience import score_and_store
        for pid in touched_ids:
            sal, _ = score_and_store(db, pid)
            kachra_counts[sal] = kachra_counts.get(sal, 0) + 1
    except Exception as e:
        logger.warning("salience scoring skipped: %s", e)

    # Phase B humanizer — outcome-shaped rewrite on salient pages.
    # Never humanizes kachra (would waste a call on what we hide).
    humanized = 0
    if hyperparams.get("humanize_auto_on_sync", True):
        try:
            from sync.humanizer import humanize_page
            for pid in touched_ids:
                page = db.get_work_page(pid)
                if not page or (page.get("salience") or "") != "salient":
                    continue
                res = humanize_page(db, page, hyperparams)
                if res.get("outcome"):
                    humanized += 1
        except Exception as e:
            logger.warning("humanize skipped: %s", e)

    # Phase C narrator — compose 1–4 journal stories from today's salient
    # humanized pages. Skips entirely if there's nothing new or if disabled.
    narratives_written = 0
    if hyperparams.get("narrate_auto_on_sync", True) and touched_ids:
        try:
            from sync.narrator import compose_narratives_for_date
            nres = compose_narratives_for_date(
                db, hyperparams, target_date=target_date,
                user_id=user_id, generated_by="auto_sync",
            )
            narratives_written = int(nres.get("narratives_written") or 0)
        except Exception as e:
            logger.warning("narrate skipped: %s", e)

    return {
        "status": "ok",
        "date": target_date,
        "sync_turn": sync_turn,
        "segments": len(new_segments),
        "clusters": len(clusters),
        "pages_created": pages_created,
        "pages_appended": pages_appended,
        "watermark": latest_ts,
        "salient": kachra_counts.get("salient", 0),
        "kachra": kachra_counts.get("kachra", 0),
        "humanized": humanized,
        "narratives_written": narratives_written,
    }


def _read_new_segments(target_date: str, last_ts: str) -> List[Dict]:
    conn = sqlite3.connect(TRACKER_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, detailed_summary, window_name, platform,
               target_segment_length_secs, worker, timestamp_start
        FROM context_1
        WHERE DATE(timestamp_start) = ?
          AND (? = '' OR timestamp_start > ?)
        ORDER BY timestamp_start ASC
        """,
        (target_date, last_ts, last_ts),
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "summary": (r["detailed_summary"] or "").strip(),
            "window": r["window_name"] or "",
            "platform": r["platform"] or "",
            "duration_secs": r["target_segment_length_secs"] or 10,
            "worker": r["worker"] or "human",
            "timestamp_start": r["timestamp_start"] or "",
        }
        for r in rows
        if r["detailed_summary"]
    ]


def _read_page_embedding(page: Dict) -> Optional[np.ndarray]:
    emb = page.get("embedding")
    if isinstance(emb, np.ndarray):
        return emb
    blob = page.get("embedding_blob")
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _blend_embeddings(
    old: Optional[np.ndarray], new: np.ndarray, page_weight: float = 0.7
) -> np.ndarray:
    if old is None:
        return new.astype(np.float32)
    blended = page_weight * old + (1.0 - page_weight) * new
    return blended.astype(np.float32)


def main() -> None:
    from core import config
    from db.manager import DBManager

    os.chdir(str(PMIS_DIR))
    hp = config.get_all()
    db = DBManager("data/memory.db")

    target = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        with sync_lock():
            result = run_sync(db, hp, target_date=target)
    except SyncBusy as e:
        print(f"skipped: {e}")
        return

    print("sync complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
