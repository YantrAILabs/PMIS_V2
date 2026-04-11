"""
Hourly aggregator — runs every 60 minutes.
Groups segments by SC/Context/Anchor, sums times, generates embeddings.
Produces temp hourly_memory rows used by daily rollup.
"""

import json
import logging
from datetime import datetime

from src.storage.db import Database
try:
    from src.storage.chromadb_store import ChromaDBStore
except ImportError:
    ChromaDBStore = None

logger = logging.getLogger("tracker.hourly_agg")


class HourlyAggregator:
    """Aggregates the last hour of segments into hourly memory."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        self.config = config
        self.chroma = ChromaDBStore(config)

    async def run(self, target_hour: int | None = None, target_date: str | None = None):
        """Run hourly aggregation for a specific hour (or current hour)."""
        now = datetime.now()
        target_date = target_date or now.strftime("%Y-%m-%d")
        current_hour = target_hour if target_hour is not None else now.hour

        logger.info(f"Running hourly aggregation: {target_date} hour {current_hour}")

        segments = self.db.get_segments_for_hour(target_date, current_hour)
        if not segments:
            logger.info(f"No segments for {target_date} hour {current_hour}.")
            return 0

        # Group by (supercontext, context, anchor)
        groups = {}
        for seg in segments:
            key = (
                seg.get("supercontext", "Unclassified"),
                seg.get("context", "Unclassified"),
                seg.get("anchor", "Unclassified"),
            )
            if key not in groups:
                groups[key] = {
                    "segments": [],
                    "total_secs": 0,
                    "human_secs": 0,
                    "agent_secs": 0,
                }
            g = groups[key]
            duration = seg.get("target_segment_length_secs", 0)
            g["segments"].append(seg["target_segment_id"])
            g["total_secs"] += duration
            if seg.get("worker") in ("agent", "ai"):
                g["agent_secs"] += duration
            else:
                g["human_secs"] += duration

        # Store each group as hourly memory entry
        for (sc, ctx, anchor), data in groups.items():
            time_mins = round(data["total_secs"] / 60, 2)
            human_mins = round(data["human_secs"] / 60, 2)
            agent_mins = round(data["agent_secs"] / 60, 2)

            # Build text for embedding
            embed_text = f"{sc} > {ctx} > {anchor} ({time_mins} mins)"
            from src.pipeline.segmenter import sanitize_id
            entry_id = sanitize_id(f"hourly-{target_date}-{current_hour}-{sc}-{ctx}-{anchor}")

            # Store embedding
            try:
                embedding_id = self.chroma.store_hourly(
                    entry_id=entry_id,
                    text=embed_text,
                    metadata={
                        "date": target_date,
                        "hour": current_hour,
                        "supercontext": sc,
                        "context": ctx,
                        "anchor": anchor,
                        "time_mins": time_mins,
                    },
                )
            except Exception as e:
                logger.error(f"Embedding storage failed: {e}")
                embedding_id = None

            # Store in SQLite hourly table
            self.db.insert_hourly(
                date=target_date,
                hour=current_hour,
                supercontext=sc,
                context=ctx,
                anchor=anchor,
                time_mins=time_mins,
                human_mins=human_mins,
                agent_mins=agent_mins,
                segment_ids=json.dumps(data["segments"]),
                embedding_id=embedding_id,
            )

        logger.info(f"Hourly aggregation complete: {len(groups)} groups from {len(segments)} segments.")
        return len(segments)

    async def run_all_pending(self, target_date: str | None = None):
        """Catch up all missed hours. If no date given, process ALL dates with unprocessed hours."""
        if target_date:
            dates = [target_date]
        else:
            dates = self.db.get_dates_with_segments()

        total_processed = 0
        for d in dates:
            processed_hours = self.db.get_processed_hours(d)
            for hour in range(24):
                if hour not in processed_hours:
                    segments = self.db.get_segments_for_hour(d, hour)
                    if segments:
                        # Delete any existing hourly data for this hour (clean rebuild)
                        seg_count = await self.run(target_hour=hour, target_date=d)
                        self.db.log_pipeline_run(
                            date=d, stage="hourly", hour=hour,
                            segments=seg_count or len(segments),
                        )
                        total_processed += 1

        if total_processed:
            logger.info(f"Catch-up complete: processed {total_processed} pending hours across {len(dates)} dates.")
        return total_processed
