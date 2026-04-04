"""
Daily rollup — runs at end of day (default 11 PM).
Merges all hourly tables into a final daily memory hierarchy.
Stores in PMIS v2 and triggers central memory merge.
"""

import json
import logging
from datetime import datetime

from openai import AsyncOpenAI

from src.storage.db import Database
try:
    from src.storage.chromadb_store import ChromaDBStore
except ImportError:
    ChromaDBStore = None
from src.pipeline.prompts import DAILY_SYNTHESIS_PROMPT

logger = logging.getLogger("tracker.daily_rollup")


class DailyRollup:
    """Builds the final daily memory from hourly aggregations."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        self.config = config
        self.chroma = ChromaDBStore(config)
        self.openai = AsyncOpenAI()

    async def run(self, target_date: str = None):
        """Run daily rollup for the given date (default: today)."""
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Running daily rollup for {target_date}")

        hourly_entries = self.db.get_hourly_for_date(target_date)
        if not hourly_entries:
            logger.info("No hourly entries to roll up.")
            return

        # ─── Build hierarchy by aggregating hourly data ─────────────────

        # Level 1: Group by supercontext
        sc_groups = {}
        for entry in hourly_entries:
            sc = entry["supercontext"]
            ctx = entry["context"]
            anchor = entry["anchor"]

            if sc not in sc_groups:
                sc_groups[sc] = {"contexts": {}, "total_mins": 0, "human_mins": 0, "agent_mins": 0}

            sc_data = sc_groups[sc]
            sc_data["total_mins"] += entry["time_mins"]
            sc_data["human_mins"] += entry["human_mins"]
            sc_data["agent_mins"] += entry["agent_mins"]

            if ctx not in sc_data["contexts"]:
                sc_data["contexts"][ctx] = {"anchors": {}, "total_mins": 0, "human_mins": 0, "agent_mins": 0}

            ctx_data = sc_data["contexts"][ctx]
            ctx_data["total_mins"] += entry["time_mins"]
            ctx_data["human_mins"] += entry["human_mins"]
            ctx_data["agent_mins"] += entry["agent_mins"]

            if anchor not in ctx_data["anchors"]:
                ctx_data["anchors"][anchor] = {"total_mins": 0, "human_mins": 0, "agent_mins": 0, "segments": []}

            anc_data = ctx_data["anchors"][anchor]
            anc_data["total_mins"] += entry["time_mins"]
            anc_data["human_mins"] += entry["human_mins"]
            anc_data["agent_mins"] += entry["agent_mins"]
            seg_ids = json.loads(entry.get("segment_ids", "[]"))
            anc_data["segments"].extend(seg_ids)

        # ─── Store each node at every level ─────────────────────────────

        from src.pipeline.segmenter import sanitize_id

        for sc, sc_data in sc_groups.items():
            # Store SC level
            sc_text = f"{sc} ({sc_data['total_mins']:.0f} mins, human: {sc_data['human_mins']:.0f}, agent: {sc_data['agent_mins']:.0f})"
            sc_embed_id = self.chroma.store_daily(
                entry_id=sanitize_id(f"daily-{target_date}-SC-{sc}"),
                text=sc_text,
                metadata={"date": target_date, "level": "SC", "supercontext": sc},
            )
            self.db.insert_daily(
                date=target_date,
                supercontext=sc,
                context=None,
                anchor=None,
                level="SC",
                time_mins=round(sc_data["total_mins"], 2),
                human_mins=round(sc_data["human_mins"], 2),
                agent_mins=round(sc_data["agent_mins"], 2),
                segment_count=sum(
                    len(a["segments"])
                    for c in sc_data["contexts"].values()
                    for a in c["anchors"].values()
                ),
                embedding_id=sc_embed_id,
            )

            for ctx, ctx_data in sc_data["contexts"].items():
                # Store Context level
                ctx_text = f"{sc} > {ctx} ({ctx_data['total_mins']:.0f} mins)"
                ctx_embed_id = self.chroma.store_daily(
                    entry_id=sanitize_id(f"daily-{target_date}-CTX-{sc}-{ctx}"),
                    text=ctx_text,
                    metadata={"date": target_date, "level": "context", "supercontext": sc, "context": ctx},
                )
                self.db.insert_daily(
                    date=target_date,
                    supercontext=sc,
                    context=ctx,
                    anchor=None,
                    level="context",
                    time_mins=round(ctx_data["total_mins"], 2),
                    human_mins=round(ctx_data["human_mins"], 2),
                    agent_mins=round(ctx_data["agent_mins"], 2),
                    segment_count=sum(len(a["segments"]) for a in ctx_data["anchors"].values()),
                    embedding_id=ctx_embed_id,
                )

                for anchor, anc_data in ctx_data["anchors"].items():
                    # Store Anchor level
                    anc_text = f"{sc} > {ctx} > {anchor} ({anc_data['total_mins']:.0f} mins, {'agent' if anc_data['agent_mins'] > anc_data['human_mins'] else 'human'})"
                    anc_embed_id = self.chroma.store_daily(
                        entry_id=sanitize_id(f"daily-{target_date}-ANC-{sc}-{ctx}-{anchor}"),
                        text=anc_text,
                        metadata={
                            "date": target_date,
                            "level": "anchor",
                            "supercontext": sc,
                            "context": ctx,
                            "anchor": anchor,
                        },
                    )
                    self.db.insert_daily(
                        date=target_date,
                        supercontext=sc,
                        context=ctx,
                        anchor=anchor,
                        level="anchor",
                        time_mins=round(anc_data["total_mins"], 2),
                        human_mins=round(anc_data["human_mins"], 2),
                        agent_mins=round(anc_data["agent_mins"], 2),
                        segment_count=len(anc_data["segments"]),
                        embedding_id=anc_embed_id,
                    )

        # ─── Cleanup hourly temp data ───────────────────────────────────

        if self.config["storage"]["hourly_temp_delete_after_rollup"]:
            self.db.delete_hourly_for_date(target_date)
            self.chroma.delete_hourly_for_date(target_date)
            logger.info("Hourly temp data deleted.")

        total_mins = sum(sc["total_mins"] for sc in sc_groups.values())
        logger.info(
            f"Daily rollup complete: {len(sc_groups)} supercontexts, "
            f"{total_mins:.0f} total minutes tracked."
        )

    def get_daily_hierarchy(self, target_date: str) -> dict:
        """Reconstruct the hierarchy from daily_memory rows for display."""
        entries = self.db.get_daily_for_date(target_date)
        hierarchy = {}

        for entry in entries:
            sc = entry["supercontext"]
            ctx = entry.get("context")
            anchor = entry.get("anchor")

            if sc not in hierarchy:
                hierarchy[sc] = {"time_mins": 0, "human_mins": 0, "agent_mins": 0, "contexts": {}}

            if entry["level"] == "SC":
                hierarchy[sc]["time_mins"] = entry["time_mins"]
                hierarchy[sc]["human_mins"] = entry["human_mins"]
                hierarchy[sc]["agent_mins"] = entry["agent_mins"]
            elif entry["level"] == "context" and ctx:
                if ctx not in hierarchy[sc]["contexts"]:
                    hierarchy[sc]["contexts"][ctx] = {"time_mins": 0, "human_mins": 0, "agent_mins": 0, "anchors": {}}
                hierarchy[sc]["contexts"][ctx]["time_mins"] = entry["time_mins"]
                hierarchy[sc]["contexts"][ctx]["human_mins"] = entry["human_mins"]
                hierarchy[sc]["contexts"][ctx]["agent_mins"] = entry["agent_mins"]
            elif entry["level"] == "anchor" and ctx and anchor:
                if ctx not in hierarchy[sc]["contexts"]:
                    hierarchy[sc]["contexts"][ctx] = {"time_mins": 0, "human_mins": 0, "agent_mins": 0, "anchors": {}}
                hierarchy[sc]["contexts"][ctx]["anchors"][anchor] = {
                    "time_mins": entry["time_mins"],
                    "human_mins": entry["human_mins"],
                    "agent_mins": entry["agent_mins"],
                    "contributed": entry.get("contributed_to_delivery", False),
                    "deliverable_id": entry.get("deliverable_id"),
                }

        return hierarchy
