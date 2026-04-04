"""
Central memory merge — orchestrates the daily-to-central merge at end of day.
Called after daily rollup completes.
"""

import logging
from src.storage.db import Database
from src.memory.pmis_integration import PMISIntegration

logger = logging.getLogger("tracker.central_merge")


class CentralMerge:
    """Merges daily productivity memory into central PMIS v2."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        self.pmis = PMISIntegration(config)

    def run(self, target_date: str):
        """Merge all daily memory entries for a date into central memory."""
        entries = self.db.get_daily_for_date(target_date)
        if not entries:
            logger.info("No daily entries to merge.")
            return

        merged = 0
        created = 0

        for entry in entries:
            # Only merge SC and context levels (anchors are too granular for central)
            if entry["level"] not in ("SC", "context"):
                continue

            result_id = self.pmis.merge_to_central(entry)
            if result_id:
                if result_id.startswith("prod-"):
                    created += 1
                else:
                    merged += 1

        logger.info(
            f"Central merge complete for {target_date}: "
            f"{merged} updated, {created} new nodes."
        )

    def get_unmatched_work(self, target_date: str) -> list[dict]:
        """Find work that didn't contribute to any deliverable."""
        entries = self.db.get_daily_for_date(target_date)
        return [
            e for e in entries
            if e["level"] == "anchor"
            and not e.get("contributed_to_delivery")
        ]

    def get_wasted_effort_report(self, start_date: str, end_date: str) -> dict:
        """
        Identify work over a date range that never contributed to delivery.
        Useful for "what not to do next project" analysis.
        """
        entries = self.db.get_daily_range(start_date, end_date)
        contributed = [e for e in entries if e.get("contributed_to_delivery")]
        not_contributed = [e for e in entries if not e.get("contributed_to_delivery") and e["level"] == "anchor"]

        total_mins = sum(e["time_mins"] for e in entries if e["level"] == "anchor")
        wasted_mins = sum(e["time_mins"] for e in not_contributed)

        return {
            "period": f"{start_date} to {end_date}",
            "total_anchor_time_mins": total_mins,
            "contributed_mins": total_mins - wasted_mins,
            "wasted_mins": wasted_mins,
            "wasted_pct": round(wasted_mins / total_mins * 100, 1) if total_mins > 0 else 0,
            "top_wasted_anchors": sorted(
                not_contributed, key=lambda x: x["time_mins"], reverse=True
            )[:10],
        }
