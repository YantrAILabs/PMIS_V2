"""
Contribution tracker — determines what work delivered value
and what was wasted effort. Powers the "what not to do" analysis.
"""

import logging
from datetime import date, timedelta

from src.storage.db import Database
from src.memory.pmis_integration import PMISIntegration

logger = logging.getLogger("tracker.contribution")


class ContributionTracker:
    """Tracks and reports on work contribution to deliverables."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        self.pmis = PMISIntegration(config)

    def mark_deliverable_complete(self, deliverable_id: str):
        """Mark a deliverable as complete and propagate to central memory."""
        with self.db.get_session() as s:
            from src.storage.models import Deliverable
            d = s.query(Deliverable).filter_by(id=deliverable_id).first()
            if d:
                d.status = "complete"
                s.commit()

        # FIX C4: Use proper query — get entries matched to this deliverable
        entries = self.db.get_daily_by_deliverable(deliverable_id)
        for e in entries:
            embedding_id = e.get("embedding_id", "")
            if embedding_id:
                self.pmis.mark_contribution(
                    entry_id=embedding_id,
                    deliverable_id=deliverable_id,
                    delivered=True,
                )

        logger.info(f"Deliverable {deliverable_id} marked complete. {len(entries)} entries propagated.")

    def get_contribution_report(self, deliverable_id: str = None, date_range: tuple = None) -> dict:
        """
        Generate a contribution report.
        Requires either deliverable_id or date_range — never operates on "all data."
        """
        if date_range:
            start, end = date_range
            entries = self.db.get_daily_range(start, end)
        elif deliverable_id:
            entries = self.db.get_daily_by_deliverable(deliverable_id)
        else:
            # Default: last 30 days
            end = date.today().strftime("%Y-%m-%d")
            start = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
            entries = self.db.get_daily_range(start, end)

        anchors = [e for e in entries if e["level"] == "anchor"]

        if not anchors:
            return {
                "total_time_mins": 0,
                "contributed_mins": 0,
                "contributed_pct": 0,
                "wasted_mins": 0,
                "wasted_pct": 0,
                "top_wasted_anchors": [],
                "wasted_by_supercontext": {},
                "agent_in_contributed": 0,
                "human_in_contributed": 0,
            }

        contributed = [a for a in anchors if a.get("contributed_to_delivery")]
        not_contributed = [a for a in anchors if not a.get("contributed_to_delivery")]

        total = sum(a["time_mins"] for a in anchors)
        contributed_mins = sum(a["time_mins"] for a in contributed)
        wasted_mins = sum(a["time_mins"] for a in not_contributed)

        # Group wasted by supercontext
        wasted_by_sc = {}
        for a in not_contributed:
            sc = a.get("supercontext", "Other")
            if sc not in wasted_by_sc:
                wasted_by_sc[sc] = {"time_mins": 0, "anchors": []}
            wasted_by_sc[sc]["time_mins"] += a["time_mins"]
            wasted_by_sc[sc]["anchors"].append(a.get("anchor", "?"))

        return {
            "total_time_mins": round(total, 1),
            "contributed_mins": round(contributed_mins, 1),
            "contributed_pct": round(contributed_mins / total * 100, 1) if total > 0 else 0,
            "wasted_mins": round(wasted_mins, 1),
            "wasted_pct": round(wasted_mins / total * 100, 1) if total > 0 else 0,
            "top_wasted_anchors": sorted(
                not_contributed, key=lambda x: x["time_mins"], reverse=True
            )[:10],
            "wasted_by_supercontext": wasted_by_sc,
            "agent_in_contributed": round(sum(a["agent_mins"] for a in contributed), 1),
            "human_in_contributed": round(sum(a["human_mins"] for a in contributed), 1),
        }
