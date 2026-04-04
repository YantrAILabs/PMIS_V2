"""
Matching engine — maps tracked work (daily memory) to assigned deliverables.
Uses exact SC match first, then semantic similarity via embeddings.
"""

import json
import logging
from dataclasses import dataclass

from src.storage.db import Database
from src.storage.chromadb_store import ChromaDBStore

logger = logging.getLogger("tracker.matching")


@dataclass
class Match:
    daily_entry: dict
    deliverable: dict
    method: str             # "exact" or "semantic"
    similarity: float = 1.0


class MatchingEngine:
    """Matches daily productivity data against assigned deliverables."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        self.config = config
        self.chroma = ChromaDBStore(config)
        self.threshold = config["memory"]["semantic_match_threshold"]

    def match_day(self, target_date: str) -> dict:
        """
        Match all daily memory entries for a date against active deliverables.
        Returns a structured report.
        """
        daily_entries = self.db.get_daily_for_date(target_date)
        deliverables = self.db.get_active_deliverables()

        if not daily_entries:
            return {"date": target_date, "deliverables": [], "unmatched": [], "total_mins": 0}

        matched_results = {}  # deliverable_id → {deliverable, entries}
        unmatched = []

        # Only match anchor-level entries (most granular)
        anchor_entries = [e for e in daily_entries if e["level"] == "anchor"]

        for entry in anchor_entries:
            match = self._find_match(entry, deliverables)
            if match:
                did = match.deliverable["id"]
                if did not in matched_results:
                    matched_results[did] = {
                        "deliverable": match.deliverable,
                        "entries": [],
                        "total_mins": 0,
                        "human_mins": 0,
                        "agent_mins": 0,
                    }
                matched_results[did]["entries"].append(entry)
                matched_results[did]["total_mins"] += entry["time_mins"]
                matched_results[did]["human_mins"] += entry["human_mins"]
                matched_results[did]["agent_mins"] += entry["agent_mins"]

                # Mark contribution in DB
                self._mark_contributed(entry, did)
            else:
                unmatched.append(entry)

        # Build report
        report = {
            "date": target_date,
            "deliverables": [],
            "unmatched": unmatched,
            "total_tracked_mins": sum(e["time_mins"] for e in anchor_entries),
            "total_matched_mins": sum(r["total_mins"] for r in matched_results.values()),
        }

        for did, data in matched_results.items():
            d = data["deliverable"]
            # Group entries by context
            contexts = {}
            for entry in data["entries"]:
                ctx = entry.get("context", "Other")
                if ctx not in contexts:
                    contexts[ctx] = {"anchors": [], "time_mins": 0, "human_mins": 0, "agent_mins": 0}
                contexts[ctx]["anchors"].append({
                    "name": entry.get("anchor", "?"),
                    "time_mins": entry["time_mins"],
                    "human_mins": entry["human_mins"],
                    "agent_mins": entry["agent_mins"],
                })
                contexts[ctx]["time_mins"] += entry["time_mins"]
                contexts[ctx]["human_mins"] += entry["human_mins"]
                contexts[ctx]["agent_mins"] += entry["agent_mins"]

            report["deliverables"].append({
                "id": did,
                "name": d["name"],
                "supercontext": d.get("supercontext", ""),
                "total_mins": data["total_mins"],
                "human_mins": data["human_mins"],
                "agent_mins": data["agent_mins"],
                "agent_leverage_pct": round(
                    data["agent_mins"] / data["total_mins"] * 100, 1
                ) if data["total_mins"] > 0 else 0,
                "contexts": [
                    {"name": name, **vals} for name, vals in contexts.items()
                ],
            })

        return report

    def _find_match(self, entry: dict, deliverables: list[dict]) -> Match | None:
        """Find the best matching deliverable for a daily entry."""
        entry_sc = entry.get("supercontext", "").lower().strip()

        # Pass 1: Exact SC match
        for d in deliverables:
            if d.get("supercontext", "").lower().strip() == entry_sc:
                return Match(daily_entry=entry, deliverable=d, method="exact")

        # Pass 2: Semantic match via embeddings
        text = f"{entry.get('supercontext', '')} > {entry.get('context', '')} > {entry.get('anchor', '')}"
        result = self.chroma.match_to_deliverable(text, threshold=self.threshold)
        if result:
            # Find the deliverable dict
            for d in deliverables:
                if d["id"] == result["id"]:
                    return Match(
                        daily_entry=entry,
                        deliverable=d,
                        method="semantic",
                        similarity=result["similarity"],
                    )

        return None

    def _mark_contributed(self, entry: dict, deliverable_id: str):
        """Mark a daily memory entry as contributing to a deliverable."""
        self.db.mark_daily_contributed(entry["id"], deliverable_id)
