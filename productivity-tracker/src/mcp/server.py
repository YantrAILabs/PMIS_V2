"""
MCP server for Claude Desktop — exposes all productivity tracker tools.
This is the primary interface for interacting with the tracker.

Run via: python -m src.mcp.server
"""

import json
import logging
from datetime import datetime, date, timedelta

from mcp.server.fastmcp import FastMCP

from src.storage.db import Database
try:
    from src.storage.chromadb_store import ChromaDBStore
except ImportError:
    ChromaDBStore = None
from src.memory.daily_rollup import DailyRollup
from src.matching.matching_engine import MatchingEngine
from src.matching.deliverables_loader import DeliverablesLoader
from src.matching.contribution_tracker import ContributionTracker
from src.memory.pmis_integration import PMISIntegration

logger = logging.getLogger("tracker.mcp")

# ─── Initialize ─────────────────────────────────────────────────────────

import yaml
with open("config/settings.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db = Database()
db.initialize()
chroma = ChromaDBStore(config)
rollup = DailyRollup(db, config)
matching = MatchingEngine(db, config)
loader = DeliverablesLoader(db, config)
contrib = ContributionTracker(db, config)
pmis = PMISIntegration(config)

# Load deliverables on startup
loader.load_from_yaml()

mcp = FastMCP("productivity-tracker")


# ─── Status tools ───────────────────────────────────────────────────────

@mcp.tool()
def get_status() -> str:
    """Get current tracking status — active segment, window, running time."""
    today = date.today().strftime("%Y-%m-%d")
    segments = db.get_segments_for_date(today)

    if not segments:
        return json.dumps({"status": "No segments tracked today yet."})

    latest = segments[-1]
    total_secs = sum(s.get("target_segment_length_secs", 0) for s in segments)

    return json.dumps({
        "status": "tracking",
        "today_segments": len(segments),
        "total_tracked_mins": round(total_secs / 60, 1),
        "current_segment": {
            "id": latest["target_segment_id"],
            "window": latest["window_name"],
            "platform": latest["platform"],
            "supercontext": latest["supercontext"],
            "context": latest["context"],
            "anchor": latest["anchor"],
            "worker": latest["worker"],
        },
    }, indent=2)


@mcp.tool()
def pause_tracking() -> str:
    """Pause the productivity tracker. No screenshots will be taken."""
    # Signal the daemon via a flag file
    from pathlib import Path
    flag = Path.home() / ".productivity-tracker" / ".paused"
    flag.touch()
    return "Tracking paused. Use resume_tracking to continue."


@mcp.tool()
def resume_tracking() -> str:
    """Resume the productivity tracker after pause."""
    from pathlib import Path
    flag = Path.home() / ".productivity-tracker" / ".paused"
    flag.unlink(missing_ok=True)
    return "Tracking resumed."


# ─── Daily summary ──────────────────────────────────────────────────────

@mcp.tool()
def get_daily_summary(target_date: str = None) -> str:
    """
    Get time breakdown for a day: SC → Context → Anchor with human/agent split.
    Defaults to today. Format: YYYY-MM-DD.
    """
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    hierarchy = rollup.get_daily_hierarchy(target_date)

    if not hierarchy:
        # Fall back to live segment data
        segments = db.get_segments_for_date(target_date)
        if not segments:
            return f"No data for {target_date}."

        summary = {"date": target_date, "segments": len(segments), "breakdown": {}}
        for seg in segments:
            sc = seg.get("supercontext", "Unclassified")
            if sc not in summary["breakdown"]:
                summary["breakdown"][sc] = {"time_mins": 0, "human_mins": 0, "agent_mins": 0}
            duration = seg.get("target_segment_length_secs", 0) / 60
            summary["breakdown"][sc]["time_mins"] += duration
            if seg.get("worker") == "agent":
                summary["breakdown"][sc]["agent_mins"] += duration
            else:
                summary["breakdown"][sc]["human_mins"] += duration

        return json.dumps(summary, indent=2)

    return json.dumps({"date": target_date, "hierarchy": hierarchy}, indent=2, default=str)


# ─── Deliverable tools ──────────────────────────────────────────────────

@mcp.tool()
def get_deliverable_progress(deliverable_name: str = None) -> str:
    """
    Get progress on deliverables with time + human/agent split.
    Optionally filter by deliverable name.
    """
    today = date.today().strftime("%Y-%m-%d")
    report = matching.match_day(today)

    if deliverable_name:
        report["deliverables"] = [
            d for d in report["deliverables"]
            if deliverable_name.lower() in d["name"].lower()
        ]

    return json.dumps(report, indent=2, default=str)


@mcp.tool()
def add_deliverable(name: str, supercontext: str, contexts: str,
                    owner: str = "", deadline: str = "") -> str:
    """
    Add a new deliverable. Contexts should be comma-separated.
    Example: add_deliverable("GTM Strategy", "Sales", "Outreach, Collateral", "Rohit", "2026-05-01")
    """
    context_list = [c.strip() for c in contexts.split(",") if c.strip()]
    new_id = loader.add_deliverable(name, supercontext, context_list, owner, deadline)
    return f"Added deliverable {new_id}: {name}"


@mcp.tool()
def complete_deliverable(deliverable_id: str) -> str:
    """Mark a deliverable as complete. Propagates contribution tracking."""
    contrib.mark_deliverable_complete(deliverable_id)
    return f"Deliverable {deliverable_id} marked complete. Contributions propagated to central memory."


# ─── Memory & search tools ──────────────────────────────────────────────

@mcp.tool()
def search_productivity_memory(query: str, days_back: int = 7) -> str:
    """Search productivity data semantically. Returns matching work entries."""
    results = chroma.search_daily(query, n_results=10)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def get_contribution_report(days_back: int = 7) -> str:
    """
    Show what work contributed to deliverables vs what didn't.
    Includes wasted effort analysis and recommendations.
    """
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    report = contrib.get_contribution_report(date_range=(start, end))
    return json.dumps(report, indent=2, default=str)


@mcp.tool()
def get_trends(days_back: int = 30) -> str:
    """
    Get productivity trends: daily totals, human/agent ratios,
    top contexts, agent leverage over time.
    """
    end = date.today()
    start = end - timedelta(days=days_back)

    daily_totals = []
    d = start
    while d <= end:
        ds = d.strftime("%Y-%m-%d")
        entries = db.get_daily_for_date(ds)
        sc_entries = [e for e in entries if e["level"] == "SC"]

        total = sum(e["time_mins"] for e in sc_entries)
        human = sum(e["human_mins"] for e in sc_entries)
        agent = sum(e["agent_mins"] for e in sc_entries)

        if total > 0:
            daily_totals.append({
                "date": ds,
                "total_mins": round(total, 1),
                "human_mins": round(human, 1),
                "agent_mins": round(agent, 1),
                "agent_pct": round(agent / total * 100, 1) if total > 0 else 0,
            })
        d += timedelta(days=1)

    return json.dumps({
        "period": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
        "daily_totals": daily_totals,
        "avg_daily_mins": round(
            sum(d["total_mins"] for d in daily_totals) / len(daily_totals), 1
        ) if daily_totals else 0,
        "avg_agent_pct": round(
            sum(d["agent_pct"] for d in daily_totals) / len(daily_totals), 1
        ) if daily_totals else 0,
    }, indent=2)


# ─── Entry point ────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
