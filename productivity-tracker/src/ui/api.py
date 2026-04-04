"""
FastAPI backend for the productivity dashboard.
Serves data to the React frontend running on localhost:3000.
"""

import json
from datetime import date, timedelta

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.storage.db import Database
from src.memory.daily_rollup import DailyRollup
from src.matching.matching_engine import MatchingEngine

with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

db = Database()
db.initialize()
rollup = DailyRollup(db, config)
matching = MatchingEngine(db, config)

app = FastAPI(title="Productivity Tracker API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/status")
def status():
    today = date.today().strftime("%Y-%m-%d")
    segments = db.get_segments_for_date(today)
    total = sum(s.get("target_segment_length_secs", 0) for s in segments)
    return {"segments": len(segments), "total_mins": round(total / 60, 1)}


@app.get("/api/daily/{target_date}")
def daily_summary(target_date: str):
    hierarchy = rollup.get_daily_hierarchy(target_date)
    return {"date": target_date, "hierarchy": hierarchy}


@app.get("/api/deliverables/{target_date}")
def deliverable_progress(target_date: str):
    return matching.match_day(target_date)


@app.get("/api/segments/{target_date}")
def segments(target_date: str):
    return {"segments": db.get_segments_for_date(target_date)}


@app.get("/api/trends")
def trends(days: int = 30):
    end = date.today()
    start = end - timedelta(days=days)
    totals = []
    d = start
    while d <= end:
        ds = d.strftime("%Y-%m-%d")
        entries = db.get_daily_for_date(ds)
        sc_entries = [e for e in entries if e["level"] == "SC"]
        total = sum(e["time_mins"] for e in sc_entries)
        human = sum(e["human_mins"] for e in sc_entries)
        agent = sum(e["agent_mins"] for e in sc_entries)
        if total > 0:
            totals.append({"date": ds, "total": round(total, 1),
                           "human": round(human, 1), "agent": round(agent, 1)})
        d += timedelta(days=1)
    return {"totals": totals}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3001)
