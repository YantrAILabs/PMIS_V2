"""
Database setup, session management, and convenience query methods.
"""

import json
import logging
import os
from datetime import datetime, date
from pathlib import Path

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from src.storage.models import Base, Context1, Context2, HourlyMemory, DailyMemory, Deliverable, PipelineLog

logger = logging.getLogger("tracker.db")


class Database:
    """SQLite database manager for the productivity tracker."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.environ.get(
                "SQLITE_DB_PATH",
                str(Path.home() / ".productivity-tracker" / "tracker.db"),
            )
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def initialize(self):
        """Create all tables if they don't exist, then migrate existing tables."""
        Base.metadata.create_all(self.engine)
        self._migrate()
        logger.info("Database initialized.")

    def _migrate(self):
        """Add new columns to existing tables (safe for fresh DBs too)."""
        import sqlite3
        raw = sqlite3.connect(str(self.engine.url).replace("sqlite:///", ""))

        # Context1 new columns
        c1_cols = {r[1] for r in raw.execute("PRAGMA table_info(context_1)").fetchall()}
        c1_new = {
            "human_frame_count": "INTEGER DEFAULT 0",
            "ai_frame_count": "INTEGER DEFAULT 0",
            "is_productive": "INTEGER DEFAULT -1",
            "project_id": "TEXT DEFAULT ''",
            "deliverable_id": "TEXT DEFAULT ''",
            "synced_to_memory": "INTEGER DEFAULT 0",
            "sc_node_id": "TEXT DEFAULT ''",
            "context_node_id": "TEXT DEFAULT ''",
            "anchor_node_id": "TEXT DEFAULT ''",
            "match_score": "REAL DEFAULT 0.0",
            "segment_dir": "TEXT DEFAULT ''",
            "frame_paths_json": "TEXT DEFAULT ''",
            "short_title": "TEXT DEFAULT ''",
        }
        for col, typedef in c1_new.items():
            if col not in c1_cols:
                try:
                    raw.execute(f"ALTER TABLE context_1 ADD COLUMN {col} {typedef}")
                except Exception:
                    pass

        # Context2 new columns
        c2_cols = {r[1] for r in raw.execute("PRAGMA table_info(context_2)").fetchall()}
        c2_new = {
            "has_keyboard_activity": "BOOLEAN DEFAULT 0",
            "has_mouse_activity": "BOOLEAN DEFAULT 0",
            "screenshot_path": "TEXT DEFAULT ''",
        }
        for col, typedef in c2_new.items():
            if col not in c2_cols:
                try:
                    raw.execute(f"ALTER TABLE context_2 ADD COLUMN {col} {typedef}")
                except Exception:
                    pass

        # Create index on synced_to_memory if missing
        try:
            raw.execute("CREATE INDEX IF NOT EXISTS idx_c1_synced ON context_1(synced_to_memory)")
        except Exception:
            pass

        # ─── Idempotent pipeline: dedup + unique indexes ───────────────

        # Deduplicate daily_memory before adding unique index (keeps row with highest time_mins)
        try:
            raw.execute("""
                DELETE FROM daily_memory WHERE id NOT IN (
                    SELECT id FROM (
                        SELECT id, ROW_NUMBER() OVER (
                            PARTITION BY date, level, supercontext, COALESCE(context,''), COALESCE(anchor,'')
                            ORDER BY time_mins DESC
                        ) as rn FROM daily_memory
                    ) WHERE rn = 1
                )
            """)
        except Exception:
            pass

        # Unique indexes — prevent future duplicates at the natural key level
        for stmt in [
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_hourly_key ON hourly_memory(date, hour, supercontext, context, anchor)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_daily_key ON daily_memory(date, level, supercontext, COALESCE(context,''), COALESCE(anchor,''))",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_pipeline_key ON pipeline_log(date, stage, COALESCE(hour, -1))",
        ]:
            try:
                raw.execute(stmt)
            except Exception:
                pass

        raw.commit()
        raw.close()

    def close(self):
        self.engine.dispose()

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ─── Context 1 (segment-level) ──────────────────────────────────────

    def insert_context1(self, **kwargs):
        with self.get_session() as s:
            row = Context1(**kwargs)
            s.add(row)
            s.commit()
            return row.id

    def get_segments_for_date(self, target_date: str) -> list[dict]:
        """Get all segments for a given date (YYYY-MM-DD)."""
        with self.get_session() as s:
            rows = (
                s.query(Context1)
                .filter(func.date(Context1.timestamp_start) == target_date)
                .order_by(Context1.timestamp_start)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def get_segments_for_hour(self, target_date: str, hour: int) -> list[dict]:
        """Get segments within a specific hour."""
        with self.get_session() as s:
            rows = (
                s.query(Context1)
                .filter(
                    func.date(Context1.timestamp_start) == target_date,
                    func.strftime("%H", Context1.timestamp_start) == f"{hour:02d}",
                )
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def get_max_segment_number(self, date_str: str) -> int:
        """Get the highest segment number for a date (for restart recovery)."""
        prefix = f"TS-{date_str}-"
        with self.get_session() as s:
            row = (
                s.query(func.max(Context1.target_segment_id))
                .filter(Context1.target_segment_id.like(f"{prefix}%"))
                .scalar()
            )
            if row:
                return int(row.split("-")[-1])
            return 0

    # ─── Context 2 (frame-level) ────────────────────────────────────────

    def insert_context2(self, **kwargs):
        with self.get_session() as s:
            row = Context2(**kwargs)
            s.add(row)
            s.commit()

    def get_frames_for_segment(self, segment_id: str) -> list[dict]:
        with self.get_session() as s:
            rows = (
                s.query(Context2)
                .filter(Context2.target_segment_id == segment_id)
                .order_by(Context2.target_frame_number)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    # ─── Hourly memory ──────────────────────────────────────────────────

    def upsert_hourly(self, **kwargs):
        """Insert or update hourly memory entry. Idempotent on (date, hour, sc, ctx, anchor)."""
        with self.get_session() as s:
            existing = s.query(HourlyMemory).filter_by(
                date=kwargs["date"], hour=kwargs["hour"],
                supercontext=kwargs["supercontext"],
                context=kwargs["context"], anchor=kwargs["anchor"],
            ).first()
            if existing:
                existing.time_mins = kwargs["time_mins"]
                existing.human_mins = kwargs["human_mins"]
                existing.agent_mins = kwargs["agent_mins"]
                existing.segment_ids = kwargs.get("segment_ids", existing.segment_ids)
                existing.embedding_id = kwargs.get("embedding_id", existing.embedding_id)
            else:
                s.add(HourlyMemory(**kwargs))
            s.commit()

    def insert_hourly(self, **kwargs):
        """Backward-compatible alias for upsert_hourly."""
        self.upsert_hourly(**kwargs)

    def get_hourly_for_date(self, target_date: str) -> list[dict]:
        with self.get_session() as s:
            rows = (
                s.query(HourlyMemory)
                .filter(HourlyMemory.date == target_date)
                .order_by(HourlyMemory.hour)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def delete_hourly_for_date(self, target_date: str):
        with self.get_session() as s:
            s.query(HourlyMemory).filter(HourlyMemory.date == target_date).delete()
            s.commit()

    # ─── Daily memory ───────────────────────────────────────────────────

    def upsert_daily(self, **kwargs):
        """Insert or update daily memory entry. Idempotent on (date, level, sc, ctx, anchor)."""
        with self.get_session() as s:
            existing = s.query(DailyMemory).filter_by(
                date=kwargs["date"], level=kwargs["level"],
                supercontext=kwargs["supercontext"],
                context=kwargs.get("context"), anchor=kwargs.get("anchor"),
            ).first()
            if existing:
                existing.time_mins = kwargs["time_mins"]
                existing.human_mins = kwargs["human_mins"]
                existing.agent_mins = kwargs["agent_mins"]
                existing.segment_count = kwargs.get("segment_count", existing.segment_count)
                existing.embedding_id = kwargs.get("embedding_id", existing.embedding_id)
            else:
                s.add(DailyMemory(**kwargs))
            s.commit()

    def insert_daily(self, **kwargs):
        """Backward-compatible alias for upsert_daily."""
        self.upsert_daily(**kwargs)

    def delete_daily_for_date(self, target_date: str):
        """Delete all daily memory entries for a date (clean rebuild)."""
        with self.get_session() as s:
            s.query(DailyMemory).filter(DailyMemory.date == target_date).delete()
            s.commit()

    # ─── Pipeline log ───────────────────────────────────────────────────

    def log_pipeline_run(self, date: str, stage: str, hour: int = None,
                         segments: int = 0, time_mins: float = 0, status: str = "done"):
        """Record pipeline completion. Idempotent — updates if (date, stage, hour) exists."""
        with self.get_session() as s:
            existing = s.query(PipelineLog).filter_by(
                date=date, stage=stage, hour=hour,
            ).first()
            if existing:
                existing.segments_processed = segments
                existing.time_mins = time_mins
                existing.status = status
                existing.processed_at = datetime.utcnow()
            else:
                s.add(PipelineLog(
                    date=date, stage=stage, hour=hour,
                    segments_processed=segments, time_mins=time_mins, status=status,
                ))
            s.commit()

    def get_processed_hours(self, target_date: str) -> set:
        """Return set of hours already processed for a date."""
        with self.get_session() as s:
            rows = (
                s.query(PipelineLog.hour)
                .filter(PipelineLog.date == target_date,
                        PipelineLog.stage == "hourly",
                        PipelineLog.status == "done")
                .all()
            )
            return {r[0] for r in rows if r[0] is not None}

    def get_processed_daily_dates(self) -> set:
        """Return set of dates that have completed daily rollup."""
        with self.get_session() as s:
            rows = (
                s.query(PipelineLog.date)
                .filter(PipelineLog.stage == "daily", PipelineLog.status == "done")
                .distinct()
                .all()
            )
            return {r[0] for r in rows}

    def get_dates_with_segments(self) -> list:
        """Return all distinct dates that have context_1 data."""
        with self.get_session() as s:
            rows = (
                s.query(func.date(Context1.timestamp_start))
                .distinct()
                .order_by(func.date(Context1.timestamp_start))
                .all()
            )
            return [r[0] for r in rows if r[0]]

    def clear_pipeline_log_for_date(self, target_date: str, stage: str = None):
        """Clear pipeline log entries for a date (and optionally a specific stage)."""
        with self.get_session() as s:
            q = s.query(PipelineLog).filter(PipelineLog.date == target_date)
            if stage:
                q = q.filter(PipelineLog.stage == stage)
            q.delete()
            s.commit()

    def get_daily_for_date(self, target_date: str) -> list[dict]:
        with self.get_session() as s:
            rows = (
                s.query(DailyMemory)
                .filter(DailyMemory.date == target_date)
                .order_by(DailyMemory.supercontext, DailyMemory.context)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def get_daily_range(self, start_date: str, end_date: str) -> list[dict]:
        with self.get_session() as s:
            rows = (
                s.query(DailyMemory)
                .filter(DailyMemory.date >= start_date, DailyMemory.date <= end_date)
                .order_by(DailyMemory.date, DailyMemory.supercontext)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def get_daily_by_deliverable(self, deliverable_id: str) -> list[dict]:
        """Get all daily entries matched to a specific deliverable."""
        with self.get_session() as s:
            rows = (
                s.query(DailyMemory)
                .filter(DailyMemory.deliverable_id == deliverable_id)
                .order_by(DailyMemory.date)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def mark_daily_contributed(self, entry_id: str, deliverable_id: str):
        """Mark a daily memory entry as contributing to a deliverable."""
        with self.get_session() as s:
            row = s.query(DailyMemory).filter_by(id=entry_id).first()
            if row:
                row.contributed_to_delivery = True
                row.deliverable_id = deliverable_id
                s.commit()

    # ─── Deliverables ───────────────────────────────────────────────────

    def upsert_deliverable(self, **kwargs):
        with self.get_session() as s:
            existing = s.query(Deliverable).filter_by(id=kwargs["id"]).first()
            if existing:
                for k, v in kwargs.items():
                    setattr(existing, k, v)
            else:
                s.add(Deliverable(**kwargs))
            s.commit()

    def get_active_deliverables(self) -> list[dict]:
        with self.get_session() as s:
            rows = s.query(Deliverable).filter_by(status="active").all()
            return [self._row_to_dict(r) for r in rows]

    # ─── Productivity sync queries ─────────────────────────────────────

    def get_unsynced_segments(self) -> list[dict]:
        """Get all segments that haven't been synced to ProMe memory."""
        with self.get_session() as s:
            rows = (
                s.query(Context1)
                .filter(Context1.synced_to_memory == 0)
                .order_by(Context1.timestamp_start)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def count_new_frames_since(self, minutes: int = 30) -> int:
        """Count unique frame numbers added in the last N minutes."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(minutes=minutes)
        with self.get_session() as s:
            count = (
                s.query(func.count(func.distinct(Context2.target_frame_number)))
                .filter(Context2.frame_timestamp >= cutoff)
                .scalar()
            )
            return count or 0

    def mark_segment_synced(self, segment_id: str, sc_node_id: str = '',
                            context_node_id: str = '', anchor_node_id: str = '',
                            match_score: float = 0.0, is_productive: int = -1,
                            project_id: str = '', deliverable_id: str = ''):
        """Mark a segment as synced to ProMe with resolved node IDs."""
        with self.get_session() as s:
            row = (
                s.query(Context1)
                .filter(Context1.target_segment_id == segment_id)
                .first()
            )
            if row:
                row.synced_to_memory = 1
                row.sc_node_id = sc_node_id
                row.context_node_id = context_node_id
                row.anchor_node_id = anchor_node_id
                row.match_score = match_score
                row.is_productive = is_productive
                row.project_id = project_id
                row.deliverable_id = deliverable_id
                s.commit()

    # ─── Utility ────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {c.name: getattr(row, c.name) for c in row.__table__.columns}
