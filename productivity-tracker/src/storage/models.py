"""
Database models — SQLAlchemy ORM definitions for all tables.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text,
    ForeignKey, create_engine, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Context1(Base):
    """Segment-level tracking — one row per target segment."""
    __tablename__ = "context_1"

    id = Column(String, primary_key=True, default=gen_uuid)
    timestamp_start = Column(DateTime, nullable=False)
    timestamp_end = Column(DateTime)
    window_name = Column(Text)
    platform = Column(Text)                     # e.g., "ChatGPT", "VS Code"
    medium = Column(String(20))                 # browser|terminal|ide|chat|office|other
    context = Column(Text)                      # work area
    supercontext = Column(Text)                 # high-level objective
    anchor = Column(Text)                       # specific task
    target_segment_id = Column(String(20), unique=True, nullable=False)
    target_segment_length_secs = Column(Integer)
    worker = Column(String(10), default="human")  # "human" or "ai"
    short_title = Column(Text, default='')          # 10-word human-readable title
    detailed_summary = Column(Text)

    # Human/AI frame classification
    human_frame_count = Column(Integer, default=0)
    ai_frame_count = Column(Integer, default=0)

    # Productivity classification
    is_productive = Column(Integer, default=-1)   # -1=unclassified, 0=no, 1=yes
    project_id = Column(String, default='')
    deliverable_id = Column(String, default='')

    # Memory sync tracking
    synced_to_memory = Column(Integer, default=0)  # 0=pending, 1=synced
    sc_node_id = Column(String, default='')        # resolved ProMe memory_nodes ID
    context_node_id = Column(String, default='')
    anchor_node_id = Column(String, default='')
    match_score = Column(Float, default=0.0)       # combined match % to project tree

    # Artifact paths — where the raw screenshots for this segment live on disk
    segment_dir = Column(Text, default='')         # directory containing all frames for this segment
    frame_paths_json = Column(Text, default='')    # JSON array of per-frame paths at segment close

    # Relationships
    frames = relationship("Context2", back_populates="segment", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_c1_date", timestamp_start),
        Index("idx_c1_sc", supercontext),
        Index("idx_c1_segment", target_segment_id),
        Index("idx_c1_synced", synced_to_memory),
    )


class Context2(Base):
    """Frame-level tracking — one row per captured frame."""
    __tablename__ = "context_2"

    id = Column(String, primary_key=True, default=gen_uuid)
    target_segment_id = Column(String(20), ForeignKey("context_1.target_segment_id"), nullable=False)
    target_frame_number = Column(Integer, nullable=False)
    frame_timestamp = Column(DateTime)
    raw_text = Column(Text)
    detailed_summary = Column(Text)
    worker_type = Column(String(10), default="human")  # "human" or "ai"

    # Keyboard/mouse activity detection per frame
    has_keyboard_activity = Column(Boolean, default=False)
    has_mouse_activity = Column(Boolean, default=False)

    # Artifact path — where the raw screenshot JPEG lives on disk
    screenshot_path = Column(Text, default='')

    segment = relationship("Context1", back_populates="frames")

    __table_args__ = (
        Index("idx_c2_segment", target_segment_id),
        Index("idx_c2_frame", target_frame_number),
    )


class HourlyMemory(Base):
    """Temp table — hourly aggregation, deleted after daily rollup."""
    __tablename__ = "hourly_memory"

    id = Column(String, primary_key=True, default=gen_uuid)
    date = Column(String(10), nullable=False)   # YYYY-MM-DD
    hour = Column(Integer, nullable=False)      # 0-23
    supercontext = Column(Text)
    context = Column(Text)
    anchor = Column(Text)
    time_mins = Column(Float, default=0)
    human_mins = Column(Float, default=0)
    agent_mins = Column(Float, default=0)
    segment_ids = Column(Text)                  # JSON array
    embedding_id = Column(String)

    __table_args__ = (
        Index("idx_hourly_date", date, hour),
    )


class DailyMemory(Base):
    """Final daily memory — persists indefinitely."""
    __tablename__ = "daily_memory"

    id = Column(String, primary_key=True, default=gen_uuid)
    date = Column(String(10), nullable=False)
    supercontext = Column(Text)
    context = Column(Text)
    anchor = Column(Text)
    level = Column(String(10))                  # SC | context | anchor
    time_mins = Column(Float, default=0)
    human_mins = Column(Float, default=0)
    agent_mins = Column(Float, default=0)
    segment_count = Column(Integer, default=0)
    embedding_id = Column(String)
    contributed_to_delivery = Column(Boolean, default=False)
    deliverable_id = Column(String)

    __table_args__ = (
        Index("idx_daily_date", date),
        Index("idx_daily_sc", supercontext),
        Index("idx_daily_deliverable", deliverable_id),
    )


class PipelineLog(Base):
    """Tracks which hours and days have been processed by the pipeline."""
    __tablename__ = "pipeline_log"

    id = Column(String, primary_key=True, default=gen_uuid)
    date = Column(String(10), nullable=False)       # YYYY-MM-DD
    stage = Column(String(20), nullable=False)       # "hourly" or "daily"
    hour = Column(Integer, nullable=True)            # 0-23 for hourly, NULL for daily
    status = Column(String(10), default="done")      # "done" or "failed"
    segments_processed = Column(Integer, default=0)
    time_mins = Column(Float, default=0)
    processed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_plog_date_stage", date, stage),
        Index("idx_plog_date_hour", date, hour),
    )


class Deliverable(Base):
    """Assigned work / deliverables to match against."""
    __tablename__ = "deliverables"

    id = Column(String, primary_key=True)       # D-001, D-002...
    name = Column(Text, nullable=False)
    supercontext = Column(Text)
    expected_contexts = Column(Text)            # JSON array
    owner = Column(String(100))
    deadline = Column(String(10))               # YYYY-MM-DD
    status = Column(String(20), default="active")
    source = Column(String(20), default="yaml") # yaml | asana | notion
    embedding_id = Column(String)
