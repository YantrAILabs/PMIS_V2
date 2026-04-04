"""
Temporal encoding for PMIS v2.

Multi-scale sinusoidal encoding (like transformer positional encoding)
that captures hours, days, weeks, months in a single vector.
"""

import numpy as np
import math
from datetime import datetime
from typing import Dict, Optional


def temporal_encode(timestamp: datetime, dim: int = 16) -> np.ndarray:
    """
    Multi-scale sinusoidal temporal encoding.
    
    Uses human-meaningful time scales instead of a single geometric progression:
      - Dims 0-1:   hour-of-day cycle (period = 24 hours)
      - Dims 2-3:   day-of-week cycle (period = 7 days)
      - Dims 4-5:   day-of-month cycle (period = 30 days)
      - Dims 6-7:   month-of-year cycle (period = 365 days)
      - Dims 8+:    absolute position (geometric progression from seconds to years)
    
    Nearby timestamps produce similar vectors.
    """
    position = timestamp.timestamp()
    encodings = np.zeros(dim)
    
    # Cyclic features (human-meaningful periods in seconds)
    cycles = [
        24 * 3600,          # daily cycle
        7 * 24 * 3600,      # weekly cycle
        30 * 24 * 3600,     # monthly cycle
        365 * 24 * 3600,    # yearly cycle
    ]
    
    idx = 0
    for period in cycles:
        if idx + 1 >= dim:
            break
        freq = 2 * math.pi / period
        encodings[idx] = math.sin(position * freq)
        encodings[idx + 1] = math.cos(position * freq)
        idx += 2
    
    # Absolute position features (geometric progression)
    # These capture "how far apart in time" at multiple scales
    remaining = dim - idx
    for i in range(remaining // 2):
        # Periods from ~1 hour to ~3 years
        period = 3600 * (10 ** (i * 3.0 / max(remaining // 2, 1)))
        freq = 2 * math.pi / period
        encodings[idx + 2 * i] = math.sin(position * freq)
        if idx + 2 * i + 1 < dim:
            encodings[idx + 2 * i + 1] = math.cos(position * freq)
    
    return encodings


def temporal_similarity(t1: np.ndarray, t2: np.ndarray) -> float:
    """Cosine similarity between two temporal encodings."""
    dot = np.dot(t1, t2)
    norm1 = np.linalg.norm(t1)
    norm2 = np.linalg.norm(t2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(dot / (norm1 * norm2))


def temporal_distance_hours(dt1: datetime, dt2: datetime) -> float:
    """Absolute time difference in hours."""
    return abs((dt1 - dt2).total_seconds()) / 3600


def compute_era(timestamp: datetime, era_boundaries: Dict[str, str]) -> str:
    """
    Assign a strategic era label based on timestamp.
    era_boundaries maps era_name → boundary_date_string.
    """
    sorted_eras = sorted(
        era_boundaries.items(),
        key=lambda x: datetime.fromisoformat(x[1])
    )
    for era_name, boundary_str in sorted_eras:
        boundary = datetime.fromisoformat(boundary_str)
        if timestamp <= boundary:
            return era_name
    return "current"
