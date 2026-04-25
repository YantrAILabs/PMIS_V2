"""Tree-shaped consolidation proposals (F2a).

Converts a segment cluster into a candidate tree: one SC (domain),
optionally one CTX (phase/skill), and N ANCs (atomic insights grouped
by window).

Shape is deterministic — no LLM calls. F5 renders wiki prose at display
time; this layer just structures the data so Review, auto-attach, and
rejection-fingerprint code all see the same tree.

Flattening rule: clusters with <= FLATTEN_THRESHOLD segments skip the
CTX layer. Saves a pointless hierarchy level when there's only a handful
of segments to summarize.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("pmis.consolidation.tree_builder")


# Clusters with this many segments or fewer skip the CTX layer — SC sits
# directly above its anchors. Matches the "tiny cluster, flat tree"
# intuition so small Review items aren't visually bloated.
FLATTEN_THRESHOLD = 3


@dataclass
class Anchor:
    title: str
    body: str
    segment_ids: List[str] = field(default_factory=list)
    window: str = ""
    duration_mins: float = 0.0


@dataclass
class TreeCandidate:
    """Structured tree produced from a segment cluster.

    `ctx_title` is None when the cluster is too small to justify a CTX
    layer. `centroid_embedding` is None when the embedder is unavailable
    — F4 treats that as 'fingerprint comparison skipped for this one'.
    """
    sc_title: str
    sc_summary: str
    ctx_title: Optional[str]
    ctx_summary: str
    anchors: List[Anchor]
    segment_ids: List[str]
    centroid_embedding: Optional[List[float]]
    duration_mins: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def build_tree_candidate(
    cluster: List[Dict[str, Any]],
    sc_title: str,
    hp: Optional[Dict[str, Any]] = None,
) -> TreeCandidate:
    """Build a TreeCandidate from a cluster of activity segments.

    Args:
        cluster: Segment dicts with optional keys `id`, `summary`,
          `short_title`, `detailed_summary`, `window`, `duration_secs`.
        sc_title: The one-liner domain title (typically from
          `DailyActivityMerger._extract_pattern`).
        hp: Hyperparameters forwarded to the embedder for the centroid.
    """
    if not cluster:
        raise ValueError("cluster must not be empty")
    hp = hp or {}

    # ANCs are grouped by window_name. Segments in the same app share
    # an anchor — avoids N anchors for N segments of the same activity.
    by_window: Dict[str, List[Dict[str, Any]]] = {}
    for seg in cluster:
        w = (seg.get("window") or "").strip() or "Unknown"
        by_window.setdefault(w, []).append(seg)

    anchors: List[Anchor] = []
    for window, segs in by_window.items():
        body_parts = [
            (s.get("short_title") or s.get("summary")
             or s.get("detailed_summary") or "")
            for s in segs
        ]
        body = "\n".join(p[:200] for p in body_parts if p)[:1500]
        if len(segs) == 1:
            title = (body_parts[0] or window)[:80]
        else:
            title = f"{len(segs)} segments in {window}"[:80]
        dur_mins = sum(s.get("duration_secs", 10) for s in segs) / 60.0
        anchors.append(Anchor(
            title=title,
            body=body,
            segment_ids=[s.get("id", "") for s in segs],
            window=window,
            duration_mins=round(dur_mins, 2),
        ))

    if len(cluster) <= FLATTEN_THRESHOLD:
        ctx_title: Optional[str] = None
        ctx_summary = ""
    else:
        windows_sorted = sorted(by_window.keys())
        if len(windows_sorted) > 2:
            ctx_title = f"Across {len(by_window)} contexts"
        else:
            ctx_title = " / ".join(windows_sorted)
        ctx_summary = f"{len(cluster)} segments across {len(by_window)} apps"

    centroid = _compute_centroid(cluster, hp)
    duration_mins = sum(s.get("duration_secs", 10) for s in cluster) / 60.0

    return TreeCandidate(
        sc_title=sc_title,
        sc_summary=f"{len(cluster)} segments, {duration_mins:.1f} mins",
        ctx_title=ctx_title,
        ctx_summary=ctx_summary,
        anchors=anchors,
        segment_ids=[s.get("id", "") for s in cluster],
        centroid_embedding=centroid,
        duration_mins=round(duration_mins, 2),
    )


def _compute_centroid(
    cluster: List[Dict[str, Any]], hp: Dict[str, Any],
) -> Optional[List[float]]:
    """Mean-embed the cluster's summaries. Returns None on any failure.
    Callers (F4 fingerprint match) should treat None as 'fingerprint
    comparison unavailable for this proposal'."""
    try:
        from ingestion.embedder import Embedder
        embedder = Embedder(hyperparams=hp)
        embs: List[np.ndarray] = []
        for seg in cluster:
            text = (seg.get("summary") or seg.get("short_title")
                    or seg.get("detailed_summary") or "")
            if not text:
                continue
            try:
                emb = embedder.embed_text(text)
                if emb is not None:
                    embs.append(emb)
            except Exception:
                continue
        if not embs:
            return None
        centroid = np.mean(np.array(embs), axis=0)
        return centroid.tolist()
    except Exception:
        logger.debug("centroid computation skipped", exc_info=True)
        return None
