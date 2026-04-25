"""Build and update work_pages from fresh tracker segments.

Clustering is greedy cosine-distance (same algo family as daily_activity_merge).
LLM summary calls hit local Ollama (qwen2.5:14b by default); failures fall
back to a concatenation so the sync never crashes.

Thresholds (cluster + append-vs-new decision) are owned by the runner and
read from `hyperparameters.yaml`:
  sync_cluster_threshold  — within-sync greedy clustering
  sync_append_distance    — below this → append to existing page, else new
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("pmis.sync.page_builder")

# Tracker's LLM summaries begin with boilerplate like "During the segment, ..."
# or "In this segment, ...". Stripping it before embedding lets nomic key on
# the distinguishing content instead of the shared lead-in phrasing.
_SUMMARY_PREAMBLE = re.compile(
    r"^\s*(during|in|throughout|within)\s+(this|the)\s+(segment|work segment|period|time|session)[\s,:;.\-]*",
    re.IGNORECASE,
)


def cluster_segments(
    segments: List[Dict], embedder, threshold: float = 0.35
) -> List[List[Dict]]:
    """Greedy cosine clustering. Segments without embeddings are dropped."""
    if not segments:
        return []

    embeddings: List[np.ndarray] = []
    valid: List[Dict] = []
    for seg in segments:
        try:
            emb = embedder.embed_text(_embed_text_for(seg))
        except Exception as e:
            logger.warning("embed error: %s", e)
            continue
        if emb is None:
            continue
        embeddings.append(np.asarray(emb, dtype=np.float32))
        valid.append(seg)

    if not valid:
        return []
    if len(valid) == 1:
        return [valid]

    arr = np.stack(embeddings)
    assigned: set = set()
    clusters: List[List[Dict]] = []

    for i in range(len(valid)):
        if i in assigned:
            continue
        cluster = [valid[i]]
        assigned.add(i)
        for j in range(i + 1, len(valid)):
            if j in assigned:
                continue
            if _cosine_distance(arr[i], arr[j]) < threshold:
                cluster.append(valid[j])
                assigned.add(j)
        clusters.append(cluster)

    return clusters


def compute_cluster_centroid(
    cluster: List[Dict], embedder
) -> Optional[np.ndarray]:
    """Mean-pool of cluster segment embeddings. Returns None on total failure."""
    embs: List[np.ndarray] = []
    for seg in cluster:
        try:
            e = embedder.embed_text(_embed_text_for(seg))
        except Exception:
            continue
        if e is not None:
            embs.append(np.asarray(e, dtype=np.float32))
    if not embs:
        return None
    return np.mean(np.stack(embs), axis=0).astype(np.float32)


def find_matching_page(
    centroid: np.ndarray, open_pages: List[Dict]
) -> Tuple[Optional[str], float]:
    """Return (page_id, distance) of the closest open page, or (None, inf)."""
    best_id: Optional[str] = None
    best_dist = float("inf")
    for page in open_pages:
        emb = page.get("embedding")
        if emb is None:
            continue
        d = _cosine_distance(centroid, emb)
        if d < best_dist:
            best_dist = d
            best_id = page["id"]
    return best_id, best_dist


def llm_generate_title_and_summary(
    cluster: List[Dict], model: str = "qwen2.5:14b"
) -> Tuple[str, str]:
    """Ask local Ollama for (title, summary). Falls back on failure."""
    duration_mins = sum(s.get("duration_secs", 10) for s in cluster) / 60.0
    windows = list({s.get("window", "") for s in cluster if s.get("window")})[:5]
    descs = [s["summary"][:150] for s in cluster[:10]]

    prompt = (
        "You are summarizing a cluster of activity segments as a wiki-style page.\n"
        f"Segments: {len(cluster)} | Duration: {duration_mins:.1f} min | "
        f"Apps: {', '.join(windows[:3])}\n\n"
        "Activity descriptions:\n"
        + "\n".join(f"- {d}" for d in descs)
        + "\n\nOutput in this exact format (no extra lines):\n"
        "TITLE: <one concise noun-phrase title, under 60 chars>\n"
        "SUMMARY: <one paragraph, 2-3 sentences, outcome-shaped>"
    )

    text = _call_ollama(prompt, model)
    if not text:
        return _fallback_title_and_summary(cluster, duration_mins)

    title, summary = _parse_title_summary(text)
    if not summary:
        summary = text[:400]
    if not title:
        title = (summary.split(".")[0] or "Untitled work")[:120]
    return title, summary


def llm_restitch_page(
    page_title: str,
    page_summary: str,
    new_cluster: List[Dict],
    model: str = "qwen2.5:14b",
) -> Tuple[str, str]:
    """Regenerate title + summary when new activity extends an existing page."""
    new_descs = [s["summary"][:150] for s in new_cluster[:10]]

    prompt = (
        "Extend an existing wiki-style page with new activity from the same thread.\n\n"
        f"Existing title: {page_title}\n"
        f"Existing summary: {page_summary}\n\n"
        "New activity descriptions:\n"
        + "\n".join(f"- {d}" for d in new_descs)
        + "\n\nOutput in this exact format (no extra lines):\n"
        "TITLE: <updated title if theme shifted; otherwise same>\n"
        "SUMMARY: <updated paragraph, 2-3 sentences, outcome-shaped, integrates new>"
    )

    text = _call_ollama(prompt, model)
    if not text:
        return page_title, page_summary

    title, summary = _parse_title_summary(text)
    return (title or page_title, summary or page_summary)


def _embed_text_for(seg: Dict) -> str:
    """Build the text we embed for a segment.

    The tracker's LLM summaries all begin with templated phrasing like
    "During the segment, a human worker...", which dominates nomic-embed-text
    and collapses distances between genuinely different activities. Prefixing
    the window name gives the embedder a distinguishing anchor even when the
    summary bodies rhyme.
    """
    raw = (seg.get("summary") or "").strip()
    summary = _SUMMARY_PREAMBLE.sub("", raw).strip() or raw
    window = (seg.get("window") or "").strip()
    platform = (seg.get("platform") or "").strip()
    if window:
        return f"{window} | {summary}"
    if platform:
        return f"{platform} | {summary}"
    return summary


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a, b) / denom)


def _call_ollama(prompt: str, model: str, timeout_s: int = 60) -> str:
    try:
        import httpx

        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout_s,
        )
        if resp.status_code != 200:
            logger.warning("ollama %s returned %d", model, resp.status_code)
            return ""
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("ollama call failed: %s", e)
        return ""


def _parse_title_summary(text: str) -> Tuple[str, str]:
    title, summary = "", ""
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("TITLE:"):
            title = stripped.split(":", 1)[1].strip()[:120]
        elif upper.startswith("SUMMARY:"):
            summary = stripped.split(":", 1)[1].strip()
    return title, summary


def _fallback_title_and_summary(
    cluster: List[Dict], duration_mins: float
) -> Tuple[str, str]:
    top = [s["summary"][:80] for s in cluster[:3] if s.get("summary")]
    summary = (
        f"{len(cluster)} activity segments ({duration_mins:.0f} min): "
        + ". ".join(top)
    )[:500]
    title = (top[0] if top else "Activity cluster")[:120]
    return title, summary
