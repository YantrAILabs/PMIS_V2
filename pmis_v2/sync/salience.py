"""Phase A kachra filter — interpretable heuristic scorer.

A work_page is classified into one of:
  - 'salient'   — shown in the Unassigned lane, eligible for tagging/narratives
  - 'kachra'    — hidden under the folded footer; reversible via "Not kachra"

Scoring inputs come from the segments linked to the page (window_name,
detailed_summary text, durations) — NO LLM calls here, pure rules. The
reason that fired is stored verbatim so the UI can show "hidden because:
passive feed" and the user can correct specific patterns, not a black box.

v0 rules are intentionally conservative. "Revive" actions from the UI get
logged elsewhere and can down-weight specific patterns in later versions.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("pmis.sync.salience")

TRACKER_DB = os.path.expanduser("~/.productivity-tracker/tracker.db")

# URL / window substrings that indicate passive feed browsing
_PASSIVE_FEED_PATTERNS = [
    r"twitter\.com(?!/[A-Za-z0-9_]+/status)",  # twitter home/feed
    r"x\.com(?!/[A-Za-z0-9_]+/status)",        # x.com home/feed
    r"youtube\.com/?$",
    r"youtube\.com/feed",
    r"reddit\.com/?$",
    r"reddit\.com/r/popular",
    r"reddit\.com/r/all",
    r"linkedin\.com/feed",
    r"linkedin\.com/?$",
    r"instagram\.com/?$",
    r"instagram\.com/reels",
    r"facebook\.com/?$",
    r"news\.ycombinator\.com/?$",
]
_PASSIVE_FEED_RE = re.compile(
    "|".join(f"({p})" for p in _PASSIVE_FEED_PATTERNS), re.IGNORECASE
)

# Inbox domains — kachra if no outgoing draft evidence
_INBOX_PATTERNS = [r"mail\.google\.com", r"gmail\.com", r"outlook\.", r"mail\."]
_INBOX_RE = re.compile("|".join(_INBOX_PATTERNS), re.IGNORECASE)
_COMPOSE_SIGNALS = ["compose", "reply", "sent", "drafted", "sending an email", "wrote an email"]

# "physical act" verbs that suggest the tracker captured motion, not outcome
_PASSIVE_VERBS = [
    r"\bscroll(?:ed|ing)?\b",
    r"\bbrows(?:ed|ing)\b",
    r"\bread(?:ing)?\b",
    r"\bviewing\b",
    r"\bwatched?\b",
    r"\bstar(?:ed|ing)\b",
    r"\blooking at\b",
]
_PASSIVE_VERB_RE = re.compile(
    "|".join(_PASSIVE_VERBS), re.IGNORECASE
)

# Micro-fragment threshold (seconds)
_MICRO_FRAGMENT_MAX_SECS = 120  # < 2 min

_ACTIVE_VERBS = [
    # past-tense action verbs (preferred signal)
    "wrote", "drafted", "created", "built", "fixed", "implemented",
    "added", "removed", "refactored", "committed", "merged", "pushed",
    "composed", "sent", "replied", "deployed", "debugged", "tested",
    "analyzed", "designed", "configured", "edited", "reviewed",
    "updated", "launched", "executed", "shipped", "integrated",
    "migrated", "documented", "pitched", "presented", "planned",
    "investigated", "resolved", "renamed", "restructured",
    # noun-form outcomes that imply active work
    "review ", "update ", "implementation", "deployment", "integration",
    "fix ", "merge ", "pull request", "commit", "refactor",
]


def classify_work_page(page: Dict, segments: List[Dict]) -> Tuple[str, str]:
    """Return (salience, reason).

    page    — a work_pages row dict (title, summary, id, ...)
    segments — list of dicts from tracker.context_1 for this page's anchors.
               Each dict should have window_name, detailed_summary,
               target_segment_length_secs.

    If no rule fires, returns ('salient', '').
    """
    if not segments:
        # No segment data to judge from — err salient so we don't hide work
        # produced by an odd sync path.
        return "salient", ""

    summary_text = " ".join(
        [(s.get("detailed_summary") or "") for s in segments]
    ).lower()
    window_text = " ".join(
        [(s.get("window_name") or "") for s in segments]
    ).lower()
    title_text = (page.get("title") or "").lower()

    total_secs = sum(
        (s.get("target_segment_length_secs") or 10) for s in segments
    )

    # Rule 1 — passive_feed
    if _PASSIVE_FEED_RE.search(window_text) or _PASSIVE_FEED_RE.search(summary_text):
        return "kachra", "passive_feed"

    # Rule 2 — inbox_no_draft
    if _INBOX_RE.search(window_text) or "gmail" in window_text or "inbox" in title_text:
        if not any(sig in summary_text for sig in _COMPOSE_SIGNALS):
            return "kachra", "inbox_no_draft"

    # Rule 3 — micro_fragment (short and no active verb in summary)
    if total_secs < _MICRO_FRAGMENT_MAX_SECS:
        if not any(v in summary_text for v in _ACTIVE_VERBS):
            return "kachra", "micro_fragment"

    # Rule 4 — passive_verbs_only (physical-act-only narration)
    if (_PASSIVE_VERB_RE.search(summary_text)
            and not any(v in summary_text for v in _ACTIVE_VERBS)):
        return "kachra", "passive_only"

    # Rule 5 — terminal_idle (terminal window with no command output)
    if ("terminal" in window_text or "iterm" in window_text or "-zsh" in window_text):
        if "executed" not in summary_text and "command" not in summary_text:
            return "kachra", "terminal_idle"

    return "salient", ""


def fetch_segments_for_page(db, page_id: str) -> List[Dict]:
    """Load the tracker segments backing a work_page."""
    segment_ids = [
        r["segment_id"] for r in db._conn.execute(
            "SELECT segment_id FROM work_page_anchors WHERE page_id = ?",
            (page_id,),
        ).fetchall()
    ]
    if not segment_ids:
        return []

    if not os.path.exists(TRACKER_DB):
        return []

    conn = sqlite3.connect(TRACKER_DB)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join(["?"] * len(segment_ids))
    rows = conn.execute(
        f"""SELECT id, window_name, detailed_summary,
                  target_segment_length_secs, timestamp_start
           FROM context_1 WHERE id IN ({placeholders})""",
        segment_ids,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def score_and_store(db, page_id: str) -> Tuple[str, str]:
    """Classify a single page and persist the result. Returns (salience, reason)."""
    page = db.get_work_page(page_id)
    if not page:
        return "pending", ""
    segments = fetch_segments_for_page(db, page_id)
    salience, reason = classify_work_page(page, segments)
    db._conn.execute(
        """UPDATE work_pages
           SET salience = ?, kachra_reason = ?
           WHERE id = ?""",
        (salience, reason, page_id),
    )
    db._conn.commit()
    return salience, reason


def rescan_all(db, date_local: Optional[str] = None,
               user_id: str = "local") -> Dict:
    """Score every work_page in scope. Retroactive tool + nightly sweep."""
    if date_local:
        rows = db._conn.execute(
            """SELECT id FROM work_pages
               WHERE user_id = ? AND date_local = ?""",
            (user_id, date_local),
        ).fetchall()
    else:
        rows = db._conn.execute(
            "SELECT id FROM work_pages WHERE user_id = ?",
            (user_id,),
        ).fetchall()

    counts = {"salient": 0, "kachra": 0, "by_reason": {}}
    for r in rows:
        sal, reason = score_and_store(db, r["id"])
        counts[sal] = counts.get(sal, 0) + 1
        if reason:
            counts["by_reason"][reason] = counts["by_reason"].get(reason, 0) + 1
    counts["total_scored"] = len(rows)
    return counts


def revive_page(db, page_id: str) -> Optional[Dict]:
    """User says "this isn't kachra" — flip to salient, clear reason."""
    page = db.get_work_page(page_id)
    if not page:
        return None
    db._conn.execute(
        """UPDATE work_pages
           SET salience = 'salient', kachra_reason = ''
           WHERE id = ?""",
        (page_id,),
    )
    db._conn.commit()
    return db.get_work_page(page_id)
