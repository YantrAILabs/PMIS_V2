"""Phase B — outcome-shaped rewrite of work_page summaries.

The tracker's raw summaries describe motion ("browsed Chrome", "typed in
terminal"). This module calls an LLM to rewrite them as outcomes, grounded
strictly in the given data — no fabrication.

Primary model: Gemini 2.5 Flash (fast + cheap).
  Requires env var GOOGLE_API_KEY or GEMINI_API_KEY.
Fallback: local Ollama (qwen2.5:7b) — runs fully offline if Gemini is
  unavailable or the user disables cloud calls via hyperparameters.yaml.

NO vision here — only text inputs (title, summary, window, platform,
duration). Phase B.2 will add frame-level reading via local qwen2.5-vl
when we need the extra depth.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("pmis.sync.humanizer")

# Words the rewrite must not lead with — these describe motion, not outcomes.
_BANNED_LEAD_VERBS = {
    "browsed", "scrolled", "viewed", "looked", "watched", "observed",
    "read", "reading", "browsing", "scrolling",
}

# Words that indicate a valid outcome-shaped lead.
_GOOD_LEAD_VERBS = {
    "drafted", "wrote", "reviewed", "shipped", "fixed", "debugged",
    "built", "created", "designed", "added", "removed", "refactored",
    "committed", "merged", "pushed", "composed", "sent", "replied",
    "deployed", "tested", "analyzed", "investigated", "resolved",
    "renamed", "restructured", "integrated", "migrated", "documented",
    "pitched", "presented", "planned", "configured", "edited", "updated",
    "launched", "executed", "researched", "compared", "discussed",
    "prepared", "identified", "monitored", "searched", "explored",
}

EMPTY_MARKER = "[low-signal segment]"


def humanize_page(
    db,
    page: Dict,
    hp: Dict,
    force: bool = False,
) -> Dict:
    """Rewrite one work_page's summary. Returns a dict of what changed.

    Skips pages that are already humanized unless force=True.
    Skips kachra pages by default (no point rewriting noise).
    """
    if not force and page.get("humanized_summary"):
        return {"page_id": page["id"], "skipped": "already_humanized"}
    if (page.get("salience") or "pending") == "kachra" and not force:
        return {"page_id": page["id"], "skipped": "kachra"}

    prompt = _build_prompt(page)

    # Try Gemini first unless hp explicitly disables cloud.
    cloud_ok = bool(hp.get("humanize_use_cloud", True))
    api_key = _resolve_gemini_key()
    model_used = ""
    text = ""
    if cloud_ok and api_key:
        text = _call_gemini(prompt, api_key,
                            model=hp.get("humanize_model_cloud", "gemini-2.5-flash"))
        if text:
            model_used = "gemini_flash"

    if not text:
        text = _call_ollama(prompt,
                            model=hp.get("humanize_model_local", "qwen2.5:7b"))
        if text:
            model_used = "qwen_local"

    if not text:
        return {"page_id": page["id"], "skipped": "llm_unavailable"}

    cleaned = _sanitize(text, page)
    if not cleaned or cleaned == EMPTY_MARKER:
        # Leave humanized_summary blank — UI falls back to raw summary.
        db._conn.execute(
            """UPDATE work_pages
               SET humanized_summary = '', humanized_at = datetime('now'),
                   humanized_by = ?
               WHERE id = ?""",
            (f"{model_used}:empty", page["id"]),
        )
        db._conn.commit()
        return {"page_id": page["id"], "humanized_by": model_used,
                "outcome": "", "skipped": "empty_or_rejected"}

    db._conn.execute(
        """UPDATE work_pages
           SET humanized_summary = ?, humanized_at = datetime('now'),
               humanized_by = ?
           WHERE id = ?""",
        (cleaned, model_used, page["id"]),
    )
    db._conn.commit()
    return {
        "page_id": page["id"],
        "humanized_by": model_used,
        "outcome": cleaned,
    }


def humanize_all(
    db,
    hp: Dict,
    date_local: Optional[str] = None,
    user_id: str = "local",
    force: bool = False,
    only_salient: bool = True,
) -> Dict:
    """Humanize every eligible work_page. Returns counts by outcome."""
    where = ["user_id = ?"]
    params: List = [user_id]
    if date_local:
        where.append("date_local = ?")
        params.append(date_local)
    if only_salient:
        where.append("salience = 'salient'")
    if not force:
        where.append("(humanized_summary IS NULL OR humanized_summary = '')")

    rows = db._conn.execute(
        f"SELECT * FROM work_pages WHERE {' AND '.join(where)}",
        params,
    ).fetchall()

    counts = {"total": len(rows), "humanized": 0, "skipped": 0,
              "by_model": {}}
    for r in rows:
        result = humanize_page(db, dict(r), hp, force=force)
        if result.get("outcome"):
            counts["humanized"] += 1
            m = result.get("humanized_by", "unknown")
            counts["by_model"][m] = counts["by_model"].get(m, 0) + 1
        else:
            counts["skipped"] += 1
    return counts


def _resolve_gemini_key() -> Optional[str]:
    """Find a Gemini API key without requiring it in chat or shell profile.

    Lookup order:
      1. GOOGLE_API_KEY env var
      2. GEMINI_API_KEY env var
      3. pmis_v2/data/.gemini_key file (mode 600 recommended)
      4. ~/.config/pmis/gemini_key file
    """
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()

    pmis_dir = Path(__file__).resolve().parent.parent
    for candidate in (
        pmis_dir / "data" / ".gemini_key",
        Path.home() / ".config" / "pmis" / "gemini_key",
    ):
        try:
            if candidate.is_file():
                content = candidate.read_text(encoding="utf-8").strip()
                if content:
                    return content
        except Exception:
            continue
    return None


def _build_prompt(page: Dict) -> str:
    title = (page.get("title") or "").strip()
    summary = (page.get("summary") or "").strip()
    return (
        "You rewrite a work tracker entry as an OUTCOME, not a motion.\n"
        "\n"
        "Input:\n"
        f"  Title:   {title}\n"
        f"  Summary: {summary}\n"
        "\n"
        "Rules:\n"
        "- Second-person past-tense (\"You reviewed...\", \"You drafted...\").\n"
        "- Start with an active outcome verb (drafted, reviewed, shipped, fixed,\n"
        "  debugged, researched, monitored, compared, documented, pitched, etc.).\n"
        "- Do NOT use: scrolled, browsed, read, viewed, looked at, stared, "
        "observed.\n"
        "- Ground STRICTLY in the title+summary. Do not invent specific project\n"
        "  names, people, deliverables, or content the input doesn't mention.\n"
        "- One sentence, under 22 words.\n"
        "- If the input describes truly passive activity only, output the exact\n"
        "  string: " + EMPTY_MARKER + "\n"
        "\n"
        "Output only the sentence (or the marker). No prefix. No quotes."
    )


def _call_gemini(prompt: str, api_key: str,
                 model: str = "gemini-2.5-flash",
                 timeout_s: int = 20) -> str:
    try:
        import httpx
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        resp = httpx.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 400,
                    # Gemini 2.5 models burn "thinking" tokens against the
                    # same budget; zero it out for this tiny-output task.
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            },
            timeout=timeout_s,
        )
        if resp.status_code != 200:
            logger.warning("gemini %s returned %d: %s",
                           model, resp.status_code, resp.text[:200])
            return ""
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        out = "".join(p.get("text", "") for p in parts).strip()
        return out
    except Exception as e:
        logger.warning("gemini call failed: %s", e)
        return ""


def _call_ollama(prompt: str, model: str = "qwen2.5:7b",
                 timeout_s: int = 60) -> str:
    try:
        import httpx
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.3, "num_predict": 120}},
            timeout=timeout_s,
        )
        if resp.status_code != 200:
            return ""
        return (resp.json().get("response") or "").strip()
    except Exception as e:
        logger.warning("ollama call failed: %s", e)
        return ""


def _sanitize(text: str, page: Dict) -> str:
    """Keep the single rewrite line; reject if it leads with a banned verb."""
    if not text:
        return ""
    # Some models wrap with quotes or add explanatory prefix — take first line.
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")
    # Strip wrapping quotes.
    for ch in ['"', "'", "“", "”", "‘", "’"]:
        if first.startswith(ch) and first.endswith(ch):
            first = first[1:-1].strip()
    # Marker passthrough.
    if EMPTY_MARKER in first:
        return EMPTY_MARKER
    # Reject if empty or too short to be useful.
    if len(first) < 6:
        return ""
    # Lead-verb check.
    lead = first.split()[0].lower().strip(",.;:")
    if lead in _BANNED_LEAD_VERBS:
        return ""
    # Truncate over-long outputs to first sentence.
    for stop in [". ", "? ", "! "]:
        if stop in first:
            first = first.split(stop, 1)[0] + stop.strip()
            break
    return first[:240]
