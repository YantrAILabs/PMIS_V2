"""Compose a project digest for one (project, window) combo.

Inputs:
  - DB handle
  - project_id
  - project_title (for the header) — optional; falls back to id
  - window_type ('day' | 'week' | 'custom')
  - window_start, window_end (YYYY-MM-DD, inclusive)

Output:
  - Writes a row to `project_digests` (upsert keyed by user+project+type+start)
  - Returns the composed digest dict (markdown + metadata)

If no confirmed pages exist in the window, returns status='empty' and skips
the DB write — per the architecture rule "no digest, not a stub."

LLM runs via local Ollama (consolidation_model_local). On LLM failure we fall
back to a deterministic template so digest generation never crashes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("pmis.reports.digest")


def compose_digest(
    db,
    project_id: str,
    window_start: str,
    window_end: str,
    window_type: str = "day",
    project_title: Optional[str] = None,
    model: str = "qwen2.5:14b",
    user_id: str = "local",
    generated_by: str = "manual",
    save: bool = True,
) -> Dict:
    """Build and (optionally) persist a project digest."""
    pages = db.get_confirmed_pages_for_project_window(
        project_id=project_id,
        window_start=window_start,
        window_end=window_end,
        user_id=user_id,
    )

    if not pages:
        return {
            "status": "empty",
            "project_id": project_id,
            "window_type": window_type,
            "window_start": window_start,
            "window_end": window_end,
            "reason": "no confirmed work_pages in window",
        }

    total_segs = sum(p.get("segment_count", 0) for p in pages)
    total_minutes = round(total_segs * 10 / 60.0, 1)  # 10-sec frames → minutes

    title = project_title or project_id
    window_label = _format_window_label(window_type, window_start, window_end)

    llm_body = _llm_compose(
        title=title,
        window_label=window_label,
        total_minutes=total_minutes,
        pages=pages,
        model=model,
    )
    markdown, key_points = _parse_llm_body(llm_body, title, window_label,
                                            total_minutes, pages)

    if save:
        digest_id = db.upsert_project_digest(
            project_id=project_id,
            window_type=window_type,
            window_start=window_start,
            window_end=window_end,
            summary_markdown=markdown,
            key_points=key_points,
            total_minutes=total_minutes,
            source_page_ids=[p["id"] for p in pages],
            generated_by=generated_by,
            user_id=user_id,
        )
    else:
        digest_id = None

    return {
        "status": "ok",
        "digest_id": digest_id,
        "project_id": project_id,
        "project_title": title,
        "window_type": window_type,
        "window_start": window_start,
        "window_end": window_end,
        "window_label": window_label,
        "page_count": len(pages),
        "total_minutes": total_minutes,
        "summary_markdown": markdown,
        "key_points": key_points,
        "source_page_ids": [p["id"] for p in pages],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generated_by": generated_by,
    }


def _format_window_label(window_type: str, start: str, end: str) -> str:
    if window_type == "day" or start == end:
        return start
    if window_type == "week":
        return f"{start} → {end}"
    return f"{start} → {end} ({window_type})"


def _llm_compose(
    title: str,
    window_label: str,
    total_minutes: float,
    pages: List[Dict],
    model: str,
) -> str:
    """Call Ollama to produce digest markdown. Returns empty string on failure."""
    import httpx

    page_lines = []
    for p in pages:
        mins = round(p.get("segment_count", 0) * 10 / 60.0, 1)
        page_lines.append(
            f"- **{p['title']}** ({mins} min, {p['date_local']})\n"
            f"  {p['summary'][:400]}"
        )

    prompt = (
        "You are composing an employee-facing daily/weekly project digest.\n"
        "Tone: second-person, narrative, outcome-shaped, no utilization metrics.\n"
        "No comparisons, no ranking.\n\n"
        f"Project: {title}\n"
        f"Window: {window_label}\n"
        f"Tracked time: {total_minutes} min across {len(pages)} work pages\n\n"
        "Tagged work pages (each with its own summary):\n"
        + "\n".join(page_lines)
        + "\n\n"
        "Output markdown in EXACTLY this shape (no intro, no outro):\n\n"
        f"## {title} — {window_label}\n\n"
        "**Focus**: <one-line headline>\n"
        f"**Time**: {total_minutes} min across {len(pages)} work threads\n\n"
        "### What you did\n"
        "- <concrete outcome bullet 1>\n"
        "- <bullet 2>\n"
        "- <bullet 3>\n\n"
        "### Open threads\n"
        "- <anything incomplete or follow-up worth revisiting>\n\n"
        "### Next likely step\n"
        "- <single inferred next action>\n"
    )

    try:
        r = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=90,
        )
        if r.status_code != 200:
            logger.warning("ollama %s returned %d", model, r.status_code)
            return ""
        return r.json().get("response", "").strip()
    except Exception as e:
        logger.warning("ollama call failed: %s", e)
        return ""


def _parse_llm_body(
    body: str,
    title: str,
    window_label: str,
    total_minutes: float,
    pages: List[Dict],
) -> tuple:
    """Extract (markdown, key_points). Falls back to deterministic template."""
    if body and "## " in body:
        key_points = _extract_bullets(body, section="### What you did")
        return body.strip(), key_points

    md_lines = [
        f"## {title} — {window_label}",
        "",
        "**Focus**: see linked work pages",
        f"**Time**: {total_minutes} min across {len(pages)} work threads",
        "",
        "### What you did",
    ]
    key_points: List[str] = []
    for p in pages[:10]:
        mins = round(p.get("segment_count", 0) * 10 / 60.0, 1)
        bullet = f"- {p['title']} ({mins} min)"
        md_lines.append(bullet)
        key_points.append(p["title"])
    md_lines.extend([
        "",
        "### Open threads",
        "- _(none tracked)_",
        "",
        "### Next likely step",
        "- _(add one when you re-open this project)_",
    ])
    return "\n".join(md_lines), key_points


def _extract_bullets(markdown: str, section: str) -> List[str]:
    """Pull '- ...' lines under a named heading until the next blank+heading."""
    if section not in markdown:
        return []
    chunk = markdown.split(section, 1)[1]
    bullets: List[str] = []
    for line in chunk.splitlines():
        s = line.strip()
        if s.startswith("###"):
            break
        if s.startswith("- "):
            bullets.append(s[2:].strip())
    return bullets
