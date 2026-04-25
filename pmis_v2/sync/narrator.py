"""Phase C — narrative composer. 1–4 daily stories from the day's salient
humanized work_pages.

Each narrative is a short journal entry with a heading and 2–4 bullets.
Every bullet cites exactly one source page id so a bullet can always be
traced back to the raw work_page it came from (prevents hallucination).

Model: Gemini 2.5 Flash primary (keyfile via sync.humanizer._resolve_gemini_key),
qwen2.5:7b fallback. thinkingBudget=0 so short-output calls don't get
truncated.

Regeneration rule: wipe the day's narratives, then write the fresh set —
simpler than diffing ordinals. Safe because narratives are derived data.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date as _date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("pmis.sync.narrator")

_STORY_SEP = "---STORY---"


def compose_narratives_for_date(
    db,
    hp: Dict,
    target_date: Optional[str] = None,
    user_id: str = "local",
    generated_by: str = "manual",
) -> Dict:
    """Build narratives over today's salient humanized work_pages. Returns
    a summary dict. Skips (no DB write) when there's nothing to narrate."""
    target_date = target_date or _date.today().isoformat()

    rows = db._conn.execute(
        """SELECT id, title, summary, humanized_summary, segment_count_hint,
                  project_id, deliverable_id
           FROM (
             SELECT wp.*,
                    (SELECT COUNT(*) FROM work_page_anchors
                     WHERE page_id = wp.id) as segment_count_hint
             FROM work_pages wp
             WHERE wp.user_id = ? AND wp.date_local = ?
               AND wp.salience = 'salient'
           )
           ORDER BY segment_count_hint DESC""",
        (user_id, target_date),
    ).fetchall()

    pages: List[Dict] = []
    for r in rows:
        d = dict(r)
        d["outcome"] = (d.get("humanized_summary") or d.get("summary") or "").strip()
        if not d["outcome"]:
            continue
        pages.append(d)

    if not pages:
        return {
            "status": "empty",
            "date": target_date,
            "reason": "no salient pages for this date",
            "narratives_written": 0,
        }

    text = _call_model(pages, hp)
    if not text:
        return {
            "status": "llm_unavailable",
            "date": target_date,
            "pages_considered": len(pages),
            "narratives_written": 0,
        }

    stories = _parse_stories(text, pages)
    if not stories:
        return {
            "status": "no_stories_parsed",
            "date": target_date,
            "pages_considered": len(pages),
            "raw_llm_output": text[:500],
            "narratives_written": 0,
        }

    db.clear_narratives_for_date(target_date, user_id=user_id)
    written = 0
    for ordinal, story in enumerate(stories):
        db.upsert_narrative(
            date_local=target_date,
            ordinal=ordinal,
            heading=story["heading"],
            body_markdown=story["body"],
            source_page_ids=story["source_page_ids"],
            project_id=story.get("project_id", ""),
            generated_by=generated_by,
            user_id=user_id,
        )
        written += 1

    return {
        "status": "ok",
        "date": target_date,
        "pages_considered": len(pages),
        "narratives_written": written,
        "generated_by": generated_by,
        "stories": [
            {"heading": s["heading"], "source_page_ids": s["source_page_ids"]}
            for s in stories
        ],
    }


def _call_model(pages: List[Dict], hp: Dict) -> str:
    prompt = _build_prompt(pages)
    cloud_ok = bool(hp.get("humanize_use_cloud", True))

    from sync.humanizer import _resolve_gemini_key, _call_gemini, _call_ollama
    api_key = _resolve_gemini_key()
    if cloud_ok and api_key:
        text = _call_gemini(
            prompt, api_key,
            model=hp.get("humanize_model_cloud", "gemini-2.5-flash"),
            timeout_s=45,
        )
        if text:
            return text

    text = _call_ollama(
        prompt,
        model=hp.get("humanize_model_local", "qwen2.5:7b"),
        timeout_s=90,
    )
    return text or ""


def _build_prompt(pages: List[Dict]) -> str:
    bullets = []
    for i, p in enumerate(pages):
        pid = p.get("id") or ""
        outcome = p.get("outcome") or ""
        title = p.get("title") or ""
        bullets.append(f"[{i + 1}] id={pid} · {title}: {outcome}")

    joined = "\n".join(bullets)

    return (
        "You compose a small number (1 to 4) of daily JOURNAL STORIES from a\n"
        "worker's outcomes for the day. Each story groups thematically related\n"
        "outcomes into a short entry with a heading and 2–4 bullets.\n"
        "\n"
        "Outcomes (one per line, numbered):\n"
        f"{joined}\n"
        "\n"
        "Rules (strict):\n"
        "1. Produce 1 to 4 stories. Fewer is better if everything is one theme.\n"
        "2. Each story begins with a line: \"## <heading>\" (under 60 chars, noun-phrase).\n"
        "3. Each story has 2 to 4 bullets. Each bullet is a single second-person\n"
        "   past-tense sentence, under 25 words, referencing the outcome.\n"
        "4. EVERY bullet must end with its source reference in square brackets,\n"
        "   citing exactly one numbered outcome, e.g. \"[3]\". No bullet without a\n"
        "   citation. No bullet citing more than one outcome.\n"
        "5. Ground strictly in the outcomes above. Do NOT invent people, projects,\n"
        "   or work that isn't present. If an outcome is vague, keep the bullet vague.\n"
        "6. Between stories, output a line that is exactly: ---STORY---\n"
        "\n"
        "Output only the stories in this format. No preamble, no outro.\n"
    )


def _parse_stories(text: str, pages: List[Dict]) -> List[Dict]:
    """Split Gemini output on ---STORY--- markers, extract heading + bullets
    + source citations. Bullets without citations are dropped."""
    id_by_num = {i + 1: p["id"] for i, p in enumerate(pages)}
    bullet_ref = re.compile(r"\[(\d+)\]")

    # Models sometimes prepend or append prose; trim to the first ## line.
    first_h = text.find("## ")
    if first_h > 0:
        text = text[first_h:]

    blocks = [b.strip() for b in text.split(_STORY_SEP)]
    stories: List[Dict] = []
    for block in blocks:
        if not block:
            continue
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        heading = ""
        bullets: List[str] = []
        cited_ids: List[str] = []
        seen: set = set()
        for line in lines:
            if line.startswith("## "):
                heading = line[3:].strip()[:120]
                continue
            if not (line.startswith("- ") or line.startswith("* ")):
                continue
            body = line[2:].strip()
            m = bullet_ref.search(body)
            if not m:
                continue
            num = int(m.group(1))
            pid = id_by_num.get(num)
            if not pid:
                continue
            # strip the [n] marker from the final stored bullet for readability
            clean = bullet_ref.sub("", body).strip().rstrip(",.;:") + "."
            bullets.append(f"- {clean}")
            if pid not in seen:
                cited_ids.append(pid)
                seen.add(pid)

        if not heading or not bullets:
            continue
        stories.append({
            "heading": heading,
            "body": "\n".join(bullets),
            "source_page_ids": cited_ids,
        })

    return stories[:4]
