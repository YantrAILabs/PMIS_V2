"""
Prompt templates for vision + text analysis.

These are intentionally terse. The OpenAI call already passes
`response_format={"type": "json_object"}` so we don't need to reiterate
"Return JSON" or show the schema more than once.
"""

# ─── Frame-level analysis (vision) ────────────────────────────────────

FRAME_BATCH_PROMPT = """For each of {count} screenshots, extract:
- text: key visible content (code/chat/doc text; not exhaustive OCR)
- app: app or site (e.g. "VS Code", "Chrome/Gmail")
- task: short phrase of what the user is doing

JSON: {{"frames":[{{"text":"...","app":"...","task":"..."}}, ...]}}"""


# ─── Segment-level synthesis (text) ───────────────────────────────────

SEGMENT_SYNTHESIS_PROMPT = """Summarize this work segment.
Window: {window_name} ({platform}) · Duration: {duration}s · Agent active: {agent_active}

Per-frame extractions:
{frame_jsons}

Output JSON fields:
- short_title: ≤10 words, no prefix label, no trailing punctuation (e.g. "Drafting CISO email in Gmail")
- detailed_summary: 2-3 sentences on what was accomplished
- full_text: complete record — file names, topics, pages visited
- worker: "agent" if agent flag true AND work looks agent-driven, else "human"
- medium: one of browser|terminal|ide|chat|office|other"""
