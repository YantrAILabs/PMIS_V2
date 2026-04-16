"""
Prompt templates for Ollama Vision and text analysis.
Centralized here for easy iteration and tuning.
"""

# ─── Frame-level analysis (Ollama Vision) ─────────────────────────────

FRAME_BATCH_PROMPT = """Analyze these {count} consecutive screenshots from a user's work session on macOS.

For EACH frame, extract:
1. text: Key visible text content (code, chat messages, document text — not exhaustive OCR, just the meaningful content)
2. app: The application or website visible (e.g., "VS Code", "ChatGPT in Chrome", "Terminal")
3. task: What specific task the user is performing in this frame (e.g., "Writing a Python function", "Reading API documentation", "Reviewing a pull request")

Return ONLY valid JSON with no other text:
{{
  "frames": [
    {{"text": "...", "app": "...", "task": "..."}},
    ...
  ]
}}

Be concise. Focus on what's actionable, not decorative UI elements."""


# ─── Segment-level synthesis (Ollama text) ─────────────────────────────

SEGMENT_SYNTHESIS_PROMPT = """You are analyzing a work session segment from a productivity tracker.

Segment ID: {segment_id}
Duration: {duration} seconds
Active window: {window_name} ({platform})
An autonomous agent was running: {agent_active}

Frame-by-frame extractions from this segment:
{frame_jsons}

Based on this data, produce:

1. detailed_summary: 2-3 sentence summary of what was accomplished in this segment.

2. full_text: Complete description of ALL work done during this segment. Include specific details like file names, functions, topics discussed, pages visited. This serves as a permanent record.

3. worker: Based on whether an autonomous agent was running:
   - "agent" if the agent flag is true AND the work appears to be agent-generated
   - "human" if the user is actively directing the work

4. medium: One of: browser, terminal, ide, chat, office, other

Return ONLY valid JSON with no other text:
{{
  "detailed_summary": "...",
  "full_text": "...",
  "worker": "human|agent",
  "medium": "..."
}}"""
