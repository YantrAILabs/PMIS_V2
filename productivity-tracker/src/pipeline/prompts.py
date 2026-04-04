"""
Prompt templates for ChatGPT Vision and text analysis.
Centralized here for easy iteration and tuning.
"""

# ─── Frame-level analysis (ChatGPT Vision) ─────────────────────────────

FRAME_BATCH_PROMPT = """Analyze these {count} consecutive screenshots from a user's work session on macOS.

For EACH frame, extract:
1. text: Key visible text content (code, chat messages, document text — not exhaustive OCR, just the meaningful content)
2. app: The application or website visible (e.g., "VS Code", "ChatGPT in Chrome", "Terminal")
3. task: What specific task the user is performing in this frame (e.g., "Writing a Python function", "Reading API documentation", "Reviewing a pull request")

Return as JSON:
{{
  "frames": [
    {{"text": "...", "app": "...", "task": "..."}},
    ...
  ]
}}

Be concise. Focus on what's actionable, not decorative UI elements."""


# ─── Segment-level synthesis (ChatGPT text) ─────────────────────────────

SEGMENT_SYNTHESIS_PROMPT = """You are analyzing a work session segment from a productivity tracker.

Segment ID: {segment_id}
Duration: {duration} seconds
Active window: {window_name} ({platform})
An autonomous agent was running: {agent_active}

Frame-by-frame extractions from this segment:
{frame_jsons}

Based on this data, classify this work segment:

1. supercontext: The HIGH-LEVEL work objective this contributes to.
   Examples: "Product Development", "Client Deliverable", "Internal Tooling", "Learning & Research", "Communication", "Operations"
   
2. context: The SPECIFIC work area within the supercontext.
   Examples: "Frontend Development", "API Integration", "Client Communication", "Architecture Design", "Code Review"

3. anchor: The PRECISE task being performed in this segment.
   Examples: "Building chart component for dashboard", "Debugging authentication flow", "Researching competitor pricing"

4. detailed_summary: 2-3 sentence summary of what was accomplished in this segment.

5. worker: Based on whether an autonomous agent was running:
   - "agent" if the agent flag is true AND the work appears to be agent-generated (code appearing without typing, automated test runs, etc.)
   - "human" if the user is actively directing the work (even if chatting with AI — that's human work)

6. medium: One of: browser, terminal, ide, chat, office, other

Return as JSON:
{{
  "supercontext": "...",
  "context": "...",
  "anchor": "...",
  "detailed_summary": "...",
  "worker": "human|agent",
  "medium": "..."
}}

Be precise with the hierarchy. The supercontext should be broad enough to group multiple contexts. The anchor should be specific enough to distinguish individual tasks."""


# ─── Hourly synthesis (for memory) ──────────────────────────────────────

HOURLY_SYNTHESIS_PROMPT = """Analyze this hour of work from a productivity tracker:

{segments_summary}

Produce a refined hierarchical summary:
1. Group related segments into supercontexts and contexts
2. For each anchor, note the total time and whether it was human or agent work
3. Identify any context that should be merged (e.g., "Debugging auth" and "Fixing auth tests" are both under "Authentication")

Return as JSON:
{{
  "hierarchy": [
    {{
      "supercontext": "...",
      "contexts": [
        {{
          "context": "...",
          "anchors": [
            {{"anchor": "...", "time_mins": ..., "worker": "human|agent"}}
          ]
        }}
      ]
    }}
  ]
}}"""


# ─── Daily synthesis (for final memory) ──────────────────────────────────

DAILY_SYNTHESIS_PROMPT = """Analyze a full day of work from a productivity tracker:

Date: {date}
Total tracked time: {total_mins} minutes

Hourly summaries:
{hourly_summaries}

Produce the final daily memory:
1. Merge all hourly hierarchies into a coherent day summary
2. Consolidate duplicate or overlapping contexts
3. Calculate final time totals per SC, context, and anchor
4. Identify the top 3 most significant accomplishments
5. Note any patterns (heavy context switching, long deep work sessions, etc.)

Return as JSON:
{{
  "date": "{date}",
  "total_tracked_mins": ...,
  "hierarchy": [...],
  "top_accomplishments": ["...", "...", "..."],
  "patterns": ["...", "..."]
}}"""
