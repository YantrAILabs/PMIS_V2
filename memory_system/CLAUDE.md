# Memory System — Claude Desktop Instructions

You are operating inside a **Personal Memory Intelligence System (PMIS)**. This folder IS the user's second brain. Every conversation here builds, retrieves, and strengthens structured memory.

## Folder Structure

```
Memory/
├── CLAUDE.md              ← You're reading this
├── scripts/memory.py      ← Memory operations (zero external deps)
├── Graph_DB/
│   ├── graph.db           ← SQLite knowledge graph
│   ├── .session.json      ← Active session state (auto-managed)
│   └── tasks/             ← Individual task logs (JSON)
└── Vector_DB/             ← Serialized embeddings (managed by script)
```

## How Memory Works

Memory is hierarchical:
- **Super Context** = broad work domain (e.g., "B2B Cold Outreach")
- **Context** = specific phase (e.g., "Email Copywriting")
- **Anchor** = atomic reusable insight (e.g., "CISOs respond to threat language over ROI")

Every anchor has a **weight** (0-1) that evolves through user feedback. Higher weight = retrieves first.

## Session-Aware Behavior

The memory system tracks **conversation sessions** automatically. A session = a continuous conversation about a related topic. Sessions end when the topic changes, the user leaves for hours, or the user says thanks.

### On EVERY user message:

**Step 1: Session Begin** — Run this FIRST, on every message:
```
python3 memory_system/scripts/memory.py session begin "<user's message or task description>"
```

This returns JSON with:
- `memories` — retrieved context (same as old retrieve)
- `session.is_new_session` — true if this is a fresh start
- `session.needs_rating` — true if previous conversation ended (topic change or time gap)
- `session.previous_session` — info about what the previous conversation was about
- `session.suggested_scs` — top 2-3 matching super contexts (only on new sessions)
- `session.should_ask_rating` — true every 3rd conversation turn (proactive feedback)
- `session.conversation_count` — how many turns in this session

**Step 2: Handle session signals** — Before presenting context, handle these in order:

1. **If `needs_rating: true`** — The previous conversation ended. Ask:
   > "Before we start — how was our last session on **[previous_session.sc_title]**? 👍 or 👎"
   - If user says 👍 or positive: `python3 memory_system/scripts/memory.py session rate up`
   - If user says 👎 or negative: `python3 memory_system/scripts/memory.py session rate down`
     - This returns a list of anchors used. Ask: "What specifically didn't work?"
     - Match their answer to anchor titles, then: `python3 memory_system/scripts/memory.py session rate down "anchor title 1, anchor title 2"`

2. **If `suggested_scs` has 2-3 close matches** — Ask:
   > "Are we working on **[SC1]** or **[SC2]**? Or is this something new?"
   - This helps narrow retrieval to the right domain immediately

3. **If `should_ask_rating: true`** — Proactively check in:
   > "Quick check — is the advice I'm pulling from memory helpful so far? 👍 or 👎"
   - Only ask if it feels natural in the conversation flow

**Step 3: Present context** — Show retrieved memories briefly:
- Lead with anchors marked `in_winning_structure: true`
- Mention temporal stage: "established pattern" vs "new untested insight"
- Note winning task score if available

**Step 4: Do the work** — Execute the task using retrieved context.

**Step 5: Store learnings** — After completing work, use `session store` (NOT bare `store`):
```
python3 memory_system/scripts/memory.py session store '{"super_context": "...", "contexts": [...]}'
```
This wraps the normal store AND tracks the task_id in the session for scoring.

### End-of-Conversation Detection

The system detects conversation endings **three ways**:

1. **Context shift** — If user's new message diverges >60% from the session topic, the system flags it automatically. You'll see `needs_rating: true` in the session begin response.

2. **Time gap** — If >3 hours pass between messages, the system treats the previous session as ended. You'll see `needs_rating: true` on the next session begin.

3. **Gratitude** — When user says "thanks", "thank you", "that's all", "goodbye", or similar:
   - Prompt for rating: "Glad to help! Quick — was this session helpful? 👍 or 👎"
   - Run `session rate up/down` based on response
   - Run `session end` to clean up: `python3 memory_system/scripts/memory.py session end`

### Commands the user can say:

| User says | You do |
|-----------|--------|
| "retrieve context for X" | Run: `python3 memory_system/scripts/memory.py session begin "X"` |
| "store this" or "remember this" | Run: `python3 memory_system/scripts/memory.py session store "..."` |
| "ingest this file" | Read file, then store key learnings via session store |
| "show my memory" or "browse" | Run: `python3 memory_system/scripts/memory.py browse` |
| "show memory tree" | Run: `python3 memory_system/scripts/memory.py tree` |
| "visualize" | Run: `python3 memory_system/scripts/memory.py viz` → tell user to open memory.html |
| "stats" | Run: `python3 memory_system/scripts/memory.py stats` |
| "rebuild weights" | Run: `python3 memory_system/scripts/memory.py rebuild` |
| "what do you know about X" | Run session begin, then synthesize an answer |

### When storing memory:

You must do the STRUCTURING WORK. The script stores exactly what you give it. If you give flat text, you get flat memory. If you give intelligent structure, you get intelligent memory.

Pass a **single JSON argument** with this format:

```
python3 memory_system/scripts/memory.py session store '{
    "super_context": "B2B Cold Outreach",
    "description": "Enterprise sales campaigns for Vision AI",
    "contexts": [
        {
            "title": "Email copywriting",
            "weight": 0.8,
            "anchors": [
                {"title": "Threat-intel language wins", "content": "CISOs respond to threat language over ROI — 3x higher reply rate", "weight": 0.9},
                {"title": "Subject under 6 words", "content": "Short subjects get 40% higher open rates", "weight": 0.7}
            ]
        }
    ],
    "summary": "Q1 cold outreach campaign learnings"
}'
```

### The 4 structuring decisions you MUST make:

**1. Super context name** — the broad work domain. Reusable across many tasks.
- Good: "B2B Cold Outreach", "Kiran AI Marketing", "Security Demo Page"
- Bad: "Email to Rajesh", "Tuesday's meeting", "Campaign v3"
- Check existing SCs first: `python3 memory_system/scripts/memory.py browse`

**2. Context grouping** — group related anchors under named subtasks.
- Ask: "If I were teaching someone this domain, what are the distinct skills/phases?"
- "Email copywriting" and "Target research" are different skills = separate contexts
- "Subject lines" and "Body copy" are both email writing = same context
- Never create date-stamped contexts like "Mar 18 session"

**3. Anchor specificity** — each anchor is ONE atomic, reusable insight.
- Good: "CISOs respond to threat-intel language — 3x higher reply rate"
- Bad: "Good email copy" (too vague to reuse)
- Include numbers: "40% higher open rate" not "higher open rate"
- Include the WHY when you know it

**4. Weight assignment** — how important is each piece relative to its siblings?
- 0.9 = critical, key driver of success
- 0.7 = important, consistently useful
- 0.5 = moderate, helpful but not decisive
- 0.3 = minor, nice to know

### Dedup behavior:

The script automatically handles duplicates:
- Same super context name exists → reuses it, adds new contexts underneath
- Same context name under same SC → reuses it, adds new anchors
- Same anchor title exists → updates use count, keeps the higher weight

### Weight system (evidence-based):

Weights evolve through user feedback:
- Initial weight = Claude's estimate when storing
- After thumbs up (4.0) or thumbs down (2.0), weights recompute
- With few ratings: initial estimate matters more (70% initial, 30% evidence)
- With 5+ ratings: evidence dominates (15% initial, 85% evidence)
- Thumbs down with follow-up selectively penalizes only the bad anchors

### Temporal stages:

Every memory node is automatically classified:
- **impulse** — just created, used once, unproven. Retrieval boost: 0.8x
- **active** — used multiple times recently. Retrieval boost: 1.2x
- **established** — used consistently over time. Retrieval boost: 1.5x
- **fading** — not used recently, low frequency. Retrieval boost: 0.5x

Stages update automatically as nodes are retrieved and stored.

## Important

- Never ask "should I store this?" — just store key learnings automatically
- Always run `session begin` before doing any work — never skip this
- When user says thanks → prompt for rating before closing
- When user expresses frustration or says something didn't work → treat as thumbs down, ask what specifically failed
- When user says "that worked well" → treat as thumbs up, run session rate up
- Keep anchor titles short and specific — they're reusable knowledge atoms
- The memory.py script has ZERO external dependencies — it uses only Python stdlib
- All old commands (retrieve, store, browse, tree, stats, score, rebuild, viz) still work as before
