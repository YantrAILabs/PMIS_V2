# Memory System — Claude Desktop Instructions

You are operating inside a **Personal Memory Intelligence System**. This folder IS the user's second brain. Every conversation here builds, retrieves, and strengthens structured memory.

## Folder Structure

```
Memory/
├── CLAUDE.md              ← You're reading this
├── scripts/memory.py      ← Memory operations (zero external deps)
├── Graph_DB/
│   ├── graph.db           ← SQLite knowledge graph
│   └── tasks/             ← Individual task logs (JSON)
└── Vector_DB/             ← Serialized embeddings (managed by script)
```

## How Memory Works

Memory is hierarchical:
- **Super Context** = broad work domain (e.g., "B2B Cold Outreach")
- **Context** = specific phase (e.g., "Email Copywriting")  
- **Anchor** = atomic reusable insight (e.g., "CISOs respond to threat language over ROI")

Every anchor has a **weight** (0-1) representing how much it contributed to successful outcomes. Higher weight = retrieves first.

## Your Behavior Rules

### On EVERY new conversation in this folder:

1. **Auto-retrieve**: Before doing any work, run `python3 scripts/memory.py retrieve "<user's task description>"` to fetch relevant memories
2. **Show context**: Present the retrieved memories to the user briefly — what you know, what past approaches worked
3. **Do the work**: Execute the task using retrieved context
4. **Auto-store**: After completing work, run `python3 scripts/memory.py store "<task title>" "<key learnings and decisions, one per line>"` to save new memories

### Commands the user can say:

| User says | You do |
|-----------|--------|
| "retrieve context for X" | Run: `python3 scripts/memory.py retrieve "X"` |
| "store this" or "remember this" | Run: `python3 scripts/memory.py store "title" "content"` |
| "ingest this file" | Read file, then store key learnings via the store command |
| "show my memory" or "browse" | Run: `python3 scripts/memory.py browse` |
| "show memory tree" | Run: `python3 scripts/memory.py tree` |
| "visualize" | Run: `python3 scripts/memory.py viz` → then tell user to open Memory/memory.html |
| "stats" | Run: `python3 scripts/memory.py stats` |
| "score task X" | Run: `python3 scripts/memory.py score "task_id" "score"` |
| "rebuild weights" | Run: `python3 scripts/memory.py rebuild` |
| "what do you know about X" | Run retrieve, then synthesize an answer from the results |

### When storing memory:

You must do the STRUCTURING WORK. The script stores exactly what you give it. If you give flat text, you get flat memory. If you give intelligent structure, you get intelligent memory.

Pass a **single JSON argument** with this format:

```
python3 scripts/memory.py store '{
    "super_context": "B2B Cold Outreach",
    "description": "Enterprise sales campaigns for Vision AI",
    "contexts": [
        {
            "title": "Email copywriting",
            "weight": 0.8,
            "anchors": [
                {"title": "Threat-intel language wins", "content": "CISOs respond to threat language over ROI — 3x higher reply rate", "weight": 0.9},
                {"title": "Subject under 6 words", "content": "Short subjects get 40% higher open rates", "weight": 0.7},
                {"title": "Blind spot framing", "content": "Best subject line: Your blind spots are showing", "weight": 0.85}
            ]
        },
        {
            "title": "Target research",
            "weight": 0.6,
            "anchors": [
                {"title": "VP Security > CISO", "content": "VP Security title responds 2x more than CISO", "weight": 0.8},
                {"title": "LinkedIn Sales Navigator", "content": "Use LinkedIn to find 50+ enterprise targets", "weight": 0.6}
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
- Check existing SCs first: `python3 scripts/memory.py browse`

**2. Context grouping** — group related anchors under named subtasks.
- Ask: "If I were teaching someone this domain, what are the distinct skills/phases?"
- "Email copywriting" and "Target research" are different skills → separate contexts
- "Subject lines" and "Body copy" are both email writing → same context
- Never create date-stamped contexts like "Mar 18 session" — that's meaningless structure

**3. Anchor specificity** — each anchor is ONE atomic, reusable insight.
- Good: "CISOs respond to threat-intel language — 3x higher reply rate"
- Bad: "Good email copy" (too vague to reuse)
- Include numbers: "40% higher open rate" not "higher open rate"
- Include the WHY when you know it: "VP Security responds 2x more because they handle vendor eval directly"

**4. Weight assignment** — how important is each piece relative to its siblings?
- 0.9 = critical, this insight was the key driver of success
- 0.7 = important, consistently useful
- 0.5 = moderate, helpful but not decisive
- 0.3 = minor, nice to know
- Ask: "If I could only remember 2 things from this context, which would they be?" → those get 0.8-0.9

### Dedup behavior:

The script automatically handles duplicates:
- If a super context with the same name exists → reuses it, adds new contexts underneath
- If a context with the same name exists under the same SC → reuses it, adds new anchors
- If an anchor with the same title exists → updates use count, keeps the higher weight
- This means you can store learnings about "B2B Cold Outreach" 50 times and they all merge into one growing tree

### Example: storing from a conversation

After a conversation about writing cold emails to hospitals, you would store:

```
python3 scripts/memory.py store '{"super_context": "B2B Cold Outreach", "description": "Enterprise sales campaigns", "contexts": [{"title": "Healthcare vertical", "weight": 0.7, "anchors": [{"title": "NABH compliance drives decisions", "content": "Hospital camera placement is driven by NABH accreditation requirements, not security needs", "weight": 0.85}, {"title": "CFO signs but IT head evaluates", "content": "Technical decision maker is IT head, but budget approval comes from CFO", "weight": 0.7}]}, {"title": "Pricing strategy", "weight": 0.6, "anchors": [{"title": "Bundle maintenance for 30% uplift", "content": "Adding 3-year maintenance contract increases deal size by 30%", "weight": 0.8}]}], "summary": "Hospital security system proposal learnings"}'
```

Notice: the super context is "B2B Cold Outreach" (reuses existing domain), but the contexts are new ("Healthcare vertical", "Pricing strategy") with specific, weighted anchors.

### When retrieving:

The script returns a context pack with three signals per anchor:

1. **weight** — evidence-based, not just Claude's initial guess. Anchors that appeared in high-scoring tasks have higher weights. Anchors only in low-scoring tasks have lower weights.
2. **stage** — temporal classification: `impulse` (new, unproven), `active` (currently in use), `established` (battle-tested), `fading` (old, rarely used).
3. **in_winning_structure** — `true` if this anchor was present in the highest-scoring task for this super context. This is the "winning recipe."

When presenting retrieved context to the user:
- Lead with anchors marked `in_winning_structure: true` — these are proven
- Mention the temporal stage: "This is an established pattern" vs "This is a new untested insight"
- Note the winning task score: "This combination scored 4.5/5 last time"
- Show lower-weighted anchors too but mark them as less proven

### Weight system (evidence-based):

Weights are NOT static. They evolve through evidence:

- Initial weight comes from Claude's estimate when storing (the `weight` field you set)
- After tasks are scored, weights recompute: `weight = blend(initial, evidence)`
- With few scored tasks: initial estimate matters more (70% initial, 30% evidence)
- With 5+ scored tasks: evidence dominates (30% initial, 70% evidence)
- An anchor in only 5/5 tasks rises fast. An anchor in only 2/5 tasks drops fast.

### Scoring tasks:

After the user completes a task, prompt them to score it. Then run:
```
python3 scripts/memory.py score "TASK_ID" "4.5"
```

This does three things:
1. Captures a **structure snapshot** — the exact tree that produced this outcome
2. Updates **evidence-based weights** for every anchor in the task
3. Updates **super context quality** (running average of all task scores)

The task_id is printed when you run the store command. Always note it for the user.

### Temporal stages:

Every memory node is automatically classified:
- **impulse** — just created, used once, unproven. Retrieval boost: 0.8x (slight penalty)
- **active** — used multiple times recently. Retrieval boost: 1.2x (hot knowledge)
- **established** — used consistently over time. Retrieval boost: 1.5x (highest — battle-tested)
- **fading** — not used recently, low frequency. Retrieval boost: 0.5x (deprioritized)

Stages update automatically as nodes are used. Run `python3 scripts/memory.py rebuild` to force-recompute all temporal scores.

## Important

- Never ask "should I store this?" — just store key learnings automatically
- Never ask "should I retrieve?" — always retrieve before starting work
- Keep anchor titles short and specific — they're reusable knowledge atoms
- Always note the task_id after storing — the user needs it for scoring
- When the user says "that worked well" or "that went great" — immediately offer to score the task
- When the user says "that didn't work" — score it low so bad anchors get deprioritized
- Run `python3 scripts/memory.py rebuild` periodically (or at end of day) to refresh all weights
- The memory.py script has ZERO external dependencies — it uses only Python stdlib
