# Productivity Tracker — Architecture & Implementation Plan

## 1. System summary

A Mac-native productivity system that captures screen activity, extracts work context via ChatGPT Vision API, records it in a PMIS v2 hierarchy (SC → Context → Anchor), matches it against assigned deliverables, and surfaces time allocation + human vs agent contribution via a dashboard. Managed and operated entirely through Claude Desktop via MCP.

---

## 2. Core distinction: Human vs Agent

The system tracks a single binary: **is the user doing the work, or is an autonomous agent acting on their behalf?**

| Classification | Definition | Examples |
|----------------|-----------|----------|
| **Human** | User is in the driver's seat — typing, navigating, reading, chatting with AI, copy-pasting, editing AI output | Writing code in VS Code, chatting with ChatGPT, browsing docs, reviewing PRs, manual terminal commands |
| **Agent** | An autonomous process is executing tasks without continuous user input | Claude Code running a task, Cursor agent mode, Aider auto-fix, GitHub Copilot Workspace, any MCP-driven automation |

**Why this distinction matters:** Using ChatGPT to get an answer and then applying it is *human work* — the user is directing, evaluating, deciding. An agent running `claude code` that writes 15 files and runs tests autonomously is *agent work* — the user kicked it off but the machine is executing.

**Detection method:** Process-level monitoring. Check for known agent process signatures running alongside the active window. See `src/agent/agent_detector.py` for implementation.

---

## 3. AI pipeline: ChatGPT Vision (single provider)

All visual analysis runs through ChatGPT API. Two-stage, same provider:

| Stage | Model | Trigger | Purpose | Est. cost/day |
|-------|-------|---------|---------|---------------|
| Frame extraction | `gpt-4o-mini` (vision) | Per frame (batched) | OCR, task ID, app detection | ~$0.30-0.50 |
| Segment synthesis | `gpt-4o-mini` | Per segment completion | SC/Context/Anchor, summary | ~$0.02-0.05 |

**Batching strategy:** Don't send every frame individually. Batch 3-5 frames per API call with a multi-image prompt. Cuts API calls by 3-5x while maintaining context.

**Cost optimization:**
- Skip near-identical frames (SSIM > 0.95 with previous = skip)
- Lower resolution for OCR (resize to 1024px wide before sending)
- Use `detail: "low"` for initial classification, `detail: "high"` only when text is small

---

## 4. Layer-by-layer architecture

### Layer 1 — Tracking agent (Mac M2)

| Signal | Method | Frequency |
|--------|--------|-----------|
| Screenshots | `screencapture -x` or CGWindowListCreateImage | Every 10s (active), 30s (low activity) |
| Active window | NSWorkspace + Accessibility API (window title, bundle ID) | Event-driven (on change) |
| Activity density | CGEventTap — keystrokes/sec, mouse movement magnitude | Continuous |
| Agent detection | Process table scan for known agent signatures | Every 10s (aligned with screenshots) |

Runs as Python daemon via `launchd`. Privacy controls in `config/privacy.yaml`.

### Layer 2 — Context extraction pipeline

#### Stage 1: Target frame segmentation

New segment starts when:
1. Active window changes (app or title)
2. Visual diff between frames is significant (SSIM < 0.7)
3. Idle gap > 5 minutes
4. Agent process starts or stops (agent boundary = segment boundary)

Segment ID: `TS-{YYYYMMDD}-{4-digit-seq}`

#### Stage 2: ChatGPT Vision analysis

**Per-frame batch (3-5 frames):**
```
Analyze these consecutive screenshots from a work session.
For each frame extract:
1. Visible text/code (key content, not exhaustive OCR)
2. Application and website/page visible  
3. Specific task the user is performing
Return as JSON array.
```

**Per-segment synthesis (on segment completion):**
```
Given frame extractions from work segment {segment_id}:
{frame_jsons}

Agent was active during this segment: {yes/no}

Produce:
1. supercontext: High-level work objective
2. context: Specific work area
3. anchor: Precise task being performed
4. detailed_summary: 2-3 sentence summary
5. worker: "human" or "agent"
Return as JSON.
```

### Layer 3 — Recording

**Table Context 1 (segment-level):**

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | PK |
| timestamp_start | datetime | Segment start |
| timestamp_end | datetime | Segment end |
| window_name | text | Active window |
| platform | text | App/website |
| medium | text | browser/terminal/ide/chat/office/other |
| context | text | Work area |
| supercontext | text | High-level objective |
| anchor | text | Specific task |
| target_segment_id | text | TS-YYYYMMDD-NNNN (unique) |
| target_segment_length_secs | int | Duration |
| worker | text | "human" or "agent" |
| detailed_summary | text | ChatGPT summary |

**Table Context 2 (frame-level):**

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | PK |
| target_segment_id | text | FK to context_1 |
| target_frame_number | int | Sequence in segment |
| frame_timestamp | datetime | Capture time |
| raw_text | text | Extracted text |
| detailed_summary | text | Frame task |
| worker | text | "human" or "agent" |

### Layer 4 — Memory pipeline

**Hourly → temp table:**
- Aggregate segments by (SC, Context, Anchor)
- Sum times, split human/agent minutes
- Generate embeddings, store in ChromaDB

**Daily → final memory:**
- Merge hourly tables into hierarchy
- SC₁ → Context₁ → Anchor₁ (with times + worker split)
- Store in PMIS v2 (ChromaDB + SQLite)
- Delete hourly temps

**Time roll-up:** Anchor time flows up to Context, Context time flows up to SC. Each level's time = sum of children.

### Layer 5 — Matching engine

**Deliverables from two sources:**
1. `config/deliverables.yaml` — manual definition
2. Asana/Notion API — project tool sync

**Matching:** exact SC match first, then embedding similarity (>0.8 threshold).

**Output per deliverable:** total time, human time, agent time, anchor breakdown, unmatched work.

### Layer 6 — Central memory (PMIS v2)

Daily memory merges into central PMIS v2. User's SC may attach at any level in central memory (SC/Context/Anchor) based on embedding similarity. Contribution tracking marks what delivered value vs what didn't.

### Layer 7 — Claude Desktop interface (MCP)

All interaction goes through Claude Desktop via MCP server. Available tools:

| Tool | Description |
|------|-------------|
| `get_status` | Current tracking state, active window, running segment |
| `get_daily_summary` | Today's time breakdown by SC/Context/Anchor |
| `get_deliverable_progress` | Progress per deliverable with time + worker split |
| `search_productivity_memory` | Semantic search across productivity data |
| `add_deliverable` | Add a new deliverable |
| `pause_tracking` / `resume_tracking` | Control tracking |
| `get_contribution_report` | What contributed to delivery vs what didn't |
| `get_trends` | Weekly/monthly productivity patterns |

---

## 5. Project structure

```
productivity-tracker/
├── PLAN.md                          # This file
├── README.md                        # Setup + usage guide
├── pyproject.toml                   # Dependencies
├── .env.example                     # API keys template
├── config/
│   ├── settings.yaml                # Main config (intervals, thresholds)
│   ├── deliverables.yaml            # Manual deliverables definition
│   └── privacy.yaml                 # App/window exclusions
├── src/
│   ├── __init__.py
│   ├── agent/                       # Layer 1: Tracking
│   │   ├── __init__.py
│   │   ├── tracker.py               # Main daemon orchestrator
│   │   ├── screenshot.py            # Screenshot capture
│   │   ├── window_monitor.py        # Active window detection
│   │   ├── activity_monitor.py      # Keystroke/mouse density
│   │   └── agent_detector.py        # Detect autonomous agents
│   ├── pipeline/                    # Layer 2: Context extraction
│   │   ├── __init__.py
│   │   ├── segmenter.py             # Target frame segmentation
│   │   ├── frame_analyzer.py        # ChatGPT Vision frame analysis
│   │   ├── context_classifier.py    # ChatGPT segment synthesis
│   │   └── prompts.py               # All prompt templates
│   ├── storage/                     # Layer 3: Recording
│   │   ├── __init__.py
│   │   ├── models.py                # SQLAlchemy table definitions
│   │   ├── db.py                    # DB setup + session management
│   │   └── chromadb_store.py        # Embedding storage
│   ├── memory/                      # Layer 4: Memory pipeline
│   │   ├── __init__.py
│   │   ├── hourly_aggregator.py     # Hourly temp table builder
│   │   ├── daily_rollup.py          # End-of-day hierarchy builder
│   │   ├── pmis_integration.py      # PMIS v2 storage interface
│   │   └── central_merge.py         # Central memory merge logic
│   ├── matching/                    # Layer 5: Matching
│   │   ├── __init__.py
│   │   ├── deliverables_loader.py   # YAML + API deliverable input
│   │   ├── matching_engine.py       # Work ↔ deliverable matching
│   │   └── contribution_tracker.py  # Mark what contributed
│   ├── mcp/                         # Layer 7: Claude Desktop interface
│   │   ├── __init__.py
│   │   └── server.py                # MCP server with all tools
│   └── ui/
│       └── api.py                   # FastAPI for React dashboard
├── skills/
│   └── productivity-tracker/
│       └── SKILL.md                 # Claude Desktop skill definition
├── scripts/
│   ├── install.sh                   # One-command setup
│   └── start_tracker.sh             # Launch daemon + MCP server
├── tests/
│   ├── test_segmenter.py
│   ├── test_analyzer.py
│   ├── test_matching.py
│   └── test_memory.py
└── claude_desktop_config.json       # Config snippet for Claude Desktop
```

---

## 6. Implementation phases

| Phase | Scope | Duration |
|-------|-------|----------|
| P1 | Tracking agent + screenshot + window + agent detector + tables | Week 1-2 |
| P2 | ChatGPT Vision pipeline + segmentation + classification | Week 3-4 |
| P3 | Memory pipeline (hourly/daily) + PMIS v2 integration | Week 5-6 |
| P4 | Matching engine + deliverables + MCP server for Claude Desktop | Week 7-8 |
| P5 | Central memory merge + contribution tracking + dashboard UI | Week 9-10 |

---

## 7. Cost estimate (daily, 8hr workday)

| Component | Volume | Cost |
|-----------|--------|------|
| ChatGPT Vision (frame batches) | ~600 batches × 3-5 frames | ~$0.30-0.50 |
| ChatGPT text (segment synthesis) | ~50-80 calls | ~$0.02-0.05 |
| ChromaDB (local) | ~200-300 embeddings | $0 |
| SQLite (local) | ~10MB/day | $0 |
| **Total** | | **~$0.35-0.55/day** |
