# Productivity Tracker — Implementation Plan

**Project path:** `~/Desktop/Memory/productivity-tracker/`
**Execution environment:** Claude Desktop + Mac M2 terminal
**Total files:** 42 | **Total lines:** ~4,200

---

## Phase 0: Environment setup (Day 1, ~30 mins)

### 0.1 Extract and position the project

```bash
cd ~/Desktop/Memory/
# If zip was downloaded, extract it here
# The folder should be: ~/Desktop/Memory/productivity-tracker/
```

### 0.2 Create virtual environment and install dependencies

```bash
cd ~/Desktop/Memory/productivity-tracker
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Validate:** `python3 -c "import openai, sqlalchemy, chromadb, PIL, skimage; print('OK')"` → should print OK

### 0.3 Set API key

```bash
cp .env.example .env
# Edit .env → set OPENAI_API_KEY=sk-...
```

### 0.4 Grant macOS permissions

Open **System Settings → Privacy & Security**:
- **Screen Recording** → add Terminal (or your terminal app)
- **Accessibility** → add Terminal

Without these, `screencapture` and `osascript` for window detection will fail silently.

### 0.5 Create data directories

```bash
mkdir -p ~/.productivity-tracker
mkdir -p /tmp/productivity-tracker/frames
```

### 0.6 Verify PMIS v2 path

The tracker's central memory merge expects your PMIS v2 ChromaDB at the path configured in `config/settings.yaml → pmis.chromadb_path`. Default is `~/.pmis-v2/chromadb`.

```bash
# Check your actual PMIS v2 path and update settings.yaml if different
ls ~/.pmis-v2/chromadb 2>/dev/null || echo "Path doesn't exist yet — will be created on first merge"
```

**Edit `config/settings.yaml`** if your PMIS v2 ChromaDB lives elsewhere.

---

## Phase 1: Tracking agent — capture layer (Days 2-4)

**Goal:** Screenshot capture + window detection + idle detection + agent detection working independently.

### 1.1 Test screenshot capture

**File:** `src/agent/screenshot.py`

```bash
cd ~/Desktop/Memory/productivity-tracker
source .venv/bin/activate
python3 -c "
from src.agent.screenshot import ScreenshotCapture
cap = ScreenshotCapture({'tracking': {'screenshot_max_width': 1024}})
path = cap.capture()
print(f'Screenshot saved: {path}')
import os
print(f'Size: {os.path.getsize(path)} bytes')
"
```

**Expected:** A `.jpg` file at `/tmp/productivity-tracker/frames/frame_*.jpg`, resized to 1024px wide.

**If it fails:** Check Screen Recording permission. Try `screencapture -x /tmp/test.png` manually.

### 1.2 Test window detection

**File:** `src/agent/window_monitor.py`

```bash
python3 -c "
from src.agent.window_monitor import WindowMonitor
wm = WindowMonitor()
wm.start()
info = wm.get_active_window()
print(f'App: {info[\"app_name\"]}')
print(f'Bundle: {info[\"bundle_id\"]}')
print(f'Title: {info[\"title\"]}')
print(f'Platform: {wm.extract_platform(info)}')
print(f'Is browser: {wm.is_browser(info)}')
"
```

**Expected:** Prints your current active window info. Switch to Chrome, run again — should show Chrome details with page title.

**If it fails:** Check Accessibility permission. The `osascript` call needs it for `System Events`.

### 1.3 Test idle detection

**File:** `src/agent/activity_monitor.py`

```bash
python3 -c "
import time
from src.agent.activity_monitor import ActivityMonitor
am = ActivityMonitor({'tracking': {'idle_threshold_secs': 5}})
am.start()
print('Move your mouse...')
time.sleep(2)
print(f'Idle: {am.is_idle()} (should be False)')
print(f'Idle duration: {am.idle_duration():.1f}s')
print('Now stop touching everything for 6 seconds...')
time.sleep(7)
print(f'Idle: {am.is_idle()} (should be True)')
am.stop()
"
```

### 1.4 Test agent detection

**File:** `src/agent/agent_detector.py`

```bash
python3 -c "
from src.agent.agent_detector import AgentDetector
ad = AgentDetector()
print(f'Agent active: {ad.is_agent_active()}')
print(f'Agent name: {ad.get_active_agent_name()}')
# Test window-based detection
print(AgentDetector.is_agent_window({'title': 'Claude Code — fixing auth'}))  # True
print(AgentDetector.is_agent_window({'title': 'VS Code'}))                     # False
"
```

**To test with a real agent:** Open a terminal, run `claude code` (or just `sleep 100 &` to simulate), then re-run the detector.

### 1.5 Phase 1 validation

All four subsystems work independently. Run the tests:

```bash
cd ~/Desktop/Memory/productivity-tracker
source .venv/bin/activate
python3 -m pytest tests/test_segmenter.py tests/test_analyzer.py -v
```

---

## Phase 2: Context extraction pipeline (Days 5-8)

**Goal:** Frame segmentation + ChatGPT Vision analysis + SC/Context/Anchor classification working.

### 2.1 Test segmentation logic

**File:** `src/pipeline/segmenter.py`

The segmenter doesn't need real images for logic testing — the unit tests mock image loading:

```bash
python3 -m pytest tests/test_segmenter.py -v
```

All 9 tests should pass. Key behaviors to verify:
- First frame always starts a new segment
- Window change triggers new segment
- Agent state flip triggers new segment
- State updates after every call (the C2 fix)
- Counter persists across `load_last_segment_counter`

### 2.2 Test ChatGPT Vision frame analysis

**File:** `src/pipeline/frame_analyzer.py`
**Costs:** ~$0.01 per test call

This is the first API call. Capture a real screenshot and analyze it:

```bash
python3 -c "
import asyncio
from src.agent.screenshot import ScreenshotCapture
from src.pipeline.frame_analyzer import FrameAnalyzer
from datetime import datetime

cap = ScreenshotCapture({'tracking': {'screenshot_max_width': 1024}})
path = cap.capture()
print(f'Captured: {path}')

analyzer = FrameAnalyzer({
    'chatgpt': {'vision_model': 'gpt-4o-mini', 'vision_detail': 'low', 'max_tokens': 500}
})

frame = {'path': path, 'timestamp': datetime.now(), 'frame_number': 1}

async def test():
    result = await analyzer.analyze_single(frame)
    print(f'App: {result.get(\"app\")}')
    print(f'Task: {result.get(\"task\")}')
    print(f'Text: {result.get(\"text\", \"\")[:200]}')

asyncio.run(test())
"
```

**Expected:** JSON with `app`, `task`, `text` fields matching what's on your screen.

**Tuning point:** If text extraction is poor, try changing `vision_detail` to `"high"` in `config/settings.yaml`. Costs more but reads small text better.

### 2.3 Test context classification

**File:** `src/pipeline/context_classifier.py`
**File:** `src/pipeline/prompts.py`

Feed the classifier a mock segment with multiple frame results:

```bash
python3 -c "
import asyncio, json
from src.pipeline.context_classifier import ContextClassifier

classifier = ContextClassifier({
    'chatgpt': {'text_model': 'gpt-4o-mini', 'max_tokens': 500, 'temperature': 0.1}
})

mock_frames = [
    {'target_frame_number': 1, 'detailed_summary': 'Reading Python docs about asyncio', 'worker_type': 'human'},
    {'target_frame_number': 2, 'detailed_summary': 'Writing async function in tracker.py', 'worker_type': 'human'},
    {'target_frame_number': 3, 'detailed_summary': 'Running pytest in terminal', 'worker_type': 'human'},
]

async def test():
    result = await classifier.classify_segment(
        segment_id='TS-20260402-0001',
        frame_results=mock_frames,
        window_info={'title': 'VS Code — tracker.py', 'app_name': 'VS Code'},
        agent_active=False,
    )
    print(json.dumps(result, indent=2))

asyncio.run(test())
"
```

**Expected:** JSON with `supercontext`, `context`, `anchor`, `worker: "human"`, `medium: "ide"`.

### 2.4 Prompt tuning checkpoint

This is where you decide if the SC/Context/Anchor assignments are sensible. Run 5-10 real segments through the classifier and check:

- Are supercontexts too broad? ("Work" is useless. "Product Development" is good.)
- Are anchors too vague? ("Coding" is useless. "Building chart component for dashboard" is good.)
- Does the human/agent split match reality?

**Edit `src/pipeline/prompts.py`** if the hierarchy isn't landing right. The prompts are the most important tuning surface in the system.

---

## Phase 3: Storage layer (Days 9-10)

**Goal:** SQLite tables + ChromaDB collections working, data persists correctly.

### 3.1 Initialize database and verify schema

**Files:** `src/storage/models.py`, `src/storage/db.py`

```bash
python3 -c "
from src.storage.db import Database
db = Database()
db.initialize()

# Verify tables exist
from sqlalchemy import inspect
insp = inspect(db.engine)
tables = insp.get_table_names()
print(f'Tables: {tables}')
assert 'context_1' in tables
assert 'context_2' in tables
assert 'hourly_memory' in tables
assert 'daily_memory' in tables
assert 'deliverables' in tables
print('All 5 tables created.')
"
```

### 3.2 Test insert/query cycle

```bash
python3 -c "
from datetime import datetime
from src.storage.db import Database

db = Database()
db.initialize()

# Insert a test segment
db.insert_context1(
    timestamp_start=datetime.now(),
    timestamp_end=datetime.now(),
    window_name='VS Code — test.py',
    platform='VS Code',
    medium='ide',
    context='Testing',
    supercontext='Internal Tooling',
    anchor='Writing unit tests',
    target_segment_id='TS-20260402-9999',
    target_segment_length_secs=120,
    worker='human',
    detailed_summary='Test insert',
)

# Query it back
from datetime import date
segments = db.get_segments_for_date(date.today().strftime('%Y-%m-%d'))
print(f'Segments today: {len(segments)}')
print(f'Latest: {segments[-1][\"anchor\"]}')

# Clean up test data
with db.get_session() as s:
    from src.storage.models import Context1
    s.query(Context1).filter_by(target_segment_id='TS-20260402-9999').delete()
    s.commit()
print('Cleanup done.')
"
```

### 3.3 Test ChromaDB embeddings

**File:** `src/storage/chromadb_store.py`

```bash
python3 -c "
from src.storage.chromadb_store import ChromaDBStore
import yaml

with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

store = ChromaDBStore(config)

# Test embed + store + search
store.store_daily(
    entry_id='test-001',
    text='Product Development > Frontend > Building chart components',
    metadata={'date': '2026-04-02', 'level': 'anchor'},
)

results = store.search_daily('frontend chart work')
print(f'Found: {len(results)} results')
print(f'Best match: {results[0][\"document\"]}')

# Clean up
store.daily.delete(ids=['test-001'])
print('Cleanup done.')
"
```

### 3.4 Verify PMIS isolation

```bash
python3 -c "
import os
# Tracker ChromaDB
tracker_path = os.path.expanduser('~/.productivity-tracker/chromadb')
# PMIS ChromaDB
pmis_path = os.path.expanduser('~/.pmis-v2/chromadb')

print(f'Tracker DB: {tracker_path}')
print(f'PMIS DB:    {pmis_path}')
print(f'Same path?  {os.path.realpath(tracker_path) == os.path.realpath(pmis_path)}')
# Must be False
"
```

### 3.5 Load deliverables

```bash
python3 -c "
from src.storage.db import Database
from src.matching.deliverables_loader import DeliverablesLoader
import yaml

with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

db = Database()
db.initialize()
loader = DeliverablesLoader(db, config)
loader.load_from_yaml()

deliverables = db.get_active_deliverables()
for d in deliverables:
    print(f'  {d[\"id\"]}: {d[\"name\"]} ({d[\"supercontext\"]})')
"
```

**Edit `config/deliverables.yaml`** with your actual current deliverables before moving to Phase 4.

---

## Phase 4: Memory pipeline (Days 11-14)

**Goal:** Hourly aggregation + daily rollup + PMIS merge working end-to-end.

### 4.1 Seed test data

Before testing the memory pipeline, seed some realistic segments:

```bash
python3 -c "
from datetime import datetime, timedelta
from src.storage.db import Database

db = Database()
db.initialize()

# Simulate 2 hours of work with 8 segments
segments = [
    ('Product Development', 'Frontend', 'Building chart components', 'human', 600),
    ('Product Development', 'Frontend', 'Building chart components', 'agent', 900),
    ('Product Development', 'Frontend', 'Styling dashboard layout', 'human', 450),
    ('Product Development', 'API Integration', 'Debugging auth flow', 'human', 1200),
    ('Client Deliverable', 'First Club', 'QC pipeline testing', 'human', 300),
    ('Client Deliverable', 'First Club', 'QC pipeline testing', 'agent', 600),
    ('Internal Tooling', 'PMIS v2', 'Memory retrieval pipeline', 'human', 480),
    ('Learning', 'Reading Papers', 'Attention mechanism survey', 'human', 360),
]

now = datetime.now()
for i, (sc, ctx, anchor, worker, secs) in enumerate(segments):
    start = now - timedelta(hours=2) + timedelta(minutes=i*15)
    db.insert_context1(
        timestamp_start=start,
        timestamp_end=start + timedelta(seconds=secs),
        window_name=f'Test Window {i}',
        platform='VS Code',
        medium='ide',
        context=ctx,
        supercontext=sc,
        anchor=anchor,
        target_segment_id=f'TS-{now.strftime(\"%Y%m%d\")}-{8000+i:04d}',
        target_segment_length_secs=secs,
        worker=worker,
        detailed_summary=f'Test segment: {anchor}',
    )
print(f'Seeded {len(segments)} test segments.')
"
```

### 4.2 Test hourly aggregation

**File:** `src/memory/hourly_aggregator.py`

```bash
python3 -c "
import asyncio, yaml
from src.storage.db import Database
from src.memory.hourly_aggregator import HourlyAggregator

with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

db = Database()
db.initialize()
agg = HourlyAggregator(db, config)

asyncio.run(agg.run())

# Check results
from datetime import date
hourly = db.get_hourly_for_date(date.today().strftime('%Y-%m-%d'))
print(f'Hourly entries: {len(hourly)}')
for h in hourly:
    print(f'  {h[\"supercontext\"]} > {h[\"context\"]} > {h[\"anchor\"]}: {h[\"time_mins\"]:.1f} min (H:{h[\"human_mins\"]:.1f} A:{h[\"agent_mins\"]:.1f})')
"
```

**Expected:** Groups of segments aggregated by SC/Context/Anchor with summed times.

### 4.3 Test daily rollup

**File:** `src/memory/daily_rollup.py`

```bash
python3 -c "
import asyncio, json, yaml
from src.storage.db import Database
from src.memory.daily_rollup import DailyRollup
from datetime import date

with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

db = Database()
db.initialize()
rollup = DailyRollup(db, config)

asyncio.run(rollup.run())

# Check hierarchy
hierarchy = rollup.get_daily_hierarchy(date.today().strftime('%Y-%m-%d'))
print(json.dumps(hierarchy, indent=2, default=str))
"
```

**Expected:** Nested hierarchy with SC → Context → Anchor, times rolled up correctly.

### 4.4 Test central merge

**File:** `src/memory/central_merge.py`

```bash
python3 -c "
import yaml
from src.storage.db import Database
from src.memory.central_merge import CentralMerge
from datetime import date

with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

db = Database()
db.initialize()
merge = CentralMerge(db, config)

today = date.today().strftime('%Y-%m-%d')
merge.run(today)
# Check what's in central memory now
from src.memory.pmis_integration import PMISIntegration
pmis = PMISIntegration(config)
count = pmis.central.count()
print(f'Central memory nodes: {count}')
"
```

### 4.5 Run matching tests

```bash
python3 -m pytest tests/test_matching.py tests/test_memory.py -v
```

---

## Phase 5: Matching + MCP server (Days 15-18)

**Goal:** Deliverable matching works, MCP server runs, Claude Desktop can talk to it.

### 5.1 Test matching engine

**File:** `src/matching/matching_engine.py`

```bash
python3 -c "
import json, yaml
from src.storage.db import Database
from src.matching.matching_engine import MatchingEngine
from datetime import date

with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

db = Database()
db.initialize()
engine = MatchingEngine(db, config)

report = engine.match_day(date.today().strftime('%Y-%m-%d'))
print(json.dumps(report, indent=2, default=str))
"
```

**Expected:** Deliverables matched to work with time breakdowns. "Learning > Reading Papers" should appear in unmatched.

### 5.2 Test MCP server standalone

**File:** `src/mcp/server.py`

```bash
cd ~/Desktop/Memory/productivity-tracker
source .venv/bin/activate

# Test that MCP server imports and initializes without errors
python3 -c "
import src.mcp.server as mcp
print('MCP server initialized.')
print(f'Tools: {[t.name for t in mcp.mcp._tools.values()]}')
"
```

**Expected:** Lists all 9 tools (get_status, pause_tracking, resume_tracking, get_daily_summary, get_deliverable_progress, add_deliverable, complete_deliverable, search_productivity_memory, get_contribution_report, get_trends).

### 5.3 Wire Claude Desktop

**Edit:** `~/Library/Application Support/Claude/claude_desktop_config.json`

Add to the `mcpServers` block:

```json
"productivity-tracker": {
    "command": "/Users/<your-username>/Desktop/Memory/productivity-tracker/.venv/bin/python",
    "args": ["-m", "src.mcp.server"],
    "cwd": "/Users/<your-username>/Desktop/Memory/productivity-tracker",
    "env": {
        "OPENAI_API_KEY": "sk-your-key-here"
    }
}
```

**Restart Claude Desktop.** The MCP server should appear in the tools list.

### 5.4 Validate via Claude Desktop

Open Claude Desktop and try:
- "What's my tracking status?" → should call `get_status`
- "Show me today's summary" → should call `get_daily_summary`
- "Add a deliverable: GTM Strategy, supercontext Sales, contexts Outreach and Collateral" → should call `add_deliverable`

---

## Phase 6: Full daemon (Days 19-21)

**Goal:** The tracker daemon runs continuously, captures real work, and the full pipeline fires.

### 6.1 First live run (manual, 10 minutes)

```bash
cd ~/Desktop/Memory/productivity-tracker
source .venv/bin/activate
python3 -m src.agent.tracker
```

Do some real work for 10 minutes — switch between Chrome, VS Code, terminal. Then Ctrl+C.

Check what was captured:

```bash
python3 -c "
from src.storage.db import Database
from datetime import date

db = Database()
db.initialize()
segments = db.get_segments_for_date(date.today().strftime('%Y-%m-%d'))
print(f'Segments captured: {len(segments)}')
for s in segments[-5:]:
    print(f'  {s[\"target_segment_id\"]}: {s[\"anchor\"]} ({s[\"target_segment_length_secs\"]}s, {s[\"worker\"]})')
"
```

### 6.2 Tune parameters

Based on the live run, adjust `config/settings.yaml`:

| Problem | Fix |
|---------|-----|
| Too many segments (micro-switches) | Raise `ssim_threshold` from 0.7 → 0.8 |
| Missed context switches | Lower `ssim_threshold` to 0.6 |
| Too many API calls | Raise `skip_similar_threshold` from 0.95 → 0.98 |
| SC/Context too vague | Edit prompts in `src/pipeline/prompts.py` |
| Wrong human/agent split | Check `src/agent/agent_detector.py` signatures |

### 6.3 Set up launchd for auto-start

```bash
chmod +x ~/Desktop/Memory/productivity-tracker/scripts/install.sh
~/Desktop/Memory/productivity-tracker/scripts/install.sh
```

This creates a launchd plist at `~/Library/LaunchAgents/com.yantra.productivity-tracker.plist`.

To start on boot:

```bash
launchctl load ~/Library/LaunchAgents/com.yantra.productivity-tracker.plist
launchctl start com.yantra.productivity-tracker
```

To stop:

```bash
launchctl stop com.yantra.productivity-tracker
```

### 6.4 Run for a full day

Let the tracker run for a full workday. At 11 PM, the daily rollup + central merge should fire automatically.

Next morning, check:

```bash
python3 -c "
import json, yaml
from src.storage.db import Database
from src.memory.daily_rollup import DailyRollup

with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

db = Database()
db.initialize()
rollup = DailyRollup(db, config)

# Check yesterday's hierarchy
from datetime import date, timedelta
yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
hierarchy = rollup.get_daily_hierarchy(yesterday)
print(json.dumps(hierarchy, indent=2, default=str))
"
```

---

## Phase 7: Dashboard UI (Days 22-25)

**Goal:** React dashboard showing time breakdown, deliverable progress, human/agent split.

### 7.1 Start the API server

**File:** `src/ui/api.py`

```bash
cd ~/Desktop/Memory/productivity-tracker
source .venv/bin/activate
python3 -m src.ui.api
# Runs on http://127.0.0.1:3001
```

Test: `curl http://127.0.0.1:3001/api/status`

### 7.2 Build React dashboard

This phase creates files in `src/ui/dashboard/`. The dashboard reads from the FastAPI backend.

**Views to build (in order):**

1. **Daily summary** — tree view: SC → Context → Anchor with time bars + human/agent color split
2. **Deliverable tracker** — cards per deliverable with progress and agent leverage %
3. **Timeline** — horizontal bars showing what was done when (reads from segments endpoint)
4. **Contribution map** — green (contributed) vs red (didn't) visualization
5. **Trends** — line charts for daily totals, agent % over time

### 7.3 Serve dashboard

```bash
cd src/ui/dashboard
npm create vite@latest . -- --template react-ts
npm install recharts
npm run dev
# Runs on http://localhost:3000
```

---

## Dependency chain (what blocks what)

```
Phase 0: Environment
  ↓
Phase 1: Tracking agent (screenshot, window, idle, agent)
  ↓
Phase 2: Pipeline (segmenter, frame analyzer, classifier)
  ↓
Phase 3: Storage (SQLite + ChromaDB)
  ↓ (all three above merge here)
Phase 4: Memory pipeline (hourly → daily → PMIS merge)
  ↓
Phase 5: Matching + MCP (deliverables, matching, Claude Desktop)
  ↓
Phase 6: Full daemon (live tracking, tuning, launchd)
  ↓
Phase 7: Dashboard UI (React + FastAPI)
```

Phases 1, 2, and 3 can be partially parallelized. Phase 4 needs all three. Phase 5 needs Phase 4. Phase 6 needs Phase 5. Phase 7 can start anytime after Phase 3 (API reads from DB directly).

---

## File-by-file implementation order

This is the exact sequence to implement and test each file:

| # | File | Phase | Test method |
|---|------|-------|-------------|
| 1 | `config/settings.yaml` | P0 | Read by all modules |
| 2 | `config/privacy.yaml` | P0 | Manual review |
| 3 | `.env` | P0 | API key check |
| 4 | `src/agent/screenshot.py` | P1 | Capture and check file exists |
| 5 | `src/agent/window_monitor.py` | P1 | Print active window |
| 6 | `src/agent/activity_monitor.py` | P1 | Idle detection timing |
| 7 | `src/agent/agent_detector.py` | P1 | Process scan + unit tests |
| 8 | `src/pipeline/segmenter.py` | P2 | `pytest test_segmenter.py` |
| 9 | `src/pipeline/prompts.py` | P2 | Manual review |
| 10 | `src/pipeline/frame_analyzer.py` | P2 | Real screenshot → ChatGPT |
| 11 | `src/pipeline/context_classifier.py` | P2 | Mock frames → classification |
| 12 | `src/storage/models.py` | P3 | DB initialize |
| 13 | `src/storage/db.py` | P3 | Insert/query cycle |
| 14 | `src/storage/chromadb_store.py` | P3 | Embed + search cycle |
| 15 | `config/deliverables.yaml` | P3 | Load and verify |
| 16 | `src/matching/deliverables_loader.py` | P3 | YAML load test |
| 17 | `src/memory/hourly_aggregator.py` | P4 | Seed data → aggregate |
| 18 | `src/memory/daily_rollup.py` | P4 | Hourly → daily hierarchy |
| 19 | `src/memory/pmis_integration.py` | P4 | Isolation check + merge |
| 20 | `src/memory/central_merge.py` | P4 | End-to-end merge |
| 21 | `src/matching/matching_engine.py` | P5 | Daily → deliverable match |
| 22 | `src/matching/contribution_tracker.py` | P5 | Contribution report |
| 23 | `src/mcp/server.py` | P5 | Tool listing + Claude Desktop |
| 24 | `skills/productivity-tracker/SKILL.md` | P5 | Claude Desktop interaction |
| 25 | `src/agent/tracker.py` | P6 | Full daemon run |
| 26 | `scripts/install.sh` | P6 | launchd setup |
| 27 | `src/ui/api.py` | P7 | curl endpoints |
| 28 | `src/ui/dashboard/*` | P7 | React dev server |

---

## Key risk areas

| Risk | Mitigation |
|------|------------|
| ChatGPT Vision misreads screens | Tune prompts, raise detail to "high", add retry logic |
| SSIM threshold too sensitive/insensitive | Calibrate with 1-day real data, adjust in settings.yaml |
| Agent detection misses new tools | Add signatures to `AGENT_SIGNATURES` in agent_detector.py |
| Embedding match gives wrong deliverable | Lower `semantic_match_threshold`, add explicit context keywords to deliverables.yaml |
| Daily rollup crashes mid-run | Add try/except per SC group so one failure doesn't kill the whole rollup |
| macOS permissions silently revoked | Add permission check on startup in tracker.py |

---

## Cost monitoring

After Phase 6 (full daemon running), check daily API cost:

```bash
# Check OpenAI usage dashboard at https://platform.openai.com/usage
# Expected: ~$0.35-0.55/day for an 8-hour workday
# If higher: raise skip_similar_threshold, lower frame_batch_size
```
