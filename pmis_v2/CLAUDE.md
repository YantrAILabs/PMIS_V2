# ProMe (PMIS v2) — Surprise-Minimizing Personal Memory Intelligence System

## What This Is
This is Rohit's personal memory system built on the Free Energy Principle.
It uses surprise minimization to decide what to remember, what to forget,
how to retrieve, and how to respond. Every conversation turn computes a
surprise score and gamma (γ) value that controls the explore-exploit balance.

## Architecture
```
User Message → Embed → Surprise → Gamma → Retrieve (γ-blended) → Compose Prompt → Respond → Store/Skip
```

## Per-Turn Behavior
The `<memory_context>` block in the system prompt tells you:
- **MODE**: ASSOCIATIVE (γ>0.7), BALANCED (0.4-0.7), or PREDICTIVE (γ<0.4)
- **SURPRISE**: How novel this turn is relative to stored memory
- **RETRIEVED MEMORIES**: Ranked by combined score (semantic + hierarchy + temporal + precision)
- **PREDICTIVE**: What typically follows the current topic in past conversations
- **BEHAVIORAL GUIDANCE**: How to respond — go deep vs explore vs ask questions

Follow the behavioral guidance precisely. It is computed from theory, not arbitrary.

## Slash Commands
- `/memory status` — Current γ, surprise, active Context, session stats
- `/memory explore` — Force exploration mode (γ=0.2) for next turns
- `/memory exploit` — Force exploitation mode (γ=0.9) for next turns
- `/memory orphans` — List unattached Anchor memories awaiting consolidation
- `/memory tree <name>` — Switch active tree context
- `/memory consolidate` — Run nightly consolidation manually
- `/memory surprise` — Show surprise history for this session

## Key Files
```
orchestrator.py              — Main entry point (call process_turn on every message)
core/surprise.py             — Surprise computation (raw × precision = effective)
core/gamma.py                — γ calculator (sigmoid of effective surprise)
core/poincare.py             — Poincaré ball math (hierarchy = geometry)
retrieval/engine.py          — γ-weighted blended retrieval
ingestion/pipeline.py        — Full ingestion with surprise-gated storage
consolidation/nightly.py     — 4-pass nightly: compress, promote, birth, prune
hyperparameters.yaml         — All 50+ tunable knobs
```

## Running
```bash
# Initialize database (ChromaDB ANN index auto-created alongside)
cd pmis_v2
python -c "from orchestrator import Orchestrator; o = Orchestrator('data/memory.db')"

# Migrate all data sources at once
python -m migration.unified_migrate \
  --claude ~/exports/claude_conversations.json \
  --chatgpt ~/exports/chatgpt_export.zip \
  --neo4j ~/exports/neo4j_graph.json \
  --db data/memory.db

# Run nightly consolidation (ChromaDB + stats cache auto-rebuilt)
python -c "
from db.manager import DBManager
from db.chroma_store import ChromaStore
from consolidation.nightly import NightlyConsolidation
chroma = ChromaStore(persist_dir='data/chroma')
db = DBManager('data/memory.db', chroma_store=chroma)
engine = NightlyConsolidation(db)
results = engine.run()
print(results)
"

# Run test suites
python tests/test_e2e.py      # 23 end-to-end tests
python tests/test_p1_p2.py    # P1/P2 feature validation
```

## Auto-Wired Features (zero manual steps)
- **ChromaDB ANN Index**: Auto-created on startup, auto-synced on every create/delete,
  auto-rebuilt after nightly consolidation. Falls back to linear scan if ChromaDB unavailable.
- **Batch Embedding**: Auto-used during migration. OpenAI batches natively; Ollama pipelines.
- **Materialized Stats**: Context precision stats cached in SQLite, auto-refreshed when
  children are attached/detached, bulk-refreshed after nightly consolidation.
- **Model Version Tracking**: Embedding model name recorded on first use, checked on every
  startup. Warns if model changes (prevents silent incompatibility).

## Integration Hook
```python
from orchestrator import Orchestrator

orch = Orchestrator(db_path="data/memory.db")

# On every user message:
result = orch.process_turn(content="user's message", conversation_id="abc123")
system_prompt_injection = result.system_prompt
# Prepend system_prompt_injection to Claude's context

# On conversation end:
orch.close_session("abc123")
```

## Hyperparameter Tuning Guide
After 2 weeks of usage, check:
1. **Retrieval quality**: Are the right memories surfacing? Adjust `score_weight_*`
2. **Storage volume**: Too many Anchors? Raise `surprise_low_threshold`. Too few? Lower it.
3. **Exploration balance**: Always in exploit mode? Lower `gamma_bias`. Never exploiting? Raise it.
4. **Consolidation**: Too aggressive pruning? Raise `prune_min_age_days`. Orphans piling up? Lower `birth_min_orphans`.
5. **Staleness**: Never triggers? Lower `staleness_threshold`. Triggers too often? Raise `staleness_window`.
