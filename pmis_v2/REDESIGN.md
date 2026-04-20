# PMIS V2 Redesign — Learned Poincare + Dynamic Hierarchy + Wiki + Goals

**Date:** 2026-04-14 (updated)
**Status:** Approved, Phase 0 complete
**Evidence:** Replay harness on 196 turns, 3,455 nodes, 3,428 edges across 23 trees

---

## Why This Redesign

The replay harness (2026-04-13) measured every scoring component's contribution:

| Problem | Data | Impact |
|---------|------|--------|
| Poincare embeddings are random projections | Discriminative power: 0.080 (lowest of 4 components) | Hierarchy scoring is near-constant noise |
| RSGD never converges | Loss oscillates 496-552 across 160 runs | 8.5 hours total compute wasted |
| Gamma is stuck | 76% of turns in 0.4-0.5 band, ASSOCIATIVE never fires | System is one-mode |
| Fat contexts | Top CTX has 471 children, top 10 have 1,718 | Graph is flat where it should be deep |
| All sessions single-turn | 179/180 conversations are 1 turn | Session engine and gamma accumulation can't work |
| No legibility | Knowledge trapped in memory.db | Can't inspect why a node matters |
| No goals/feedback | No intent linking, no validation loop | Can't learn what works |

---

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Poincare dimensions | **16d** (down from 32d) | Nickel & Kiela: 5d works for 130K-node WordNet. 3,455 nodes needs far less. |
| GNN architecture | **2-layer HGCN** (Chami et al. 2019) | Tangent space aggregation for stability. |
| Training stack | **geoopt + torch_geometric** | geoopt: Riemannian optimizer. torch_geometric: message passing. |
| Training schedule | **Nightly consolidation only** | ~3-5 min on 3,455 nodes. No per-turn latency. |
| Training rollout | **Phased: 4a structural → 4b co-retrieval → 4c feedback** | One variable at a time. Nickel & Kiela (2017) used structural only. Co-retrieval is our addition — validate separately. |
| Co-retrieval filtering | **Use all pairs, let GNN learn** | 88.8% are cross-branch (novel signal). |
| Co-retrieval negatives | **Never co-retrieved with either member** | Conservative true negatives. |
| Level naming | **Numeric depth: CTX-1, CTX-2, CTX-3...** | Extensible. DB stores depth as integer. |
| Cell division depth | **No limit** | "Client Communication" (471 children) splits to ~6 levels. |
| Norm bands | **Computed from depth: `0.05 + (depth/max_depth) * 0.85`** | Auto-adapts. HGCN refines from initialization. |
| Retrieval weights | **Keep 40/30/15/15 until post-HGCN harness validates** | Can't change model AND weights simultaneously. |
| Wiki format | **LLM-generated prose (front) + backend data panel (on click)** | Following LLM Wiki pattern: readable pages, not dashboards. |
| Wiki caching | **Generate on change, cache until data changes** | LLM generates prose once, served from cache until invalidated. |

---

## Page Value Function

Every wiki page has a computable value from 6 weighted components. Components activate as data becomes available. The same PageValue flows into both wiki rendering (display) and HGCN training (edge weights).

### Formula

```
PageValue(node) = Σ (normalized_wᵢ × Cᵢ)    for each ACTIVE component i
```

Weights are renormalized across active components only — inactive components don't dilute.

### Components (Ordered by Value)

```
Component     Default Weight   Activates When              What It Measures
──────────────────────────────────────────────────────────────────────────────
1. Goal        0.30            Goal linked to node          Intent alignment
2. Feedback    0.25            First feedback received       User validation
3. Usage       0.20            First retrieval              Retrieval frequency
4. Recency     0.10            Always (has timestamp)       Freshness
5. Linkage     0.10            Co-retrieval data exists     Lateral connections
6. Structural  0.05            Always (has parent)          Tree position
```

### Component Calculations

```python
# 1. GOAL — weighted avg of goal weights × status multiplier
goal_score = avg(gl.weight * STATUS_MULT[gl.status] for gl in goal_links)
# STATUS_MULT: active=1.0, achieved=0.3, paused=0.1, abandoned=0.0

# 2. FEEDBACK — asymmetric: negatives hit harder
raw = (positive_count * 0.10) - (negative_count * 0.15)
feedback_score = sigmoid(raw * 5.0)  # normalize to [0, 1]

# 3. USAGE — log scale, saturates at ~50 accesses
usage_score = min(log(1 + access_count) / log(51), 1.0)

# 4. RECENCY — exponential decay, 30-day halflife
recency_score = exp(-hours_since_access / 720)

# 5. LINKAGE — log of co-retrieval pair count
linkage_score = min(log(1 + co_retrieval_count) / log(21), 1.0)

# 6. STRUCTURAL — tree depth position score
structural_score = depth_based_score(node)
```

### Activation Cascade (Natural Lifecycle of Knowledge)

```
Day 1  (fresh):      Recency(0.67) + Structural(0.33)
Day 3  (retrieved):  Usage(0.57) + Recency(0.29) + Structural(0.14)
Day 7  (validated):  Feedback(0.36) + Usage(0.29) + Recency(0.14) + Linkage(0.14) + Structural(0.07)
Day 14 (intentional): Goal(0.30) + Feedback(0.25) + Usage(0.20) + Recency(0.10) + Linkage(0.10) + Structural(0.05)
```

### PageValue → HGCN Training

PageValue modulates edge importance in the GNN loss:
```
L_edge(a, b) = avg(PageValue(a), PageValue(b)) × base_loss(a, b)
```
High-value nodes have their edges amplified. Low-value nodes barely influence geometry.

### Hyperparameters (in hyperparameters.yaml)

```yaml
page_weight_goal: 0.30
page_weight_feedback: 0.25
page_weight_usage: 0.20
page_weight_recency: 0.10
page_weight_linkage: 0.10
page_weight_structural: 0.05
goal_status_active: 1.0
goal_status_achieved: 0.3
goal_status_paused: 0.1
goal_status_abandoned: 0.0
feedback_positive_weight: 0.10
feedback_negative_weight: 0.15
feedback_sigmoid_scale: 5.0
recency_halflife: 720
```

---

## Feedback System

### Three Types

| Type | Source | Strength | Example |
|------|--------|----------|---------|
| **Explicit** | User CLI command | 1.0 | `feedback <node> positive "Got meeting"` |
| **Session** | Thumbs up/down at session end | 1/N (N = retrieved nodes) | `session rate up` → attributed to all retrieved |
| **Implicit** | Behavioral inference | 0.1-0.3 | Topic continuation = weak positive; topic pivot = weak negative |

### Three Channels of Impact

**Channel 1 — Immediate (retrieval boost):**
```python
adjusted_score = base_score * (1.0 + feedback_score)
# feedback_score = accumulated (pos * 0.10) - (neg * 0.15), clamped
```

**Channel 2 — Nightly (HGCN geometry):**
Feedback edges in L_feedback term. Positive pulls nodes closer in Poincare space. Negative pushes apart.

**Channel 3 — Wiki (legibility):**
Feedback renders on node pages with timestamps, notes, and goal attribution.

### Feedback Table Schema

```sql
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT REFERENCES memory_nodes(id),
    goal_id TEXT REFERENCES goals(id),
    polarity TEXT CHECK(polarity IN ('positive','negative','correction')),
    content TEXT,
    source TEXT DEFAULT 'explicit',  -- 'explicit'|'session'|'implicit'
    strength REAL DEFAULT 1.0,
    timestamp TEXT
);
```

---

## Phase 0: Diagnostics Monitor [DONE]

73-column `turn_diagnostics` table. Captures every pipeline touchpoint per turn.

**Files:** `core/diagnostics.py`, `db/schema.sql`, `db/manager.py`, `orchestrator.py`

**Baseline:** hierarchy_disc=0.012, gamma_mode=BALANCED only, narrow_candidates=0

---

## Phase 1: Fix Gamma + K-Nearest Surprise

**1a.** Temperature 3.0→6.0, bias 0.5→0.3
**1b.** Precision floor: contexts with 15+ anchors, 10+ accesses get floor 0.7
**1c.** K=5 weighted nearest surprise (replaces single-point)
**1d.** Session gamma accumulation via EMA precision accumulator

**Files:** `hyperparameters.yaml`, `core/surprise.py`, `core/gamma.py`, `orchestrator.py`
**Target:** gamma std >0.10, ASSOCIATIVE mode appears

---

## Phase 1b: Fix Conversation Model

Add `session continue` CLI command. Follow-up messages reuse conversation_id.

**Files:** `cli.py`, `orchestrator.py`
**Target:** session_turn_count > 1 in diagnostics

---

## Phase 2: Goals + Feedback Schema

Goals table, goal_links table, feedback table. CLI commands for create/link/list/rate.

**Files:** `db/schema.sql`, `db/manager.py`, `cli.py`

---

## Phase 3: Cell Division (Before HGCN)

Recursive split when CTX has 25+ children. K-means in tangent space, k=2-3 by silhouette. No depth limit. Numeric level naming (CTX-1, CTX-2...).

Expected: "Client Communication" (471) → 6 levels, ~19 leaf clusters. ~65 new intermediate nodes total.

**Files:** `consolidation/cell_division.py` (new), `consolidation/nightly.py`, `db/manager.py`, `db/schema.sql`

---

## Phase 4: Hyperbolic GCN — Phased Rollout

### Architecture
```
768d semantic → [Layer 1: 128d tangent] → [Layer 2: 16d Poincare ball]
Curvature: learnable per layer, init c=-1.0
Aggregation: attention-weighted gyromidpoint
```

### Phase 4a: Structural Only (Pure Nickel & Kiela 2017)

Train on ~3,493 child_of edges only. This is the original Poincare embeddings formulation.
- Loss: `d(parent,child)^2 + margin negatives`
- Negatives: 50% cross-branch, 50% random
- **Run harness → measure hierarchy_score_discriminative**
- Baseline: "does Poincare geometry help at all?"

### Phase 4b: Add Co-Retrieval

Add 3,048 cross-branch co-retrieval pairs to training.
- Loss: `0.6 * L_structural + 0.3 * L_co_retrieval`
- Co-retrieval negatives: nodes never co-retrieved with either member
- **Run harness → measure delta from 4a**
- Co-retrieval covers 300/3,457 nodes (8.7%). Monitor two-tier geometry effect.

### Phase 4c: Add Feedback (When Data Exists)

Add feedback edges from Phase 2 CLI usage.
- Loss: `0.6 * L_structural + 0.3 * L_co_retrieval + 0.1 * L_feedback`
- Positive feedback → pull closer. Negative → push apart.

### Edge Weighting by PageValue

All loss terms modulated by PageValue of endpoints:
```
L_edge(a,b) = avg(PageValue(a), PageValue(b)) × base_loss(a,b)
```

### Training Data Profile

| Signal | Edges | Node Coverage | Origin |
|--------|-------|---------------|--------|
| Structural | 3,493 | 100% | Nickel & Kiela 2017 |
| Co-retrieval | 3,048 cross-branch | 8.7% (300 nodes) | Our addition (node2vec-inspired) |
| Feedback | 0 (growing) | Growing | Our addition |

### Migration

HGCN outputs 16d → overwrite hyperbolic blobs (32d→16d) → recompute centroids → rebuild ChromaDB → retrieval uses new embeddings immediately.

**Files:** `core/hgcn.py` (new), `core/co_retrieval.py` (new), `core/poincare.py` (rewrite), `core/rsgd.py` (delete), `consolidation/nightly.py`, `hyperparameters.yaml`, `requirements.txt`

---

## Phase 5: Wiki Dashboard

### Architecture: Two Layers

**Front (wiki prose):** LLM-generated markdown pages. Headings, subheadings, inline cross-references, synthesis. Following LLM Wiki pattern — reads like a knowledge base, not a dashboard.

**Backend (on click):** PageValue decomposition, component bars, diagnostics, retrieval stats, HGCN training impact. Accessed via "View backend data" link on each page.

### Page Generation Pipeline

```
DB Data → WikiRenderer queries → Structured Context → LLM Prompt → Prose → Cache
```

The LLM writes each page as narrative prose from structured data. Pages are cached in `wiki_page_cache` table and regenerated only when underlying data changes (new feedback, consolidation, goal change).

### Wiki Page Format (LLM Wiki Style)

**SC pages (~1200 words):** Domain overview → H2 per child CTX with key insights → inline cross-references to other SCs → Related Pages section → backend link.

**CTX pages (~800 words):** Skill/phase overview → top anchors woven into narrative → contradictions noted inline → co-retrieval neighbors → Related Pages → backend link.

**ANC pages (~500 words):** The core insight stated clearly → evidence (feedback, usage, source) → context (parent, goals) → related anchors → backend link.

### Page Schema (wiki_schema.yaml)

```yaml
page_types:
  SC:
    max_words: 1200
    sections: [overview, children_narrative, cross_references, related_pages]
  CTX:
    max_words: 800
    sections: [overview, key_anchors_narrative, contradictions, co_retrieval, related_pages]
  ANC:
    max_words: 500
    sections: [insight, evidence, context, related]
cross_reference_style: inline  # [Page Name](link) in prose, not footnotes
contradiction_style: inline    # "However, X suggests..."
```

### Special Pages

**Index (`/wiki/`):** Categorized catalog of all SCs with one-line summaries, goal tags, staleness warnings. Organized by domain (Sales, Engineering, Research, Other). LLM Wiki's `index.md` pattern.

**Log (`/wiki/log`):** Append-only chronological record. Consolidation events, feedback events, session events. Parseable with simple grep. LLM Wiki's `log.md` pattern.

**Goals (`/wiki/goals`):** All goals with linked nodes, progress, feedback summary.

**Lint (`8200/lint`):** Lint report — orphans, contradictions, stale contexts, unvalidated nodes. _Moved from 8100/wiki/health._

**Diagnostics (`8200/diagnostics`):** Turn diagnostic trends, gamma distribution, score spreads. _Moved from 8100/wiki/diagnostics._

### Backend Panel (Per Page, On Click)

```
PAGE VALUE: 0.82 / 1.00
Components:
  Goal:       0.90  (w=0.30)
  Feedback:   0.72  (w=0.25)
  Usage:      0.68  (w=0.20)
  Recency:    0.55  (w=0.10)
  Linkage:    0.48  (w=0.10)
  Structural: 0.35  (w=0.05)

Retrieval Stats | Poincare Geometry | Diagnostics History
[Edit Goal Weight] [Add Feedback] [View Raw JSON]
```

### Wiki Cache Table

```sql
CREATE TABLE wiki_page_cache (
    node_id TEXT PRIMARY KEY,
    prose_markdown TEXT,
    context_hash TEXT,
    generated_at TEXT,
    llm_model TEXT,
    word_count INTEGER,
    FOREIGN KEY (node_id) REFERENCES memory_nodes(id)
);
```

### Cache Invalidation Triggers

| Trigger | Pages Regenerated |
|---------|-------------------|
| New feedback on node | Node page + parent CTX page |
| Goal linked/unlinked | Node page + goal page |
| Consolidation (nightly) | All pages with changed children |
| First HGCN training | All pages (geometry changed globally) |
| User navigates to stale page | That page on-demand |
| Manual "Regenerate" click | Any page |

### Link Types in Wiki Pages

| Link Type | Source | Rendering |
|-----------|--------|-----------|
| Hierarchy | `relations.child_of` | Parent/child links in tree structure |
| Co-retrieval | `turn_retrieved_memories` | "Often retrieved alongside [Page]" inline |
| Goal | `goal_links` | "Serves goal: [Goal Title]" |
| Feedback | `feedback` | Feedback entries with timestamps |
| Sequence | `relations.followed_by` | "Typically followed by [Page]" |

All rendered as clickable cross-references woven into prose — not separate link sections.

### Routes

```
/wiki/                    → Index (categorized SC listing)
/wiki/sc/<id>             → SC prose page
/wiki/sc/<id>/backend     → SC backend panel (PageValue, stats)
/wiki/ctx/<id>            → CTX prose page
/wiki/ctx/<id>/backend    → CTX backend panel
/wiki/anc/<id>            → ANC prose page
/wiki/anc/<id>/backend    → ANC backend panel
/wiki/goals               → Goals page
/wiki/log                 → Chronological event log

# Ops UI (port 8200 — health_dashboard.py)
/                         → Monitor (7-organ health)
/dashboard                → Main dashboard (stats + conversations + hyperparams)
/feedback                 → Feedback log
/diagnostics              → Turn diagnostics trends
/lint                     → Lint report (orphans/stale/oversized/unvalidated)
```

**Files:** `server.py`, `wiki_renderer.py` (new), `wiki_schema.yaml` (new), `templates/wiki_*.html` (new), `db/schema.sql` (wiki_page_cache table)

---

## Phase 6: Consolidation Pipeline Update

```
Pass 1: COMPRESS        (unchanged)
Pass 2: PROMOTE         (unchanged)
Pass 3: BIRTH           (unchanged)
Pass 4: CELL DIVISION   (new — recursive split of fat CTXs)
Pass 5: PRUNE           (unchanged)
Pass 6: HGCN TRAIN      (new — replaces RSGD, always runs)
Pass 7: REINDEX         (new — rebuild ChromaDB ANN index)
Pass 8: WIKI REGEN      (new — regenerate wiki pages for changed nodes)
Pass 9: WIKI HEALTH     (new — lint checks, health report)
```

**Files:** `consolidation/nightly.py`

---

## Implementation Order

| Phase | What | Est. LOC | Status |
|-------|------|----------|--------|
| **0** | Diagnostics monitor | 200 | **DONE** |
| **1** | Gamma fix + K-nearest surprise | 80 | Next |
| **1b** | Conversation model (session continue) | 50 | Next |
| **2** | Goals/feedback schema + CLI | 300 | Pending |
| **3** | Cell division (recursive, no depth limit) | 250 | Pending |
| **4a** | HGCN structural only (pure Nickel & Kiela) | 600 new, 400 deleted | Pending |
| **4b** | Add co-retrieval to HGCN | 50 | Pending (after 4a harness) |
| **4c** | Add feedback to HGCN | 30 | Pending (when data exists) |
| **5** | Wiki dashboard (prose + backend) | 600 | Pending |
| **6** | Consolidation pipeline update | 100 | Pending |
| **Total** | | **~2,260 new, ~800 deleted** | |

---

## Validation Targets (vs. Baseline 2026-04-13)

| Metric | Baseline | Target | Measured By |
|--------|----------|--------|-------------|
| Hierarchy discriminative power | 0.080 | >0.15 | Replay harness |
| Gamma std | 0.053 | >0.10 | Diagnostics table |
| Gamma range | 0.35-0.62 | 0.15-0.85 | Diagnostics table |
| ASSOCIATIVE mode % | 0% | >10% | Diagnostics table |
| HGCN loss | Oscillates 496-552 | Converges | Consolidation log |
| Max tree depth | 3 (SC/CTX/ANC) | 6+ | Node level query |
| Session convergence rate | 0.7% | >20% | Session engine harness |
| narrow_candidates_found | 0 | >3 per turn | Diagnostics table |
| Wiki page value coverage | 0% | 100% of SCs | wiki_page_cache |

---

## Key Files Reference

| File | Role |
|------|------|
| `orchestrator.py` | Pipeline controller — all phases touch this |
| `core/poincare.py` | Poincare distance, exp/log maps (keep), projection (delete) |
| `core/hgcn.py` | NEW: HGCN model, mixed loss, training loop |
| `core/co_retrieval.py` | NEW: Build co-retrieval graph from turn_retrieved_memories |
| `core/surprise.py` | Surprise computation — K-nearest, precision floor |
| `core/gamma.py` | Gamma sigmoid — temperature, session boost |
| `core/diagnostics.py` | Turn diagnostics capture (DONE) |
| `core/rsgd.py` | DELETE: Replaced by HGCN |
| `consolidation/nightly.py` | All consolidation passes |
| `consolidation/cell_division.py` | NEW: Recursive tree splitting |
| `retrieval/engine.py` | Scoring formula (weights unchanged initially) |
| `db/schema.sql` | All table definitions |
| `db/manager.py` | All DB operations |
| `cli.py` | CLI commands: session continue, goal, feedback |
| `server.py` | FastAPI server + wiki routes |
| `wiki_renderer.py` | NEW: DB queries → structured context for LLM |
| `wiki_schema.yaml` | NEW: Page format conventions for LLM generation |
| `replay_harness.py` | Component ablation analysis |
| `run_both_harnesses.py` | Unified replay runner |
| `hyperparameters.yaml` | All tunable parameters (page weights + engine params) |
