# PMIS v3 Architecture — 3-Iteration Design

## The 4 Problems

| # | Problem | Current State | Target |
|---|---------|--------------|--------|
| P1 | Similar memories with different outcomes aren't differentiated | Flat weight per anchor, no variant tracking | Distinguish "what" from "how well" |
| P2 | Long conversations degrade (AI overfitting on linear context) | No conversation structure awareness | Convergent tree pruning across sessions |
| P3 | Memories lack environmental context (work/travel/home) | No environment dimension at all | Environment-aware retrieval |
| P4 | Cross-domain relational transfer (website design → new website) | Only word-overlap retrieval | Skill-pattern transfer across domains |

---

# ITERATION 1: First Principles Design

## P1: Differentiating Similar Memories — Variant Chains

### The Core Insight

Currently, PMIS stores "CISOs respond to threat language" as one anchor with one weight. But in reality, you might have tried this insight 5 times with 5 variations:
- v1: "Used threat language in email subject" → score 4.5
- v2: "Used threat language in body only" → score 2.0
- v3: "Used threat-intel stats in opening" → score 4.8
- v4: "Used threat language + ROI hybrid" → score 3.0
- v5: "Used threat-intel with personalized data" → score 4.9

The anchor "threat language works" is TRUE — but the *variant* matters enormously. Currently all 5 collapse into one node.

### Iteration 1 Solution: Variant Chain Table

```
┌──────────────────────────────────────────────────┐
│ anchor: "Threat-intel language works on CISOs"    │
│ weight: 0.85 (aggregate)                          │
│                                                    │
│  ┌─ variant_1: "threat in subject"     score: 4.5 │
│  ├─ variant_2: "threat in body only"   score: 2.0 │
│  ├─ variant_3: "threat stats opening"  score: 4.8 │
│  ├─ variant_4: "threat + ROI hybrid"   score: 3.0 │
│  └─ variant_5: "threat + personalized" score: 4.9 │
│                                                    │
│  best_variant: v5 (4.9)                           │
│  worst_variant: v2 (2.0)                          │
│  signal: specificity matters more than presence   │
└──────────────────────────────────────────────────┘
```

**New table: `anchor_variants`**
```sql
CREATE TABLE anchor_variants (
    id TEXT PRIMARY KEY,
    anchor_id TEXT NOT NULL,       -- parent anchor
    task_id TEXT NOT NULL,         -- which task produced this
    description TEXT NOT NULL,     -- what specifically was done
    parameters TEXT DEFAULT '{}',  -- JSON: specific choices made
    score REAL DEFAULT 0,          -- outcome score
    created_at TEXT NOT NULL
);
```

**Quantitative metrics:**
- **Variant Spread** = `max(variant_scores) - min(variant_scores)` — high spread = the "how" matters more than the "what"
- **Best Variant Weight** = `best_variant_score / 5.0` — the ceiling of this insight
- **Variant Consistency** = `1 / (stdev(variant_scores) + 1)` — low stdev = insight is robust regardless of execution

### Rohit's suggestion (trained model) — Assessment:

A small model trained on patterns would be overkill at current scale (177 anchors). The variant chain gives you 80% of the value because it captures *what specifically differed* in structured form. A model becomes necessary when:
1. You have 1000+ variants and need automatic pattern extraction
2. You want the system to *generate* the next best variant to try
3. You need cross-anchor pattern recognition ("personalization helps everywhere, not just threat emails")

**Verdict for Iter 1:** Variant chains first, model later as a retrieval optimizer on top.

---

## P2: Long Conversation Degradation — Convergent Tree Building

### The Core Insight

When a human has 10 conversations about "website design", each conversation CONSTRAINS the solution space:
- Conv 1: "I want a dark theme" → eliminates light theme options
- Conv 2: "Hero should be full-bleed" → constrains layout
- Conv 3: "No gradients" → constrains styling
- ...
- Conv 10: Full spec is clear, design is tight

An AI doesn't do this. Each conversation starts fresh, or worse, in a long conversation it starts *reinforcing* its own previous outputs (overfitting on its linear context window).

### Iteration 1 Solution: Hypothesis Trees with Convergence Score

For each super context, maintain a set of **hypothesis trees** — alternative structural arrangements of contexts and anchors that represent different "theories" about the right approach.

```
SC: "Yantra AI Website Design"
│
├── Hypothesis A (confidence: 0.82, sessions: 7)
│   ├── [CTX] Dark theme + Sora font
│   ├── [CTX] 9-section canonical layout
│   └── [CTX] Red strikethrough + amber accent
│
├── Hypothesis B (confidence: 0.35, sessions: 2)
│   ├── [CTX] Light minimalist theme
│   ├── [CTX] 5-section short layout
│   └── [CTX] Blue accent color
│
└── Hypothesis C (confidence: 0.15, sessions: 1)
    ├── [CTX] Gradient-heavy design
    └── [CTX] Animation-first approach
```

**New table: `hypotheses`**
```sql
CREATE TABLE hypotheses (
    id TEXT PRIMARY KEY,
    sc_id TEXT NOT NULL,
    structure TEXT NOT NULL,       -- JSON: the tree structure
    confidence REAL DEFAULT 0.5,
    session_count INTEGER DEFAULT 1,
    last_reinforced TEXT,
    created_at TEXT NOT NULL
);
```

**Convergence Score** = `1 - (active_hypotheses - 1) / max_hypotheses`
- 1 hypothesis left → convergence = 1.0 (fully decided)
- 5 hypotheses → convergence = 0.0 (still exploring)

**Session Update Rules:**
1. After each conversation, check which hypothesis the session's decisions align with
2. Reinforced hypothesis: `confidence += 0.1 * (1 - confidence)` (diminishing returns)
3. Contradicted hypothesis: `confidence *= 0.7` (30% decay)
4. When `confidence < 0.1`: prune hypothesis
5. When only 1 hypothesis remains: flag SC as "converged"

---

## P3: Environmental Context — The Environment Dimension

### The Core Insight

"Check email" at work = open Outlook, triage by priority.
"Check email" while traveling = quick scan on phone, reply only urgent.
"Check email" at home = personal inbox, relaxed browsing.

Same memory, completely different retrieval and execution based on *environment*.

### Iteration 1 Solution: Environment Tags

```sql
ALTER TABLE nodes ADD COLUMN environment TEXT DEFAULT 'general';
-- Values: 'work', 'home', 'travel', 'creative', 'learning', 'social', 'general'

ALTER TABLE tasks ADD COLUMN environment TEXT DEFAULT 'general';
```

**Retrieval modifier:**
```python
env_boost = 1.5 if node.environment == current_env else 0.7
final_score = base_score * temporal_boost * env_boost
```

**Environment detection (at store time):**
- User explicitly tags: `"environment": "work"`
- Or infer from SC name: "B2B Cold Outreach" → work, "Travel Planning" → travel
- Or infer from time: 9am-6pm weekday → work, evening/weekend → personal

---

## P4: Cross-Domain Relational Transfer — Skill Patterns

### The Core Insight

When you design the Schools website and then start the Health website, you don't just retrieve "Schools page memories." You transfer the *pattern* — "9-section architecture works for proposal pages" — which is ABSTRACT, not domain-specific.

Currently, retrieval is pure word overlap. "Health website" shares zero words with "Schools website" so those memories score 0.

### Iteration 1 Solution: Skill Pattern Nodes

A new node type — `pattern` — that lives ABOVE super contexts:

```
[PATTERN] "Proposal page architecture"
├── applies_to: [SC: Schools Website, SC: Health Website, SC: Security Website]
├── core_structure: "9-section canonical layout with hero→benefits→intelligence→..."
├── transfer_count: 3
└── transfer_score: 4.2 (avg score across domains)

[PATTERN] "Cold outreach methodology"
├── applies_to: [SC: B2B Cold Outreach, SC: Healthcare Sales]
├── core_structure: "threat-language→personalization→48hr-followup→..."
└── transfer_count: 2
```

```sql
CREATE TABLE patterns (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    core_structure TEXT,          -- JSON: abstract structure
    transfer_count INTEGER DEFAULT 1,
    avg_transfer_score REAL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE pattern_links (
    pattern_id TEXT NOT NULL,
    sc_id TEXT NOT NULL,
    score REAL DEFAULT 0          -- how well this pattern worked in this SC
);
```

**Transfer retrieval:**
When searching for "Health website design", after finding no direct word matches, check patterns:
1. Get patterns linked to any SC that scores > 0 on the query
2. Get OTHER SCs linked to those patterns
3. Retrieve their structures as transfer candidates

---

# ITERATION 2: Critique & Redesign

## What's Wrong With Iteration 1

### P1 Critique: Variant Chains are Redundant

The variant chain table duplicates what `task_anchors` + `tasks` already capture. Each task already records which anchors were present and what score it got. The REAL missing piece isn't a new table — it's a **comparison function** that can look at two tasks using the same anchor and identify what differed.

**Redesign:** Instead of a new table, add a `parameters` JSON field to `task_anchors` that captures the *specific execution details* of each anchor in each task. Then compute variant metrics from the existing task history.

```sql
ALTER TABLE task_anchors ADD COLUMN execution_params TEXT DEFAULT '{}';
-- Captures: {"approach": "threat in subject", "target": "VP Security", "tone": "urgent"}
```

**New metric — Anchor Discrimination Power (ADP):**
```
ADP(anchor) = variance(scores of tasks containing this anchor) /
              variance(all task scores)
```
- ADP > 1.0 = this anchor's presence is more variable than average → execution details matter
- ADP < 0.5 = this anchor contributes consistently regardless of how you use it → reliable baseline
- ADP ≈ 1.0 = typical contribution

This eliminates the need for a separate variant table. The existing task log IS the variant history. We just need to query it better.

### P2 Critique: Hypothesis Trees Are Over-Engineered

The hypothesis tree approach requires:
1. Alignment detection (which hypothesis does this session match?)
2. Confidence updating
3. Pruning logic
4. A separate table with complex JSON structures

This is essentially building a decision-making framework on top of memory. The simpler truth: **the winning structure snapshot already captures the best tree**. What's missing isn't hypothesis tracking — it's **decision logging**.

**Redesign: Decision Log**

Every conversation makes decisions that constrain future work. Capture these as a new edge type:

```sql
-- Using existing edges table with new type
INSERT INTO edges (id, src, tgt, type, weight, created_at)
VALUES (?, anchor_id, anchor_id, 'decision', confidence, now);
```

But actually, decisions are richer than edges. New table:

```sql
CREATE TABLE decisions (
    id TEXT PRIMARY KEY,
    sc_id TEXT NOT NULL,
    session_id TEXT NOT NULL,      -- which conversation
    decision TEXT NOT NULL,        -- "Use dark theme"
    eliminates TEXT DEFAULT '[]',  -- JSON: alternatives ruled out
    confidence REAL DEFAULT 0.5,   -- how certain
    reversible INTEGER DEFAULT 1,  -- can this be undone?
    created_at TEXT NOT NULL
);
```

**Convergence is now simple:**
```
convergence(SC) = count(irreversible decisions) / total_decision_slots
```

Where `total_decision_slots` is estimated from the SC type (a website has ~15 design decisions, a sales campaign has ~8).

**Long conversation fix:** At the start of each session, retrieve ALL decisions for this SC and present them as hard constraints. The AI doesn't drift because the constraint list is explicit.

### P3 Critique: Environment Tags Are Too Static

Binary environment matching (work/home/travel) is crude. The same anchor might be relevant in work AND creative environments. And "environment" is really a proxy for a richer concept: **the user's current mode of operation**.

**Redesign: Mode Vectors Instead of Tags**

Each node gets a vector of mode relevances:
```json
{
    "work": 0.9, "creative": 0.3, "learning": 0.7,
    "social": 0.1, "travel": 0.0, "home": 0.2
}
```

Retrieval uses dot product between query mode vector and node mode vector, giving continuous weighting instead of binary on/off.

**But this requires knowing the user's current mode.** Detection:
- Explicit: user says "I'm at work" or system prompt includes context
- Implicit from SC: "B2B Cold Outreach" → {work: 0.95, ...}
- Time-based prior: weekday 9-6 → work mode likely

### P4 Critique: Pattern Nodes Create a 4th Level of Hierarchy

Adding patterns above super contexts breaks the 3-level architecture (SC > Context > Anchor). It introduces complexity and a new retrieval path.

**Redesign: Lateral Edges Instead of Vertical Hierarchy**

Don't add a new level. Add a new EDGE TYPE between existing SCs:

```sql
INSERT INTO edges (id, src, tgt, type, weight, created_at)
VALUES (?, sc_1_id, sc_2_id, 'transfers_to', strength, now);
```

**Transfer strength** computed automatically:
1. Two SCs share structure → compute structural similarity of their winning snapshots
2. Jaccard(context_titles_SC1, context_titles_SC2) = title overlap
3. If SCs share >40% context titles → strong transfer edge

**Retrieval update:** When a query matches SC_A weakly but SC_A has a `transfers_to` edge to SC_B which matches strongly, include SC_B's patterns in retrieval with `transfer_weight` discount.

```python
transfer_score = direct_score * 0.6  # transferred knowledge is 60% as reliable
```

---

# ITERATION 3: Final Optimized Architecture

After two iterations of design-then-critique, here is the convergent architecture.

---

## Final P1 Solution: Execution Signatures + Anchor Discrimination Power

### Schema Changes

```sql
-- Add execution detail capture to task_anchors
ALTER TABLE task_anchors ADD COLUMN execution_params TEXT DEFAULT '{}';

-- Add discrimination metric to nodes
ALTER TABLE nodes ADD COLUMN discrimination_power REAL DEFAULT 0.0;
-- ADP > 1.0 = variant-sensitive, ADP < 0.5 = reliably consistent
```

### Algorithm: Compute Discrimination Power

```python
def compute_discrimination_power(conn, anchor_id):
    """How much does this anchor's outcome vary based on execution?
    High ADP = the HOW matters. Low ADP = the WHAT is enough."""

    # Scores of tasks containing this anchor
    anchor_scores = conn.execute("""
        SELECT t.score FROM tasks t
        JOIN task_anchors ta ON ta.task_id = t.id
        WHERE ta.anchor_id = ? AND t.score > 0
    """, (anchor_id,)).fetchall()

    if len(anchor_scores) < 3:
        return 0.0  # insufficient data

    # All task scores (baseline variance)
    all_scores = conn.execute(
        "SELECT score FROM tasks WHERE score > 0"
    ).fetchall()

    anchor_var = statistics.variance([r["score"] for r in anchor_scores])
    all_var = statistics.variance([r["score"] for r in all_scores]) or 0.001

    return round(anchor_var / all_var, 3)
```

### Quantitative Framework

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **ADP** | `var(anchor_scores) / var(all_scores)` | >1.0: execution matters; <0.5: reliable |
| **Best Execution Score** | `max(task.score WHERE anchor in task)` | Ceiling potential |
| **Execution Range** | `max - min of task scores` | How much variation exists |
| **Consistency Index** | `1 / (stdev(scores) + 1)` | How predictable outcomes are |

### How It Changes Retrieval

When presenting an anchor with ADP > 1.0:
```
⚠️ "Threat-intel language works on CISOs" (w=0.85, ADP=1.4)
   Variant-sensitive: outcomes range from 2.0 to 4.9
   Best execution: "personalized threat data in opening" (4.9/5)
   Worst execution: "threat language in body only" (2.0/5)
   → Use the specific variant, not just the general insight
```

When presenting an anchor with ADP < 0.5:
```
✅ "9-section canonical layout" (w=0.88, ADP=0.3)
   Reliably consistent: always scores 4.0-4.8
   → Safe to use as-is, execution details don't matter much
```

---

## Final P2 Solution: Decision Log + Constraint-Driven Sessions

### Schema Changes

```sql
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    sc_id TEXT NOT NULL,
    anchor_id TEXT,               -- which anchor this decision relates to (optional)
    session_id TEXT NOT NULL,     -- conversation/task that made this decision
    decision TEXT NOT NULL,       -- "Use dark theme with Sora font"
    alternatives_eliminated TEXT DEFAULT '[]', -- JSON: ["light theme", "gradient design"]
    confidence REAL DEFAULT 0.5,  -- 0-1: how certain
    reversible INTEGER DEFAULT 1, -- 0 = locked in, 1 = can change
    evidence TEXT DEFAULT '',     -- why this decision was made
    created_at TEXT NOT NULL
);

-- Index for fast SC-based retrieval
CREATE INDEX IF NOT EXISTS idx_dec_sc ON decisions(sc_id);
```

### Algorithm: Session Convergence Protocol

```python
def get_convergence_state(conn, sc_id):
    """Compute how converged a super context's decisions are."""
    decisions = conn.execute(
        "SELECT * FROM decisions WHERE sc_id=? ORDER BY created_at", (sc_id,)
    ).fetchall()

    if not decisions:
        return {"convergence": 0.0, "decisions": 0, "locked": 0, "open": 0}

    locked = sum(1 for d in decisions if not d["reversible"])
    total = len(decisions)
    avg_confidence = sum(d["confidence"] for d in decisions) / total

    # Convergence: combination of decision count, lock ratio, and confidence
    lock_ratio = locked / total if total > 0 else 0
    convergence = (avg_confidence * 0.4) + (lock_ratio * 0.4) + (min(total / 10, 1.0) * 0.2)

    return {
        "convergence": round(convergence, 3),
        "total_decisions": total,
        "locked": locked,
        "open": total - locked,
        "avg_confidence": round(avg_confidence, 3)
    }
```

### How It Fixes Long Conversations

**Start of every session:**
```python
def session_preamble(conn, sc_id):
    """Generate constraint list for the AI at conversation start."""
    state = get_convergence_state(conn, sc_id)
    decisions = conn.execute(
        "SELECT decision, confidence, reversible FROM decisions WHERE sc_id=? AND confidence > 0.3 ORDER BY confidence DESC",
        (sc_id,)
    ).fetchall()

    constraints = []
    for d in decisions:
        lock = "🔒" if not d["reversible"] else "🔓"
        constraints.append(f"{lock} {d['decision']} (confidence: {d['confidence']:.0%})")

    return {
        "convergence": state["convergence"],
        "constraints": constraints,
        "instruction": "These decisions are established. Do not contradict locked decisions. Open decisions can be revisited with evidence."
    }
```

**During conversation:** When the AI makes a design choice, Claude stores it:
```python
memory.py decision "sc_id" '{"decision": "Use red strikethrough for reactive word", "confidence": 0.9, "reversible": 0, "evidence": "Tested on 3 pages, consistently strong"}'
```

**After conversation:** Decisions with score > 4.0 get confidence boost. Decisions in failed tasks (score < 2.5) get marked reversible again.

### Quantitative Framework

| Metric | Formula | Range | Meaning |
|--------|---------|-------|---------|
| **Convergence Score** | `0.4×avg_confidence + 0.4×lock_ratio + 0.2×min(decisions/10, 1)` | 0-1 | How "decided" is this domain |
| **Decision Confidence** | Starts at store-time estimate, evolves with task scores | 0-1 | Trust in this specific decision |
| **Decision Entropy** | `-Σ(p_i × log(p_i))` over alternative counts | 0-∞ | How many open questions remain |
| **Session Drift** | `count(contradictions with locked decisions) / total_new_decisions` | 0-1 | Is the AI respecting constraints? |

### Why This Is Better Than Hypothesis Trees

1. **No alignment detection needed** — decisions are explicit, not inferred
2. **No complex pruning** — individual decisions strengthen/weaken independently
3. **Composable** — decisions from different sessions naturally accumulate
4. **Queryable** — "Show me all open decisions for Website Design" is a simple SQL query
5. **Prevents overfitting** — the AI gets explicit constraints, not its own prior outputs fed back

---

## Final P3 Solution: Mode Vectors on Nodes + Dot-Product Retrieval

### Schema Changes

```sql
ALTER TABLE nodes ADD COLUMN mode_vector TEXT DEFAULT '{}';
-- JSON: {"work": 0.9, "creative": 0.3, "learning": 0.7, "social": 0.1, "travel": 0.0, "home": 0.2}

ALTER TABLE tasks ADD COLUMN mode_vector TEXT DEFAULT '{}';
```

### The 6 Modes

| Mode | Description | Example SCs |
|------|-------------|-------------|
| **work** | Professional execution, deadlines, deliverables | B2B Outreach, Website Design |
| **creative** | Open-ended exploration, design, writing | Editorial Writing, UI Design |
| **learning** | Structured knowledge acquisition | UPSC Prep, Technical Research |
| **social** | Communication, relationships, networking | Team Management, Client Relations |
| **travel** | Mobile context, limited attention, quick actions | Travel Planning, Quick Reference |
| **home** | Personal projects, relaxed pace, hobbies | Personal Finance, Home Projects |

### Algorithm: Mode Inference and Scoring

```python
# Predefined mode templates for common SC patterns
MODE_TEMPLATES = {
    "outreach": {"work": 0.9, "creative": 0.2, "social": 0.5},
    "website":  {"work": 0.7, "creative": 0.8, "learning": 0.2},
    "editorial":{"creative": 0.8, "learning": 0.6, "work": 0.3},
    "research": {"learning": 0.9, "work": 0.4, "creative": 0.2},
    "sales":    {"work": 0.9, "social": 0.6, "creative": 0.1},
}

def infer_mode_vector(sc_title, contexts):
    """Infer mode vector from SC title and context keywords."""
    title_lower = sc_title.lower()
    for keyword, template in MODE_TEMPLATES.items():
        if keyword in title_lower:
            return template
    # Default: balanced
    return {"work": 0.5, "creative": 0.5, "learning": 0.5,
            "social": 0.3, "travel": 0.1, "home": 0.3}

def mode_similarity(vec_a, vec_b):
    """Dot product similarity between two mode vectors."""
    all_keys = set(list(vec_a.keys()) + list(vec_b.keys()))
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in all_keys)
    mag_a = sum(v**2 for v in vec_a.values()) ** 0.5 or 1
    mag_b = sum(v**2 for v in vec_b.values()) ** 0.5 or 1
    return dot / (mag_a * mag_b)  # cosine similarity, range [0, 1]
```

### Integration with Retrieval

```python
# In cmd_retrieve, after computing word-overlap score:
query_mode = infer_mode_vector(query, [])
node_mode = json.loads(sc.get("mode_vector", "{}"))
mode_boost = 0.5 + mode_similarity(query_mode, node_mode)  # range [0.5, 1.5]

final_score = word_overlap * temporal_boost * mode_boost
```

### Quantitative Framework

| Metric | Formula | Range | Meaning |
|--------|---------|-------|---------|
| **Mode Similarity** | `cosine(query_mode, node_mode)` | 0-1 | Environmental alignment |
| **Mode Boost** | `0.5 + mode_similarity` | 0.5-1.5 | Retrieval multiplier |
| **Mode Entropy** | `-Σ(p_i × log(p_i))` of normalized vector | 0-log(6) | Specificity: low = niche, high = general |
| **Cross-Mode Score** | `avg(mode_similarity across all SCs)` | 0-1 | How universally applicable a memory is |

---

## Final P4 Solution: Transfer Edges + Structural Similarity

### Schema Changes

No new tables needed — uses existing `edges` table with new type:

```sql
-- Transfer edges between super contexts
INSERT INTO edges (id, src, tgt, type, weight, created_at)
VALUES (?, sc_1, sc_2, 'transfer', similarity_score, now);
```

### Algorithm: Automatic Transfer Detection

```python
def compute_structural_similarity(conn, sc_id_a, sc_id_b):
    """How similar are the internal structures of two super contexts?"""
    # Get context titles for each SC
    ctx_a = {c["title"].lower() for c in get_children(conn, sc_id_a)}
    ctx_b = {c["title"].lower() for c in get_children(conn, sc_id_b)}

    if not ctx_a or not ctx_b:
        return 0.0

    # Fuzzy Jaccard on context titles (reuse find_similar_node's fuzzy logic)
    matched = 0
    for ta in ctx_a:
        for tb in ctx_b:
            words_a = set(ta.split())
            words_b = set(tb.split())
            if len(words_a & words_b) / max(len(words_a | words_b), 1) > 0.3:
                matched += 1
                break

    structural_sim = matched / max(len(ctx_a), len(ctx_b))

    # Also check anchor overlap
    anc_a = set()
    for ctx in get_children(conn, sc_id_a):
        for anc in get_children(conn, ctx["id"]):
            anc_a.add(anc["title"].lower())

    anc_b = set()
    for ctx in get_children(conn, sc_id_b):
        for anc in get_children(conn, ctx["id"]):
            anc_b.add(anc["title"].lower())

    anchor_overlap = len(anc_a & anc_b) / max(len(anc_a | anc_b), 1)

    # Combined: 60% structural, 40% anchor overlap
    return round(structural_sim * 0.6 + anchor_overlap * 0.4, 3)


def detect_transfers(conn):
    """Find all SC pairs with structural similarity > threshold."""
    scs = conn.execute("SELECT id, title FROM nodes WHERE type='super_context'").fetchall()
    THRESHOLD = 0.25

    transfers = []
    for i, sc_a in enumerate(scs):
        for sc_b in scs[i+1:]:
            sim = compute_structural_similarity(conn, sc_a["id"], sc_b["id"])
            if sim >= THRESHOLD:
                # Create bidirectional transfer edges
                link_transfer(conn, sc_a["id"], sc_b["id"], sim)
                transfers.append((sc_a["title"], sc_b["title"], sim))
    return transfers
```

### Integration with Retrieval

```python
# In cmd_retrieve, after direct SC scoring:
def retrieve_with_transfers(conn, query, direct_results):
    """Augment direct retrieval with transfer knowledge."""
    seen_sc_ids = {r["sc_id"] for r in direct_results}

    for result in direct_results:
        # Find transfer edges from this SC
        transfers = conn.execute("""
            SELECT e.tgt as transfer_sc_id, e.weight as transfer_strength,
                   n.title as transfer_sc_title
            FROM edges e
            JOIN nodes n ON n.id = e.tgt
            WHERE e.src = ? AND e.type = 'transfer'
            ORDER BY e.weight DESC
        """, (result["sc_id"],)).fetchall()

        for t in transfers:
            if t["transfer_sc_id"] not in seen_sc_ids:
                # Retrieve from transferred SC at discounted weight
                transfer_pack = retrieve_sc(conn, t["transfer_sc_id"])
                transfer_pack["transfer_from"] = result["sc_title"]
                transfer_pack["transfer_strength"] = t["transfer_strength"]
                transfer_pack["relevance_discount"] = 0.6
                direct_results.append(transfer_pack)
                seen_sc_ids.add(t["transfer_sc_id"])

    return direct_results
```

### Quantitative Framework

| Metric | Formula | Range | Meaning |
|--------|---------|-------|---------|
| **Structural Similarity** | `0.6×context_jaccard + 0.4×anchor_overlap` | 0-1 | How alike two SCs are |
| **Transfer Strength** | Structural similarity score stored on edge | 0-1 | How much to trust transferred knowledge |
| **Transfer Discount** | Fixed at 0.6 (adjustable) | 0-1 | Retrieval weight for transferred anchors |
| **Transfer Success Rate** | `avg(task_score where transferred anchors used) / avg(all task scores)` | 0-∞ | Does transferred knowledge actually help? |
| **Transfer Reach** | `count(SCs reachable via transfer edges from SC_X)` | 0-N | How connected is this domain's knowledge |

---

## Complete Quantitative Parameter Map

### All Parameters in One Table

| Parameter | Location | Default | Tunable Range | What It Controls |
|-----------|----------|---------|---------------|-----------------|
| `evidence_power` | `compute_evidence_weight` | 0.7 | 0.5-1.0 | Score amplification curve |
| `decay_schedule` | `compute_evidence_weight` | [0.7, 0.5, 0.3, 0.15] | Per-tier | Initial vs evidence trust |
| `established_freq_threshold` | `classify_stage` | 0.15 | 0.1-0.3 | Min frequency for "established" |
| `active_freq_threshold` | `classify_stage` | 0.05 | 0.03-0.1 | Min frequency for "active" |
| `coincidence_guard` | `classify_stage` | f<0.1 ∧ c>0.8 | Adjustable | Prevents false "active" |
| `jaccard_threshold` | `find_similar_node` | 0.35 | 0.25-0.50 | Dedup sensitivity |
| `fuzzy_prefix_len` | `find_similar_node` | 4 | 3-5 | Stem matching strictness |
| **NEW: `adp_threshold`** | Variant system | 1.0 | 0.5-2.0 | When to flag variant-sensitive anchors |
| **NEW: `convergence_weights`** | Decision system | [0.4, 0.4, 0.2] | Triplet summing to 1 | Confidence vs locks vs count balance |
| **NEW: `decision_reinforce`** | Decision system | 0.1 | 0.05-0.2 | Confidence gain per supporting session |
| **NEW: `decision_decay`** | Decision system | 0.7 | 0.5-0.9 | Confidence loss per contradicting session |
| **NEW: `mode_boost_range`** | Mode system | [0.5, 1.5] | [0.3, 1.8] | Environmental influence on retrieval |
| **NEW: `transfer_threshold`** | Transfer system | 0.25 | 0.15-0.40 | Min similarity for transfer edge |
| **NEW: `transfer_discount`** | Transfer system | 0.6 | 0.3-0.8 | How much to trust transferred knowledge |

### Derived System-Level Metrics

| Metric | Formula | Healthy Range | Alert If |
|--------|---------|--------------|----------|
| **Memory Health** | `avg(weight) × (1 - fading_ratio)` | 0.4-0.8 | < 0.3 |
| **Knowledge Velocity** | `new_anchors_per_week / total_anchors` | 0.02-0.15 | > 0.20 (too fast) or < 0.01 (stagnant) |
| **Convergence Rate** | `Δconvergence / session_count` per SC | 0.05-0.15/session | < 0.02 (indecisive) |
| **Transfer Utilization** | `transferred_anchors_used / total_retrievals` | 0.1-0.3 | 0 (no cross-pollination) |
| **Discrimination Coverage** | `count(anchors with ADP computed) / total_anchors` | > 0.5 | < 0.2 (not enough scored tasks) |

---

## Implementation Priority

| Phase | What | Effort | Impact | Dependencies |
|-------|------|--------|--------|-------------|
| **Phase 1** | Decision Log + Convergence | Medium | High | None — standalone |
| **Phase 2** | Transfer Edges + Structural Similarity | Medium | High | Need 3+ SCs with structure |
| **Phase 3** | ADP (Discrimination Power) | Low | Medium | Need scored tasks |
| **Phase 4** | Mode Vectors | Low | Medium | Mode templates |
| **Phase 5** | Execution Parameters on task_anchors | Low | Medium | Need Claude to capture params |
| **Future** | Small model for pattern extraction | High | Very High | Need 500+ variants for training data |

### When Does the Small Model Become Necessary?

Your suggestion of a trained model is the right end-state. Here's the quantitative trigger:

| Trigger | Threshold | Why |
|---------|-----------|-----|
| Total anchors | > 500 | Rule-based dedup Jaccard breaks down at scale |
| Unique SCs | > 20 | Transfer detection becomes O(N²) expensive |
| Scored tasks | > 100 | Enough training data for meaningful patterns |
| ADP computed anchors | > 200 | Pattern extraction across variants becomes viable |

At that point, a small model (fine-tuned sentence transformer, ~30M params) would:
1. Replace fuzzy Jaccard dedup with learned semantic similarity
2. Auto-detect transfer edges via embedding proximity
3. Generate "next best variant" suggestions from ADP patterns
4. Infer mode vectors from content instead of keyword templates

But today, with 177 anchors and 0 scored tasks — the rule-based system with these 4 extensions is optimal.

---

## Architecture Diagram: PMIS v3

```
                    ┌─────────────────────────────┐
                    │     RETRIEVAL ENGINE         │
                    │                              │
                    │  word_overlap × temporal_boost│
                    │  × mode_boost × transfer     │
                    │  + ADP annotation             │
                    │  + decision constraints       │
                    └──────────┬──────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                      │
    ┌────▼────┐         ┌──────▼──────┐        ┌─────▼─────┐
    │ SCORING │         │  TEMPORAL   │        │  TRANSFER │
    │ ENGINE  │         │  ENGINE     │        │  ENGINE   │
    │         │         │             │        │           │
    │ weight  │         │ recency     │        │ structural│
    │ ADP     │         │ frequency   │        │ similarity│
    │ evidence│         │ consistency │        │ discount  │
    │ variants│         │ stage       │        │ reach     │
    └────┬────┘         └──────┬──────┘        └─────┬─────┘
         │                     │                      │
         └─────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │      GRAPH DATABASE          │
                    │                              │
                    │  nodes (SC > CTX > Anchor)   │
                    │  edges (parent_child +       │
                    │         transfer)            │
                    │  decisions (per SC)           │
                    │  tasks + task_anchors         │
                    │    (+ execution_params)       │
                    │  mode_vector (per node)       │
                    └──────────────────────────────┘
```
