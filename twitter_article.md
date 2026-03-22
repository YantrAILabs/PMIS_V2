# We built a memory system that scores 91.4% on LongMemEval — with zero LLM calls.

No GPT. No Gemini. No API keys. No cloud. Just Python stdlib and a 464KB SQLite file.

Here's how we got there, what it means, and why we think the future of agent memory isn't in bigger models — it's in better structure.

---

## The Problem

Every AI memory system today — Supermemory, Zep, Mastra, EmergenceMem — relies on frontier LLMs to function. They send your conversation history to GPT-4o or Gemini, pay $0.01-0.05 per query, wait 2-5 seconds, and hope the model extracts the right memories.

This works. Supermemory hits 85.2%. Mastra hits 94.9%. But you're paying for inference on every recall. You need internet. You need API keys. And your memories live on someone else's server.

We asked: what if the structure itself was intelligent enough that you didn't need an LLM to retrieve?

---

## What We Built

PMIS — Personal Memory Intelligence System.

The architecture is simple: a three-level tree.

```
Super Context → Context → Anchor
     ↓              ↓         ↓
  "Domain"      "Skill"    "Fact"
```

Every piece of knowledge is an atomic anchor. Anchors group into contexts (skills/phases). Contexts group into super contexts (work domains). The tree is stored in a single SQLite database.

Retrieval is pure algorithmic — no neural networks, no embeddings API, no LLM judge. Just BM25 keyword search, tag-based Jaccard matching, and a fusion layer that blends signals.

That's it. That's the entire system.

---

## The Benchmark

LongMemEval is the standard benchmark for memory systems. Published at ICLR 2025, it tests whether a system can find the right information buried in 50+ conversation sessions spanning 115,000+ tokens.

500 questions. Six categories. Every major memory system has been evaluated on it.

We ran PMIS against the exact same 500 questions.

**Result: 91.4%**

```
Category                    PMIS     Questions
─────────────────────────────────────────────
Single-session Assistant   100.0%      56
Knowledge Update            98.7%      78
Multi-session               92.5%     133
Temporal Reasoning          87.2%     133
Single-session User         87.1%      70
Single-session Preference   80.0%      30
─────────────────────────────────────────────
OVERALL                     91.4%     500
```

---

## Where That Puts Us

```
System                  Score    LLM Required    Cost/Query
──────────────────────────────────────────────────────────
Chronos High            95.6%    Yes (frontier)   $$$
Mastra OM (gpt-5-mini)  94.9%    Yes (gpt-5)      $$$
Mastra OM (gemini-3)    93.3%    Yes (gemini-3)    $$
PMIS P11               91.4%    No                $0
Hindsight (gemini-3)    91.4%    Yes (gemini-3)    $$
EmergenceMem            86.0%    Yes (gpt-4o)      $$
Supermemory (gemini-3)  85.2%    Yes (gemini-3)    $$
Supermemory (gpt-4o)    81.6%    Yes (gpt-4o)      $
Zep/Graphiti            71.2%    Yes (gpt-4o)      $
Full Context GPT-4o     60.2%    Yes (gpt-4o)     $$$
```

PMIS ties with Hindsight and beats every version of Supermemory, EmergenceMem, and Zep.

The only systems above us use frontier models that cost real money on every query.

---

## The Journey: 50% → 91.4%

We didn't start at 91%. We started at 50%.

The system evolved through four versions, each adding structural innovations — not more parameters, not bigger models, but better algorithms for how memory is organized and searched.

**P9 (50%)** — The baseline. Triple fusion: BM25 + tag matching + optional TF-IDF vectors. Session tracking with cosine divergence detection. Goal alignment vectors. All the fundamentals.

**P10 (74%)** — Three structural fixes.
- Feedback reinforcement became multiplicative instead of additive. User scores now genuinely reorder what surfaces first.
- Session context started filtering results, not just boosting them. Each turn narrows the search space.
- Short queries got a direct-match bypass. "MediaMTX" shouldn't go through the full BM25 pipeline.

**P11 (94% internal, 91.4% LongMemEval)** — The semantic layer.
- Domain synonym expansion. 30 synonym groups mapping terms like "camera" to "cctv/surveillance/cam" and "memory" to "recall/retrieve/pmis."
- Morphological stemming. 14 suffix rules so "deploying" matches "deployment."
- Compound query splitting. "Cold email for construction safety" becomes two sub-queries scored independently, enabling cross-domain retrieval.
- SC size normalization. Large memory domains were drowning out smaller, more relevant ones. A square-root penalty fixed this.

---

## The AutoResearch Loop

We borrowed an idea from Karpathy's autoresearch: instead of manually tuning parameters, let an automated loop mutate → benchmark → keep/discard → repeat.

We ran 6,000+ experiments across three rounds. The system explored 28 tunable parameters — BM25 constants, fusion weights, temporal decay rates, feedback amplifiers, tightening thresholds.

Key discovery: **BM25 should dominate retrieval, not tags or vectors.** The optimized fusion is 56% BM25, 28% tags, 15% vectors. Most memory systems over-index on embeddings. For structured knowledge, keyword matching with proper IDF weighting beats dense retrieval.

Other findings:
- Established memories should get 1.87x retrieval boost (higher than default 1.5x)
- Active memories should actually get *less* boost than default (0.8x vs 1.2x)
- Returning 10 results instead of 3 improved recall without hurting precision
- Feedback amplification works best at 0.8x with a quality floor of 0.3

---

## The Architecture That Makes It Work

Five engines, all running locally in pure Python:

**1. BM25 Engine** — Classic keyword search with tunable k1/b parameters. Builds inverted index over every node's title + description + content. Each node includes its full path (SC → Context → Anchor) for richer matching.

**2. Tag Engine** — Every anchor has auto-generated domain tags. Query tags are matched via Jaccard similarity. Tags are generated from the full SC→Context→Anchor path, not just the anchor title.

**3. Synonym Engine** — 30 domain-specific synonym groups. Query tokens are expanded before both BM25 and tag search. Abbreviations (PPE, ANPR, GPU) are expanded to full forms.

**4. Session Engine** — Tracks conversation context as a word-frequency vector. Detects topic divergence via cosine similarity. Progressively tightens retrieval scope across turns. Auto-breaks on divergence.

**5. Feedback Engine** — Multiplicative quality scoring. SC quality (from user-scored tasks) scales with evidence depth. More scored tasks → more the system trusts evidence over initial estimates.

The fusion layer merges all five signals using Reciprocal Rank Fusion (RRF) blended with raw weighted scores (70/30 split). Transfer edges connect structurally similar domains for cross-pollination.

---

## The Versioning System

Every version of the engine is snapshotted:

```
versions/
  P9/       # 50%  — baseline
  P10/      # 74%  — structural fixes
  P11/      # 94%  — semantic layer
  P11-final/ # 100% on internal, 91.4% on LongMemEval
```

Each snapshot stores: all source files, optimized parameters, benchmark scores, SHA256 checksums, database stats, architecture notes, and known limitations.

Single-command rollback:
```
python3 scripts/version.py restore P9
```

---

## What We Learned

**1. Structure beats scale.** A well-organized 464KB SQLite file retrieves better than a million-token context window. The tree hierarchy (SC → Context → Anchor) means every retrieval is a tree traversal, not a needle-in-haystack search.

**2. Synonyms matter more than embeddings.** Our biggest single gain came from a 30-entry synonym dictionary. Not a 768-dimensional embedding space. Not a fine-tuned retriever. Just "camera = cctv = surveillance = cam."

**3. Feedback must be multiplicative.** Additive quality bonuses (+0.13 max) don't move rankings. Multiplicative quality multipliers (0.3x to 1.5x) genuinely separate proven knowledge from unproven.

**4. Large domains dominate unless normalized.** Without square-root size normalization, our largest SC appeared in the top-3 of every retrieval regardless of relevance. This is the memory equivalent of a popularity bias.

**5. Context tightening is how humans think.** Each question in a conversation should narrow the search space, not restart it. Tracking a session vector and penalizing off-topic SCs mimics how human memory progressively focuses.

**6. Short queries need special handling.** One-word queries like "MediaMTX" produce near-zero BM25 signal because BM25 depends on document frequency ratios. A direct substring bypass catches these.

---

## What's Next

91.4% is strong. But three categories are below 90%:

- **Single-session Preference (80%)** — Extracting implicit preferences from conversation. This needs better attention to hedging language ("I prefer," "I usually").
- **Temporal Reasoning (87.2%)** — Understanding event sequences and time calculations. This needs explicit temporal indexing of conversation dates.
- **Single-session User (87.1%)** — Recalling user-stated facts. This needs better entity extraction at store time.

Each of these is a structural problem — not a parameter tuning problem. We have 85 more experiments designed and ready to implement.

The system runs on a MacBook Air M2 with 16GB RAM. No GPU. No cloud. 1,500 queries per hour. $0.00 per query.

The code is structured, versioned, and benchmarked. Every improvement is measured against the same 500-question standard that Supermemory, Mastra, and Chronos use.

---

## The Takeaway

The memory layer for AI agents doesn't have to be expensive. It doesn't have to be cloud-dependent. It doesn't have to call GPT-4o on every retrieval.

With the right structural innovations — hierarchical trees, domain synonyms, compound query splitting, multiplicative feedback, progressive context tightening — a zero-dependency local system can compete with the best in the world.

91.4% on LongMemEval. Zero LLM calls. 464KB of SQLite.

That's the pitch.

---

*Built at Yantra AI Labs. Open to collaboration — reach out if you're working on agent memory.*
