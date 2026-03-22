# PMIS AutoResearch Program

## Objective
Optimize PMIS retrieval to be goal-oriented, context-aware, and session-continuous.

## What the agent can modify
- `scripts/p9_retrieve.py` — retrieval algorithm (tag weights, BM25 params, fusion strategy)
- `scripts/autoresearch_retrieve.py` — experimental retrieval variants

## What is fixed
- `Graph_DB/graph.db` — the memory database (231 anchors, 71 contexts, 12 SCs)
- `scripts/memory.py` — core store/score/rebuild operations
- `scripts/ground_truth.py` — benchmark queries and expected results

## Metric
**Composite score** (lower is better, like val_bpb):
```
score = 1.0 - (0.5 * retrieval_f1 + 0.3 * session_continuity + 0.2 * goal_alignment)
```

- retrieval_f1: precision/recall of returned anchors vs ground truth
- session_continuity: does the system maintain context across related queries?
- goal_alignment: are returned anchors relevant to the user's current goal?

## Constraints
- Each experiment must complete in < 60 seconds on M2 CPU
- No external API calls (everything local)
- No modifications to the database during experiments
- All changes must be in a single file

## Strategy
1. Start with current P9+ as baseline
2. Experiment with: tag weight ratios, BM25 k1/b parameters,
   context window for session continuity, goal vector dimensions,
   divergence threshold for session breaks
3. Each experiment modifies ONE parameter at a time
4. Log all results for analysis
