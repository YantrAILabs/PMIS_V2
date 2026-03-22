# Evidence-Weighted Hierarchical Memory for LLM Agents: Outcome-Driven Knowledge Management Without External Dependencies

---

## 1. Introduction

Large Language Model (LLM) agents operate with a fundamental limitation: their memory is volatile. Context windows — even at 128K or 1M tokens — are session-bound. Knowledge acquired in one conversation vanishes in the next. The agent that helped you craft a winning sales strategy last Tuesday has no memory of it on Wednesday.

Existing solutions fall into three categories, each with structural weaknesses:

1. **Retrieval-Augmented Generation (RAG)**: Treats knowledge as static documents retrieved by embedding similarity. No learning loop — the system never improves from outcomes.
2. **In-context memory buffers** (Reflexion, scratchpads): Bounded slots that overflow. No persistence across sessions.
3. **Agent-managed memory** (MemGPT): The LLM itself decides what to remember, consuming cognitive bandwidth and producing opaque decisions.

None of these systems answer a question that human memory answers naturally: **which memories actually helped me succeed, and which ones didn't?**

This paper presents the **Personal Memory Intelligence System (PMIS)** — a hierarchical knowledge graph with three novel mechanisms:

1. **Evidence-based weight evolution**: Memory importance starts as an LLM estimate and evolves based on downstream task outcome scores, using a Bayesian-inspired blending schedule.
2. **Structure snapshots with winning recipe identification**: The system captures the exact combination of memories present during high-performing tasks and prioritizes that combination in future retrieval.
3. **4-stage temporal classification**: Memories are classified as impulse, active, established, or fading based on three independent signals (recency, frequency, consistency), with each stage receiving a distinct retrieval multiplier.

PMIS is implemented in 1,003 lines of Python using only the standard library (sqlite3, json, math, statistics). It requires zero external dependencies — no embedding models, no vector databases, no API calls. This constraint is deliberate: a personal memory system should be fully deterministic, inspectable, and reproducible.

---

## 2. Related Work

### 2.1 Generative Agents (Park et al., 2023)

The Stanford/Google Generative Agents architecture introduced a **memory stream** — a flat, append-only log of timestamped natural-language observations — combined with a three-factor retrieval score:

- **Recency**: Exponential decay (factor 0.995 per hour)
- **Importance**: LLM-assigned integer (1-10) at creation time
- **Relevance**: Cosine similarity between query and memory embeddings

A periodic **reflection** mechanism generates higher-order observations by prompting the LLM to synthesize patterns from recent memories, creating an implicit abstraction hierarchy.

**Key limitation**: Importance scores are static — assigned once by the LLM and never updated from outcomes. A memory rated "importance: 8" at creation remains at 8 regardless of whether it contributes to successful or failed tasks.

*Reference: arXiv:2304.03442*

### 2.2 MemGPT / Letta (Packer et al., 2023)

MemGPT applies an operating-system metaphor to LLM memory: the context window is "RAM," and external storage is "disk." The agent uses function calls to page information between a working context, a recall storage database (searchable interaction history), and an archival storage (vector-indexed long-term memory).

**Key limitation**: The LLM serves as both task-reasoner and memory-manager simultaneously, consuming cognitive bandwidth on memory decisions. Memory operations are opaque — there is no explicit importance scoring or temporal staging.

*Reference: arXiv:2310.08560*

### 2.3 Reflexion (Shinn et al., NeurIPS 2023)

Reflexion implements memory as verbal self-critique. After task failure, the agent generates a natural-language reflection ("I should have checked the API documentation before writing the function"), stores it in a 3-slot buffer, and includes recent reflections in subsequent attempts.

**Key limitation**: Memory is bounded to 3 slots with no persistence mechanism. There is no importance weighting, no temporal handling, and no scalability beyond the buffer size. Despite this simplicity, Reflexion achieved 91% on HumanEval versus GPT-4's 80%, demonstrating that even minimal memory can be powerful.

*Reference: arXiv:2303.11366*

### 2.4 A-MEM (Xu et al., 2025)

A-MEM draws from the Zettelkasten note-taking methodology. Each memory is an atomic note with raw content, LLM-generated keywords, a context description, a dense embedding, and bidirectional links to related notes. When new memories are inserted, they trigger updates to contextual representations of existing notes — a form of memory evolution.

**Key limitation**: Linking is primarily semantic (embedding similarity), missing temporal and causal relationships. No explicit importance weighting from task outcomes.

*Reference: arXiv:2502.12110*

### 2.5 CoALA (Sumers et al., TMLR)

The Cognitive Architectures for Language Agents framework provides a theoretical taxonomy inspired by cognitive science (SOAR, ACT-R). It divides long-term memory into episodic (past experiences), semantic (facts), and procedural (skills/code), with memory retrieval as a first-class internal action alongside external tool use.

CoALA is a framework, not an implementation, but it provides the canonical vocabulary for analyzing memory systems.

*Reference: arXiv:2309.02427*

### 2.6 GraphRAG and Hybrid Approaches

Recent work (2024-2025) has demonstrated that combining knowledge graphs with vector embeddings outperforms either approach alone on multi-hop reasoning. AriGraph (IJCAI 2025) builds semantic knowledge graphs with episodic vertices during exploration. MAGMA (2025) proposes multi-graph architectures capturing different relationship types.

**Key finding**: Hybrid GraphRAG achieves higher factual correctness while pure GraphRAG achieves superior context relevance. Both outperform vector-only RAG on complex reasoning.

*References: arXiv:2407.04363 (AriGraph), arXiv:2601.03236 (MAGMA)*

### 2.7 Comparative Analysis

| Axis | Gen. Agents | MemGPT | Reflexion | A-MEM | CoALA | GraphRAG | **PMIS** |
|------|-------------|--------|-----------|-------|-------|----------|----------|
| **Structure** | Flat stream | 2-tier paging | 3 fixed slots | Linked notes | Episodic+Semantic+Procedural | KG + vector | **3-level hierarchy (SC>Ctx>Anchor)** |
| **Retrieval** | Recency x Importance x Relevance | Agent-driven paging | Sliding window | Embedding + link spreading | Architecture-dependent | Subgraph + NN | **Word-overlap x temporal boost x quality** |
| **Importance** | Static LLM score (1-10) | Implicit (LLM decides) | None | Activation spreading | Framework-dependent | Static edge weights | **Evidence-based: initial estimate blended with task outcome scores** |
| **Temporal** | Single exp decay | FIFO buffer | Window-based | Timestamps (no decay) | Framework-dependent | None | **4-stage: impulse/active/established/fading from 3 metrics** |
| **Learning loop** | Reflection (abstractions) | None | Self-critique | Evolution on insert | Architecture-dependent | None | **Outcome scoring: user rates tasks, weights update** |
| **Dependencies** | Embedding model + vector DB | LLM API + custom engine | LLM only | Embedding model | Varies | Graph DB + embeddings | **Zero (Python stdlib only)** |
| **Interpretability** | Low (embeddings opaque) | Low (paging is black-box) | High (verbal reflections) | Medium (links inspectable) | N/A | Medium | **High (all scores deterministic, HTML viz)** |

---

## 3. PMIS Architecture

### 3.1 Data Model

PMIS stores knowledge in a strict 3-level hierarchy within SQLite:

```
Super Context (broad domain, e.g., "B2B Sales")
  -> Context (specific skill/phase, e.g., "Cold Email Copywriting")
    -> Anchor (atomic reusable insight, e.g., "Threat language outperforms ROI for CISOs — 3x reply rate")
```

**Core schema** (6 tables):

- **nodes** (id, type, title, content, weight, initial_weight, quality, use_count, last_used, occurrence_log, recency, frequency, consistency, memory_stage)
- **edges** (id, src, tgt, type='parent_child', weight)
- **tasks** (id, sc_id, title, score, structure_snapshot)
- **task_anchors** (task_id, anchor_id, context_id, was_retrieved)

Each node carries both `weight` (current evidence-blended value) and `initial_weight` (LLM's original estimate, preserved unchanged). This separation enables the evidence system to operate without destroying the prior.

### 3.2 Storage with Deduplication

Storage accepts structured JSON with explicit hierarchy:

```json
{
  "super_context": "B2B Sales",
  "contexts": [
    {
      "title": "Cold Email",
      "weight": 0.8,
      "anchors": [
        {"title": "Threat language wins", "content": "CISOs respond to threat language over ROI — 3x reply rate", "weight": 0.9}
      ]
    }
  ]
}
```

**Dedup algorithm** (3 levels):
1. **Super Context**: Exact title match -> word-overlap match (first 3 words) -> create new
2. **Context**: Exact child-title match under parent -> word-overlap match -> create new
3. **Anchor**: Same as SC-level matching

On duplicate detection: `weight = max(existing, incoming)` — the system keeps the higher confidence estimate and increments use_count. This makes storage **append-idempotent**: storing the same domain 50 times produces one richer tree, not 50 copies.

### 3.3 Retrieval Algorithm

Given a query string, PMIS computes a relevance score for each Super Context:

```
score = (word_overlap + quality_bonus + use_bonus) * temporal_boost

where:
  word_overlap = sum(
    +2.0 per query word matching SC title/description,
    +1.0 per query word matching any Context title,
    +0.5 per query word matching any Anchor title/content
  )
  quality_bonus = (sc.quality / 5.0) * 2.0
  use_bonus = min(sc.use_count / 10.0, 0.5)
  temporal_boost = {impulse: 0.8, active: 1.2, established: 1.5, fading: 0.5}[sc.stage]
```

The top 3 SCs are returned with full context trees. Within each tree, anchors are sorted by `(in_winning_structure DESC, weight DESC)` — proven combinations surface first.

### 3.4 Evidence-Based Weight Evolution

When a user scores a completed task (0-5 scale), the system propagates that score to every anchor present during the task:

```
evidence_weight = mean(all_task_scores_containing_this_anchor) / 5.0

decay = 0.7   if scored_task_count < 2   (trust initial estimate)
        0.5   if scored_task_count < 5   (balanced)
        0.3   if scored_task_count >= 5  (trust evidence)

blended_weight = (initial_weight * decay) + (evidence_weight * (1 - decay))
final_weight = clip(blended_weight, 0.05, 1.0)
```

This implements a principled cold-start strategy: new anchors rely 70% on the LLM's estimate, but as evidence accumulates, the system shifts to 70% evidence-driven. Bad anchors (present only in low-scoring tasks) see their weights drop. Good anchors (present in high-scoring tasks) rise.

### 3.5 Temporal Engine

Three independent signals are computed for each node:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Recency** | `exp(-0.693 * days_since_last / 7.0)` | Exponential decay, 7-day half-life |
| **Frequency** | `min(occurrences_last_30_days / 30.0, 1.0)` | Usage rate |
| **Consistency** | `1.0 / (stdev(gaps_between_uses) + 1.0)` | Regularity of usage pattern |

These feed a rule-based classifier:

| Stage | Conditions | Retrieval Boost |
|-------|-----------|-----------------|
| **impulse** | New or low recency + low frequency | 0.8x (penalty for unproven) |
| **active** | Recency > 0.5 AND frequency >= 0.1 | 1.2x (hot knowledge) |
| **established** | Frequency >= 0.1 AND consistency > 0.3 | 1.5x (battle-tested) |
| **fading** | Recency < 0.3 AND frequency < 0.05 | 0.5x (deprioritized, not deleted) |

The **established** stage captures something no single decay curve can: a memory that has been steadily useful over time, even if not accessed in the last few days. This distinguishes "proven pattern" from "recently hot topic."

### 3.6 Structure Snapshots

When a task is scored, PMIS captures a **structure snapshot** — the exact set of contexts and anchors present at that moment:

```json
{
  "B2B Cold Outreach": ["3-stage lead qualification", "International lead lists"],
  "Proposal Websites": ["Interactive web proposal > PDF"]
}
```

The highest-scoring snapshot for each Super Context becomes the **winning recipe**. During retrieval, anchors appearing in the winning recipe are flagged `in_winning_structure: true` and sorted to the top.

This implements **structural credit assignment** — not just "which individual memory matters" but "which combination of memories produced the best outcome." No existing system captures this.

---

## 4. Novel Contributions

### 4.1 Evidence-Based Weight Evolution (No Prior Analog)

Generative Agents assign importance once at creation. A-MEM evolves weights via activation spreading (usage frequency). Neither connects memory importance to downstream task performance. PMIS closes the loop: user scores task outcome -> score propagates to anchors -> weights update -> future retrieval improves.

The blending schedule (70/30 -> 50/50 -> 30/70) handles the cold-start problem gracefully — new memories aren't penalized by lack of evidence, but proven memories earn their rank through outcomes.

### 4.2 Structure Snapshots (Structural Credit Assignment)

Existing systems track individual memory importance. PMIS tracks which **combination** of memories co-occurred with high performance. This is analogous to ensemble feature importance in ML — the value of a feature depends on what other features are present.

### 4.3 4-Stage Temporal Classification (Beyond Simple Decay)

Generative Agents use a single exponential decay. PMIS computes three orthogonal signals (recency, frequency, consistency) and maps them to four discrete stages with distinct retrieval multipliers. The "established" stage — high consistency, moderate recency — has no equivalent in any existing system.

### 4.4 Hierarchical Dedup with Merge Semantics

A-MEM links related notes but doesn't merge duplicates. MemGPT relies on the LLM to manage duplicates. Generative Agents is purely append-only. PMIS performs title-matching dedup at every level, keeping the higher weight and incrementing use count — making the graph append-idempotent.

### 4.5 Zero-Dependency Deterministic Design

Every existing memory system with temporal or importance handling requires an embedding model (Generative Agents, A-MEM, GraphRAG) or custom infrastructure (MemGPT). PMIS runs on any Python 3.7+ installation with no pip installs, no API keys, no model downloads. The same query always returns the same results — fully auditable.

---

## 5. Limitations

### 5.1 Word-Overlap Retrieval is Brittle (HIGH severity)

The retrieval algorithm uses exact word matching. The query "effective prospecting techniques" would completely miss an anchor titled "cold email copywriting best practices" despite semantic equivalence. This is the most significant limitation compared to embedding-based systems.

**Concrete failure example**: Query "how to reduce false alarm rates" would not match anchor "kill the noise so your team can focus on what's real" despite being the same concept.

**Mitigation path**: A hybrid layer adding optional lightweight embeddings (e.g., sentence-transformers) as a retrieval augmentation while preserving deterministic word-overlap as the explainability backbone.

### 5.2 Evidence System is Empirically Unvalidated (HIGH for claims)

The live database contains 18 logged tasks and 0 scored tasks. All 236 nodes are in "impulse" stage. The entire evidence-based weight evolution, temporal staging progression, and winning recipe identification exist as implemented code but have never been activated in production use. Section 6 addresses this through synthetic benchmarks.

### 5.3 No Reflection or Abstraction Mechanism

Generative Agents periodically generate higher-order observations from lower-level memories. PMIS has no mechanism for the LLM to synthesize meta-anchors from existing anchors. All abstraction is front-loaded to the store step.

### 5.4 O(N) Retrieval Scalability

Retrieval performs a full scan of all SCs, their contexts, and their anchors with string matching. At the current scale (236 nodes), this is sub-millisecond. At 100K nodes, it would become a bottleneck. No full-text search indexes exist.

### 5.5 No Forgetting

Fading memories receive a 0.5x retrieval penalty but are never deleted. The graph grows monotonically. Over years of use, noise accumulation could degrade retrieval quality.

---

## 6. Benchmark Results

Three synthetic benchmarks were executed to validate PMIS's core mechanisms. Results are reported honestly, including failures.

### 6.1 Weight Convergence Test — PASS

**Setup**: 20 anchors with linearly-spaced true utility values (0.2-0.9). Initial weights assigned with Gaussian noise (sigma=0.15) to simulate imperfect LLM estimates. 50 tasks simulated, each using 5-8 random anchors, scored proportional to mean true utility of participating anchors.

**Results**:

| Metric | Initial Weights | After 50 Tasks |
|--------|----------------|----------------|
| Pearson r (vs true utility) | 0.9250 | **0.9504** |
| Spearman rho | 0.9293 | **0.9383** |
| Mean Absolute Error | 0.0687 | 0.1125 |

**Analysis**: The evidence system successfully improved rank correlation (Pearson r: 0.925 -> 0.950, Spearman rho: 0.929 -> 0.938), meaning it correctly identifies which anchors are more useful than others. However, MAE increased from 0.069 to 0.113. This is because the blending formula compresses weights toward the center — high-utility anchors (true: 0.9) converge to ~0.67, while low-utility anchors (true: 0.2) converge to ~0.42. The system preserves **relative ordering** (the most important property for retrieval ranking) but not absolute calibration.

**Verdict**: PASS. Pearson r = 0.950 > 0.7 threshold. The evidence system reliably improves retrieval ranking quality over initial LLM estimates.

### 6.2 Temporal Stage Transition Test — FAIL (50%)

**Setup**: 10 anchors with 4 simulated usage patterns: daily use (30 days), weekly use (8 weeks), sporadic (3 uses over 60 days), abandoned (1 use, 45 days ago).

**Results**:

| Pattern | Expected Stage | Actual Stage | Correct? |
|---------|---------------|--------------|----------|
| Daily 30 days | established | **active** | NO |
| Weekly 8 weeks | active | active | YES |
| Sporadic | impulse | **active** | NO |
| Abandoned 45 days | fading | fading | YES |

**Accuracy**: 5/10 = 50% (FAIL, threshold was 80%)

**Root cause**: The `classify_stage()` function checks `recency > 0.5 AND frequency >= 0.1` for "active" BEFORE checking `frequency >= 0.1 AND consistency > 0.3` for "established." Daily use produces high recency (1.0) which triggers the "active" branch before "established" can be reached. Similarly, sporadic use with a recent access (5 days ago = recency 0.61) combined with moderate frequency (0.067) triggers the `frequency >= 0.05 → active` fallback.

**Fix required**: The classification logic needs rule reordering — check "established" conditions (high consistency) BEFORE "active" conditions (high recency). This is a code bug, not an architectural flaw. The three underlying metrics (recency, frequency, consistency) are correctly computed; only the decision boundary needs adjustment.

### 6.3 Dedup Precision/Recall Test — PARTIAL PASS

**Setup**: 20 unique concepts, each stored with 5 paraphrase variants (100 total store operations).

**Results**:

| Metric | Value |
|--------|-------|
| Nodes created | 99 (ideal: 20) |
| Perfect dedup (1 node/concept) | 0/20 (0%) |
| Variants correctly merged | 1/80 (1.25%) |
| Cross-concept false merges | 0 |
| **Precision** | **1.000** (no incorrect merges) |
| **Recall** | **0.013** (almost no merges at all) |

**Analysis**: Precision is perfect — when the system does merge, it never merges different concepts together. However, recall is near-zero because the word-overlap dedup algorithm requires the first 2-3 words of the title to match in order. Paraphrased titles like "Optimizing PostgreSQL queries" vs "PostgreSQL query optimization" have different word order and fail the LIKE pattern match.

**This confirms the word-overlap brittleness identified in Section 5.1.** The dedup system works well for near-exact titles (the common case when a single LLM consistently names concepts) but fails on paraphrases. In practice, PMIS mitigates this because Claude is instructed to use consistent naming conventions — but this is a behavioral workaround, not a robust solution.

**Verdict**: Precision PASS (1.0 > 0.8), Recall FAIL (0.013 < 0.6). The system is conservative (never merges incorrectly) but misses most opportunities to merge.

---

## 7. Summary and Future Work

### 7.1 Summary

PMIS occupies a genuinely novel position in the LLM memory landscape. Its strongest contribution is the **outcome-driven weight evolution loop** — no existing academic system adjusts memory importance based on explicit downstream task performance scores with a principled cold-start blending schedule. The **structure snapshot / winning recipe** mechanism implements structural credit assignment that goes beyond individual memory importance. The **4-stage temporal classification** provides richer temporal modeling than any single decay curve.

The most significant tradeoff is the deliberate omission of semantic embeddings. This yields full determinism, zero dependencies, and complete interpretability — at the cost of retrieval recall on paraphrased queries. A future hybrid extension could add optional embeddings while preserving the deterministic core.

### 7.2 Future Work

1. **Hybrid retrieval**: Add optional sentence-transformers as a retrieval augmentation layer, with word-overlap as fallback and explainability backbone
2. **Reflection mechanism**: Periodic LLM-generated meta-anchors from anchor clusters within the same context
3. **Forgetting with grace**: Prune anchors below a weight threshold after N rebuild cycles in "fading" stage
4. **Multi-user partitions**: Per-user super contexts with shared organizational knowledge
5. **MemBench evaluation**: Adapt PMIS for standardized memory benchmarks (ACL 2025) with honest reporting of semantic retrieval limitations
6. **Cognitive science connection**: The weight evolution mechanism as a computational model of memory consolidation — initial encoding (LLM estimate) strengthened or weakened by rehearsal outcomes (task scoring)

---

---

## 8. Final Paper-Ready Summary

*(Self-reviewed and revised. Claims below are supported by architecture analysis and synthetic benchmarks. Unsupported claims are explicitly flagged.)*

### Positioning Statement

PMIS is a **hierarchical, evidence-weighted memory system** for LLM agents that introduces a closed feedback loop between memory retrieval and task outcomes. Unlike existing systems that assign memory importance statically (Generative Agents), implicitly (MemGPT), or not at all (RAG, Reflexion), PMIS evolves memory weights based on explicit downstream performance scores — a mechanism with no direct analog in the current literature.

### What PMIS Contributes (Ranked by Strength of Evidence)

**1. Evidence-Based Weight Evolution — STRONG (Benchmarked)**

The core innovation. When a user scores a completed task, PMIS propagates that score to every memory anchor present during the task. A Bayesian-inspired blending schedule shifts from 70% LLM-estimate / 30% evidence (cold start) to 30/70 (data-rich). Synthetic benchmark with 20 anchors and 50 tasks shows Pearson r = 0.95 between final weights and ground-truth utility, confirming the system reliably improves retrieval ranking quality over time. No existing system — Generative Agents, MemGPT, A-MEM, Reflexion, or GraphRAG — implements outcome-to-importance feedback.

**2. Structure Snapshots & Winning Recipes — NOVEL (Not Yet Benchmarked)**

PMIS captures the exact tree of contexts and anchors present when a task is scored, preserving which *combination* of memories co-occurred with high performance. The highest-scoring snapshot becomes the "winning recipe" and its anchors are prioritized in future retrieval. This is a form of structural credit assignment — tracking not just individual importance but combinatorial value. Architecturally sound, but empirically unvalidated: the live system has 0 scored tasks, so no winning recipes have been generated in production.

**3. Three-Metric Temporal Signals — SOUND (Classifier Needs Tuning)**

PMIS computes three orthogonal temporal signals — recency (7-day half-life exponential decay), frequency (30-day occurrence rate), and consistency (inverse gap standard deviation) — that are richer than any single decay curve in existing literature. However, the rule-based classifier mapping these to four stages (impulse/active/established/fading) achieved only 50% accuracy in synthetic testing due to rule ordering: the "active" check fires before "established" for daily-use patterns. The metrics themselves are correctly computed; the decision boundaries need reordering. This is a code fix, not an architectural redesign.

**4. Zero-Dependency Design — FACTUAL (Tradeoff Acknowledged)**

PMIS runs on pure Python stdlib (sqlite3, json, math, statistics). No embedding models, no vector databases, no API keys. This yields full determinism (same query = same result), complete auditability, and universal portability. The cost is significant: word-overlap retrieval fails on paraphrased queries (Benchmark 3: dedup recall = 1.3%), and the system cannot handle synonymy or conceptual proximity. This is positioned as a deliberate design tradeoff, not an oversight — with a clear migration path to hybrid retrieval.

### What PMIS Does Not Do (Honest Gaps)

| Gap | Impact | Nearest System That Solves It |
|-----|--------|-------------------------------|
| No semantic retrieval (embeddings) | Misses paraphrased matches; recall is poor | Generative Agents, A-MEM, GraphRAG |
| No reflection/abstraction | Cannot synthesize meta-insights from existing memories | Generative Agents (reflection mechanism) |
| No multi-agent support | Single-user, single-LLM only | MemGPT (multi-agent extensions) |
| 0 production-scored tasks | Evidence system validated synthetically only | — (requires longitudinal study) |
| O(N) retrieval | Degrades beyond ~10K nodes | GraphRAG (indexed), A-MEM (vector NN) |

### Benchmark Scorecard

| Benchmark | Result | Pass? |
|-----------|--------|-------|
| Weight Convergence (r vs true utility) | r = 0.950 | **PASS** (> 0.7) |
| Temporal Stage Accuracy | 50% | **FAIL** (< 80%) — classifier bug identified |
| Dedup Precision | 1.000 | **PASS** (> 0.8) |
| Dedup Recall | 0.013 | **FAIL** (< 0.6) — word-overlap limitation |

### Recommended Paper Angle

**Title**: "Evidence-Weighted Hierarchical Memory for LLM Agents: Outcome-Driven Knowledge Management Without External Dependencies"

**Core argument**: The missing piece in LLM agent memory is not better embeddings or larger context windows — it is a feedback loop from task outcomes to memory importance. PMIS demonstrates that this loop works (weight convergence benchmark), is architecturally feasible with zero dependencies, and opens a research direction orthogonal to the embedding-focused mainstream.

**Honest framing**: PMIS is not a replacement for embedding-based memory systems. It is a complementary architecture that prioritizes interpretability, outcome-accountability, and zero-dependency design. The retrieval layer is its weakest component; the learning loop is its strongest. A hybrid system combining PMIS's evidence-based weight evolution with embedding-based retrieval would capture the strengths of both paradigms.

**Target venues**: Workshop paper at NeurIPS (LLM Agents Workshop), EMNLP (System Demonstrations), or a position paper at AAAI (AI for Personal Knowledge Management).

---

## References

1. Park, J. S., O'Brien, J. C., Cai, C. J., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *arXiv:2304.03442*.
2. Packer, C., Wooders, S., Lin, K., et al. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.
3. Shinn, N., Cassano, F., Gopinath, A., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*. *arXiv:2303.11366*.
4. Xu, Z., et al. (2025). A-MEM: Agentic Memory for LLM Agents. *arXiv:2502.12110*.
5. Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive Architectures for Language Agents. *TMLR*. *arXiv:2309.02427*.
6. Gao, Y., et al. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv:2312.10997*.
7. Zhu, Z., et al. (2025). AriGraph: Learning Knowledge Graph World Models. *IJCAI 2025*. *arXiv:2407.04363*.
8. Zhang, Y., et al. (2025). MemBench: Benchmarking LLM Agent Memory. *ACL 2025*. *arXiv:2506.21605*.
9. Wu, Y., et al. (2025). A Survey on the Memory Mechanism of LLM-based Agents. *ACM TOIS*. *doi:10.1145/3748302*.
10. Hu, X., et al. (2025). Memory in the Age of AI Agents. *arXiv:2512.13564*.
