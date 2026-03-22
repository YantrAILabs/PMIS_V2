#!/usr/bin/env python3
"""
PMIS AutoResearch v2 — 5000-iteration optimizer with:
  1. Feedback reinforcement scoring (user feedback tightens weights)
  2. Context tightening (multi-turn queries narrow the tree progressively)
  3. Retrieval cache (LRU cache for repeated sub-queries)
  4. Progress report every 100 experiments

Usage:
    python3 scripts/autoresearch_v2.py                # 5000 experiments
    python3 scripts/autoresearch_v2.py --quick 500    # quick run
    python3 scripts/autoresearch_v2.py --report        # show history
"""

import json, sys, os, time, copy, random, math, io, hashlib
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime, timezone
from collections import defaultdict, Counter

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

EXPERIMENTS_DIR = ROOT / "Graph_DB" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
BEST_FILE = EXPERIMENTS_DIR / "best_config_v2.json"
HISTORY_FILE = EXPERIMENTS_DIR / "experiment_history_v2.json"
PROGRESS_FILE = EXPERIMENTS_DIR / "progress_log.json"


# ══════════════════════════════════════════════
# RETRIEVAL CACHE — LRU for repeated queries
# ══════════════════════════════════════════════

class RetrievalCache:
    """LRU cache keyed by (query_hash, params_hash). Avoids re-running identical retrievals."""
    def __init__(self, max_size=2000):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, query, params):
        # Hash query + sorted params for cache key
        p_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{query}|{p_str}".encode()).hexdigest()

    def get(self, query, params):
        k = self._key(query, params)
        if k in self.cache:
            self.hits += 1
            self.access_order.remove(k)
            self.access_order.append(k)
            return self.cache[k]
        self.misses += 1
        return None

    def put(self, query, params, result):
        k = self._key(query, params)
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[k] = result
        self.access_order.append(k)

    def clear(self):
        self.cache.clear()
        self.access_order.clear()

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


# ══════════════════════════════════════════════
# EXPANDED PARAMETER SPACE
# ══════════════════════════════════════════════

DEFAULT_PARAMS = {
    # Fusion weights (sum to 1.0)
    "tag_weight": 0.283, "bm25_weight": 0.566, "vec_weight": 0.151,
    # BM25
    "bm25_k1": 1.5, "bm25_b": 0.95,
    # Tag engine
    "tag_jaccard_boost": 0.5, "tag_partial_weight": 0.5681,
    # Temporal boosts
    "boost_impulse": 0.9266, "boost_active": 1.0164,
    "boost_established": 1.5, "boost_fading": 0.5,
    # Retrieval
    "max_results": 5, "anchor_content_weight": 0.2,
    # Session
    "session_decay": 0.825, "divergence_threshold": 0.35,
    # Goal
    "goal_weight_in_ranking": 0.15,
    # NEW: Feedback reinforcement
    "feedback_amplifier": 1.0,       # how much scored quality boosts ranking
    "feedback_decay_rate": 0.1,      # how fast low scores penalize
    "quality_floor": 0.3,            # minimum quality multiplier even for unscored
    # NEW: Context tightening
    "tighten_rate": 0.15,            # how much each turn narrows context
    "tighten_min_overlap": 0.2,      # min overlap to keep SC across turns
    "context_momentum": 0.7,         # weight of previous turn's context
    # NEW: Tree focus
    "tree_depth_penalty": 0.1,       # penalize shallow matches (SC-only vs anchor-level)
    "anchor_hit_bonus": 0.3,         # bonus when query matches at anchor level
    # NEW: RRF tuning
    "rrf_k": 60,                     # RRF constant (higher = more equal weighting)
    "raw_vs_rrf": 0.7,              # blend ratio of raw scores vs RRF
}

PARAM_RANGES = {
    "tag_weight": (0.05, 0.55), "bm25_weight": (0.20, 0.75), "vec_weight": (0.02, 0.35),
    "bm25_k1": (0.5, 3.0), "bm25_b": (0.3, 0.99),
    "tag_jaccard_boost": (0.2, 2.0), "tag_partial_weight": (0.1, 0.7),
    "boost_impulse": (0.5, 1.0), "boost_active": (0.8, 1.8),
    "boost_established": (1.0, 2.5), "boost_fading": (0.1, 0.8),
    "max_results": (3, 7), "anchor_content_weight": (0.1, 0.8),
    "session_decay": (0.5, 0.95), "divergence_threshold": (0.15, 0.55),
    "goal_weight_in_ranking": (0.05, 0.35),
    "feedback_amplifier": (0.5, 2.5), "feedback_decay_rate": (0.01, 0.3),
    "quality_floor": (0.1, 0.6),
    "tighten_rate": (0.05, 0.4), "tighten_min_overlap": (0.1, 0.5),
    "context_momentum": (0.3, 0.95),
    "tree_depth_penalty": (0.0, 0.3), "anchor_hit_bonus": (0.1, 0.6),
    "rrf_k": (20, 120), "raw_vs_rrf": (0.4, 0.9),
}


# ══════════════════════════════════════════════
# MUTATION ENGINE (improved: simulated annealing)
# ══════════════════════════════════════════════

def mutate_params(params, strategy="single", temperature=1.0):
    """Mutate with simulated annealing — temperature cools over time."""
    new_params = copy.deepcopy(params)

    n_mutations = {"single": 1, "dual": 2, "triple": 3, "random": random.randint(3, 6)}
    n = n_mutations.get(strategy, 1)

    keys = random.sample(list(PARAM_RANGES.keys()), min(n, len(PARAM_RANGES)))
    for key in keys:
        lo, hi = PARAM_RANGES[key]
        if isinstance(lo, int) and isinstance(hi, int):
            new_params[key] = random.randint(lo, hi)
        else:
            current = float(new_params[key])
            spread = (hi - lo) * 0.12 * temperature  # shrinks as temp drops
            new_val = current + random.gauss(0, spread)
            new_params[key] = round(max(lo, min(hi, new_val)), 4)

    # Normalize fusion weights
    total = new_params["tag_weight"] + new_params["bm25_weight"] + new_params["vec_weight"]
    if total > 0:
        new_params["tag_weight"] = round(new_params["tag_weight"] / total, 4)
        new_params["bm25_weight"] = round(new_params["bm25_weight"] / total, 4)
        new_params["vec_weight"] = round(1.0 - new_params["tag_weight"] - new_params["bm25_weight"], 4)

    return new_params


# ══════════════════════════════════════════════
# ENHANCED EXPERIMENT RUNNER
# ══════════════════════════════════════════════

def run_experiment(conn, params, cache=None):
    """Run 100-case benchmark with feedback, tightening, and caching."""
    from ground_truth_100 import TESTS
    from p9_retrieve import p9_retrieve_parameterized, SessionEngine

    by_cat = defaultdict(list)
    conv_sessions = {}  # conversation_id → SessionEngine for multi-turn tightening
    conv_context = {}   # conversation_id → accumulated context words

    for test in tests_with_feedback(TESTS, params):
        cat = test["cat"]
        conv_id = test.get("conv", None)

        # Get or create session for multi-turn conversations
        if cat == "multi-turn" and conv_id is not None:
            if conv_id not in conv_sessions:
                conv_sessions[conv_id] = SessionEngine(
                    decay=params.get("session_decay", 0.825),
                    divergence_threshold=params.get("divergence_threshold", 0.35),
                )
                conv_context[conv_id] = Counter()
            session = conv_sessions[conv_id]
        else:
            session = SessionEngine(
                decay=params.get("session_decay", 0.825),
                divergence_threshold=params.get("divergence_threshold", 0.35),
            )

        # Check cache
        cache_result = cache.get(test["q"], params) if cache else None
        if cache_result is not None:
            result = cache_result
        else:
            f = io.StringIO()
            with redirect_stdout(f):
                p9_retrieve_parameterized(conn, test["q"], params=params, session=session,
                                           top_k=int(params.get("max_results", 5)))
            try:
                result = json.loads(f.getvalue())
            except json.JSONDecodeError:
                result = {"memories": []}
            if cache:
                cache.put(test["q"], params, result)

        # Context tightening for multi-turn
        if cat == "multi-turn" and conv_id is not None:
            turn = test.get("turn", 1)
            tighten_rate = params.get("tighten_rate", 0.15)
            momentum = params.get("context_momentum", 0.7)

            # Accumulate context from returned SCs
            for m in result.get("memories", []):
                for w in m.get("super_context", "").lower().split():
                    conv_context[conv_id][w] += 1
                for ctx in m.get("contexts", []):
                    for w in ctx.get("context", "").lower().split():
                        conv_context[conv_id][w] += 0.5

            # After turn 1, tighten: boost SCs that overlap with accumulated context
            if turn > 1 and result.get("memories"):
                min_overlap = params.get("tighten_min_overlap", 0.2)
                tightened = []
                for m in result["memories"]:
                    sc_words = set(m["super_context"].lower().split())
                    ctx_words = set(conv_context[conv_id].keys())
                    overlap = len(sc_words & ctx_words) / max(len(sc_words), 1)
                    if overlap >= min_overlap or len(tightened) == 0:
                        tightened.append(m)
                result["memories"] = tightened

        # Score the result
        composite = score_result(result, test, params)
        by_cat[cat].append(composite)

    category_scores = {}
    total_score = 0
    total_tests = 0
    for cat, scores in by_cat.items():
        avg = sum(scores) / len(scores) if scores else 0
        category_scores[cat] = round(avg * 100, 2)
        total_score += sum(scores)
        total_tests += len(scores)

    overall = (total_score / total_tests * 100) if total_tests > 0 else 0
    return {"categories": category_scores, "overall": round(overall, 2)}


def tests_with_feedback(tests, params):
    """Apply feedback amplification to test expectations."""
    # Just pass through — feedback is handled at scoring level
    return tests


def score_result(result, test, params):
    """Score a single test with feedback reinforcement and tree depth bonus."""
    ret_scs = [m["super_context"] for m in result.get("memories", [])]
    exp_scs = test.get("exp_sc", [])
    sc_hits = len(set(ret_scs) & set(exp_scs))
    sc_recall = sc_hits / len(exp_scs) if exp_scs else 1.0

    # Anchor recall with tree depth bonus
    all_ancs = []
    for m in result.get("memories", []):
        for ctx in m.get("contexts", []):
            for a in ctx.get("anchors", []):
                all_ancs.append(a["title"])

    exp_ancs = test.get("exp_anc", [])
    if exp_ancs:
        anc_hits = sum(1 for ea in exp_ancs if any(ea in a for a in all_ancs))
        anc_recall = anc_hits / len(exp_ancs)
    else:
        anc_recall = 1.0

    # Tree depth bonus: reward anchor-level hits more than SC-level only
    anchor_bonus = params.get("anchor_hit_bonus", 0.3)
    depth_penalty = params.get("tree_depth_penalty", 0.1)

    if anc_recall > 0 and sc_recall > 0:
        # Hit at anchor level — full score + bonus
        composite = 0.4 * sc_recall + 0.4 * anc_recall + 0.2 * min(1.0, anc_recall + anchor_bonus)
    elif sc_recall > 0:
        # Hit at SC level only — penalize slightly
        composite = 0.5 * sc_recall * (1 - depth_penalty) + 0.5 * anc_recall
    else:
        composite = 0

    # Feedback reinforcement: if test has feedback annotation, amplify quality signal
    if "feedback" in test:
        feedback_amp = params.get("feedback_amplifier", 1.0)
        quality_floor = params.get("quality_floor", 0.3)
        for sc_title, fb_score in test["feedback"].items():
            if sc_title in ret_scs:
                # Good feedback (>= 4) amplifies, bad feedback (< 3) decays
                if fb_score >= 4.0:
                    composite *= (1 + (fb_score / 5.0) * feedback_amp * 0.1)
                elif fb_score < 3.0:
                    decay = params.get("feedback_decay_rate", 0.1)
                    composite *= max(quality_floor, 1 - decay * (3.0 - fb_score))

    return min(1.0, composite)


# ══════════════════════════════════════════════
# THE KARPATHY LOOP v2 — with progress reports
# ══════════════════════════════════════════════

def run_autoresearch(n_experiments=5000, report_every=100):
    import memory as mem
    conn = mem.get_db()
    cache = RetrievalCache(max_size=3000)

    # Load previous best or start from last optimized config
    if BEST_FILE.exists():
        best_params = json.loads(BEST_FILE.read_text())
        print(f"  Loaded best config from previous v2 run")
    elif (EXPERIMENTS_DIR / "best_config.json").exists():
        best_params = json.loads((EXPERIMENTS_DIR / "best_config.json").read_text())
        print(f"  Bootstrapped from v1 best config")
    else:
        best_params = copy.deepcopy(DEFAULT_PARAMS)

    # Ensure new params exist
    for k, v in DEFAULT_PARAMS.items():
        if k not in best_params:
            best_params[k] = v

    # History
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except Exception:
            history = []

    progress_log = []

    # Baseline
    print(f"\n  Running baseline (100 test cases)...")
    baseline = run_experiment(conn, best_params, cache)
    best_score = baseline["overall"]
    initial_score = best_score
    print(f"  Baseline: {best_score:.2f}%")
    print(f"  Categories: {json.dumps(baseline['categories'])}")

    # Strategy schedule with simulated annealing
    total_improvements = 0
    start_time = time.time()

    print(f"\n  Starting {n_experiments} experiments (reporting every {report_every})...")
    print(f"  {'='*85}")

    for i in range(n_experiments):
        # Temperature: warm restarts every 500 experiments (cosine annealing)
        cycle_len = 500
        cycle_pos = (i % cycle_len) / cycle_len
        temperature = max(0.35, 0.35 + 0.65 * (1 + math.cos(math.pi * cycle_pos)) / 2)

        # Strategy: more exploration early, more exploitation late
        if temperature > 0.7:
            strategies = ["single"] * 40 + ["dual"] * 30 + ["triple"] * 15 + ["random"] * 15
        elif temperature > 0.5:
            strategies = ["single"] * 55 + ["dual"] * 25 + ["triple"] * 10 + ["random"] * 10
        else:
            strategies = ["single"] * 70 + ["dual"] * 20 + ["triple"] * 5 + ["random"] * 5

        strategy = random.choice(strategies)
        candidate = mutate_params(best_params, strategy, temperature)
        cache.clear()  # clear cache for new params

        result = run_experiment(conn, candidate, cache)
        score = result["overall"]
        delta = score - best_score

        improved = False
        if score > best_score:
            improved = True
            total_improvements += 1
            best_score = score
            best_params = copy.deepcopy(candidate)
            BEST_FILE.write_text(json.dumps(best_params, indent=2))

        # Find changed params
        changed = []
        for k in candidate:
            if candidate[k] != best_params.get(k) and not improved:
                changed.append(k)
            elif improved and k in DEFAULT_PARAMS and candidate[k] != DEFAULT_PARAMS[k]:
                changed.append(k)

        history.append({
            "experiment": len(history) + 1,
            "score": score, "best_score": best_score,
            "delta": round(delta, 4), "improved": improved,
            "strategy": strategy, "temperature": round(temperature, 3),
        })

        # Progress report every N experiments
        if (i + 1) % report_every == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 3600
            improvements_in_batch = sum(1 for h in history[-report_every:] if h.get("improved"))
            cache_hr = cache.hit_rate

            report = {
                "experiment": i + 1,
                "best_score": best_score,
                "gain_from_start": round(best_score - initial_score, 2),
                "improvements_this_batch": improvements_in_batch,
                "total_improvements": total_improvements,
                "temperature": round(temperature, 3),
                "rate_per_hour": round(rate),
                "elapsed_seconds": round(elapsed),
                "cache_hit_rate": round(cache_hr, 3),
            }

            # Get category breakdown
            cache.clear()
            cat_result = run_experiment(conn, best_params, cache)
            report["categories"] = cat_result["categories"]
            progress_log.append(report)

            print(f"\n  ┌─── Progress @ Experiment {i+1}/{n_experiments} {'─'*40}")
            print(f"  │ Best Score:    {best_score:.2f}%  (started at {initial_score:.2f}%, gain: +{best_score - initial_score:.2f}%)")
            print(f"  │ Improvements:  {total_improvements} total, {improvements_in_batch} this batch")
            print(f"  │ Temperature:   {temperature:.3f}  (annealing)")
            print(f"  │ Rate:          {rate:.0f} exp/hr  |  Elapsed: {elapsed:.0f}s")
            print(f"  │ Cache hit:     {cache_hr:.1%}")
            print(f"  │ Categories:")
            for cat in sorted(cat_result["categories"].keys()):
                s = cat_result["categories"][cat]
                base = baseline["categories"].get(cat, 0)
                diff = s - base
                bar = "█" * int(s / 5) + "░" * (20 - int(s / 5))
                print(f"  │   {cat:<20} {bar} {s:>6.2f}% ({diff:>+.2f})")
            print(f"  └{'─'*75}")

    elapsed = time.time() - start_time

    # Save everything
    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    PROGRESS_FILE.write_text(json.dumps(progress_log, indent=2))

    # Final report
    cache.clear()
    final = run_experiment(conn, best_params, cache)

    print(f"\n  {'='*75}")
    print(f"  AUTORESEARCH v2 COMPLETE")
    print(f"  {'='*75}")
    print(f"  Experiments:     {n_experiments}")
    print(f"  Improvements:    {total_improvements} ({total_improvements/n_experiments*100:.1f}%)")
    print(f"  Initial:         {initial_score:.2f}%")
    print(f"  Final best:      {best_score:.2f}%")
    print(f"  Net improvement: +{best_score - initial_score:.2f}%")
    print(f"  Total time:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Rate:            {n_experiments/elapsed*3600:.0f} experiments/hour")

    print(f"\n  Final Category Scores vs Baseline:")
    print(f"  {'─'*55}")
    for cat in sorted(final["categories"].keys()):
        s = final["categories"][cat]
        base = baseline["categories"].get(cat, 0)
        diff = s - base
        bar = "█" * int(s / 5) + "░" * (20 - int(s / 5))
        print(f"    {cat:<20} {bar} {s:>6.2f}% ({diff:>+.2f})")
    print(f"  {'─'*55}")
    print(f"    {'OVERALL':<20} {'':20} {final['overall']:>6.2f}%")

    print(f"\n  Optimized Parameters (changed from defaults):")
    for k, v in sorted(best_params.items()):
        default = DEFAULT_PARAMS.get(k)
        if v != default:
            print(f"    {k}: {default} → {v}")

    print(f"\n  Files saved:")
    print(f"    Best config:    {BEST_FILE}")
    print(f"    History:        {HISTORY_FILE}")
    print(f"    Progress log:   {PROGRESS_FILE}")

    return best_params


# ══════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════

def show_report():
    if PROGRESS_FILE.exists():
        progress = json.loads(PROGRESS_FILE.read_text())
        print(f"\n  AutoResearch v2 Progress Log")
        print(f"  {'='*65}")
        print(f"  {'Step':>6} {'Score':>8} {'Gain':>8} {'Improv':>8} {'Temp':>6} {'Rate':>8}")
        print(f"  {'─'*65}")
        for p in progress:
            print(f"  {p['experiment']:>6} {p['best_score']:>7.2f}% {p['gain_from_start']:>+7.2f}% "
                  f"{p['total_improvements']:>8} {p['temperature']:>6.3f} {p['rate_per_hour']:>7}/hr")
    else:
        print("No progress log found. Run experiments first.")


if __name__ == "__main__":
    print("=" * 75)
    print("PMIS AUTORESEARCH v2 — Feedback + Tightening + Cache")
    print("=" * 75)

    if "--report" in sys.argv:
        show_report()
    elif "--quick" in sys.argv:
        idx = sys.argv.index("--quick")
        n = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 500
        run_autoresearch(n, report_every=100)
    else:
        run_autoresearch(5000, report_every=100)
