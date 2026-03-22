#!/usr/bin/env python3
"""
PMIS AutoResearch — Karpathy Loop for Memory Retrieval Optimization

Adapted from karpathy/autoresearch:
- Instead of training GPT models, we optimize retrieval algorithms
- Instead of val_bpb, we optimize retrieval F1 + session continuity
- Instead of 5-min GPU runs, we do 30-sec CPU benchmark runs
- The agent modifies retrieval parameters, not model architecture

Usage:
    python3 scripts/autoresearch.py              # Run full loop (100 experiments)
    python3 scripts/autoresearch.py --quick 10   # Run 10 experiments
    python3 scripts/autoresearch.py --report      # Show experiment history
"""

import json
import sys
import os
import time
import copy
import random
import math
import io
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

EXPERIMENTS_DIR = ROOT / "Graph_DB" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_FILE = EXPERIMENTS_DIR / "best_config.json"
HISTORY_FILE = EXPERIMENTS_DIR / "experiment_history.json"

# ══════════════════════════════════════════════
# PARAMETER SPACE
# ══════════════════════════════════════════════

DEFAULT_PARAMS = {
    # P9+ fusion weights (must sum to 1.0)
    "tag_weight": 0.40,
    "bm25_weight": 0.35,
    "vec_weight": 0.25,

    # BM25 tuning
    "bm25_k1": 1.5,     # term frequency saturation
    "bm25_b": 0.75,     # length normalization

    # Tag engine
    "tag_jaccard_boost": 1.0,  # multiplier for Jaccard matches
    "tag_partial_weight": 0.3,  # weight for partial tag matches

    # Temporal boosts
    "boost_impulse": 0.8,
    "boost_active": 1.2,
    "boost_established": 1.5,
    "boost_fading": 0.5,

    # Retrieval
    "max_results": 3,
    "anchor_content_weight": 0.5,  # how much anchor content matters vs title

    # Session continuity (new)
    "session_decay": 0.85,          # how much previous query context carries forward
    "divergence_threshold": 0.35,   # % drift before session break

    # Goal alignment (new)
    "goal_weight_in_ranking": 0.15,  # how much goal alignment affects final score
}

# Ranges for each parameter
PARAM_RANGES = {
    "tag_weight": (0.15, 0.60),
    "bm25_weight": (0.15, 0.60),
    "vec_weight": (0.05, 0.40),
    "bm25_k1": (0.5, 3.0),
    "bm25_b": (0.3, 0.95),
    "tag_jaccard_boost": (0.5, 2.0),
    "tag_partial_weight": (0.1, 0.7),
    "boost_impulse": (0.4, 1.0),
    "boost_active": (0.8, 1.8),
    "boost_established": (1.0, 2.5),
    "boost_fading": (0.1, 0.8),
    "max_results": (2, 5),
    "anchor_content_weight": (0.2, 0.8),
    "session_decay": (0.5, 0.95),
    "divergence_threshold": (0.15, 0.55),
    "goal_weight_in_ranking": (0.05, 0.35),
}


# ══════════════════════════════════════════════
# PARAMETER MUTATION
# ══════════════════════════════════════════════

def mutate_params(params, strategy="single"):
    """Mutate parameters. Strategy: single, dual, or random."""
    new_params = copy.deepcopy(params)

    if strategy == "single":
        # Change exactly ONE parameter
        key = random.choice(list(PARAM_RANGES.keys()))
        lo, hi = PARAM_RANGES[key]
        if isinstance(lo, int):
            new_params[key] = random.randint(lo, hi)
        else:
            # Gaussian perturbation centered on current value
            current = new_params[key]
            spread = (hi - lo) * 0.15  # 15% of range
            new_val = current + random.gauss(0, spread)
            new_params[key] = round(max(lo, min(hi, new_val)), 4)

    elif strategy == "dual":
        # Change TWO related parameters
        keys = random.sample(list(PARAM_RANGES.keys()), 2)
        for key in keys:
            lo, hi = PARAM_RANGES[key]
            if isinstance(lo, int):
                new_params[key] = random.randint(lo, hi)
            else:
                current = new_params[key]
                spread = (hi - lo) * 0.2
                new_val = current + random.gauss(0, spread)
                new_params[key] = round(max(lo, min(hi, new_val)), 4)

    elif strategy == "random":
        # Randomize 3-5 parameters from scratch
        keys = random.sample(list(PARAM_RANGES.keys()), random.randint(3, 5))
        for key in keys:
            lo, hi = PARAM_RANGES[key]
            if isinstance(lo, int):
                new_params[key] = random.randint(lo, hi)
            else:
                new_params[key] = round(random.uniform(lo, hi), 4)

    # Normalize fusion weights to sum to 1.0
    total = new_params["tag_weight"] + new_params["bm25_weight"] + new_params["vec_weight"]
    if total > 0:
        new_params["tag_weight"] = round(new_params["tag_weight"] / total, 4)
        new_params["bm25_weight"] = round(new_params["bm25_weight"] / total, 4)
        new_params["vec_weight"] = round(1.0 - new_params["tag_weight"] - new_params["bm25_weight"], 4)

    return new_params


# ══════════════════════════════════════════════
# PARAMETERIZED RETRIEVAL
# ══════════════════════════════════════════════

def retrieve_with_params(conn, query, params):
    """Run retrieval using parameterized P9+ engine."""
    from p9_retrieve import p9_retrieve_parameterized, SessionEngine

    # Fresh session per experiment (no cross-contamination)
    session = SessionEngine(
        decay=params.get("session_decay", 0.85),
        divergence_threshold=params.get("divergence_threshold", 0.35),
    )

    f = io.StringIO()
    with redirect_stdout(f):
        p9_retrieve_parameterized(conn, query, params=params, session=session,
                                   top_k=int(params.get("max_results", 3)))
    output = f.getvalue()

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"memories": []}


# ══════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════

def run_experiment(conn, params):
    """Run one complete experiment with given params. Returns composite score."""
    from ground_truth import BENCHMARK, score_retrieval, score_session_continuity

    total_score = 0
    total_tests = 0
    category_scores = {}

    for category, tests in BENCHMARK.items():
        cat_scores = []

        for test in tests:
            if "query_sequence" in test:
                seq_results = []
                for q in test["query_sequence"]:
                    r = retrieve_with_params(conn, q, params)
                    seq_results.append(r)
                continuity = score_session_continuity(seq_results, test)
                cat_scores.append(continuity)
            else:
                query = test["query"]
                result = retrieve_with_params(conn, query, params)
                scores = score_retrieval(result, test)
                composite = 0.5 * scores["f1"] + 0.25 * scores["sc_hit"] + 0.25 * scores["negative_pass"]
                cat_scores.append(composite)
                total_score += composite
                total_tests += 1

        cat_avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
        category_scores[category] = round(cat_avg * 100, 2)

    overall = (total_score / total_tests * 100) if total_tests > 0 else 0

    return {
        "categories": category_scores,
        "overall": round(overall, 2),
    }


# ══════════════════════════════════════════════
# THE KARPATHY LOOP
# ══════════════════════════════════════════════

def run_autoresearch(n_experiments=100):
    """The main loop. Mutate → Benchmark → Keep/Discard → Repeat."""
    import memory as mem

    conn = mem.get_db()

    # Load best known config or start with defaults
    if BEST_FILE.exists():
        best_params = json.loads(BEST_FILE.read_text())
        print(f"  Loaded best config from previous run")
    else:
        best_params = copy.deepcopy(DEFAULT_PARAMS)

    # Load history
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except Exception:
            history = []

    # Baseline
    print(f"\n  Running baseline...")
    baseline = run_experiment(conn, best_params)
    best_score = baseline["overall"]
    print(f"  Baseline score: {best_score:.2f}%")

    # Strategy schedule: mostly single mutations, occasional exploration
    strategies = ["single"] * 60 + ["dual"] * 25 + ["random"] * 15
    random.shuffle(strategies)

    improvements = 0
    start_time = time.time()

    print(f"\n  Starting {n_experiments} experiments...")
    print(f"  {'#':>4} {'Strategy':>8} {'Score':>8} {'Best':>8} {'Delta':>8} {'Status':>10} {'Param Changed'}")
    print(f"  {'-'*80}")

    for i in range(n_experiments):
        strategy = strategies[i % len(strategies)]

        # Mutate from best known params
        candidate_params = mutate_params(best_params, strategy)

        # Find what changed
        changed = []
        for k in candidate_params:
            if candidate_params[k] != best_params[k]:
                changed.append(f"{k}={candidate_params[k]}")

        # Run experiment
        exp_start = time.time()
        result = run_experiment(conn, candidate_params)
        exp_time = time.time() - exp_start
        score = result["overall"]

        # Compare
        delta = score - best_score
        if score > best_score:
            status = "★ IMPROVED"
            improvements += 1
            best_score = score
            best_params = copy.deepcopy(candidate_params)
            BEST_FILE.write_text(json.dumps(best_params, indent=2))
        else:
            status = "  discard"

        # Log
        changed_str = ", ".join(changed[:2]) if changed else "none"
        if len(changed) > 2:
            changed_str += f" +{len(changed)-2}"
        print(f"  {i+1:>4} {strategy:>8} {score:>7.2f}% {best_score:>7.2f}% {delta:>+7.2f}% {status:>10}  {changed_str}")

        history.append({
            "experiment": len(history) + 1,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "strategy": strategy,
            "score": score,
            "best_score": best_score,
            "delta": round(delta, 4),
            "improved": score > best_score - delta,  # was it an improvement?
            "params_changed": changed[:3],
            "categories": result["categories"],
            "elapsed": round(exp_time, 3),
        })

    elapsed = time.time() - start_time

    # Save history
    HISTORY_FILE.write_text(json.dumps(history, indent=2))

    # Final report
    print(f"\n  {'='*60}")
    print(f"  AUTORESEARCH COMPLETE")
    print(f"  {'='*60}")
    print(f"  Experiments:    {n_experiments}")
    print(f"  Improvements:   {improvements} ({improvements/n_experiments*100:.1f}%)")
    print(f"  Baseline:       {baseline['overall']:.2f}%")
    print(f"  Final best:     {best_score:.2f}%")
    print(f"  Net improvement: {best_score - baseline['overall']:+.2f}%")
    print(f"  Total time:     {elapsed:.1f}s ({elapsed/n_experiments:.2f}s per experiment)")
    print(f"  Rate:           {n_experiments/elapsed*3600:.0f} experiments/hour")
    print(f"\n  Best config saved to: {BEST_FILE}")
    print(f"  History saved to: {HISTORY_FILE}")

    # Category breakdown of best
    final = run_experiment(conn, best_params)
    print(f"\n  Final Category Scores:")
    print(f"  {'-'*45}")
    for cat, score in sorted(final["categories"].items()):
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        improvement = score - baseline["categories"].get(cat, 0)
        print(f"    {cat:<28} {bar} {score:>6.2f}% ({improvement:>+.2f})")
    print(f"  {'-'*45}")
    print(f"    {'OVERALL':<28} {'':20} {final['overall']:>6.2f}%")

    # Best params summary
    print(f"\n  Optimized Parameters (vs default):")
    for k, v in best_params.items():
        default = DEFAULT_PARAMS.get(k)
        if v != default:
            print(f"    {k}: {default} → {v}")

    return best_params


# ══════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════

def show_report():
    """Show experiment history."""
    if not HISTORY_FILE.exists():
        print("No experiments run yet.")
        return

    history = json.loads(HISTORY_FILE.read_text())
    print(f"\n  PMIS AutoResearch History")
    print(f"  {'='*60}")
    print(f"  Total experiments: {len(history)}")

    improvements = [h for h in history if h.get("improved")]
    print(f"  Improvements found: {len(improvements)}")

    if history:
        scores = [h["score"] for h in history]
        print(f"  Score range: {min(scores):.2f}% - {max(scores):.2f}%")
        print(f"  Current best: {history[-1]['best_score']:.2f}%")

    if BEST_FILE.exists():
        best = json.loads(BEST_FILE.read_text())
        print(f"\n  Best Config:")
        for k, v in best.items():
            default = DEFAULT_PARAMS.get(k)
            marker = " ◀ changed" if v != default else ""
            print(f"    {k}: {v}{marker}")


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PMIS AUTORESEARCH — Karpathy Loop for Memory Retrieval")
    print("=" * 60)

    if "--report" in sys.argv:
        show_report()
    elif "--quick" in sys.argv:
        idx = sys.argv.index("--quick")
        n = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 10
        run_autoresearch(n)
    else:
        run_autoresearch(100)
