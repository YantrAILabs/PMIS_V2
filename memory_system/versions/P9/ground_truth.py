#!/usr/bin/env python3
"""
PMIS Ground Truth Benchmark
Defines test queries with expected results for measuring retrieval quality.
Modeled after Supermemory's benchmark categories.
"""

import json
import sys
import os
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# ══════════════════════════════════════════════
# GROUND TRUTH TEST CASES
# ══════════════════════════════════════════════
# Each test: query, expected_sc (super context title substring),
# expected_anchors (title substrings that SHOULD appear),
# negative_anchors (title substrings that should NOT appear)

BENCHMARK = {
    # ── Category 1: Knowledge Update ──
    # Can the system find updated/specific knowledge?
    "knowledge_update": [
        {
            "query": "What font should I use for the security page?",
            "expected_sc": ["Design Principles", "Sales & Marketing"],
            "expected_anchors": ["typography", "font", "sora", "inter"],
            "negative_anchors": ["silver etf", "trading"],
        },
        {
            "query": "How does the GPU cost calculator work?",
            "expected_sc": ["Sales & Marketing", "Vision AI"],
            "expected_anchors": ["gpu", "calculator", "cost", "inference"],
            "negative_anchors": ["dairy", "iqra"],
        },
        {
            "query": "What camera brands work with our CCTV system?",
            "expected_sc": ["Vision AI"],
            "expected_anchors": ["camera", "rtsp", "onvif", "stream"],
            "negative_anchors": ["etf", "food"],
        },
    ],

    # ── Category 2: Single-session Assistant ──
    # Can it retrieve the right context for a work task?
    "single_session_assistant": [
        {
            "query": "Help me write a cold email to a CISO",
            "expected_sc": ["Sales & Marketing"],
            "expected_anchors": ["cold", "email", "outreach", "ciso", "threat"],
            "negative_anchors": ["dairy", "gutzy"],
        },
        {
            "query": "Build a proposal page for a new client",
            "expected_sc": ["Sales & Marketing", "Design Principles"],
            "expected_anchors": ["proposal", "webpage", "interactive", "demo"],
            "negative_anchors": ["trading", "etf"],
        },
        {
            "query": "Design a dashboard for camera monitoring",
            "expected_sc": ["Vision AI", "Design Principles"],
            "expected_anchors": ["dashboard", "camera", "monitoring", "ui"],
            "negative_anchors": ["food", "gutzy"],
        },
    ],

    # ── Category 3: Single-session User ──
    # Personal context retrieval
    "single_session_user": [
        {
            "query": "What do I know about Kiran AI retail product?",
            "expected_sc": ["Kiran AI"],
            "expected_anchors": ["retail", "kiran", "vision"],
            "negative_anchors": ["iqra", "etf"],
        },
        {
            "query": "Show me my memory system architecture",
            "expected_sc": ["PMIS Architecture", "AI Agent Memory"],
            "expected_anchors": ["memory", "architecture", "anchor", "weight"],
            "negative_anchors": ["dairy", "food"],
        },
        {
            "query": "What's the IQRA academy project about?",
            "expected_sc": ["IQRA"],
            "expected_anchors": ["iqra", "academy", "ias"],
            "negative_anchors": ["etf", "gpu"],
        },
    ],

    # ── Category 4: Temporal Reasoning ──
    # Does it respect recency, frequency, established patterns?
    "temporal_reasoning": [
        {
            "query": "What have I been working on recently?",
            "expected_sc": ["Sales & Marketing", "Design Principles", "PMIS Architecture"],
            "expected_anchors": [],
            "negative_anchors": [],
            "temporal_check": True,  # Should return most recent SCs first
        },
        {
            "query": "What patterns are proven and established?",
            "expected_sc": [],
            "expected_anchors": [],
            "negative_anchors": [],
            "stage_check": "established",  # Should prefer established > impulse
        },
    ],

    # ── Category 5: Multi-session ──
    # Continuity across related queries (simulated session)
    "multi_session": [
        {
            "query_sequence": [
                "I'm working on the security solutions page",
                "Now add a track record section",
                "What about the GPU calculator?",
            ],
            "expected_continuity": True,
            # All 3 queries should stay within Security/Sales domain
            "expected_sc": ["Sales & Marketing", "Design Principles"],
        },
        {
            "query_sequence": [
                "Let's work on cold outreach emails",
                "Actually, let me check my silver ETF trades",
            ],
            "expected_continuity": False,
            # Should detect divergence between outreach and ETF
            "expected_divergence_at": 1,
        },
    ],

    # ── Category 6: Single-session Preferences ──
    # Does it remember user-specific preferences?
    "single_session_preferences": [
        {
            "query": "What design style do I prefer?",
            "expected_sc": ["Design Principles"],
            "expected_anchors": ["dark", "futuristic", "industrial", "cinematic"],
            "negative_anchors": [],
        },
        {
            "query": "What's my preferred tech stack?",
            "expected_sc": ["Vision AI", "PMIS Architecture"],
            "expected_anchors": ["python", "sqlite", "react", "fastapi"],
            "negative_anchors": [],
        },
    ],
}


# ══════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════

def score_retrieval(result, test_case):
    """Score a single retrieval result against ground truth."""
    if not result or "memories" not in result:
        return {"precision": 0, "recall": 0, "f1": 0, "sc_hit": 0, "negative_pass": 1.0}

    memories = result.get("memories", [])
    if not memories:
        return {"precision": 0, "recall": 0, "f1": 0, "sc_hit": 0, "negative_pass": 1.0}

    # Flatten all returned content
    returned_scs = [m.get("super_context", "").lower() for m in memories]
    returned_anchors = []
    for m in memories:
        for ctx in m.get("contexts", []):
            for anc in ctx.get("anchors", []):
                returned_anchors.append(anc.get("title", "").lower())
                returned_anchors.append(anc.get("content", "").lower())

    all_returned_text = " ".join(returned_scs + returned_anchors)

    # SC hit rate
    expected_scs = test_case.get("expected_sc", [])
    if expected_scs:
        sc_hits = sum(1 for sc in expected_scs if any(sc.lower() in rsc for rsc in returned_scs))
        sc_hit = sc_hits / len(expected_scs)
    else:
        sc_hit = 1.0  # No SC expectation = pass

    # Anchor precision/recall
    expected_anchors = test_case.get("expected_anchors", [])
    if expected_anchors:
        hits = sum(1 for ea in expected_anchors if ea.lower() in all_returned_text)
        recall = hits / len(expected_anchors)
        # Precision: what fraction of returned content is relevant?
        # Approximate: if we got hits, precision = hits / (hits + noise)
        total_returned = max(len(returned_anchors), 1)
        precision = min(hits / min(total_returned, len(expected_anchors) * 2), 1.0)
    else:
        precision = 1.0
        recall = 1.0

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    # Negative check: these should NOT appear
    negative_anchors = test_case.get("negative_anchors", [])
    if negative_anchors:
        neg_hits = sum(1 for na in negative_anchors if na.lower() in all_returned_text)
        negative_pass = 1.0 - (neg_hits / len(negative_anchors))
    else:
        negative_pass = 1.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "sc_hit": round(sc_hit, 4),
        "negative_pass": round(negative_pass, 4),
    }


def score_session_continuity(results_sequence, test_case):
    """Score multi-query session continuity."""
    if not results_sequence or len(results_sequence) < 2:
        return 0.5

    # Check if all queries return SCs from the same domain family
    sc_sets = []
    for result in results_sequence:
        scs = set()
        for m in result.get("memories", []):
            scs.add(m.get("super_context", "").lower())
        sc_sets.append(scs)

    if test_case.get("expected_continuity", True):
        # All queries should share at least one SC
        common = sc_sets[0]
        for s in sc_sets[1:]:
            common = common & s
        return 1.0 if common else 0.3
    else:
        # Should detect divergence
        diverge_at = test_case.get("expected_divergence_at", -1)
        if diverge_at > 0 and diverge_at < len(sc_sets):
            overlap = sc_sets[diverge_at - 1] & sc_sets[diverge_at]
            return 1.0 if not overlap else 0.3  # Good if they DON'T overlap
        return 0.5


def run_benchmark(retrieve_fn):
    """Run full benchmark suite. retrieve_fn(query) -> result dict."""
    results = {}
    total_f1 = 0
    total_tests = 0
    category_scores = {}

    for category, tests in BENCHMARK.items():
        cat_scores = []

        for test in tests:
            if "query_sequence" in test:
                # Multi-session test
                seq_results = []
                for q in test["query_sequence"]:
                    r = retrieve_fn(q)
                    seq_results.append(r)
                continuity = score_session_continuity(seq_results, test)
                cat_scores.append(continuity)
            else:
                # Single query test
                query = test["query"]
                result = retrieve_fn(query)
                scores = score_retrieval(result, test)
                # Composite: 50% F1 + 25% SC hit + 25% negative pass
                composite = 0.5 * scores["f1"] + 0.25 * scores["sc_hit"] + 0.25 * scores["negative_pass"]
                cat_scores.append(composite)
                total_f1 += composite
                total_tests += 1

        cat_avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
        category_scores[category] = round(cat_avg * 100, 2)

    overall = (total_f1 / total_tests * 100) if total_tests > 0 else 0

    return {
        "categories": category_scores,
        "overall": round(overall, 2),
        "total_tests": total_tests,
    }


# ══════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import memory as mem

    conn = mem.get_db()

    def retrieve_fn(query):
        """Capture retrieve output as JSON."""
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            mem.cmd_retrieve(conn, query)
        output = f.getvalue()
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"memories": []}

    print("=" * 60)
    print("PMIS GROUND TRUTH BENCHMARK")
    print("=" * 60)

    start = time.time()
    results = run_benchmark(retrieve_fn)
    elapsed = time.time() - start

    print(f"\nTime: {elapsed:.1f}s")
    print(f"\nCategory Scores (%):")
    print("-" * 40)
    for cat, score in sorted(results["categories"].items()):
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        print(f"  {cat:<30} {bar} {score:>6.2f}%")
    print("-" * 40)
    print(f"  {'OVERALL':30} {'':20} {results['overall']:>6.2f}%")
    print(f"\n  Total test cases: {results['total_tests']}")
    print("=" * 60)

    # Save result
    result_file = ROOT / "Graph_DB" / "benchmark_results.json"
    history = []
    if result_file.exists():
        try:
            history = json.loads(result_file.read_text())
        except Exception:
            history = []

    history.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "current",
        "results": results,
        "elapsed_seconds": round(elapsed, 2),
    })
    result_file.write_text(json.dumps(history, indent=2))
    print(f"\nResults saved to {result_file}")
