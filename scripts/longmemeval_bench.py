#!/usr/bin/env python3
"""
PMIS LongMemEval Benchmark — Run PMIS against the official 500-question benchmark.

Evaluates RETRIEVAL ACCURACY: can PMIS find the session(s) containing the answer?

Metrics:
  - Recall@K: % of answer sessions found in top K retrieved results
  - Session Hit Rate: did we retrieve at least one answer session?
  - Per-category breakdown matching LongMemEval's 6 categories

Usage:
    python3 scripts/longmemeval_bench.py                # Run full 500
    python3 scripts/longmemeval_bench.py --quick 50     # Quick test
"""

import json, sys, os, time, sqlite3, io, hashlib
from pathlib import Path
from contextlib import redirect_stdout
from collections import defaultdict, Counter

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

LONGMEMEVAL_FILE = ROOT / "longmemeval_s.json"
RESULTS_FILE = ROOT / "longmemeval_results.json"

# Separate DB for LongMemEval (don't pollute main graph.db)
BENCH_DB = ROOT / "Graph_DB" / "longmemeval_bench.db"


def get_bench_db():
    """Create a separate SQLite DB for LongMemEval evaluation."""
    import memory as mem
    # Temporarily override the DB path
    original_db = mem.GRAPH_DB
    mem.GRAPH_DB = BENCH_DB
    conn = mem.get_db()
    mem.GRAPH_DB = original_db
    return conn


def ingest_question(conn, question_data):
    """Ingest one question's haystack sessions into the memory graph.
    Each session becomes an SC, each turn becomes an anchor."""
    import memory as mem

    sessions = question_data["haystack_sessions"]
    session_ids = question_data["haystack_session_ids"]
    dates = question_data.get("haystack_dates", [])

    for i, (session, sid) in enumerate(zip(sessions, session_ids)):
        # Build structured memory from this session
        anchors = []
        for turn in session:
            if isinstance(turn, dict):
                role = turn.get("role", "user")
                content = turn.get("content", "")
            elif isinstance(turn, list) and len(turn) >= 2:
                role, content = turn[0], turn[1]
            else:
                continue

            if not content or len(content) < 10:
                continue

            # Extract key info from content (first 200 chars as anchor)
            title = content[:80].replace("\n", " ").strip()
            if len(title) < 10:
                continue

            anchors.append({
                "title": f"[{role}] {title}",
                "content": content[:500],
                "weight": 0.7 if role == "user" else 0.5,
            })

        if not anchors:
            continue

        # Create SC for this session
        date_str = dates[i] if i < len(dates) else ""
        sc_data = {
            "super_context": f"Session_{sid}",
            "description": f"Conversation session {sid} on {date_str}",
            "contexts": [{
                "title": f"Conversation_{sid}",
                "weight": 0.7,
                "anchors": anchors[:20],  # Cap at 20 anchors per session
            }],
            "summary": f"Session {sid}",
        }

        f = io.StringIO()
        with redirect_stdout(f):
            mem.cmd_store(conn, json.dumps(sc_data))


def retrieve_for_question(conn, question_data):
    """Run PMIS retrieval for a question and return matched session IDs."""
    from p9_retrieve import p9_retrieve_parameterized, SessionEngine

    query = question_data["question"]

    # Load config
    cfg_path = ROOT / "Graph_DB" / "experiments" / "best_config_v2.json"
    params = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    params["max_results"] = 10

    session = SessionEngine(
        decay=params.get("session_decay", 0.825),
        divergence_threshold=params.get("divergence_threshold", 0.35),
    )

    f = io.StringIO()
    with redirect_stdout(f):
        p9_retrieve_parameterized(conn, query, params=params, session=session, top_k=10)

    try:
        result = json.loads(f.getvalue())
    except json.JSONDecodeError:
        return []

    # Extract session IDs from retrieved SC titles
    retrieved_sids = []
    for m in result.get("memories", []):
        sc_title = m.get("super_context", "")
        if sc_title.startswith("Session_"):
            sid = sc_title[len("Session_"):]
            retrieved_sids.append(sid)

    return retrieved_sids


def evaluate_retrieval(retrieved_sids, answer_sids, k_values=[1, 3, 5, 10]):
    """Compute Recall@K and hit rate."""
    results = {}
    for k in k_values:
        top_k = set(retrieved_sids[:k])
        hits = len(top_k & set(answer_sids))
        recall = hits / len(answer_sids) if answer_sids else 0
        results[f"recall@{k}"] = recall

    # Session hit: did we find at least one answer session?
    results["hit"] = 1 if any(sid in retrieved_sids for sid in answer_sids) else 0
    return results


def run_benchmark(n_questions=None):
    """Run the full LongMemEval benchmark."""
    import memory as mem

    if not LONGMEMEVAL_FILE.exists():
        print(f"ERROR: {LONGMEMEVAL_FILE} not found. Download first.")
        return

    with open(LONGMEMEVAL_FILE) as f:
        all_questions = json.load(f)

    if n_questions:
        all_questions = all_questions[:n_questions]

    print(f"=" * 70)
    print(f"PMIS LongMemEval Benchmark — {len(all_questions)} questions")
    print(f"=" * 70)

    # Category distribution
    cats = Counter(q["question_type"] for q in all_questions)
    print(f"\nCategories:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    results_by_cat = defaultdict(list)
    all_results = []
    start_time = time.time()

    for qi, q in enumerate(all_questions):
        q_start = time.time()

        # Create fresh DB for each question (clean evaluation)
        if BENCH_DB.exists():
            os.remove(BENCH_DB)

        conn = get_bench_db()

        # Ingest all sessions
        ingest_question(conn, q)

        # Retrieve
        retrieved = retrieve_for_question(conn, q)

        # Evaluate
        answer_sids = q.get("answer_session_ids", [])
        metrics = evaluate_retrieval(retrieved, answer_sids)

        q_time = time.time() - q_start
        cat = q["question_type"]

        result = {
            "question_id": q["question_id"],
            "category": cat,
            "question": q["question"][:80],
            "hit": metrics["hit"],
            "recall@1": metrics["recall@1"],
            "recall@3": metrics["recall@3"],
            "recall@5": metrics["recall@5"],
            "recall@10": metrics["recall@10"],
            "n_sessions": len(q["haystack_sessions"]),
            "n_answer_sessions": len(answer_sids),
            "time": round(q_time, 2),
        }
        results_by_cat[cat].append(result)
        all_results.append(result)

        # Progress every 10 questions
        if (qi + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (qi + 1) / elapsed * 3600
            total_hits = sum(r["hit"] for r in all_results)
            print(f"  [{qi+1}/{len(all_questions)}] Hit rate: {total_hits}/{qi+1} = {total_hits/(qi+1)*100:.1f}%  "
                  f"({elapsed:.0f}s, {rate:.0f}/hr)")

        conn.close()

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")

    print(f"\n{'Category':<30} {'Hit%':>8} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'Count':>6}")
    print(f"{'─' * 78}")

    overall_hits = 0
    overall_r1 = 0
    overall_r3 = 0
    overall_r5 = 0
    overall_r10 = 0
    overall_n = 0

    for cat in ["single-session-user", "single-session-assistant", "single-session-preference",
                "knowledge-update", "temporal-reasoning", "multi-session"]:
        items = results_by_cat.get(cat, [])
        if not items:
            continue
        hits = sum(r["hit"] for r in items)
        r1 = sum(r["recall@1"] for r in items) / len(items)
        r3 = sum(r["recall@3"] for r in items) / len(items)
        r5 = sum(r["recall@5"] for r in items) / len(items)
        r10 = sum(r["recall@10"] for r in items) / len(items)

        hit_pct = hits / len(items) * 100
        bar = "█" * int(hit_pct / 5) + "░" * (20 - int(hit_pct / 5))

        print(f"  {cat:<28} {hit_pct:>7.1f}% {r1*100:>7.1f}% {r3*100:>7.1f}% {r5*100:>7.1f}% {r10*100:>7.1f}% {len(items):>6}")

        overall_hits += hits
        overall_r1 += sum(r["recall@1"] for r in items)
        overall_r3 += sum(r["recall@3"] for r in items)
        overall_r5 += sum(r["recall@5"] for r in items)
        overall_r10 += sum(r["recall@10"] for r in items)
        overall_n += len(items)

    print(f"{'─' * 78}")
    oh = overall_hits / overall_n * 100
    print(f"  {'OVERALL':<28} {oh:>7.1f}% {overall_r1/overall_n*100:>7.1f}% "
          f"{overall_r3/overall_n*100:>7.1f}% {overall_r5/overall_n*100:>7.1f}% "
          f"{overall_r10/overall_n*100:>7.1f}% {overall_n:>6}")

    print(f"\n  Time: {elapsed:.0f}s ({elapsed/overall_n:.1f}s per question)")
    print(f"  Rate: {overall_n/elapsed*3600:.0f} questions/hr")

    # Save results
    output = {
        "benchmark": "LongMemEval-S",
        "system": "PMIS P11-final",
        "n_questions": overall_n,
        "overall_hit_rate": round(oh, 2),
        "overall_recall_at_1": round(overall_r1/overall_n*100, 2),
        "overall_recall_at_3": round(overall_r3/overall_n*100, 2),
        "overall_recall_at_5": round(overall_r5/overall_n*100, 2),
        "overall_recall_at_10": round(overall_r10/overall_n*100, 2),
        "by_category": {},
        "elapsed_seconds": round(elapsed, 1),
        "all_results": all_results,
    }

    for cat, items in results_by_cat.items():
        hits = sum(r["hit"] for r in items)
        output["by_category"][cat] = {
            "count": len(items),
            "hit_rate": round(hits/len(items)*100, 2),
            "recall_at_1": round(sum(r["recall@1"] for r in items)/len(items)*100, 2),
            "recall_at_5": round(sum(r["recall@5"] for r in items)/len(items)*100, 2),
        }

    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to: {RESULTS_FILE}")

    # Cleanup
    if BENCH_DB.exists():
        os.remove(BENCH_DB)


if __name__ == "__main__":
    if "--quick" in sys.argv:
        idx = sys.argv.index("--quick")
        n = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 50
        run_benchmark(n)
    else:
        run_benchmark()
