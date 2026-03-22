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


def _extract_key_phrases(text, max_phrases=5):
    """P12: Extract key named entities and specific facts from text."""
    import re
    phrases = set()
    # Proper nouns (capitalized words not at sentence start)
    for m in re.finditer(r'(?<=[.!?]\s)[A-Z][a-z]+|(?<=\s)[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*', text):
        phrases.add(m.group().strip())
    # Quoted strings
    for m in re.finditer(r'["\']([^"\']{3,40})["\']', text):
        phrases.add(m.group(1))
    # Dollar/rupee amounts
    for m in re.finditer(r'[\$\u20b9][\d,]+(?:\.\d{2})?', text):
        phrases.add(m.group())
    # Percentages and numbers with context
    for m in re.finditer(r'\d+(?:\.\d+)?%|\d+\s*(?:weeks?|days?|months?|hours?|years?)\s*ago', text, re.I):
        phrases.add(m.group())
    # Preference markers
    for m in re.finditer(r'(?:I (?:prefer|love|like|enjoy|hate|dislike|usually|always|never))\s+(.{5,50}?)(?:[.,;!]|$)', text, re.I):
        phrases.add("PREF: " + m.group(1).strip())
    return list(phrases)[:max_phrases]


def _extract_temporal_markers(text, session_date=""):
    """P12: Extract temporal information from text."""
    import re
    markers = []
    # Relative time references
    for m in re.finditer(r'(?:last|this|next)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)', text, re.I):
        markers.append(m.group())
    # Specific dates
    for m in re.finditer(r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}', text):
        markers.append(m.group())
    # "ago" references
    for m in re.finditer(r'\d+\s*(?:weeks?|days?|months?|hours?|years?)\s*ago', text, re.I):
        markers.append(m.group())
    if session_date:
        markers.append(f"SESSION_DATE:{session_date}")
    return markers


def ingest_question(conn, question_data):
    """P12: Enhanced ingestion with entity extraction, preference detection, and temporal indexing.
    Each session becomes an SC, each turn becomes an anchor with richer metadata."""
    import memory as mem

    sessions = question_data["haystack_sessions"]
    session_ids = question_data["haystack_session_ids"]
    dates = question_data.get("haystack_dates", [])

    for i, (session, sid) in enumerate(zip(sessions, session_ids)):
        date_str = dates[i] if i < len(dates) else ""

        # P12: Build multiple anchors per turn — title, key phrases, temporal markers
        anchors = []
        all_text_for_desc = []  # Collect all text for SC description

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

            # P12 FIX 1: Multiple chunks per long turn (every 200 chars)
            # This ensures keywords deep in a response aren't lost
            chunks = []
            clean = content.replace("\n", " ").strip()
            if len(clean) > 200:
                for ci in range(0, min(len(clean), 800), 200):
                    chunk = clean[ci:ci+200].strip()
                    if len(chunk) > 20:
                        chunks.append(chunk)
            else:
                chunks.append(clean)

            for ci, chunk in enumerate(chunks):
                title = chunk[:100]
                anchors.append({
                    "title": f"[{role}] {title}",
                    "content": chunk,
                    "weight": 0.8 if role == "user" else 0.5,
                })

            # P12 FIX 2: Extract key entities/facts as separate anchors
            key_phrases = _extract_key_phrases(content)
            for kp in key_phrases:
                anchors.append({
                    "title": f"[fact] {kp}",
                    "content": f"{kp} — mentioned in session {sid} by {role}",
                    "weight": 0.9,  # High weight for extracted entities
                })

            # P12 FIX 3: Extract temporal markers
            temporal = _extract_temporal_markers(content, date_str)
            for tm in temporal:
                anchors.append({
                    "title": f"[time] {tm}",
                    "content": f"Temporal reference: {tm} in session {sid} on {date_str}",
                    "weight": 0.85,
                })

            # Collect for description
            all_text_for_desc.append(content[:100])

        if not anchors:
            continue

        # P12 FIX 4: Richer SC description from all turns (better BM25 matching)
        desc_text = " | ".join(all_text_for_desc[:6])
        if date_str:
            desc_text = f"[{date_str}] {desc_text}"

        # P12 FIX 5: Multiple contexts per session — user turns vs assistant turns
        user_anchors = [a for a in anchors if a["title"].startswith("[user]") or a["title"].startswith("[fact]") or a["title"].startswith("[time]")]
        asst_anchors = [a for a in anchors if a["title"].startswith("[assistant]")]
        entity_anchors = [a for a in anchors if a["title"].startswith("[fact]") or a["title"].startswith("[time]")]

        contexts = []
        if user_anchors:
            contexts.append({
                "title": f"User_statements_{sid}",
                "weight": 0.85,
                "anchors": user_anchors[:30],
            })
        if asst_anchors:
            contexts.append({
                "title": f"Assistant_responses_{sid}",
                "weight": 0.6,
                "anchors": asst_anchors[:20],
            })
        if entity_anchors:
            contexts.append({
                "title": f"Key_facts_{sid}",
                "weight": 0.95,  # Highest weight — these are the specific facts
                "anchors": entity_anchors[:15],
            })

        sc_data = {
            "super_context": f"Session_{sid}",
            "description": desc_text[:300],
            "contexts": contexts,
            "summary": f"Session {sid} on {date_str}",
        }

        f = io.StringIO()
        with redirect_stdout(f):
            mem.cmd_store(conn, json.dumps(sc_data))


def retrieve_for_question(conn, question_data):
    """P12: Enhanced retrieval with query expansion for temporal/preference/entity queries."""
    from p9_retrieve import p9_retrieve_parameterized, SessionEngine
    import re

    query = question_data["question"]
    q_date = question_data.get("question_date", "")

    # P12: Query expansion for better matching
    expanded_query = query

    # Temporal expansion: add date context if query references time
    if re.search(r'(?:weeks?|days?|months?|years?)\s*ago|last\s+(?:week|month|saturday|sunday)', query, re.I):
        expanded_query += f" temporal time date {q_date}"

    # Preference expansion: add preference markers
    if re.search(r'prefer|favorite|recommend|suggest|like|enjoy|usual', query, re.I):
        expanded_query += " preference prefer like enjoy usually favorite"

    # Entity expansion: add fact markers for specific recall queries
    if re.search(r'name of|how much|where did|what was|what is the', query, re.I):
        expanded_query += " fact specific detail name amount"

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
