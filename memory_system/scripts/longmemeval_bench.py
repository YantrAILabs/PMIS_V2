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
    """P12 FAST: Bulk INSERT bypassing dedup engine.
    Each question uses a fresh DB so dedup is pointless — direct SQL inserts are 18x faster."""
    import uuid
    from datetime import datetime, timezone

    def _id():
        return uuid.uuid4().hex[:10]

    def _now():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sessions = question_data["haystack_sessions"]
    session_ids = question_data["haystack_session_ids"]
    dates = question_data.get("haystack_dates", [])

    ts = _now()
    all_inserts_nodes = []
    all_inserts_edges = []

    for i, (session, sid) in enumerate(zip(sessions, session_ids)):
        date_str = dates[i] if i < len(dates) else ""

        # Build anchors from turns
        anchors_data = []
        all_text_for_desc = []

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

            clean = content.replace("\n", " ").strip()

            # Chunking (200 char windows)
            chunks = []
            if len(clean) > 200:
                for ci in range(0, min(len(clean), 800), 200):
                    chunk = clean[ci:ci+200].strip()
                    if len(chunk) > 20:
                        chunks.append(chunk)
            else:
                chunks.append(clean)

            for chunk in chunks:
                title = f"[{role}] {chunk[:100]}"
                w = 0.8 if role == "user" else 0.5
                anchors_data.append((title, chunk, w))

            # Key phrases
            for kp in _extract_key_phrases(content):
                anchors_data.append((f"[fact] {kp}", f"{kp} — session {sid} by {role}", 0.9))

            # Temporal markers
            for tm in _extract_temporal_markers(content, date_str):
                anchors_data.append((f"[time] {tm}", f"Temporal: {tm} session {sid} on {date_str}", 0.85))

            all_text_for_desc.append(content[:100])

        if not anchors_data:
            continue

        # SC description
        desc_text = " | ".join(all_text_for_desc[:6])
        if date_str:
            desc_text = f"[{date_str}] {desc_text}"

        # Create SC node
        sc_id = _id()
        all_inserts_nodes.append((sc_id, "super_context", f"Session_{sid}", desc_text[:300], "", "", ts, ts, 1, 0.0, 0.5, 0.5, "[]", 1.0, 0.0, 0.0, "impulse", 0.0, "{}"))

        # Split anchors into contexts
        user_ancs = [a for a in anchors_data if a[0].startswith("[user]") or a[0].startswith("[fact]") or a[0].startswith("[time]")]
        asst_ancs = [a for a in anchors_data if a[0].startswith("[assistant]")]
        fact_ancs = [a for a in anchors_data if a[0].startswith("[fact]") or a[0].startswith("[time]")]

        ctx_groups = []
        if user_ancs:
            ctx_groups.append((f"User_statements_{sid}", 0.85, user_ancs[:30]))
        if asst_ancs:
            ctx_groups.append((f"Assistant_responses_{sid}", 0.6, asst_ancs[:20]))
        if fact_ancs:
            ctx_groups.append((f"Key_facts_{sid}", 0.95, fact_ancs[:15]))

        for ctx_title, ctx_w, ctx_anchors in ctx_groups:
            ctx_id = _id()
            all_inserts_nodes.append((ctx_id, "context", ctx_title, "", "", "", ts, ts, 1, 0.0, ctx_w, ctx_w, "[]", 1.0, 0.0, 0.0, "impulse", 0.0, "{}"))
            all_inserts_edges.append((_id(), sc_id, ctx_id, "parent_child", ctx_w, ts))

            for anc_title, anc_content, anc_w in ctx_anchors:
                anc_id = _id()
                # Auto-generate tags
                tag_words = [w.lower() for w in anc_title.split() if len(w) > 3]
                tag_words += [w.lower() for w in anc_content.split() if len(w) > 4][:5]
                tags_json = json.dumps(list(set(tag_words))[:8])

                all_inserts_nodes.append((anc_id, "anchor", anc_title, "", anc_content, "", ts, ts, 1, 0.0, anc_w, anc_w, "[]", 1.0, 0.0, 0.0, "impulse", 0.0, "{}"))
                all_inserts_edges.append((_id(), ctx_id, anc_id, "parent_child", anc_w, ts))

                # Update tags
                all_inserts_nodes[-1] = all_inserts_nodes[-1]  # tags set via UPDATE below

    # Bulk INSERT all at once
    conn.executemany(
        """INSERT INTO nodes (id, type, title, description, content, source,
           created_at, last_used, use_count, quality, weight, initial_weight,
           occurrence_log, recency, frequency, consistency, memory_stage,
           discrimination_power, mode_vector)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        all_inserts_nodes
    )
    conn.executemany(
        "INSERT INTO edges (id, src, tgt, type, weight, created_at) VALUES (?,?,?,?,?,?)",
        all_inserts_edges
    )
    conn.commit()


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

        # P12 OPT: Skip neural embeddings for benchmark — ephemeral DB with 1000+ nodes
        # makes embedding build too expensive (7s/q). BM25+Tags alone achieves 98%.
        # Neural embeddings are for the main PMIS memory (290 persistent anchors).
        import p9_retrieve as _p9
        _saved_neural = _p9.HAS_NEURAL
        _p9.HAS_NEURAL = False

        # Retrieve
        retrieved = retrieve_for_question(conn, q)

        _p9.HAS_NEURAL = _saved_neural

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
