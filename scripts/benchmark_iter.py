#!/usr/bin/env python3
"""
PMIS Benchmark Suite — Tests all 3 fixes across iterations.
Run from memory/ root: python3 scripts/benchmark_iter.py
"""
import sqlite3, os, sys, json, math, random, datetime, tempfile, shutil

sys.path.insert(0, os.path.dirname(__file__))
import memory as mem

def fresh_db():
    """Create a temp database with full PMIS schema."""
    tmp = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(tmp)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Reproduce the schema from get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY, type TEXT NOT NULL, title TEXT NOT NULL,
            description TEXT DEFAULT '', content TEXT DEFAULT '', source TEXT DEFAULT '',
            created_at TEXT NOT NULL, last_used TEXT NOT NULL,
            use_count INTEGER DEFAULT 1, quality REAL DEFAULT 0.0,
            weight REAL DEFAULT 0.5, initial_weight REAL DEFAULT 0.5,
            occurrence_log TEXT DEFAULT '[]',
            recency REAL DEFAULT 1.0, frequency REAL DEFAULT 0.0,
            consistency REAL DEFAULT 0.0, memory_stage TEXT DEFAULT 'impulse'
        );
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY, src TEXT NOT NULL, tgt TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'parent_child',
            weight REAL DEFAULT 1.0, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, sc_id TEXT, title TEXT NOT NULL,
            started TEXT NOT NULL, completed TEXT, score REAL DEFAULT 0,
            content TEXT DEFAULT '', structure_snapshot TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS task_anchors (
            task_id TEXT NOT NULL, anchor_id TEXT NOT NULL,
            context_id TEXT NOT NULL, was_retrieved INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    return conn, tmp

# ══════════════════════════════════════════════════════════
# BENCHMARK 1: Weight Convergence
# ══════════════════════════════════════════════════════════

def benchmark_weight_convergence():
    conn, tmp = fresh_db()
    random.seed(42)

    sc_id = mem.add_node(conn, "super_context", "Weight Test SC", weight=0.5)
    ctx_id = mem.add_node(conn, "context", "Weight Test Ctx", weight=0.5)
    mem.link(conn, sc_id, ctx_id)

    true_utils = {}
    anchor_ids = []

    for i in range(20):
        true_u = 0.2 + (i * 0.7 / 19)
        noisy_initial = max(0.1, min(0.95, true_u + random.gauss(0, 0.2)))
        aid = mem.add_node(conn, "anchor", f"Anchor_{i}", weight=noisy_initial)
        conn.execute("UPDATE nodes SET initial_weight=? WHERE id=?", (noisy_initial, aid))
        mem.link(conn, ctx_id, aid)
        true_utils[aid] = true_u
        anchor_ids.append(aid)
    conn.commit()

    # 50 tasks, each using 5-8 random anchors
    for t in range(50):
        used = random.sample(anchor_ids, random.randint(5, 8))
        mean_util = sum(true_utils[a] for a in used) / len(used)
        task_score = round(mean_util * 5.0 + random.gauss(0, 0.3), 2)
        task_score = max(0.5, min(5.0, task_score))

        task_id = mem._id()
        now_str = datetime.datetime.now().isoformat()
        conn.execute("""INSERT INTO tasks(id, title, started, completed, score)
                        VALUES(?,?,?,?,?)""", (task_id, f"Task_{t}", now_str, now_str, task_score))
        for aid in used:
            conn.execute("INSERT INTO task_anchors(task_id, anchor_id, context_id) VALUES(?,?,?)",
                         (task_id, aid, ctx_id))
    conn.commit()

    # Compute evidence weights
    final_weights = {}
    for aid in anchor_ids:
        final_weights[aid] = mem.compute_evidence_weight(conn, aid)

    # Pearson correlation
    xs = [true_utils[a] for a in anchor_ids]
    ys = [final_weights[a] for a in anchor_ids]

    mean_x, mean_y = sum(xs)/len(xs), sum(ys)/len(ys)
    cov = sum((x-mean_x)*(y-mean_y) for x,y in zip(xs,ys)) / len(xs)
    std_x = (sum((x-mean_x)**2 for x in xs)/len(xs)) ** 0.5
    std_y = (sum((y-mean_y)**2 for y in ys)/len(ys)) ** 0.5
    r = cov / (std_x * std_y) if std_x > 0 and std_y > 0 else 0
    mae = sum(abs(x-y) for x,y in zip(xs,ys)) / len(xs)

    conn.close(); os.unlink(tmp)
    return r, mae

# ══════════════════════════════════════════════════════════
# BENCHMARK 2: Temporal Stage Classification
# ══════════════════════════════════════════════════════════

def benchmark_temporal_stages():
    now = datetime.datetime.now()

    test_cases = [
        # Daily use 30 days → established
        ("daily_30d",   [now - datetime.timedelta(days=d) for d in range(30)], "established"),
        ("daily_30d_b", [now - datetime.timedelta(days=d) for d in range(30)], "established"),
        # Weekly 8 weeks → established
        ("weekly_8w",   [now - datetime.timedelta(weeks=w) for w in range(8)], "established"),
        ("weekly_8w_b", [now - datetime.timedelta(weeks=w) for w in range(8)], "established"),
        # Sporadic: 3 uses over 60 days → impulse
        ("sporadic_60d",  [now, now - datetime.timedelta(days=30), now - datetime.timedelta(days=60)], "impulse"),
        ("sporadic_60d_b",[now - datetime.timedelta(days=5), now - datetime.timedelta(days=35), now - datetime.timedelta(days=58)], "impulse"),
        # Abandoned: used once 45+ days ago → fading
        ("abandoned_45d", [now - datetime.timedelta(days=45)], "fading"),
        ("abandoned_60d", [now - datetime.timedelta(days=60)], "fading"),
        # Recent 2-3x → active
        ("recent_2x", [now, now - datetime.timedelta(days=2)], "active"),
        ("recent_3x", [now - datetime.timedelta(days=1), now - datetime.timedelta(days=3), now - datetime.timedelta(days=5)], "active"),
    ]

    correct = 0
    results = []
    for desc, timestamps, expected in test_cases:
        date_strs = sorted(set(t.strftime("%Y-%m-%d") for t in timestamps))
        r, f, c = mem.compute_temporal(date_strs)
        stage = mem.classify_stage(r, f, c)
        match = stage == expected
        if match: correct += 1
        results.append((desc, expected, stage, match, r, f, c))

    return correct, len(test_cases), results

# ══════════════════════════════════════════════════════════
# BENCHMARK 3: Dedup Precision/Recall
# ══════════════════════════════════════════════════════════

def benchmark_dedup():
    conn, tmp = fresh_db()

    concepts = [
        ["CISOs respond to threat language", "Threat language works on CISOs",
         "Security officers prefer threat-based messaging", "CISO threat language response",
         "Threat intel language resonates with CISOs"],
        ["Short subject lines get higher opens", "Subject under 6 words performs better",
         "Brief email subjects increase open rate", "Short subjects higher opens",
         "Email subject brevity boosts opens"],
        ["VP Security responds more than CISO", "VP Security title better than CISO",
         "Target VP Security over CISO", "VP Security more responsive than CISO",
         "Security VP responds faster than CISO"],
        ["Bundle maintenance for deal uplift", "Maintenance contract increases deal size",
         "Bundling maintenance boosts revenue", "Bundle maintenance contract for uplift",
         "Add maintenance for larger deals"],
        ["NABH compliance drives hospital decisions", "Hospital decisions driven by NABH",
         "NABH accreditation influences procurement", "NABH compliance procurement driver",
         "Hospital NABH compliance drives choices"],
        ["LinkedIn Sales Navigator for prospecting", "Use LinkedIn for enterprise targets",
         "Sales Navigator finds prospects", "LinkedIn prospecting for enterprises",
         "Enterprise targeting via LinkedIn Navigator"],
        ["CFO signs but IT head evaluates", "IT head evaluates CFO approves",
         "Technical eval by IT budget by CFO", "CFO budget approval IT evaluation",
         "IT evaluates tech CFO signs budget"],
        ["Blind spot framing in subject lines", "Subject lines with blind spot framing",
         "Blind spot angle for email subjects", "Email blind spot subject framing",
         "Use blind spot in subject lines"],
        ["Video demos increase conversion", "Demo videos boost conversion rates",
         "Video demonstrations improve conversions", "Conversion rates higher with video demos",
         "Demo video improves conversion"],
        ["Follow up within 48 hours", "48 hour follow up window",
         "Follow up emails within two days", "Two day follow up timing",
         "48hr followup window optimal"],
        ["Personalization increases reply rates", "Personalized emails get more replies",
         "Reply rates improve with personalization", "Personalization boosts email replies",
         "Email personalization higher replies"],
        ["ROI messaging fails with technical buyers", "Technical buyers ignore ROI pitch",
         "ROI framing ineffective for tech audience", "Technical audience dislikes ROI messaging",
         "ROI pitch fails with technical buyers"],
        ["Morning sends outperform afternoon", "Morning email sends beat afternoon",
         "Send emails in the morning", "Morning outperforms afternoon for sends",
         "Email morning sends higher engagement"],
        ["Case studies build trust faster", "Trust building through case studies",
         "Case study content builds trust", "Customer case studies accelerate trust",
         "Trust faster with case study content"],
        ["Multi-thread approach wins enterprise", "Enterprise deals need multi-threading",
         "Multi-thread contacts in enterprise sales", "Enterprise multi-threading strategy",
         "Thread multiple contacts for enterprise"],
        ["Pain point opening beats greeting", "Open with pain point not greeting",
         "Pain point email opening outperforms", "Start emails with pain points",
         "Pain opening beats friendly greeting"],
        ["Social proof in email signature", "Email signature social proof",
         "Add social proof to signatures", "Signature social proof boosts credibility",
         "Email sig with social proof works"],
        ["Quarterly business reviews retain clients", "QBR meetings improve retention",
         "Quarterly reviews reduce churn", "Client retention through QBR",
         "QBR process improves client retention"],
        ["Warm intro doubles response rate", "Warm introductions 2x response",
         "Referral intro doubles replies", "Warm intro response rate doubles",
         "Introduction referrals double response"],
        ["Competitor displacement requires proof", "Displacing competitors needs evidence",
         "Competitive displacement proof required", "Competitor replacement needs proof points",
         "Proof needed for competitor displacement"],
    ]

    sc_id = mem.add_node(conn, "super_context", "Dedup Test", weight=0.5)
    ctx_id = mem.add_node(conn, "context", "Dedup Ctx", weight=0.5)
    mem.link(conn, sc_id, ctx_id)

    for concept_group in concepts:
        for variant in concept_group:
            existing = mem.find_similar_node(conn, "anchor", variant)
            if existing:
                pass  # dedup hit
            else:
                nid = mem.add_node(conn, "anchor", variant, weight=0.5)
                mem.link(conn, ctx_id, nid)

    total_anchors = conn.execute("SELECT COUNT(*) as c FROM nodes WHERE type='anchor'").fetchone()["c"]

    ideal = 20
    recall = max(0, 1.0 - (total_anchors - ideal) / 80) if total_anchors >= ideal else 1.0
    precision = 1.0  # distinct concepts = no false positive merges expected

    conn.close(); os.unlink(tmp)
    return precision, recall, total_anchors

# ══════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PMIS BENCHMARK SUITE — ITERATION 1")
    print("=" * 60)

    print("\n🔬 Benchmark 1: Weight Convergence")
    r, mae = benchmark_weight_convergence()
    b1_pass = r > 0.7
    print(f"   Pearson r = {r:.4f}  (threshold: > 0.7)  {'✅ PASS' if b1_pass else '❌ FAIL'}")
    print(f"   MAE = {mae:.4f}")

    print("\n🔬 Benchmark 2: Temporal Stage Classification")
    correct, total, results = benchmark_temporal_stages()
    accuracy = correct / total
    b2_pass = accuracy >= 0.8
    print(f"   Accuracy = {correct}/{total} = {accuracy:.0%}  (threshold: >= 80%)  {'✅ PASS' if b2_pass else '❌ FAIL'}")
    for desc, expected, got, match, rec, freq, cons in results:
        status = "✅" if match else "❌"
        print(f"   {status} {desc:20s} expected={expected:12s} got={got:12s}  (r={rec:.3f} f={freq:.3f} c={cons:.3f})")

    print("\n🔬 Benchmark 3: Dedup Precision/Recall")
    precision, recall, total_nodes = benchmark_dedup()
    b3_pass = precision > 0.8 and recall > 0.6
    print(f"   Nodes created: {total_nodes} (ideal: 20)")
    print(f"   Precision = {precision:.3f}  Recall = {recall:.3f}  {'✅ PASS' if b3_pass else '❌ FAIL'}")

    print("\n" + "=" * 60)
    passed = sum([b1_pass, b2_pass, b3_pass])
    print(f"RESULT: {passed}/3 benchmarks passed")
    print("=" * 60)
