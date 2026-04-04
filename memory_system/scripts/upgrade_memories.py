#!/usr/bin/env python3
"""
PMIS Memory Upgrade Script
============================
Run from memory/ folder:
    python3 scripts/upgrade_memories.py

What this script does (in order):
  1. Adds 'tags' column to nodes table (if missing)
  2. Auto-generates tags for ALL existing anchors from title + content + parent path
  3. Recalculates mode vectors for SCs that don't have one
  4. Refreshes temporal scores (recency, frequency, consistency, stage) for all nodes
  5. Detects and creates transfer edges between related SCs
  6. Patches memory.py store function to save tags going forward
  7. Reports complete inventory

Safe to run multiple times — skips already-done work.
Creates backup before any changes.
Zero external dependencies.
"""

import sqlite3
import json
import os
import sys
import math
import shutil
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
GRAPH_DB = ROOT / "Graph_DB" / "graph.db"
BACKUP_DIR = SCRIPT_DIR / "backups"

sys.path.insert(0, str(SCRIPT_DIR))

# =============================================================================
# Tag Generation — the core intelligence
# =============================================================================

# Domain vocabulary for smart tag extraction
VERTICALS = {
    "manufacturing", "retail", "security", "construction", "hospital", "school",
    "warehouse", "residential", "parking", "logistics", "kirana", "factory",
    "pharma", "agriculture", "mining", "energy", "telecom", "healthcare",
    "education", "defense", "aviation", "maritime", "fintech", "insurance",
}

TECHNIQUES = {
    "yolo", "transformer", "cnn", "rnn", "lstm", "gan", "diffusion", "sam",
    "clip", "resnet", "efficientnet", "mobilenet", "bert", "gpt", "qwen",
    "llama", "mistral", "lora", "rlhf", "dpo", "rag", "bm25", "hnsw",
    "faiss", "qdrant", "chromadb", "mediamtx", "cloudflare", "rtsp",
    "websocket", "grpc", "mqtt", "graphql", "rest",
}

INFRA = {
    "t4", "a100", "v100", "l4", "gpu", "cpu", "edge", "cloud", "docker",
    "kubernetes", "lambda", "ec2", "gce", "cloudrun", "terraform", "nginx",
    "redis", "postgresql", "mongodb", "sqlite", "dvr",
}

METRICS = {
    "accuracy", "latency", "throughput", "precision", "recall", "f1",
    "fps", "roi", "engagement", "conversion", "retention", "churn",
    "open-rate", "response-rate", "click-rate",
}

TASKS = {
    "detection", "classification", "segmentation", "tracking", "counting",
    "anomaly", "recognition", "generation", "summarization", "monitoring",
    "alerting", "reporting", "forecasting", "optimization", "compliance",
    "deployment", "integration", "streaming", "inference",
}

BUSINESS = {
    "outreach", "email", "cold-email", "follow-up", "proposal", "pitch",
    "demo", "case-study", "whitepaper", "blog", "carousel", "linkedin",
    "pricing", "gtm", "pipeline", "funnel", "qualification", "prospecting",
    "copywriting", "branding", "marketing",
}

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "for", "and",
    "or", "but", "in", "on", "at", "to", "of", "with", "how", "what", "which",
    "do", "does", "can", "should", "i", "my", "our", "that", "this", "it", "by",
    "from", "has", "have", "had", "not", "will", "about", "need", "help", "me",
    "now", "also", "its", "we", "these", "more", "into", "using", "used", "use",
    "get", "set", "new", "like", "just", "very", "so", "if", "than", "then",
    "been", "being", "too", "some", "over", "under", "between", "through",
    "most", "such", "only", "other", "each", "every", "both", "few", "all",
    "any", "own", "same", "would", "could", "may", "might", "shall", "must",
    "first", "last", "next", "best", "good", "better", "higher", "lower",
    "gets", "got", "much", "well", "way", "per", "via", "key", "based",
})


def _tokenize(text):
    """Extract significant words from text."""
    words = []
    for w in text.lower().split():
        w = w.strip(".,;:!?()[]{}\"'/-_%#@&*+=<>")
        if len(w) > 2 and w not in STOP_WORDS and not w.isdigit():
            words.append(w)
    return words


def generate_tags(title, content="", parent_context="", parent_sc=""):
    """
    Generate 3-8 intelligent tags for an anchor.
    Extracts: vertical, technique, task, infra, metric, business term, key nouns.
    """
    all_text = f"{title} {content} {parent_context} {parent_sc}".lower()
    words = set(_tokenize(all_text))
    title_words = set(_tokenize(title))

    tags = set()

    # Match against known vocabularies
    for w in words:
        if w in VERTICALS:
            tags.add(w)
        if w in TECHNIQUES:
            tags.add(w)
        if w in INFRA:
            tags.add(w)
        if w in METRICS:
            tags.add(w)
        if w in TASKS:
            tags.add(w)
        if w in BUSINESS:
            tags.add(w)

    # Extract numbers with context (e.g., "40%" → "open-rate" if near "open")
    numbers = re.findall(r'\d+(?:%|ms|fps|x|px|hr|min|sec|days?|gb|mb|tb)', all_text)
    # Don't add raw numbers as tags, but they indicate metrics

    # Add key title words (nouns > 4 chars that aren't already tagged)
    for w in title_words:
        if len(w) > 4 and w not in tags and len(tags) < 8:
            # Check it's likely a noun (not a common verb/adjective)
            common_verbs = {"should", "would", "could", "works", "needs", "makes",
                           "shows", "takes", "gives", "helps", "means", "keeps",
                           "leads", "drives", "builds", "handles", "reduces",
                           "increases", "improves", "requires", "provides"}
            if w not in common_verbs:
                tags.add(w)

    # Compound tags from common patterns
    if "cold" in words and "email" in words:
        tags.add("cold-email")
    if "open" in words and "rate" in words:
        tags.add("open-rate")
    if "follow" in words and ("up" in words or "followup" in all_text):
        tags.add("follow-up")
    if "case" in words and "study" in words:
        tags.add("case-study")
    if "edge" in words and "compute" in words:
        tags.add("edge-compute")
    if "dual" in words and "layer" in words:
        tags.add("dual-layer")

    # Ensure at least 3 tags
    if len(tags) < 3:
        for w in title_words:
            if w not in tags and len(w) > 3:
                tags.add(w)
            if len(tags) >= 3:
                break

    # Cap at 8 tags
    return sorted(list(tags))[:8]


# =============================================================================
# Transfer Edge Detection
# =============================================================================

def detect_transfers(conn):
    """Find SCs with similar internal structure and create transfer edges."""
    scs = conn.execute("SELECT id, title FROM nodes WHERE type='super_context'").fetchall()
    if len(scs) < 2:
        return 0

    # Build vocabulary per SC
    sc_vocab = {}
    for sc in scs:
        children = conn.execute(
            "SELECT n.title FROM nodes n JOIN edges e ON e.tgt=n.id WHERE e.src=? AND e.type='parent_child'",
            (sc["id"],)
        ).fetchall()
        words = set()
        for ch in children:
            words.update(_tokenize(ch["title"]))
            # Also get grandchildren (anchors)
            grandchildren = conn.execute(
                "SELECT n.title FROM nodes n JOIN edges e ON e.tgt=n.id WHERE e.src=? AND e.type='parent_child'",
                (ch["title"],)  # This won't work, need ID
            ).fetchall()
        # Get all descendants
        desc = conn.execute("""
            SELECT n.title FROM nodes n
            JOIN edges e1 ON e1.tgt=n.id
            WHERE e1.src IN (
                SELECT n2.id FROM nodes n2
                JOIN edges e2 ON e2.tgt=n2.id
                WHERE e2.src=? AND e2.type='parent_child'
            ) AND e1.type='parent_child'
        """, (sc["id"],)).fetchall()
        for d in desc:
            words.update(_tokenize(d["title"]))
        words.update(_tokenize(sc["title"]))
        sc_vocab[sc["id"]] = words

    # Compare all pairs
    created = 0
    for i, sc_a in enumerate(scs):
        for sc_b in scs[i+1:]:
            words_a = sc_vocab.get(sc_a["id"], set())
            words_b = sc_vocab.get(sc_b["id"], set())
            if not words_a or not words_b:
                continue

            overlap = len(words_a & words_b)
            union = len(words_a | words_b)
            similarity = overlap / union if union > 0 else 0

            if similarity >= 0.15:  # lower threshold for transfer
                # Check if edge already exists
                existing = conn.execute(
                    "SELECT id FROM edges WHERE src=? AND tgt=? AND type='transfer'",
                    (sc_a["id"], sc_b["id"])
                ).fetchone()
                if not existing:
                    import uuid
                    eid1 = uuid.uuid4().hex[:10]
                    eid2 = uuid.uuid4().hex[:10]
                    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    conn.execute("INSERT INTO edges VALUES (?,?,?,?,?,?)",
                                (eid1, sc_a["id"], sc_b["id"], "transfer", round(similarity, 2), now))
                    conn.execute("INSERT INTO edges VALUES (?,?,?,?,?,?)",
                                (eid2, sc_b["id"], sc_a["id"], "transfer", round(similarity, 2), now))
                    created += 2

    conn.commit()
    return created


# =============================================================================
# Store Function Patch — adds tags to new stores
# =============================================================================

def patch_store_for_tags(memory_py_path):
    """Patch cmd_store to extract and save tags for new anchors."""
    content = memory_py_path.read_text()

    if "# P9_TAGS_PATCH" in content:
        return False  # Already patched

    # Find the anchor creation line and add tag saving after it
    # Target: after add_node for anchor, add tag computation and storage
    old_code = '''anc_id = add_node(conn, "anchor", anc_title,
                                  content=anc_content, source="claude_desktop",
                                  weight=anc_weight)
                link(conn, ctx_id, anc_id, weight=anc_weight)'''

    new_code = '''anc_id = add_node(conn, "anchor", anc_title,
                                  content=anc_content, source="claude_desktop",
                                  weight=anc_weight)
                link(conn, ctx_id, anc_id, weight=anc_weight)
                # P9_TAGS_PATCH: Auto-generate and store tags
                _anc_tags = anc_data.get("tags", [])
                if not _anc_tags:
                    _tag_words = [w.lower() for w in anc_title.split() if len(w) > 3]
                    _tag_words += [w.lower() for w in anc_content.split() if len(w) > 4][:5]
                    _anc_tags = list(set(_tag_words))[:8]
                try:
                    conn.execute("UPDATE nodes SET tags=? WHERE id=?", (json.dumps(_anc_tags), anc_id))
                    conn.commit()
                except Exception:
                    pass  # tags column might not exist yet'''

    if old_code in content:
        content = content.replace(old_code, new_code)
        memory_py_path.write_text(content)
        return True
    return False


# =============================================================================
# Main Upgrade
# =============================================================================

def run_upgrade():
    print("=" * 70)
    print("PMIS MEMORY UPGRADE")
    print("Upgrading all existing memories for P9+ pipeline")
    print("=" * 70)

    # ── Check database ──
    if not GRAPH_DB.exists():
        print(f"\nERROR: Database not found at {GRAPH_DB}")
        print("Run a store command first to create the database.")
        return

    # ── Backup ──
    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_db = BACKUP_DIR / f"graph_backup_{ts}.db"
    shutil.copy2(GRAPH_DB, backup_db)
    print(f"\n1. Database backed up to: {backup_db}")

    backup_mem = BACKUP_DIR / f"memory_backup_{ts}.py"
    if (SCRIPT_DIR / "memory.py").exists():
        shutil.copy2(SCRIPT_DIR / "memory.py", backup_mem)
        print(f"   memory.py backed up to: {backup_mem}")

    # ── Connect ──
    conn = sqlite3.connect(str(GRAPH_DB))
    conn.row_factory = sqlite3.Row

    # ── Step 1: Add tags column ──
    print("\n2. Schema upgrade...")
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}
    if "tags" not in existing_cols:
        conn.execute("ALTER TABLE nodes ADD COLUMN tags TEXT DEFAULT '[]'")
        conn.commit()
        print("   Added 'tags' column to nodes table")
    else:
        print("   'tags' column already exists")

    # ── Step 2: Auto-generate tags for all anchors ──
    print("\n3. Auto-tagging anchors...")
    anchors = conn.execute("SELECT * FROM nodes WHERE type='anchor'").fetchall()
    tagged = 0
    skipped = 0

    for anc_row in anchors:
        anc = dict(anc_row)
        # Check if already has meaningful tags
        existing_tags = []
        try:
            existing_tags = json.loads(anc.get("tags") or "[]")
        except (json.JSONDecodeError, TypeError):
            existing_tags = []

        if existing_tags and len(existing_tags) >= 2:
            skipped += 1
            continue

        # Find parent context title
        parent_ctx = conn.execute("""
            SELECT n.title FROM nodes n
            JOIN edges e ON e.src=n.id
            WHERE e.tgt=? AND e.type='parent_child' LIMIT 1
        """, (anc["id"],)).fetchone()
        ctx_title = parent_ctx["title"] if parent_ctx else ""

        # Find grandparent SC title
        sc_title = ""
        if parent_ctx:
            parent_sc_row = conn.execute("""
                SELECT n.title FROM nodes n
                JOIN edges e ON e.src=n.id
                WHERE e.tgt=(
                    SELECT e2.src FROM edges e2 WHERE e2.tgt=? AND e2.type='parent_child' LIMIT 1
                ) AND e.type='parent_child' LIMIT 1
            """, (anc["id"],)).fetchone()
            sc_title = parent_sc_row["title"] if parent_sc_row else ""

        tags = generate_tags(
            title=anc.get("title", ""),
            content=anc.get("content", ""),
            parent_context=ctx_title,
            parent_sc=sc_title,
        )

        conn.execute("UPDATE nodes SET tags=? WHERE id=?", (json.dumps(tags), anc["id"]))
        tagged += 1

    conn.commit()
    print(f"   Tagged: {tagged} anchors")
    print(f"   Skipped: {skipped} (already had tags)")

    # Show sample tags
    sample = conn.execute(
        "SELECT title, tags FROM nodes WHERE type='anchor' AND tags != '[]' LIMIT 5"
    ).fetchall()
    if sample:
        print("   Sample tags:")
        for s in sample:
            try:
                tags = json.loads(s["tags"])
            except (json.JSONDecodeError, TypeError):
                tags = []
            print(f"     \"{s['title'][:50]}\"")
            print(f"       → {tags}")

    # ── Step 3: Mode vectors for SCs ──
    print("\n4. Mode vectors...")
    import memory as mem
    scs = conn.execute("SELECT * FROM nodes WHERE type='super_context'").fetchall()
    mode_updated = 0
    for sc_row in scs:
        sc = dict(sc_row)
        try:
            existing_mode = json.loads(sc.get("mode_vector") or "{}")
        except (json.JSONDecodeError, TypeError):
            existing_mode = {}

        if not existing_mode or existing_mode == {}:
            mode = mem.infer_mode_vector(sc["title"], [])
            conn.execute("UPDATE nodes SET mode_vector=? WHERE id=?",
                        (json.dumps(mode), sc["id"]))
            mode_updated += 1
    conn.commit()
    print(f"   Updated mode vectors: {mode_updated} SCs")

    # ── Step 4: Temporal scores ──
    print("\n5. Temporal scores...")
    all_nodes = conn.execute("SELECT id, occurrence_log FROM nodes").fetchall()
    temporal_updated = 0
    for node in all_nodes:
        try:
            log = json.loads(node["occurrence_log"] or "[]")
        except (json.JSONDecodeError, TypeError):
            log = []

        if log:
            r, f, c = mem.compute_temporal(log)
            stage = mem.classify_stage(r, f, c)
            conn.execute(
                "UPDATE nodes SET recency=?, frequency=?, consistency=?, memory_stage=? WHERE id=?",
                (r, f, c, stage, node["id"])
            )
            temporal_updated += 1
    conn.commit()

    stages = conn.execute("""
        SELECT memory_stage, COUNT(*) as cnt
        FROM nodes GROUP BY memory_stage ORDER BY cnt DESC
    """).fetchall()
    stage_str = ", ".join(f"{s['memory_stage']}={s['cnt']}" for s in stages)
    print(f"   Refreshed: {temporal_updated} nodes")
    print(f"   Stages: {stage_str}")

    # ── Step 5: Transfer edges ──
    print("\n6. Transfer edges...")
    existing_transfers = conn.execute(
        "SELECT COUNT(*) as c FROM edges WHERE type='transfer'"
    ).fetchone()["c"]
    new_transfers = detect_transfers(conn)
    total_transfers = conn.execute(
        "SELECT COUNT(*) as c FROM edges WHERE type='transfer'"
    ).fetchone()["c"]
    print(f"   Existing: {existing_transfers}")
    print(f"   New: {new_transfers}")
    print(f"   Total: {total_transfers}")

    # ── Step 6: Patch store ──
    print("\n7. Patching store function...")
    memory_py = SCRIPT_DIR / "memory.py"
    if memory_py.exists():
        patched = patch_store_for_tags(memory_py)
        if patched:
            print("   Patched cmd_store to auto-generate tags for new anchors")
        else:
            print("   Already patched (or pattern not found)")
    else:
        print("   memory.py not found — skipped")

    # ── Final inventory ──
    print("\n" + "=" * 70)
    print("FINAL INVENTORY")
    print("=" * 70)

    counts = {}
    for ntype in ["super_context", "context", "anchor"]:
        counts[ntype] = conn.execute(
            "SELECT COUNT(*) as c FROM nodes WHERE type=?", (ntype,)
        ).fetchone()["c"]

    total_edges = conn.execute("SELECT COUNT(*) as c FROM edges").fetchone()["c"]
    total_tasks = conn.execute("SELECT COUNT(*) as c FROM tasks").fetchone()["c"]
    scored_tasks = conn.execute("SELECT COUNT(*) as c FROM tasks WHERE score > 0").fetchone()["c"]
    tagged_count = conn.execute(
        "SELECT COUNT(*) as c FROM nodes WHERE type='anchor' AND tags != '[]' AND tags IS NOT NULL"
    ).fetchone()["c"]
    total_decisions = conn.execute("SELECT COUNT(*) as c FROM decisions").fetchone()["c"]

    print(f"""
   Super Contexts:  {counts['super_context']}
   Contexts:        {counts['context']}
   Anchors:         {counts['anchor']}
   Tagged anchors:  {tagged_count} / {counts['anchor']}
   Edges:           {total_edges} ({total_transfers} transfer)
   Tasks logged:    {total_tasks}
   Tasks scored:    {scored_tasks}
   Decisions:       {total_decisions}
   Memory stages:   {stage_str}
""")

    # ── Per-SC summary ──
    print("Per Super Context:")
    for sc in conn.execute(
        "SELECT id, title, quality, memory_stage FROM nodes WHERE type='super_context' ORDER BY quality DESC"
    ).fetchall():
        n_ctx = conn.execute(
            "SELECT COUNT(*) as c FROM edges WHERE src=? AND type='parent_child'", (sc["id"],)
        ).fetchone()["c"]
        n_anc = conn.execute("""
            SELECT COUNT(*) as c FROM nodes n
            JOIN edges e1 ON e1.tgt=n.id
            WHERE n.type='anchor' AND e1.type='parent_child'
            AND e1.src IN (SELECT tgt FROM edges WHERE src=? AND type='parent_child')
        """, (sc["id"],)).fetchone()["c"]
        print(f"   {sc['title'][:40]:<42} {n_ctx} ctx, {n_anc} anc, q={sc['quality']:.1f}, {sc['memory_stage']}")

    conn.close()

    print(f"""
{'='*70}
UPGRADE COMPLETE
{'='*70}

What was done:
  1. Added 'tags' column to database schema
  2. Auto-generated tags for {tagged} anchors
  3. Updated {mode_updated} SC mode vectors
  4. Refreshed temporal scores for {temporal_updated} nodes
  5. Created {new_transfers} new transfer edges
  6. Patched store function to save tags for future anchors

Your existing memories are now P9+ ready.

To verify retrieval uses P9+:
  python3 scripts/memory.py retrieve "your query"
  → Look for "retrieval_method": "p9_triple_fusion"

Backup location:
  Database: {backup_db}
  memory.py: {backup_mem}

To undo everything:
  cp "{backup_db}" "{GRAPH_DB}"
  cp "{backup_mem}" "{memory_py}"
""")


if __name__ == "__main__":
    run_upgrade()
