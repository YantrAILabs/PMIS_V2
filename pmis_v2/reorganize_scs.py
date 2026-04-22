"""
SC Reorganization Script — 28 SCs → 8 Knowledge + 1 Activity Log

This script:
1. Creates 6 new destination SCs (Sales&GTM, Vision AI, Security, Memory&AI, Design&UX, Other)
2. Reparents all CTXs from source SCs to destinations
3. Splits "AI & ML Knowledge Base" (vision CTXs → Vision AI, research CTXs → Memory&AI)
4. Creates "Activity Log" SC, moves 5 auto-generated SCs under it
5. Soft-deletes empty source SCs
6. Creates activity_segments table

Run: python3 pmis_v2/reorganize_scs.py
     python3 pmis_v2/reorganize_scs.py --dry-run   (preview only)
"""

import sqlite3
import json
import sys
import os
import hashlib
from datetime import datetime
from pathlib import Path

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "memory.db")


def generate_id(content):
    return hashlib.sha256(content.encode()).hexdigest()[:10]


def find_sc_id(conn, pattern):
    """Find SC by content pattern."""
    row = conn.execute(
        "SELECT id FROM memory_nodes WHERE level='SC' AND is_deleted=0 AND content LIKE ?",
        (f"%{pattern}%",)
    ).fetchone()
    return row[0] if row else None


def get_children_ids(conn, parent_id):
    """Get direct child IDs via relations."""
    return [r[0] for r in conn.execute(
        "SELECT source_id FROM relations WHERE target_id=? AND relation_type='child_of'",
        (parent_id,)
    ).fetchall()]


def reparent_children(conn, from_sc_id, to_sc_id, to_tree_id, dry_run=False):
    """Move all direct children from one SC to another."""
    children = get_children_ids(conn, from_sc_id)
    if not children:
        return 0

    if dry_run:
        return len(children)

    for child_id in children:
        # Update relation: point to new parent
        conn.execute(
            "UPDATE relations SET target_id=?, tree_id=? WHERE source_id=? AND target_id=? AND relation_type='child_of'",
            (to_sc_id, to_tree_id, child_id, from_sc_id)
        )
        # Update parent_ids JSON
        row = conn.execute("SELECT parent_ids FROM memory_nodes WHERE id=?", (child_id,)).fetchone()
        if row:
            try:
                parents = json.loads(row[0]) if row[0] else []
            except (json.JSONDecodeError, TypeError):
                parents = []
            parents = [p for p in parents if p != from_sc_id]
            if to_sc_id not in parents:
                parents.append(to_sc_id)
            conn.execute("UPDATE memory_nodes SET parent_ids=? WHERE id=?", (json.dumps(parents), child_id))

        # Update tree_ids
        row2 = conn.execute("SELECT tree_ids FROM memory_nodes WHERE id=?", (child_id,)).fetchone()
        if row2:
            try:
                trees = json.loads(row2[0]) if row2[0] else []
            except (json.JSONDecodeError, TypeError):
                trees = []
            if to_tree_id not in trees:
                trees.append(to_tree_id)
            conn.execute("UPDATE memory_nodes SET tree_ids=? WHERE id=?", (json.dumps(trees), child_id))

    return len(children)


def create_sc(conn, name, description, dry_run=False):
    """Create a new SC node."""
    sc_id = generate_id(name)
    tree_id = f"tree_{sc_id}"

    if dry_run:
        return sc_id, tree_id

    # Check if already exists
    existing = conn.execute("SELECT id FROM memory_nodes WHERE id=?", (sc_id,)).fetchone()
    if existing:
        return sc_id, tree_id

    content = f"{name}. {description}"
    conn.execute("""
        INSERT INTO memory_nodes (id, content, level, parent_ids, tree_ids, precision,
                                  surprise_at_creation, access_count, is_orphan, is_tentative, is_deleted)
        VALUES (?, ?, 'SC', '[]', ?, 0.5, 0.0, 0, 0, 0, 0)
    """, (sc_id, content, json.dumps([tree_id])))

    # Create tree entry
    conn.execute("""
        INSERT OR IGNORE INTO trees (tree_id, name, description, root_node_id)
        VALUES (?, ?, ?, ?)
    """, (tree_id, name, description, sc_id))

    return sc_id, tree_id


def soft_delete_sc(conn, sc_id, dry_run=False):
    """Soft-delete an SC if it has no remaining children.

    Also drops any `trees` row whose root is this SC, so downstream queries
    (tree lists, dashboards) don't continue to surface dead trees. Any nodes
    that still reference the dead tree_id in their JSON `tree_ids` field are
    rewritten to drop it — prevents orphan references that would need a later
    reparent pass.
    """
    children = get_children_ids(conn, sc_id)
    if children:
        return False, len(children)

    if dry_run:
        return True, 0

    conn.execute("UPDATE memory_nodes SET is_deleted=1 WHERE id=?", (sc_id,))

    # Drop any trees rooted at this SC and scrub their ids from all nodes
    dead_tree_ids = [
        r[0] for r in conn.execute(
            "SELECT tree_id FROM trees WHERE root_node_id=?", (sc_id,)
        ).fetchall()
    ]
    for dt in dead_tree_ids:
        for nid, tids_json in conn.execute(
            "SELECT id, tree_ids FROM memory_nodes WHERE is_deleted=0 AND instr(tree_ids, ?) > 0",
            (dt,),
        ).fetchall():
            try:
                tids = json.loads(tids_json or "[]")
            except Exception:
                continue
            if dt in tids:
                tids = [t for t in tids if t != dt]
                conn.execute(
                    "UPDATE memory_nodes SET tree_ids=? WHERE id=?",
                    (json.dumps(tids), nid),
                )
    if dead_tree_ids:
        conn.execute(
            f"DELETE FROM trees WHERE tree_id IN ({','.join(['?'] * len(dead_tree_ids))})",
            dead_tree_ids,
        )

    return True, 0


def main():
    dry_run = "--dry-run" in sys.argv
    mode = "DRY RUN" if dry_run else "LIVE"

    print(f"{'=' * 60}")
    print(f"SC REORGANIZATION [{mode}]")
    print(f"{'=' * 60}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=OFF")  # Temporarily for bulk operations

    stats = {"created": 0, "reparented": 0, "deleted": 0, "errors": []}

    # ─── STEP 1: Create destination SCs ───
    print("\n── Step 1: Creating destination SCs ──")

    destinations = [
        ("Yantra AI — Sales & GTM",
         "All go-to-market: cold outreach, lead pipeline, proposals, LinkedIn, case studies"),
        ("Vision AI Platform",
         "Vision tech: products, pipelines, CCTV analytics, dwell time, computer vision"),
        ("Security Verticals",
         "Security use cases: demo pages, hospital, residential, construction monitoring"),
        ("Memory & AI Research",
         "PMIS design, agent memory research, ML knowledge, productivity systems, LLM architecture"),
        ("Design & UX",
         "UI/UX design patterns, visual design, website design, branding"),
        ("Other Projects",
         "Standalone projects: Gutzy food innovation, Silver ETF trading"),
        ("Activity Log",
         "Auto-generated productivity tracking data, work narration segments"),
    ]

    dest_ids = {}
    for name, desc in destinations:
        sc_id, tree_id = create_sc(conn, name, desc, dry_run)
        dest_ids[name] = (sc_id, tree_id)
        stats["created"] += 1
        print(f"  {'[DRY]' if dry_run else '[OK]'} Created: {name} (id={sc_id})")

    # ─── STEP 2: Reparent CTXs ───
    print("\n── Step 2: Reparenting CTXs ──")

    # Mapping: source SC pattern → destination name
    merge_map = [
        # Sales & GTM
        ("Yantra AI — Sales", "Yantra AI — Sales & GTM"),
        ("B2B Cold Outreach", "Yantra AI — Sales & GTM"),
        ("GTM Lead Pipeline", "Yantra AI — Sales & GTM"),
        ("LinkedIn Content", "Yantra AI — Sales & GTM"),
        ("Case Study", "Yantra AI — Sales & GTM"),

        # Vision AI (note: AI & ML Knowledge handled separately in Step 3)
        ("Vision AI Products", "Vision AI Platform"),
        ("Vision Pipeline", "Vision AI Platform"),
        ("CCTV Dwell Time", "Vision AI Platform"),

        # Security
        ("Security Demo", "Security Verticals"),
        ("Hospital CCTV", "Security Verticals"),
        ("Residential Society", "Security Verticals"),
        ("Construction Site", "Security Verticals"),

        # Memory & AI Research
        ("AI Agent Memory", "Memory & AI Research"),
        ("PMIS Architecture", "Memory & AI Research"),
        ("Personal Productivity", "Memory & AI Research"),
        ("Test Team Memory", "Memory & AI Research"),

        # Design & UX
        ("General Design", "Design & UX"),
        ("E2E Test Website", "Design & UX"),

        # Other
        ("Gutzy", "Other Projects"),
        ("Silver ETF", "Other Projects"),

        # Activity Log
        ("Operations", "Activity Log"),
        ("Product Development", "Activity Log"),
        ("Client Deliverable", "Activity Log"),
        ("Internal Tooling", "Activity Log"),
    ]

    for source_pattern, dest_name in merge_map:
        source_id = find_sc_id(conn, source_pattern)
        if not source_id:
            stats["errors"].append(f"Source SC not found: {source_pattern}")
            continue

        dest_sc_id, dest_tree_id = dest_ids[dest_name]
        count = reparent_children(conn, source_id, dest_sc_id, dest_tree_id, dry_run)
        stats["reparented"] += count
        print(f"  {'[DRY]' if dry_run else '[OK]'} {source_pattern:30s} → {dest_name:25s} ({count} CTXs)")

    # ─── STEP 3: Split AI & ML Knowledge Base ───
    print("\n── Step 3: Splitting AI & ML Knowledge Base ──")

    aiml_id = find_sc_id(conn, "AI & ML Knowledge")
    if aiml_id:
        # Get children
        children = conn.execute("""
            SELECT mn.id, mn.content FROM relations r
            JOIN memory_nodes mn ON mn.id = r.source_id
            WHERE r.target_id=? AND r.relation_type='child_of' AND mn.is_deleted=0
        """, (aiml_id,)).fetchall()

        vision_keywords = ["computer vision", "video processing", "real-time ai"]
        research_keywords = ["llm", "agent", "ai industry", "landscape"]

        for child_id, content in children:
            content_lower = content.lower()
            dest = None

            for kw in vision_keywords:
                if kw in content_lower:
                    dest = "Vision AI Platform"
                    break

            if not dest:
                for kw in research_keywords:
                    if kw in content_lower:
                        dest = "Memory & AI Research"
                        break

            if not dest:
                dest = "Memory & AI Research"  # default

            dest_sc_id, dest_tree_id = dest_ids[dest]

            if not dry_run:
                # Reparent this specific child
                conn.execute(
                    "UPDATE relations SET target_id=?, tree_id=? WHERE source_id=? AND target_id=? AND relation_type='child_of'",
                    (dest_sc_id, dest_tree_id, child_id, aiml_id)
                )

            stats["reparented"] += 1
            print(f"  {'[DRY]' if dry_run else '[OK]'} {content[:40]:40s} → {dest}")
    else:
        print("  [SKIP] AI & ML Knowledge Base not found")

    # ─── STEP 3b: Handle Learning & Research (auto-SC) ───
    print("\n── Step 3b: Learning & Research (auto-SC) ──")
    lr_id = find_sc_id(conn, "Learning & Research")
    if lr_id:
        children = conn.execute("""
            SELECT mn.id, mn.content FROM relations r
            JOIN memory_nodes mn ON mn.id = r.source_id
            WHERE r.target_id=? AND r.relation_type='child_of' AND mn.is_deleted=0
        """, (lr_id,)).fetchall()

        for child_id, content in children:
            content_lower = content.lower()
            # Research-related CTXs go to Memory & AI Research
            if any(kw in content_lower for kw in ["researching ai", "architecture design", "ai interaction"]):
                dest = "Memory & AI Research"
            else:
                dest = "Activity Log"

            dest_sc_id, dest_tree_id = dest_ids[dest]
            if not dry_run:
                conn.execute(
                    "UPDATE relations SET target_id=?, tree_id=? WHERE source_id=? AND target_id=? AND relation_type='child_of'",
                    (dest_sc_id, dest_tree_id, child_id, lr_id)
                )
            stats["reparented"] += 1
            print(f"  {'[DRY]' if dry_run else '[OK]'} {content[:40]:40s} → {dest}")

    # ─── STEP 4: Soft-delete empty source SCs ───
    print("\n── Step 4: Cleaning up empty source SCs ──")

    all_source_patterns = [p[0] for p in merge_map] + ["AI & ML Knowledge", "Learning & Research"]
    for pattern in all_source_patterns:
        sc_id = find_sc_id(conn, pattern)
        if not sc_id:
            continue
        deleted, remaining = soft_delete_sc(conn, sc_id, dry_run)
        if deleted:
            stats["deleted"] += 1
            print(f"  {'[DRY]' if dry_run else '[OK]'} Deleted: {pattern}")
        else:
            print(f"  [SKIP] {pattern} still has {remaining} children")

    # ─── STEP 5: Create activity_segments table ───
    print("\n── Step 5: Creating activity_segments table ──")
    if not dry_run:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS activity_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_text TEXT,
                category TEXT,
                duration_seconds INTEGER DEFAULT 10,
                timestamp TEXT,
                mapped_sc_id TEXT,
                mapped_ctx_id TEXT,
                source TEXT DEFAULT 'productivity_tracker'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS activity_aggregates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_sc_id TEXT,
                period TEXT,
                total_segments INTEGER,
                total_duration_mins REAL,
                top_activities TEXT,
                generated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_seg_cat ON activity_segments(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_seg_sc ON activity_segments(mapped_sc_id)")
        print("  [OK] activity_segments + activity_aggregates tables created")
    else:
        print("  [DRY] Would create activity_segments + activity_aggregates tables")

    # ─── COMMIT ───
    if not dry_run:
        conn.commit()
        print(f"\n{'=' * 60}")
        print(f"COMMITTED")
    else:
        conn.rollback()
        print(f"\n{'=' * 60}")
        print(f"DRY RUN — no changes made")

    print(f"{'=' * 60}")
    print(f"Stats:")
    print(f"  SCs created:    {stats['created']}")
    print(f"  CTXs reparented: {stats['reparented']}")
    print(f"  SCs deleted:    {stats['deleted']}")
    if stats["errors"]:
        print(f"  Errors:         {len(stats['errors'])}")
        for e in stats["errors"]:
            print(f"    - {e}")

    # ─── VERIFY ───
    if not dry_run:
        print(f"\n── Verification ──")
        conn2 = sqlite3.connect(DB_PATH)
        c = conn2.cursor()
        c.execute("SELECT level, COUNT(*) FROM memory_nodes WHERE is_deleted=0 GROUP BY level")
        print(f"  Node counts:")
        for r in c.fetchall():
            print(f"    {r[0]:6s}: {r[1]}")
        c.execute("SELECT id, content FROM memory_nodes WHERE level='SC' AND is_deleted=0 ORDER BY content")
        print(f"  Active SCs:")
        for r in c.fetchall():
            children_count = c.execute(
                "SELECT COUNT(*) FROM relations WHERE target_id=? AND relation_type='child_of'", (r[0],)
            ).fetchone()[0]
            print(f"    [{children_count:>3} CTXs] {r[1][:60]}")
        conn2.close()

    conn.close()


if __name__ == "__main__":
    main()
