#!/usr/bin/env python3
"""
PMIS Memory Engine Versioning — lightweight snapshot system.

Usage:
    python3 scripts/version.py save P9 "P9+ Triple Fusion"
    python3 scripts/version.py restore P9
    python3 scripts/version.py list
    python3 scripts/version.py info P9
    python3 scripts/version.py diff P9 P10

Zero external dependencies. Does NOT touch graph.db (DB is forward-only).
"""

import json, sys, os, shutil, hashlib
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
VERSIONS_DIR = ROOT / "versions"
REGISTRY_FILE = VERSIONS_DIR / "VERSION_REGISTRY.json"
GRAPH_DB = ROOT / "Graph_DB" / "graph.db"

# Core files to snapshot
CORE_FILES = [
    ("scripts/memory.py", "memory.py"),
    ("scripts/p9_retrieve.py", "p9_retrieve.py"),
    ("scripts/autoresearch.py", "autoresearch.py"),
    ("scripts/autoresearch_v2.py", "autoresearch_v2.py"),
    ("scripts/ground_truth_100.py", "ground_truth_100.py"),
    ("scripts/ground_truth.py", "ground_truth.py"),
]

CONFIG_FILES = [
    ("Graph_DB/experiments/best_config.json", "config/best_config.json"),
    ("Graph_DB/experiments/best_config_v2.json", "config/best_config_v2.json"),
]


def _sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_registry():
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {"versions": [], "current": None}


def _save_registry(reg):
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2))


def _get_db_stats():
    """Query graph.db for node counts."""
    import sqlite3
    if not GRAPH_DB.exists():
        return {}
    conn = sqlite3.connect(str(GRAPH_DB))
    conn.row_factory = sqlite3.Row
    try:
        stats = {}
        for t in ("super_context", "context", "anchor"):
            r = conn.execute("SELECT COUNT(*) c FROM nodes WHERE type=?", (t,)).fetchone()
            stats[t + "s"] = r["c"] if r else 0
        stats["nodes"] = conn.execute("SELECT COUNT(*) c FROM nodes").fetchone()["c"]
        stats["edges"] = conn.execute("SELECT COUNT(*) c FROM edges").fetchone()["c"]
        stats["tasks"] = conn.execute("SELECT COUNT(*) c FROM tasks").fetchone()["c"]
        stats["scored_tasks"] = conn.execute("SELECT COUNT(*) c FROM tasks WHERE score > 0").fetchone()["c"]
        return stats
    finally:
        conn.close()


def _run_benchmark():
    """Run ground_truth_100 and return scores."""
    import io
    from contextlib import redirect_stdout
    sys.path.insert(0, str(SCRIPT_DIR))

    try:
        from ground_truth_100 import TESTS, run_retrieval, evaluate_test, load_memory
        from collections import defaultdict
        import memory as mem

        conn = mem.get_db()
        by_cat = defaultdict(list)

        for test in TESTS:
            if "feedback" in test:
                for sc_title, score in test["feedback"].items():
                    sc_row = conn.execute("SELECT id FROM nodes WHERE type='super_context' AND title=?", (sc_title,)).fetchone()
                    if sc_row:
                        conn.execute("UPDATE nodes SET quality=? WHERE id=?", (score, sc_row["id"]))
                conn.commit()

            result = run_retrieval(conn, test["q"])
            metrics = evaluate_test(result, test)
            by_cat[metrics["cat"]].append(metrics)

        categories = {}
        total_pass = 0
        total = 0
        for cat, items in by_cat.items():
            p = sum(1 for i in items if i["passed"])
            categories[cat] = round(p / len(items) * 100, 2) if items else 0
            total_pass += p
            total += len(items)

        conn.close()
        return {
            "overall": round(total_pass / total * 100, 2) if total > 0 else 0,
            "categories": categories,
            "passed": total_pass,
            "total": total,
        }
    except Exception as e:
        return {"error": str(e), "overall": 0, "categories": {}}


# ══════════════════════════════════════════════
# COMMANDS
# ══════════════════════════════════════════════

def cmd_save(name, description=""):
    """Snapshot current engine state."""
    ver_dir = VERSIONS_DIR / name
    if ver_dir.exists():
        print(f"  ERROR: Version '{name}' already exists. Use a different name.")
        return

    ver_dir.mkdir(parents=True, exist_ok=True)
    (ver_dir / "config").mkdir(exist_ok=True)

    # Copy files
    checksums = {}
    copied = 0
    for src_rel, dst_rel in CORE_FILES + CONFIG_FILES:
        src = ROOT / src_rel
        dst = ver_dir / dst_rel
        if src.exists():
            shutil.copy2(str(src), str(dst))
            checksums[dst_rel] = _sha256(dst)
            copied += 1
            print(f"  ✓ {src_rel} → versions/{name}/{dst_rel}")
        else:
            print(f"  ⚠ {src_rel} not found, skipped")

    # DB stats
    db_stats = _get_db_stats()

    # Run benchmark
    print(f"\n  Running benchmark...")
    benchmark = _run_benchmark()
    print(f"  Benchmark: {benchmark.get('overall', 0):.2f}% ({benchmark.get('passed', 0)}/{benchmark.get('total', 0)} passed)")

    # Save benchmark scores
    (ver_dir / "benchmark_scores.json").write_text(json.dumps(benchmark, indent=2))

    # Load config params if available
    config_params = {}
    for cfg_src, _ in CONFIG_FILES:
        cfg_path = ROOT / cfg_src
        if cfg_path.exists():
            try:
                config_params[cfg_path.name] = json.loads(cfg_path.read_text())
            except Exception:
                pass

    # Create manifest
    manifest = {
        "name": name,
        "created": _now(),
        "description": description,
        "architecture_notes": "",
        "known_limitations": [],
        "benchmark": benchmark,
        "db_stats": db_stats,
        "optimized_params": config_params,
        "file_checksums": checksums,
        "files_copied": copied,
    }
    (ver_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Update registry
    reg = _load_registry()
    entry = {
        "name": name,
        "created": manifest["created"],
        "description": description,
        "benchmark_score": benchmark.get("overall", 0),
        "categories": benchmark.get("categories", {}),
        "key_changes": [],
        "known_limitations": [],
        "db_stats": db_stats,
    }
    # Remove existing entry with same name if any
    reg["versions"] = [v for v in reg["versions"] if v["name"] != name]
    reg["versions"].append(entry)
    reg["current"] = name
    _save_registry(reg)

    print(f"\n  ✓ Version '{name}' saved ({copied} files)")
    print(f"  ✓ Registry updated: {REGISTRY_FILE}")


def cmd_restore(name):
    """Restore engine from a saved version."""
    ver_dir = VERSIONS_DIR / name
    if not ver_dir.exists():
        print(f"  ERROR: Version '{name}' not found.")
        return

    restored = 0
    for src_rel, dst_rel in CORE_FILES:
        src = ver_dir / dst_rel
        dst = ROOT / src_rel
        if src.exists():
            # Backup current before overwrite
            if dst.exists():
                backup = dst.with_suffix(dst.suffix + ".pre_restore")
                shutil.copy2(str(dst), str(backup))
            shutil.copy2(str(src), str(dst))
            restored += 1
            print(f"  ✓ versions/{name}/{dst_rel} → {src_rel}")

    for src_rel, dst_rel in CONFIG_FILES:
        src = ver_dir / dst_rel
        dst = ROOT / src_rel
        if src.exists():
            shutil.copy2(str(src), str(dst))
            restored += 1
            print(f"  ✓ versions/{name}/{dst_rel} → {src_rel}")

    # Update registry current
    reg = _load_registry()
    reg["current"] = name
    _save_registry(reg)

    print(f"\n  ✓ Restored {restored} files from version '{name}'")
    print(f"  ⚠ Database (graph.db) was NOT changed — DB is forward-only")


def cmd_list():
    """List all saved versions."""
    reg = _load_registry()
    if not reg["versions"]:
        print("  No versions saved yet.")
        return

    current = reg.get("current", "")
    print(f"\n  {'Name':<12} {'Score':>7} {'Date':<22} {'Description'}")
    print(f"  {'─'*70}")
    for v in reg["versions"]:
        marker = " ◀" if v["name"] == current else ""
        print(f"  {v['name']:<12} {v.get('benchmark_score', 0):>6.2f}% {v['created']:<22} {v.get('description', '')}{marker}")


def cmd_info(name):
    """Show full manifest for a version."""
    manifest_path = VERSIONS_DIR / name / "manifest.json"
    if not manifest_path.exists():
        print(f"  ERROR: Version '{name}' not found.")
        return

    m = json.loads(manifest_path.read_text())
    print(f"\n  Version: {m['name']}")
    print(f"  Created: {m['created']}")
    print(f"  Description: {m.get('description', '')}")
    print(f"  Benchmark: {m.get('benchmark', {}).get('overall', 0):.2f}%")

    cats = m.get("benchmark", {}).get("categories", {})
    if cats:
        print(f"\n  Categories:")
        for cat, score in sorted(cats.items()):
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            print(f"    {cat:<20} {bar} {score:.2f}%")

    stats = m.get("db_stats", {})
    if stats:
        print(f"\n  DB Stats: {stats.get('nodes', 0)} nodes, {stats.get('anchors', 0)} anchors, "
              f"{stats.get('super_contexts', 0)} SCs, {stats.get('scored_tasks', 0)} scored tasks")

    lims = m.get("known_limitations", [])
    if lims:
        print(f"\n  Known Limitations:")
        for l in lims:
            print(f"    • {l}")

    print(f"\n  Files: {m.get('files_copied', 0)} snapshotted")
    for f, checksum in m.get("file_checksums", {}).items():
        print(f"    {f}: {checksum[:12]}...")


def cmd_diff(name_a, name_b):
    """Compare two versions side by side."""
    ma_path = VERSIONS_DIR / name_a / "manifest.json"
    mb_path = VERSIONS_DIR / name_b / "manifest.json"

    if not ma_path.exists():
        print(f"  ERROR: Version '{name_a}' not found.")
        return
    if not mb_path.exists():
        print(f"  ERROR: Version '{name_b}' not found.")
        return

    ma = json.loads(ma_path.read_text())
    mb = json.loads(mb_path.read_text())

    score_a = ma.get("benchmark", {}).get("overall", 0)
    score_b = mb.get("benchmark", {}).get("overall", 0)

    print(f"\n  {'':20} {name_a:>15} {name_b:>15} {'Delta':>10}")
    print(f"  {'─'*65}")
    print(f"  {'Overall':20} {score_a:>14.2f}% {score_b:>14.2f}% {score_b - score_a:>+9.2f}%")

    cats_a = ma.get("benchmark", {}).get("categories", {})
    cats_b = mb.get("benchmark", {}).get("categories", {})
    all_cats = sorted(set(list(cats_a.keys()) + list(cats_b.keys())))
    for cat in all_cats:
        sa = cats_a.get(cat, 0)
        sb = cats_b.get(cat, 0)
        print(f"  {cat:20} {sa:>14.2f}% {sb:>14.2f}% {sb - sa:>+9.2f}%")

    # File checksum diffs
    fa = ma.get("file_checksums", {})
    fb = mb.get("file_checksums", {})
    changed = [f for f in set(list(fa.keys()) + list(fb.keys())) if fa.get(f) != fb.get(f)]
    if changed:
        print(f"\n  Changed files:")
        for f in sorted(changed):
            status = "modified" if f in fa and f in fb else ("added" if f in fb else "removed")
            print(f"    {f}: {status}")


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python3 scripts/version.py save P9 "description"')
        print("  python3 scripts/version.py restore P9")
        print("  python3 scripts/version.py list")
        print("  python3 scripts/version.py info P9")
        print("  python3 scripts/version.py diff P9 P10")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "save":
        name = sys.argv[2] if len(sys.argv) > 2 else "unnamed"
        desc = sys.argv[3] if len(sys.argv) > 3 else ""
        cmd_save(name, desc)
    elif cmd == "restore":
        cmd_restore(sys.argv[2])
    elif cmd == "list":
        cmd_list()
    elif cmd == "info":
        cmd_info(sys.argv[2])
    elif cmd == "diff":
        cmd_diff(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {cmd}")
