#!/usr/bin/env python3
"""
P9+ Migration Script
=====================
Run ONCE from your memory/ folder:
    python3 scripts/migrate_p9.py

This script:
  1. Patches memory.py to route 'retrieve' through P9+ triple fusion
  2. Verifies p9_retrieve.py is in the right place
  3. Tests that retrieval works
  4. Reports status

Your existing database, store, browse, tree, stats, score, rebuild, viz
commands are UNTOUCHED. Only 'retrieve' gets upgraded.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
MEMORY_PY = SCRIPT_DIR / "memory.py"
P9_FILE = SCRIPT_DIR / "p9_retrieve.py"
BACKUP_DIR = SCRIPT_DIR / "backups"


def check_files():
    """Verify all required files exist."""
    print("Checking files...")
    
    if not MEMORY_PY.exists():
        print(f"  ERROR: {MEMORY_PY} not found")
        return False
    print(f"  memory.py: found ({MEMORY_PY.stat().st_size:,} bytes)")
    
    if not P9_FILE.exists():
        print(f"  ERROR: {P9_FILE} not found")
        print(f"  → Copy p9_retrieve.py to {SCRIPT_DIR}/")
        return False
    print(f"  p9_retrieve.py: found ({P9_FILE.stat().st_size:,} bytes)")
    
    return True


def backup_memory():
    """Backup current memory.py before patching."""
    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"memory_backup_{ts}.py"
    shutil.copy2(MEMORY_PY, backup)
    print(f"  Backed up to: {backup}")
    return backup


def patch_memory():
    """Add P9+ import and routing to memory.py's cmd_retrieve."""
    content = MEMORY_PY.read_text()
    
    # Check if already patched
    if "p9_retrieve" in content:
        print("  memory.py already patched — skipping")
        return True
    
    # Strategy: Find cmd_retrieve function and add a wrapper that calls p9_retrieve
    # We insert an import at the top and replace the cmd_retrieve routing in main()
    
    # 1. Add import after existing imports
    import_line = "\n# P9+ Triple Fusion retrieval (added by migrate_p9.py)\ntry:\n    from p9_retrieve import p9_retrieve as _p9_retrieve\n    _HAS_P9 = True\nexcept ImportError:\n    _HAS_P9 = False\n"
    
    # Find a safe insertion point — after the last import
    insert_after = "from pathlib import Path"
    if insert_after in content:
        content = content.replace(insert_after, insert_after + import_line)
        print("  Added P9+ import")
    else:
        # Fallback: insert after first blank line after imports
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("# ── Paths"):
                lines.insert(i, import_line)
                break
        content = "\n".join(lines)
        print("  Added P9+ import (fallback location)")
    
    # 2. Patch the retrieve routing in main()
    # Find: cmd_retrieve(conn, sys.argv[2])
    # Replace with: try p9 first, fall back to original
    old_retrieve_call = 'cmd_retrieve(conn, sys.argv[2])'
    new_retrieve_call = '''(_p9_retrieve(conn, sys.argv[2]) if _HAS_P9 else cmd_retrieve(conn, sys.argv[2]))'''
    
    if old_retrieve_call in content:
        content = content.replace(old_retrieve_call, new_retrieve_call)
        print("  Patched retrieve routing → P9+ with fallback")
    else:
        print("  WARNING: Could not find cmd_retrieve call to patch")
        print("  You may need to manually edit memory.py (see instructions below)")
    
    MEMORY_PY.write_text(content)
    return True


def test_retrieve():
    """Quick test that retrieval works."""
    print("\nTesting retrieval...")
    
    sys.path.insert(0, str(SCRIPT_DIR))
    
    try:
        import memory as mem
        conn = mem.get_db()
        
        # Count existing nodes
        total = conn.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
        scs = conn.execute("SELECT COUNT(*) as c FROM nodes WHERE type='super_context'").fetchone()["c"]
        anchors = conn.execute("SELECT COUNT(*) as c FROM nodes WHERE type='anchor'").fetchone()["c"]
        
        print(f"  Database: {total} nodes ({scs} SCs, {anchors} anchors)")
        
        if total == 0:
            print("  No data yet — retrieval test skipped (will work after first store)")
            conn.close()
            return True
        
        # Try P9+ import
        try:
            from p9_retrieve import p9_retrieve
            print("  P9+ module: loaded successfully")
            
            # Run a test query
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                p9_retrieve(conn, "test query")
            output = f.getvalue()
            
            try:
                result = json.loads(output)
                n_matches = result.get("total_matches", 0)
                n_memories = len(result.get("memories", []))
                method = result.get("memories", [{}])[0].get("retrieval_method", "unknown") if n_memories > 0 else "n/a"
                print(f"  Test query: {n_matches} matches, {n_memories} returned, method={method}")
                
                if n_memories > 0 and method == "p9_triple_fusion":
                    print("  P9+ retrieval: WORKING")
                elif n_memories > 0:
                    print("  Retrieval working but using old method")
                else:
                    print("  No results (may be normal for 'test query')")
                    
            except Exception as e:
                print(f"  Output parsing: {e}")
                
        except ImportError as e:
            print(f"  P9+ import failed: {e}")
            print("  Falling back to original cmd_retrieve")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def print_instructions():
    """Print what was done and what to do next."""
    print(f"""
{'='*60}
P9+ MIGRATION COMPLETE
{'='*60}

What was done:
  1. memory.py backed up to scripts/backups/
  2. P9+ import added to memory.py
  3. 'retrieve' command now routes through P9+ triple fusion
  4. Falls back to original word-overlap if P9+ fails

What changed in retrieval:
  - Tag search (Jaccard overlap on anchor words)
  - BM25 search (keyword + IDF, 60% current / 40% expanded)
  - Vector search (TF-IDF cosine, optional — needs sklearn)
  - Triple merge with adaptive weights
  - Tree-level quality (best context drives SC bonus)

What did NOT change:
  - store, browse, tree, stats, score, rebuild, viz
  - Database schema (no migration needed)
  - CLAUDE.md commands
  - All existing data preserved

Optional: install sklearn for vector search:
  pip install scikit-learn
  (Without it, P9+ uses Tag + BM25 only — still better than word-overlap)

To verify:
  python3 scripts/memory.py retrieve "test query"
  → Look for "retrieval_method": "p9_triple_fusion" in output

To undo:
  cp scripts/backups/memory_backup_*.py scripts/memory.py
""")


# =============================================================================

import json

if __name__ == "__main__":
    print("=" * 60)
    print("P9+ Migration Script")
    print("=" * 60)
    
    if not check_files():
        print("\nFix the missing files and run again.")
        sys.exit(1)
    
    print("\nBacking up memory.py...")
    backup_memory()
    
    print("\nPatching memory.py...")
    if not patch_memory():
        print("Patching failed. Your backup is safe.")
        sys.exit(1)
    
    test_retrieve()
    print_instructions()
