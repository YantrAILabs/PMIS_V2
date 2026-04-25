#!/usr/bin/env python3
"""Dream nightly entrypoint for launchd.

Runs the 13-phase consolidation, then the Phase 9b work-page auto-match.
Separated from the pmis_v2 cli so launchd gets a stable Python callable
with no shell involvement (avoids macOS TCC blocks on ~/Desktop scripts).
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # pmis_v2/
sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)


def main() -> int:
    print("=== dream consolidate ===", flush=True)
    try:
        from consolidation.runner import run_idempotent
        result = run_idempotent(include_today=True)
        print(f"consolidate result: {result}", flush=True)
    except Exception as e:
        print(f"consolidate failed: {e}", flush=True)

    print("=== dream match-pages ===", flush=True)
    try:
        from db.manager import DBManager
        from core import config
        from consolidation.work_page_matcher import run_work_page_matching

        db = DBManager("data/memory.db")
        hp = config.get_all()
        result = run_work_page_matching(db, hp)
        print(f"match-pages result: {result}", flush=True)
    except Exception as e:
        print(f"match-pages failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
