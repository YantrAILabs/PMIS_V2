"""
ProMe installer entry point.

Usage:
    python3 -m prome_installer            (run from repo root)

Thin shell wrappers at the repo root — install.sh / install.bat — just invoke
this module. Idempotent: safe to re-run.
"""

from __future__ import annotations

import sys

from . import paths, steps, ui


def main() -> int:
    ui.banner("ProMe Installer")
    print(f"  Repo:     {paths.REPO_DIR}")
    print(f"  Data:     {paths.DATA_DIR}")
    print(f"  Platform: {sys.platform}")
    print()

    try:
        ctx = steps.step1_preflight()
        steps.step2_venv(ctx["python_exe"])
        steps.step3_directories()
        steps.step4_env()
        steps.step5_databases()
        steps.step6_verify()
        steps.step7_daemon()
        steps.step8_summary(perm_issues=ctx["perm_issues"])
    except KeyboardInterrupt:
        print()
        ui.warn("Interrupted by user. Re-run ./install.sh (or install.bat) to resume.")
        return 130
    except SystemExit:
        raise  # ui.fail() already exited with code 1
    except Exception as e:
        print()
        ui.fail(f"Unexpected error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
