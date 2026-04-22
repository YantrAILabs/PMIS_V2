"""Regression guard — all text file I/O must pass encoding='utf-8'.

On Windows, Python's default text encoding is cp1252 (or similar), which
cannot decode valid UTF-8 files containing non-ASCII characters. Any
`Path.read_text()`, `Path.write_text(value)`, or `open(...)` text-mode call
without an explicit encoding= kwarg is a latent Windows-install bug.

This test greps the Python codebase for the bug pattern and fails if it
finds any. Run manually:

    python3 -m pytest tests/test_text_io_utf8.py -v

Or as a standalone script:

    python3 tests/test_text_io_utf8.py

If you legitimately need binary I/O (not covered by this test anyway) or
non-UTF8 encoding, this guard ignores `open(..., "rb")`, `"wb"`, `"ab"`,
and anything with `encoding=` already present.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories to audit. Skip vendored / generated trees.
AUDIT_DIRS = [
    "pmis_v2",
    "productivity-tracker/src",
    "productivity-tracker/daemon",
    "prome_installer",
    "memory_system",
    "proxy",
]

# Directories NEVER to audit.
SKIP_DIRS = {
    "__pycache__", ".venv", "venv", "node_modules", ".git",
    "data", "logs", "chromadb", "chroma",
}

# read_text() / write_text() without encoding=
RE_READ_TEXT = re.compile(r"\.read_text\(\s*\)")
RE_WRITE_TEXT_NO_ENC = re.compile(r"\.write_text\([^)]*\)")

# open(...) text-mode without encoding=
# We match `open(` followed by any chars, and rely on later heuristics to
# filter out binary-mode and encoding-present calls.
RE_OPEN = re.compile(r"\bopen\s*\(")


def _is_text_mode_open(full_call: str, preceding: str = "") -> bool:
    """Return True if this open() is text mode AND has no encoding=."""
    # Binary mode — safe, not our concern
    if re.search(r"['\"](rb|wb|ab|r\+b|w\+b|a\+b|br|bw|ba)['\"]", full_call):
        return False
    # Has explicit encoding — safe
    if "encoding=" in full_call:
        return False
    # webbrowser.open(url) — not a file open. `preceding` ends just before "open("
    # so we look for trailing "webbrowser." / "Image." / "os." / "urllib.request."
    if re.search(r"(webbrowser|Image|os|urllib\.request|_urllib|request)\.\s*$", preceding):
        return False
    # subprocess redirect: stdout=open(...) / stderr=open(...) — the subprocess
    # writes its own bytes, Python's text encoding doesn't apply
    if re.search(r"(stdout|stderr|stdin)\s*=\s*$", preceding):
        return False
    return True


def _walk_py_files() -> list[Path]:
    files: list[Path] = []
    for d in AUDIT_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            files.append(p)
    return files


def find_violations() -> list[tuple[Path, int, str]]:
    """Return list of (file, line_number, offending_line)."""
    violations: list[tuple[Path, int, str]] = []

    for py in _walk_py_files():
        try:
            lines = py.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        # Track multi-line open( ... ) calls by buffering
        buf = ""
        buf_start_line = 0
        depth = 0

        for i, line in enumerate(lines, start=1):
            # Skip comments
            if line.lstrip().startswith("#"):
                continue

            # Fast local checks first
            if RE_READ_TEXT.search(line):
                violations.append((py, i, line.rstrip()))
                continue

            # write_text — accumulate multi-line calls until balanced parens,
            # then check the full call text for encoding=
            if ".write_text(" in line and "encoding=" not in line:
                # Collect the full call, possibly across lines
                start_idx = line.find(".write_text(") + len(".write_text")
                snippet = line[start_idx:]
                j = i
                bal = snippet.count("(") - snippet.count(")")
                while bal > 0 and j < len(lines):
                    snippet += " " + lines[j]
                    bal += lines[j].count("(") - lines[j].count(")")
                    j += 1
                # Now `snippet` contains the full parenthesized call
                if "encoding=" not in snippet:
                    violations.append((py, i, line.rstrip()))

            # open( ... ) — may span lines; approximate with a per-line heuristic
            if RE_OPEN.search(line):
                # Collect the next few lines until paren balances
                snippet = line
                j = i
                bal = line.count("(") - line.count(")")
                while bal > 0 and j < len(lines):
                    snippet += " " + lines[j]
                    bal += lines[j].count("(") - lines[j].count(")")
                    j += 1
                # Extract just the open(...) call from snippet (best-effort)
                m = re.search(r"\bopen\s*\([^()]*(?:\([^()]*\)[^()]*)*\)", snippet)
                if not m:
                    continue
                call = m.group(0)
                # Pass the 30 chars BEFORE `open(` so we can tell webbrowser.open,
                # Image.open, os.open etc. from file-open.
                call_pos = snippet.find(call)
                preceding = snippet[max(0, call_pos - 30):call_pos]
                if _is_text_mode_open(call, preceding):
                    violations.append((py, i, line.rstrip()))

    return violations


def test_no_utf8_violations():
    """pytest entry point."""
    violations = find_violations()
    assert not violations, "Found {} unprotected text I/O call(s):\n{}".format(
        len(violations),
        "\n".join(f"  {p.relative_to(REPO_ROOT)}:{n}: {s}" for p, n, s in violations),
    )


def main() -> int:
    violations = find_violations()
    if not violations:
        print("✓ No unprotected text I/O calls found.")
        return 0
    print(f"✗ Found {len(violations)} unprotected text I/O call(s):")
    for p, n, s in violations:
        print(f"  {p.relative_to(REPO_ROOT)}:{n}: {s}")
    print()
    print("Fix: add `encoding='utf-8'` to each call. Examples:")
    print("  Path(...).read_text(encoding='utf-8')")
    print("  Path(...).write_text(content, encoding='utf-8')")
    print("  open(path, 'r', encoding='utf-8')")
    return 1


if __name__ == "__main__":
    sys.exit(main())
