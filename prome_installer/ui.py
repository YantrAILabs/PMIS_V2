"""
Terminal UI helpers — colored output, step banners, interactive prompts.

ANSI colors work on macOS/Linux terminals and on Windows 10+ Terminal/PowerShell.
On older Windows consoles we try to enable VT100 mode via the Win32 API;
if that fails, colors gracefully degrade (escape codes appear literally, but
the installer still runs to completion).
"""

from __future__ import annotations

import os
import sys


# ── ANSI color codes ───────────────────────────────────────────────────────
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
BOLD = "\033[1m"
NC = "\033[0m"


def _enable_windows_ansi() -> None:
    """Flip the VT100 processing bit on Windows 10+ so escape codes render."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # STD_OUTPUT_HANDLE = -11
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass  # colors will appear as raw codes — not fatal


_enable_windows_ansi()


# ── Printing primitives ────────────────────────────────────────────────────
def ok(msg: str) -> None:
    print(f"  {GREEN}\u2713{NC} {msg}")


def fail(msg: str) -> None:
    """Print and exit with code 1."""
    print(f"  {RED}\u2717{NC} {msg}")
    sys.exit(1)


def info(msg: str) -> None:
    print(f"  {BLUE}\u2192{NC} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{NC} {msg}")


def step(n: int, total: int, title: str) -> None:
    print(f"\n{BOLD}[{n}/{total}] {title}{NC}")


def banner(title: str) -> None:
    bar = "\u2550" * 39
    print()
    print(f"{BOLD}{bar}{NC}")
    print(f"{BOLD}  {title}{NC}")
    print(f"{BOLD}{bar}{NC}")
    print()


def prompt(question: str, default: str | None = None) -> str:
    """Interactive input with optional default shown in brackets."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"  {question}{suffix}: ").strip()
    except EOFError:
        answer = ""
    return answer or (default or "")


def prompt_yes_no(question: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = prompt(f"{question} ({hint})", default="y" if default else "n").lower()
    return raw.startswith("y")
