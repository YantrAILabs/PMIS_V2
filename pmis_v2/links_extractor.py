"""Phase D1 — pure-Python URL / file-path extractor for the links
pipeline. Deterministic, regex-only, no I/O.

Used downstream by:
  D2 — frame_analyzer.py to populate context_2.extracted_links
       and roll into context_1.segment_links.
  D3 — project_matcher to upsert link_bindings when a segment is
       tagged to a deliverable.
  D4 — the deliverable page's "Contributing links" slot.

Key design choices (from preview):
  - Paths are captured ONLY when their extension lands in a known
    "kind". Random /tmp/foo or system paths get dropped, keeping
    the signal high.
  - When a URL surfaces in both OCR text and the window name,
    `extract_all` keeps the window-name copy (cleaner string,
    no OCR errors) and tags source='window'.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# ─── Classification table ─────────────────────────────────────────────
# Order matters: the first matching domain or extension wins. Web is the
# fallthrough for any http(s) URL that didn't otherwise classify.

_DOMAIN_KIND: Tuple[Tuple[str, str], ...] = (
    ("figma.com", "design"),
    ("framer.com", "design"),
    ("github.com", "code"),
    ("gitlab.com", "code"),
    ("bitbucket.org", "code"),
    ("docs.google.com/document", "md"),
)

_EXT_KIND: Tuple[Tuple[str, str], ...] = (
    (".fig", "design"),
    (".sketch", "design"),
    (".pdf", "pdf"),
    (".md", "md"),
    (".markdown", "md"),
    (".py", "code"),
    (".ts", "code"),
    (".tsx", "code"),
    (".js", "code"),
    (".jsx", "code"),
    (".go", "code"),
    (".rs", "code"),
    (".java", "code"),
    (".cpp", "code"),
    (".c", "code"),
    (".h", "code"),
    (".html", "code"),
    (".css", "code"),
    (".png", "image"),
    (".jpg", "image"),
    (".jpeg", "image"),
    (".gif", "image"),
    (".svg", "image"),
    (".webp", "image"),
)

_PATH_ALLOWED_KINDS = {"design", "code", "pdf", "md", "image"}

# System / cache paths we never want as "files the user worked on".
_PATH_BLOCKLIST_SUBSTRINGS = (
    "/System/", "/Library/Caches/", "/private/var/", "/var/folders/",
    "/.Trash/", "/node_modules/", "/.git/", "/__pycache__/",
    "\\Windows\\", "\\AppData\\",
)


# ─── Regexes ──────────────────────────────────────────────────────────

# URLs: lazy match so trailing punctuation doesn't get glued on.
_URL_RE = re.compile(
    r"https?://[^\s<>\"'`\)\]]+",
    re.IGNORECASE,
)

# POSIX absolute paths (Mac / Linux). Excludes spaces — paths with
# spaces in the wild are ambiguous in OCR; we'd hit too many false
# positives.
_POSIX_PATH_RE = re.compile(
    r"(?:/Users/|/home/|/Volumes/|/opt/|/usr/local/share/)"
    r"[^\s<>\"'`]+",
)

# Windows: C:\..., D:\..., etc. (drive letter + backslash path).
# We deliberately don't include spaces in the character class — paths
# with spaces are too ambiguous against the surrounding sentence to
# disambiguate without context.
_WIN_PATH_RE = re.compile(
    r"[A-Z]:\\[A-Za-z0-9_\-\.\\]+",
)


# ─── Public API ───────────────────────────────────────────────────────

def classify_kind(url_or_path: str) -> str:
    """Map a URL or absolute path to one of:
        design / code / pdf / md / image / web / other
    `web` is the http(s) fallthrough; `other` for anything else."""
    if not url_or_path:
        return "other"
    s = url_or_path.lower()

    # Domain matches (only relevant for URLs)
    if s.startswith("http"):
        for needle, kind in _DOMAIN_KIND:
            if needle in s:
                return kind

    # Extension matches (works for both URLs and paths)
    for ext, kind in _EXT_KIND:
        if s.endswith(ext) or (ext + "?") in s or (ext + "#") in s:
            return kind

    if s.startswith("http"):
        return "web"
    return "other"


def extract_urls(text: str) -> List[str]:
    """Extract http(s) URLs from `text`. De-duped, original order
    preserved. Trailing common punctuation (. , ; : ) is trimmed."""
    if not text:
        return []
    seen: Dict[str, None] = {}
    for raw in _URL_RE.findall(text):
        cleaned = raw.rstrip(".,;:!?")
        if cleaned and cleaned not in seen:
            seen[cleaned] = None
    return list(seen.keys())


def extract_paths(text: str) -> List[str]:
    """Extract absolute file paths whose extension classifies into a
    recognized non-web kind (design/code/pdf/md/image). Random or
    system paths are dropped to keep the signal high."""
    if not text:
        return []
    candidates: List[str] = []
    candidates.extend(_POSIX_PATH_RE.findall(text))
    candidates.extend(_WIN_PATH_RE.findall(text))

    out: Dict[str, None] = {}
    for raw in candidates:
        cleaned = raw.rstrip(".,;:!?)\"'")
        if any(b in cleaned for b in _PATH_BLOCKLIST_SUBSTRINGS):
            continue
        if classify_kind(cleaned) not in _PATH_ALLOWED_KINDS:
            continue
        if cleaned not in out:
            out[cleaned] = None
    return list(out.keys())


def extract_links_from_window(window_name: str) -> List[str]:
    """Pull URLs out of a window title. Browsers (Chrome, Safari, Arc)
    sometimes append the URL after a separator. IDE patterns like
    `"file.py — project"` aren't URL-bearing — paths from those go
    through the OCR text path, not here."""
    return extract_urls(window_name or "")


def extract_all(
    text: str = "", window_name: str = "",
) -> List[Dict[str, str]]:
    """Public entry point used by the pipeline.

    Returns a list of `{url, kind, source}` dicts. Source values:
      'window' — surfaced from the window title (cleaner, prefer this
                 when the same URL appears in both)
      'ocr'    — pulled from raw_text
      'path'   — local absolute file path

    De-duped across all three sources. Order: window URLs first
    (best signal), then OCR URLs, then file paths.
    """
    out: List[Dict[str, str]] = []
    seen: Dict[str, None] = {}

    for url in extract_links_from_window(window_name):
        if url not in seen:
            seen[url] = None
            out.append({"url": url, "kind": classify_kind(url), "source": "window"})

    for url in extract_urls(text):
        if url not in seen:
            seen[url] = None
            out.append({"url": url, "kind": classify_kind(url), "source": "ocr"})

    for path in extract_paths(text):
        if path not in seen:
            seen[path] = None
            out.append({"url": path, "kind": classify_kind(path), "source": "path"})

    return out
