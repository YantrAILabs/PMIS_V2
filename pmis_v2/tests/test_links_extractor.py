"""Tests for Phase D1 — URL / file-path extractor + classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from links_extractor import (  # noqa: E402
    classify_kind,
    extract_all,
    extract_links_from_window,
    extract_paths,
    extract_urls,
)


# ─── classify_kind ────────────────────────────────────────────────────

class TestClassifyKind:
    @pytest.mark.parametrize("url, kind", [
        ("https://www.figma.com/design/abc/Foo", "design"),
        ("https://framer.com/something", "design"),
        ("https://github.com/anthropics/foo", "code"),
        ("https://gitlab.com/proj/repo", "code"),
        ("https://bitbucket.org/team/repo", "code"),
        ("https://example.com/whitepaper.pdf", "pdf"),
        ("https://example.com/spec.md", "md"),
        ("https://docs.google.com/document/d/abc", "md"),
        ("https://example.com/screenshot.png", "image"),
        ("https://example.com", "web"),  # http fallthrough
        ("https://news.example.com/article?x=1", "web"),
        ("/Users/x/design.fig", "design"),
        ("/Users/x/spec.pdf", "pdf"),
        ("/Users/x/notes.md", "md"),
        ("/Users/x/server.py", "code"),
        ("/Users/x/icon.svg", "image"),
        ("not a link", "other"),
        ("", "other"),
    ])
    def test_classifies(self, url, kind):
        assert classify_kind(url) == kind

    def test_query_string_does_not_break_extension_match(self):
        assert classify_kind("https://example.com/file.pdf?download=1") == "pdf"

    def test_fragment_does_not_break_extension_match(self):
        assert classify_kind("https://example.com/file.md#section") == "md"


# ─── extract_urls ─────────────────────────────────────────────────────

class TestExtractUrls:
    def test_simple_url(self):
        assert extract_urls("see https://example.com here") == ["https://example.com"]

    def test_url_with_query(self):
        out = extract_urls("https://example.com/path?x=1&y=2")
        assert out == ["https://example.com/path?x=1&y=2"]

    def test_url_with_fragment(self):
        out = extract_urls("https://example.com/page#anchor")
        assert out == ["https://example.com/page#anchor"]

    def test_multiple_urls_dedup_preserve_order(self):
        out = extract_urls(
            "https://a.com first then https://b.com and again https://a.com"
        )
        assert out == ["https://a.com", "https://b.com"]

    def test_no_url(self):
        assert extract_urls("plain text with no link") == []

    def test_empty_string(self):
        assert extract_urls("") == []

    def test_trailing_punctuation_trimmed(self):
        out = extract_urls("Look at https://example.com/foo, please.")
        assert out == ["https://example.com/foo"]


# ─── extract_paths ────────────────────────────────────────────────────

class TestExtractPaths:
    def test_posix_design_file(self):
        out = extract_paths("opened /Users/rohit/Designs/spec.fig today")
        assert out == ["/Users/rohit/Designs/spec.fig"]

    def test_windows_path(self):
        out = extract_paths(r"check C:\Users\bob\notes.md please")
        assert out == [r"C:\Users\bob\notes.md"]

    def test_path_without_recognized_extension_dropped(self):
        # /Users/x/notes (no extension) → other → blocked
        assert extract_paths("see /Users/x/notes for more") == []

    def test_system_path_blocked(self):
        assert extract_paths("/Users/x/Library/Caches/cache.png") == []

    def test_node_modules_blocked(self):
        assert extract_paths("/Users/x/proj/node_modules/lib/index.js") == []

    def test_relative_path_ignored(self):
        # We only capture absolute paths.
        assert extract_paths("./local/script.py was used") == []

    def test_dedup(self):
        out = extract_paths(
            "ran /Users/x/a.py and /Users/x/a.py again"
        )
        assert out == ["/Users/x/a.py"]


# ─── extract_links_from_window ────────────────────────────────────────

class TestExtractLinksFromWindow:
    def test_chrome_style_window(self):
        # Chrome sometimes shows: "Page Title - Browser - https://..."
        out = extract_links_from_window(
            "Anthropic - Google Chrome - https://www.anthropic.com"
        )
        assert out == ["https://www.anthropic.com"]

    def test_no_url_in_window(self):
        out = extract_links_from_window("Cursor - main.py")
        assert out == []

    def test_empty_window_name(self):
        assert extract_links_from_window("") == []
        assert extract_links_from_window(None) == []  # type: ignore[arg-type]


# ─── extract_all ──────────────────────────────────────────────────────

class TestExtractAll:
    def test_window_takes_priority_over_ocr_for_same_url(self):
        text = "i was on https://figma.com/file/abc"
        window = "Figma - https://figma.com/file/abc"
        out = extract_all(text, window)
        assert len(out) == 1
        assert out[0]["url"] == "https://figma.com/file/abc"
        assert out[0]["source"] == "window"
        assert out[0]["kind"] == "design"

    def test_ocr_url_when_not_in_window(self):
        out = extract_all("i opened https://github.com/x/y", "")
        assert out == [{
            "url": "https://github.com/x/y", "kind": "code", "source": "ocr",
        }]

    def test_path_captured_with_kind(self):
        out = extract_all("edited /Users/x/spec.md just now", "")
        assert out == [{
            "url": "/Users/x/spec.md", "kind": "md", "source": "path",
        }]

    def test_path_overlapping_with_url_dedup(self):
        """If somehow the same string shows up in both, dedup works."""
        out = extract_all(
            "twice: /Users/x/spec.md and /Users/x/spec.md", "",
        )
        assert len([d for d in out if d["url"] == "/Users/x/spec.md"]) == 1

    def test_empty_inputs(self):
        assert extract_all("", "") == []
        assert extract_all() == []

    def test_mixed_window_ocr_paths(self):
        out = extract_all(
            text="ran /Users/x/server.py; pushed to https://github.com/x/y",
            window_name="Cursor - server.py - https://www.figma.com/design/Z",
        )
        # Order: window URLs first, then OCR URLs, then paths.
        kinds_sources = [(d["kind"], d["source"]) for d in out]
        assert ("design", "window") in kinds_sources
        assert ("code", "ocr") in kinds_sources
        assert ("code", "path") in kinds_sources
        # Specifically, the window-source figma URL comes first.
        assert out[0]["source"] == "window"
        assert out[0]["kind"] == "design"

    def test_returned_shape(self):
        out = extract_all("https://example.com", "")
        assert set(out[0].keys()) == {"url", "kind", "source"}
