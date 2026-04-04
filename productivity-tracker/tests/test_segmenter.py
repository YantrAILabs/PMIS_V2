"""Tests for the target frame segmenter."""

import pytest
from unittest.mock import patch
from src.pipeline.segmenter import TargetFrameSegmenter, sanitize_id


@pytest.fixture
def config():
    return {
        "segmentation": {"ssim_threshold": 0.7, "min_segment_secs": 15},
        "tracking": {"skip_similar_threshold": 0.95, "frame_batch_size": 4},
    }


@pytest.fixture
def segmenter(config):
    return TargetFrameSegmenter(config)


class TestSegmenter:
    def test_first_frame_always_new_segment(self, segmenter):
        """First frame ever should always trigger a new segment."""
        # Mock image loading to avoid needing real files
        with patch.object(segmenter, "_load_image_gray", return_value=None):
            assert segmenter.should_start_new_segment(
                window_info={"bundle_id": "com.google.Chrome", "title": "Test"},
                screenshot_path="/tmp/test.jpg",
                agent_active=False,
            )

    def test_window_change_triggers_new_segment(self, segmenter):
        """Changing app or window title should start a new segment."""
        with patch.object(segmenter, "_load_image_gray", return_value=None):
            # First frame sets state
            segmenter.should_start_new_segment(
                window_info={"bundle_id": "com.google.Chrome", "title": "Page 1"},
                screenshot_path="/tmp/test1.jpg",
                agent_active=False,
            )
            # Different window — should trigger
            assert segmenter.should_start_new_segment(
                window_info={"bundle_id": "com.apple.Terminal", "title": "bash"},
                screenshot_path="/tmp/test2.jpg",
                agent_active=False,
            )

    def test_same_window_no_new_segment(self, segmenter):
        """Same window with no visual change should NOT start a new segment."""
        with patch.object(segmenter, "_load_image_gray", return_value=None):
            info = {"bundle_id": "com.google.Chrome", "title": "Same Page"}
            # First frame
            segmenter.should_start_new_segment(
                window_info=info, screenshot_path="/tmp/t1.jpg", agent_active=False,
            )
            # Same window, no image array = skip SSIM
            assert not segmenter.should_start_new_segment(
                window_info=info, screenshot_path="/tmp/t2.jpg", agent_active=False,
            )

    def test_agent_state_change_triggers_segment(self, segmenter):
        """Agent starting or stopping should create a segment boundary."""
        with patch.object(segmenter, "_load_image_gray", return_value=None):
            info = {"bundle_id": "com.apple.Terminal", "title": "bash"}
            # First frame — human
            segmenter.should_start_new_segment(
                window_info=info, screenshot_path="/tmp/t1.jpg", agent_active=False,
            )
            # Agent started — should trigger
            assert segmenter.should_start_new_segment(
                window_info=info, screenshot_path="/tmp/t2.jpg", agent_active=True,
            )

    def test_state_updates_after_each_call(self, segmenter):
        """Internal state (window, agent) should update after every call (FIX C2)."""
        with patch.object(segmenter, "_load_image_gray", return_value=None):
            segmenter.should_start_new_segment(
                window_info={"bundle_id": "a", "title": "x"},
                screenshot_path="/tmp/t.jpg",
                agent_active=False,
            )
            # State should be updated
            assert segmenter._last_window_info["bundle_id"] == "a"
            assert segmenter._last_agent_state is False

            segmenter.should_start_new_segment(
                window_info={"bundle_id": "b", "title": "y"},
                screenshot_path="/tmp/t.jpg",
                agent_active=True,
            )
            # State updated again
            assert segmenter._last_window_info["bundle_id"] == "b"
            assert segmenter._last_agent_state is True

    def test_segment_id_format(self, segmenter):
        sid = segmenter.start_new_segment(
            window_info={"bundle_id": "test", "title": "test"},
            agent_active=False,
        )
        assert sid.startswith("TS-")
        parts = sid.split("-")
        assert len(parts) == 3
        assert len(parts[1]) == 8   # YYYYMMDD
        assert len(parts[2]) == 4   # 0001

    def test_sequential_ids(self, segmenter):
        s1 = segmenter.start_new_segment({"bundle_id": "a", "title": "a"}, False)
        s2 = segmenter.start_new_segment({"bundle_id": "b", "title": "b"}, False)
        n1 = int(s1.split("-")[2])
        n2 = int(s2.split("-")[2])
        assert n2 == n1 + 1

    def test_counter_survives_load(self, segmenter):
        """Loading a counter from DB should set the right starting point (FIX M1)."""
        segmenter.load_last_segment_counter(max_counter_today=42)
        sid = segmenter.start_new_segment({"bundle_id": "a", "title": "a"}, False)
        assert sid.endswith("-0043")


class TestSanitizeId:
    def test_spaces_replaced(self):
        assert " " not in sanitize_id("Product Development - Frontend")

    def test_max_length(self):
        long = "a" * 300
        assert len(sanitize_id(long)) <= 200

    def test_special_chars_replaced(self):
        result = sanitize_id("hourly-2026-04-02-15-Product Dev/Test")
        assert "/" not in result
        assert all(c.isalnum() or c in "_-" for c in result)

    def test_no_double_underscores(self):
        result = sanitize_id("a   b   c")
        assert "__" not in result
