"""
Target frame segmenter — groups consecutive screenshots into meaningful segments.
A new segment starts when the work context changes.
"""

import logging
import re
from datetime import datetime, date

import numpy as np
from PIL import Image
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    # Fallback: simple MSE-based similarity if scikit-image not installed
    def ssim(img1, img2, **kwargs):
        """Fallback SSIM approximation using normalized MSE."""
        diff = (img1.astype(float) - img2.astype(float)) / 255.0
        mse = np.mean(diff ** 2)
        return max(0.0, 1.0 - mse * 10)  # rough approximation

logger = logging.getLogger("tracker.segmenter")


def sanitize_id(raw: str, max_len: int = 200) -> str:
    """Sanitize a string for use as a ChromaDB or DB ID."""
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:max_len]


class TargetFrameSegmenter:
    """Groups consecutive frames into target segments based on context changes."""

    def __init__(self, config: dict):
        self.ssim_threshold = config["segmentation"]["ssim_threshold"]
        self.skip_threshold = config["tracking"]["skip_similar_threshold"]
        self.min_segment_secs = config["segmentation"]["min_segment_secs"]

        # State
        self._counter = 0
        self._date = None
        self._current_segment = None
        self._last_window_info = None
        self._last_image_array = None
        self._last_agent_state = None

    def load_last_segment_counter(self, max_counter_today: int = 0):
        """
        Set counter to resume after restart.
        Caller queries db.get_max_segment_number(today) and passes the result.
        """
        today = date.today().strftime("%Y%m%d")
        self._date = today
        self._counter = max_counter_today

    def should_start_new_segment(
        self,
        window_info: dict,
        screenshot_path: str,
        agent_active: bool,
    ) -> bool:
        """
        Determine if a new segment should start.
        Updates internal state (image, window, agent) after every call.

        Triggers:
        1. Window changed (different app or title)
        2. Visual content changed significantly (SSIM < threshold)
        3. Agent state changed (agent started or stopped)
        """
        current_array = self._load_image_gray(screenshot_path)
        needs_new = False

        # First frame ever
        if self._last_window_info is None:
            needs_new = True
        else:
            # Trigger 1: Window change
            if (
                window_info.get("bundle_id") != self._last_window_info.get("bundle_id")
                or window_info.get("title") != self._last_window_info.get("title")
            ):
                logger.debug("New segment: window changed")
                needs_new = True

            # Trigger 2: Agent state boundary
            elif self._last_agent_state is not None and agent_active != self._last_agent_state:
                logger.debug(f"New segment: agent state changed to {agent_active}")
                needs_new = True

            # Trigger 3: Visual diff
            elif self._last_image_array is not None and current_array is not None:
                try:
                    score = ssim(self._last_image_array, current_array)
                    if score < self.ssim_threshold:
                        logger.debug(f"New segment: visual diff (SSIM={score:.2f})")
                        needs_new = True
                except Exception as e:
                    logger.debug(f"SSIM comparison failed: {e}")

        # FIX C2: Always update state after comparison so next call has fresh data
        self._last_window_info = window_info
        self._last_agent_state = agent_active
        if current_array is not None:
            self._last_image_array = current_array

        return needs_new

    def should_skip_frame(self, screenshot_path: str) -> bool:
        """Check if frame is near-identical to previous (skip to save API cost)."""
        try:
            current_array = self._load_image_gray(screenshot_path)
            if self._last_image_array is not None and current_array is not None:
                score = ssim(self._last_image_array, current_array)
                return score > self.skip_threshold
        except Exception:
            pass
        return False

    def start_new_segment(self, window_info: dict, agent_active: bool) -> str:
        """Create a new segment and return its ID."""
        today = date.today().strftime("%Y%m%d")
        if self._date != today:
            self._date = today
            self._counter = 0

        self._counter += 1
        segment_id = f"TS-{today}-{self._counter:04d}"

        self._current_segment = {
            "id": segment_id,
            "start_time": datetime.now(),
            "window_info": window_info,
            "agent_active": agent_active,
        }

        logger.info(f"Started segment {segment_id}: {window_info.get('app_name', '?')}")
        return segment_id

    def get_current_segment(self) -> dict | None:
        return self._current_segment

    def _load_image_gray(self, path: str) -> np.ndarray | None:
        """Load image as grayscale numpy array, resized for fast SSIM."""
        try:
            img = Image.open(path).convert("L").resize((320, 180))
            return np.array(img)
        except Exception:
            return None
