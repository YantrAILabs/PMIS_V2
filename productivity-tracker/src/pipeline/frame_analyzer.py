"""
Frame analyzer — sends batches of screenshots to ChatGPT Vision for extraction.
Extracts visible text, application, task, and initial worker classification.
"""

import asyncio
import base64
import logging
from pathlib import Path

from openai import AsyncOpenAI

from src.pipeline.prompts import FRAME_BATCH_PROMPT

logger = logging.getLogger("tracker.frame_analyzer")


class FrameAnalyzer:
    """Analyzes screenshot frames using ChatGPT Vision API."""

    def __init__(self, config: dict):
        self.model = config["chatgpt"]["vision_model"]
        self.detail = config["chatgpt"]["vision_detail"]
        self.max_tokens = config["chatgpt"]["max_tokens"]
        self.client = AsyncOpenAI()

    async def analyze_batch(self, frames: list[dict]) -> list[dict]:
        """
        Analyze a batch of frames (3-5) in a single API call.
        
        Args:
            frames: list of frame dicts with 'path', 'timestamp', 'frame_number'
        
        Returns:
            list of extraction dicts: {text, app, task, worker}
        """
        if not frames:
            return []

        # Build multi-image message
        content = [{"type": "text", "text": FRAME_BATCH_PROMPT.format(
            count=len(frames)
        )}]

        for i, frame in enumerate(frames):
            image_b64 = self._encode_image(frame["path"])
            if image_b64:
                content.append({
                    "type": "text",
                    "text": f"--- Frame {i + 1} (captured at {frame['timestamp'].strftime('%H:%M:%S')}) ---",
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": self.detail,
                    },
                })

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            import json
            result = json.loads(result_text)

            # Expecting {"frames": [{text, app, task, worker}, ...]}
            frame_results = result.get("frames", [])

            # Pad if fewer results than frames
            while len(frame_results) < len(frames):
                frame_results.append({
                    "text": "",
                    "app": "unknown",
                    "task": "unknown",
                    "worker": "human",
                })

            return frame_results

        except Exception as e:
            logger.error(f"ChatGPT Vision batch analysis failed: {e}")
            # Return empty results for each frame
            return [{"text": "", "app": "unknown", "task": "unknown", "worker": "human"}] * len(frames)

    async def analyze_single(self, frame: dict) -> dict:
        """Analyze a single frame. Wrapper around analyze_batch."""
        results = await self.analyze_batch([frame])
        return results[0]

    def _encode_image(self, path: str) -> str | None:
        """Read image file and return base64 string."""
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {path}: {e}")
            return None
