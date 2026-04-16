"""
Frame analyzer — sends batches of screenshots to local Ollama Vision model.
Extracts visible text, application, task per frame.

Uses qwen2.5vl:3b (local, free, private) instead of GPT-4o-mini.
"""

import asyncio
import base64
import json
import logging
from pathlib import Path

import httpx

from src.pipeline.prompts import FRAME_BATCH_PROMPT

logger = logging.getLogger("tracker.frame_analyzer")


class FrameAnalyzer:
    """Analyzes screenshot frames using local Ollama Vision model."""

    def __init__(self, config: dict):
        ollama_config = config.get("ollama", {})
        self.model = ollama_config.get("vision_model", "qwen2.5vl:3b")
        self.fallback_model = ollama_config.get("vision_fallback", "qwen2.5vl:7b")
        self.base_url = ollama_config.get("base_url", "http://localhost:11434")
        self.timeout = ollama_config.get("timeout", 60)

    async def analyze_batch(self, frames: list[dict]) -> list[dict]:
        """
        Analyze a batch of frames (3-5) via local Ollama vision model.

        Args:
            frames: list of frame dicts with 'path', 'timestamp', 'frame_number'

        Returns:
            list of extraction dicts: {text, app, task}
        """
        if not frames:
            return []

        # Encode all images
        images_b64 = []
        frame_labels = []
        for i, frame in enumerate(frames):
            img = self._encode_image(frame["path"])
            if img:
                images_b64.append(img)
                ts = frame["timestamp"].strftime("%H:%M:%S") if hasattr(frame["timestamp"], "strftime") else str(frame["timestamp"])
                frame_labels.append(f"Frame {i + 1} (captured at {ts})")

        if not images_b64:
            return [{"text": "", "app": "unknown", "task": "unknown"}] * len(frames)

        prompt = FRAME_BATCH_PROMPT.format(count=len(images_b64))

        # Try primary model, fall back if needed
        for model in [self.model, self.fallback_model]:
            result = await self._call_ollama(model, prompt, images_b64)
            if result is not None:
                return self._parse_result(result, len(frames))

        # All models failed
        logger.error("All Ollama vision models failed for frame batch")
        return [{"text": "", "app": "unknown", "task": "unknown"}] * len(frames)

    async def _call_ollama(self, model: str, prompt: str, images: list[str]) -> str | None:
        """Call Ollama vision API with images."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "images": images,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 500,
                        },
                    },
                )

                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    logger.warning(f"Ollama {model} returned {response.status_code}")
                    return None

        except Exception as e:
            logger.warning(f"Ollama {model} call failed: {e}")
            return None

    def _parse_result(self, text: str, expected_count: int) -> list[dict]:
        """Parse JSON response from Ollama, with fallback for malformed output."""
        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            # Find first { and last }
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

            result = json.loads(text)
            frame_results = result.get("frames", [])

            # Pad if fewer results
            while len(frame_results) < expected_count:
                frame_results.append({"text": "", "app": "unknown", "task": "unknown"})

            return frame_results[:expected_count]

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse Ollama response as JSON: {e}")
            # Best effort: use the raw text as a single summary
            return [{"text": text[:200], "app": "unknown", "task": text[:100]}] * expected_count

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
