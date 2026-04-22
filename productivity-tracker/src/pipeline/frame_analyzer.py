"""
Frame analyzer — sends batches of screenshots to a vision model.

Provider order is configurable via settings.yaml `llm.provider_order`.
Default: OpenAI gpt-4o-mini first, Ollama qwen2.5vl:3b/7b as fallback.
"""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path

import httpx

from src.pipeline.prompts import FRAME_BATCH_PROMPT

logger = logging.getLogger("tracker.frame_analyzer")


class FrameAnalyzer:
    """Analyzes screenshot frames with OpenAI → Ollama fallback."""

    def __init__(self, config: dict):
        llm_config = config.get("llm", {})
        self.provider_order = llm_config.get("provider_order", ["openai", "ollama"])

        openai_config = config.get("openai", {})
        self.openai_model = openai_config.get("vision_model", "gpt-4o-mini")
        self.openai_timeout = openai_config.get("timeout", 45)
        # "low" = 85 tokens per image flat (server resizes to 512x512).
        # "high" = up to ~765 tokens/image at 1024px (full-res, reads small text).
        # For app+task identification, "low" is plenty and ~5-8x cheaper.
        self.openai_vision_detail = openai_config.get("vision_detail", "low")
        self.openai_max_tokens = openai_config.get("max_tokens_vision", 400)

        ollama_config = config.get("ollama", {})
        self.ollama_vision_enabled = ollama_config.get("vision_enabled", False)
        self.ollama_model = ollama_config.get("vision_model", "qwen2.5vl:3b")
        self.ollama_fallback_model = ollama_config.get("vision_fallback", "qwen2.5vl:7b")
        self.ollama_base_url = ollama_config.get("base_url", "http://localhost:11434")
        self.ollama_timeout = ollama_config.get("timeout", 60)
        self.ollama_keep_alive = ollama_config.get("keep_alive", "0")

    async def analyze_batch(self, frames: list[dict]) -> list[dict]:
        """
        Analyze a batch of frames (3-5) via the configured provider chain.

        Args:
            frames: list of frame dicts with 'path', 'timestamp', 'frame_number'

        Returns:
            list of extraction dicts: {text, app, task}
        """
        if not frames:
            return []

        images_b64 = []
        for frame in frames:
            img = self._encode_image(frame["path"])
            if img:
                images_b64.append(img)

        if not images_b64:
            return [{"text": "", "app": "unknown", "task": "unknown"}] * len(frames)

        prompt = FRAME_BATCH_PROMPT.format(count=len(images_b64))

        for provider in self.provider_order:
            if provider == "openai":
                result = await self._call_openai(prompt, images_b64)
            elif provider == "ollama":
                result = await self._call_ollama_chain(prompt, images_b64)
            else:
                logger.warning(f"Unknown provider {provider!r}, skipping")
                continue

            if result is not None:
                return self._parse_result(result, len(frames))

        logger.error("All vision providers failed for frame batch")
        return [{"text": "", "app": "unknown", "task": "unknown"}] * len(frames)

    async def _call_openai(self, prompt: str, images: list[str]) -> str | None:
        """Call OpenAI vision chat completion. Returns None to trigger fallback."""
        if not os.getenv("OPENAI_API_KEY"):
            logger.info("OPENAI_API_KEY not set — skipping OpenAI vision")
            return None

        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.warning("openai package not installed — skipping")
            return None

        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img}",
                    "detail": self.openai_vision_detail,
                },
            })

        try:
            client = AsyncOpenAI(timeout=self.openai_timeout)
            response = await client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                max_tokens=self.openai_max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI vision call failed ({self.openai_model}): {e}")
            return None

    async def _call_ollama_chain(self, prompt: str, images: list[str]) -> str | None:
        """Try primary Ollama model, then fallback model. Skips entirely when disabled."""
        if not self.ollama_vision_enabled:
            logger.info("Ollama vision disabled (ollama.vision_enabled=false) — skipping")
            return None
        for model in [self.ollama_model, self.ollama_fallback_model]:
            result = await self._call_ollama(model, prompt, images)
            if result is not None:
                return result
        return None

    async def _call_ollama(self, model: str, prompt: str, images: list[str]) -> str | None:
        """Call Ollama vision API with images."""
        try:
            async with httpx.AsyncClient(timeout=self.ollama_timeout) as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "images": images,
                        "stream": False,
                        "keep_alive": self.ollama_keep_alive,
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
        """Parse JSON response, with fallback for malformed output."""
        try:
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

            result = json.loads(text)
            frame_results = result.get("frames", [])

            while len(frame_results) < expected_count:
                frame_results.append({"text": "", "app": "unknown", "task": "unknown"})

            return frame_results[:expected_count]

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse vision response as JSON: {e}")
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
