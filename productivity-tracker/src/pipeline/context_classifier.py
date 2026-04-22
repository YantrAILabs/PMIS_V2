"""
Context classifier — takes frame-level extractions and produces
a summary + full text description for a completed segment.

Provider order is configurable via settings.yaml `llm.provider_order`.
Default: OpenAI gpt-4o-mini first, Ollama qwen2.5:3b as fallback.
No SC/CTX/ANC classification — memory structure forms at nightly consolidation.
"""

import json
import logging
import os

import httpx

from src.pipeline.prompts import SEGMENT_SYNTHESIS_PROMPT

logger = logging.getLogger("tracker.context_classifier")


class ContextClassifier:
    """Classifies work segments with OpenAI → Ollama fallback."""

    def __init__(self, config: dict):
        llm_config = config.get("llm", {})
        self.provider_order = llm_config.get("provider_order", ["openai", "ollama"])

        openai_config = config.get("openai", {})
        self.openai_model = openai_config.get("text_model", "gpt-4o-mini")
        self.openai_timeout = openai_config.get("timeout", 45)

        ollama_config = config.get("ollama", {})
        self.ollama_model = ollama_config.get("text_model", "qwen2.5:3b")
        self.ollama_base_url = ollama_config.get("base_url", "http://localhost:11434")
        self.ollama_timeout = ollama_config.get("timeout", 60)

    async def classify_segment(
        self,
        segment_id: str,
        frame_results: list[dict],
        window_info: dict,
        agent_active: bool,
    ) -> dict:
        """
        Produce summary + full text for a completed segment.

        Args:
            segment_id: Target segment ID
            frame_results: List of frame extraction dicts from Context2
            window_info: Window info from the segment
            agent_active: Whether an agent was detected during this segment

        Returns:
            dict with detailed_summary, full_text, worker, medium
        """
        duration = len(frame_results) * 10  # ~10s per frame

        frame_summaries = []
        for fr in frame_results[:20]:
            frame_summaries.append({
                "frame": fr.get("target_frame_number", "?"),
                "task": fr.get("detailed_summary", fr.get("raw_text", "")[:200]),
                "worker": fr.get("worker_type", "human"),
            })

        prompt = SEGMENT_SYNTHESIS_PROMPT.format(
            segment_id=segment_id,
            duration=duration,
            window_name=window_info.get("title", "Unknown"),
            platform=window_info.get("app_name", "Unknown"),
            agent_active="Yes" if agent_active else "No",
            frame_jsons=json.dumps(frame_summaries, indent=2),
        )

        for provider in self.provider_order:
            if provider == "openai":
                text = await self._call_openai(prompt)
            elif provider == "ollama":
                text = await self._call_ollama(prompt)
            else:
                logger.warning(f"Unknown provider {provider!r}, skipping")
                continue

            if text is not None:
                return self._parse_result(text, agent_active)

        logger.error("All text providers failed for segment classification")
        return self._fallback_result(frame_summaries, window_info, agent_active)

    async def _call_openai(self, prompt: str) -> str | None:
        """Call OpenAI chat completion for segment text. Returns None to trigger fallback."""
        if not os.getenv("OPENAI_API_KEY"):
            logger.info("OPENAI_API_KEY not set — skipping OpenAI text")
            return None

        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.warning("openai package not installed — skipping")
            return None

        try:
            client = AsyncOpenAI(timeout=self.openai_timeout)
            response = await client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI text call failed ({self.openai_model}): {e}")
            return None

    async def _call_ollama(self, prompt: str) -> str | None:
        """Call local Ollama text model."""
        try:
            async with httpx.AsyncClient(timeout=self.ollama_timeout) as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 500,
                        },
                    },
                )

                if response.status_code == 200:
                    return response.json().get("response", "")
                logger.warning(f"Ollama {self.ollama_model} returned {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Ollama {self.ollama_model} call failed: {e}")
            return None

    def _parse_result(self, text: str, agent_active: bool) -> dict:
        """Parse JSON from model response."""
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

            if "detailed_summary" not in result:
                result["detailed_summary"] = result.get("summary", "Activity segment")
            if "full_text" not in result:
                result["full_text"] = result.get("detailed_summary", "")
            if "worker" not in result:
                result["worker"] = "agent" if agent_active else "human"
            if "medium" not in result:
                result["medium"] = "other"
            if "short_title" not in result or not result.get("short_title"):
                # Fall back to a truncated detailed_summary so the review UI
                # always has something human-readable to show.
                result["short_title"] = (result["detailed_summary"] or "")[:80]

            return result

        except (json.JSONDecodeError, KeyError, TypeError):
            return {
                "short_title": (text[:80] if text else "Activity"),
                "detailed_summary": text[:200] if text else "Activity segment",
                "full_text": text[:500] if text else "",
                "worker": "agent" if agent_active else "human",
                "medium": "other",
            }

    def _fallback_result(self, frames: list[dict], window_info: dict, agent_active: bool) -> dict:
        """Build result from raw frame data when all LLMs fail."""
        tasks = [f.get("task", "") for f in frames if f.get("task")]
        window = window_info.get("title", "Unknown")[:50]

        summary = f"Working in {window}"
        if tasks:
            summary = f"Working in {window}: {tasks[0][:80]}"

        full_text = ". ".join(t[:100] for t in tasks[:5]) if tasks else summary

        return {
            "short_title": summary[:80],
            "detailed_summary": summary,
            "full_text": full_text,
            "worker": "agent" if agent_active else "human",
            "medium": "other",
        }
