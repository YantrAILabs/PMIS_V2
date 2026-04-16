"""
Context classifier — takes frame-level extractions and produces
a summary + full text description for a completed segment.

Uses local Ollama qwen2.5:3b (text-only) instead of GPT-4o-mini.
No SC/CTX/ANC classification — memory structure forms at nightly consolidation.
"""

import json
import logging

import httpx

from src.pipeline.prompts import SEGMENT_SYNTHESIS_PROMPT

logger = logging.getLogger("tracker.context_classifier")


class ContextClassifier:
    """Classifies work segments using local Ollama text model."""

    def __init__(self, config: dict):
        ollama_config = config.get("ollama", {})
        self.model = ollama_config.get("text_model", "qwen2.5:3b")
        self.base_url = ollama_config.get("base_url", "http://localhost:11434")
        self.timeout = ollama_config.get("timeout", 60)

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

        # Build frame summary (cap at 20)
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

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 500,
                        },
                    },
                )

                if response.status_code == 200:
                    text = response.json().get("response", "")
                    return self._parse_result(text, agent_active)

        except Exception as e:
            logger.error(f"Segment classification failed: {e}")

        # Fallback: build from frame data directly
        return self._fallback_result(frame_summaries, window_info, agent_active)

    def _parse_result(self, text: str, agent_active: bool) -> dict:
        """Parse JSON from Ollama response."""
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

            # Ensure required fields
            if "detailed_summary" not in result:
                result["detailed_summary"] = result.get("summary", "Activity segment")
            if "full_text" not in result:
                result["full_text"] = result.get("detailed_summary", "")
            if "worker" not in result:
                result["worker"] = "agent" if agent_active else "human"
            if "medium" not in result:
                result["medium"] = "other"

            return result

        except (json.JSONDecodeError, KeyError, TypeError):
            # Use raw text as summary
            return {
                "detailed_summary": text[:200] if text else "Activity segment",
                "full_text": text[:500] if text else "",
                "worker": "agent" if agent_active else "human",
                "medium": "other",
            }

    def _fallback_result(self, frames: list[dict], window_info: dict, agent_active: bool) -> dict:
        """Build result from raw frame data when LLM fails."""
        tasks = [f.get("task", "") for f in frames if f.get("task")]
        window = window_info.get("title", "Unknown")[:50]

        summary = f"Working in {window}"
        if tasks:
            summary = f"Working in {window}: {tasks[0][:80]}"

        full_text = ". ".join(t[:100] for t in tasks[:5]) if tasks else summary

        return {
            "detailed_summary": summary,
            "full_text": full_text,
            "worker": "agent" if agent_active else "human",
            "medium": "other",
        }
