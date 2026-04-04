"""
Context classifier — takes frame-level extractions and produces
the SC/Context/Anchor hierarchy for a completed segment.
"""

import json
import logging

from openai import AsyncOpenAI

from src.pipeline.prompts import SEGMENT_SYNTHESIS_PROMPT

logger = logging.getLogger("tracker.context_classifier")


class ContextClassifier:
    """Classifies work segments using ChatGPT text model."""

    def __init__(self, config: dict):
        self.model = config["chatgpt"]["text_model"]
        self.max_tokens = config["chatgpt"]["max_tokens"]
        self.temperature = config["chatgpt"]["temperature"]
        self.client = AsyncOpenAI()

    async def classify_segment(
        self,
        segment_id: str,
        frame_results: list[dict],
        window_info: dict,
        agent_active: bool,
    ) -> dict:
        """
        Classify a completed segment into SC/Context/Anchor.

        Args:
            segment_id: Target segment ID
            frame_results: List of frame extraction dicts from Context2
            window_info: Window info from the segment
            agent_active: Whether an agent was detected during this segment

        Returns:
            dict with supercontext, context, anchor, detailed_summary, worker, medium
        """
        # Calculate duration from frame results
        duration = len(frame_results) * 10  # Approximate: 10s per frame

        # Build frame JSON summary (truncate if too many)
        frame_summaries = []
        for fr in frame_results[:20]:  # Cap at 20 frames to control token usage
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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            # Validate required fields
            required = ["supercontext", "context", "anchor", "detailed_summary", "worker", "medium"]
            for field in required:
                if field not in result:
                    result[field] = "Unclassified" if field != "worker" else "human"

            return result

        except Exception as e:
            logger.error(f"Segment classification failed: {e}")
            return {
                "supercontext": "Unclassified",
                "context": "Unclassified",
                "anchor": "Unclassified",
                "detailed_summary": f"Classification failed: {e}",
                "worker": "agent" if agent_active else "human",
                "medium": "other",
            }
