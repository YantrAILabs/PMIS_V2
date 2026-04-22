"""
feedback.py — One-bit feedback signal and extended feedback metrics.

The core insight: after each turn, the NEXT turn tells us whether the
session engine's schema was helpful. If the user continues in the schema
direction, the schema was right. If they diverge, it was wrong or irrelevant.

This module computes three levels of feedback:
1. One-bit: confirmed (1) or rejected (0)
2. Continuous: how strongly confirmed/rejected (-1 to +1)
3. Extended: additional signals like topic continuation, depth progression,
   and retrieval utilization.

The feedback is computed RETROACTIVELY — turn t's feedback comes from
turn t+1's embedding. The harness handles the timing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class FeedbackResult:
    """Complete feedback for one turn, computed retroactively."""
    turn_number: int

    # Core one-bit
    bit: Optional[int]                  # 1=confirmed, 0=rejected, None=N/A
    continuous: Optional[float]         # schema_sim - global_sim

    # Extended signals
    topic_continued: Optional[bool]     # next turn same broad topic?
    depth_progressed: Optional[bool]    # went deeper (narrowing)?
    query_refined: Optional[bool]       # next turn is refinement of this?
    schema_abandoned: Optional[bool]    # clear topic switch?

    # Retrieval utilization proxy
    retrieval_overlap: Optional[float]  # how much next turn's retrieval
                                        # overlaps with schema-boosted results

    # Confidence calibration
    confidence_at_prediction: float     # how confident was the engine?
    was_correct: Optional[bool]         # confidence > 0.5 AND bit == 1,
                                        # or confidence < 0.5 AND bit == 0


class FeedbackComputer:
    """
    Computes retroactive feedback signals.

    Usage:
        fc = FeedbackComputer()
        # After turn t+1 arrives:
        feedback = fc.compute(
            turn_number=t,
            schema_embedding=accumulated_embedding_at_t,
            global_centroid=centroid_at_t,
            next_turn_embedding=embedding_t_plus_1,
            convergence_at_t=convergence_value,
            confidence_at_t=best_tree_confidence,
            depth_at_t=depth_t,
            depth_at_t_plus_1=depth_t_plus_1,
            retrieved_at_t=retrieved_memories_t,
            retrieved_at_t_plus_1=retrieved_memories_t_plus_1,
        )
    """

    def __init__(
        self,
        topic_sim_threshold: float = 0.5,
        refinement_sim_threshold: float = 0.7,
        abandon_sim_threshold: float = 0.3,
    ):
        self.topic_sim_thresh = topic_sim_threshold
        self.refinement_sim_thresh = refinement_sim_threshold
        self.abandon_sim_thresh = abandon_sim_threshold

    def compute(
        self,
        turn_number: int,
        schema_embedding: Optional[np.ndarray],
        global_centroid: Optional[np.ndarray],
        next_turn_embedding: np.ndarray,
        convergence_at_t: float,
        confidence_at_t: float,
        depth_at_t: float = 0.0,
        depth_at_t_plus_1: float = 0.0,
        prev_turn_embedding: Optional[np.ndarray] = None,
        retrieved_ids_at_t: Optional[List[str]] = None,
        retrieved_ids_at_t_plus_1: Optional[List[str]] = None,
    ) -> FeedbackResult:
        """Compute full feedback for turn t given turn t+1 data."""

        # ── Core one-bit ──
        if schema_embedding is None or global_centroid is None:
            bit = None
            continuous = None
        elif convergence_at_t < 0.3:
            # Engine wasn't confident enough to have a schema prediction
            bit = None
            continuous = None
        else:
            schema_sim = cosine_sim(next_turn_embedding, schema_embedding)
            global_sim = cosine_sim(next_turn_embedding, global_centroid)
            continuous = round(schema_sim - global_sim, 4)
            bit = 1 if continuous > 0 else 0

        # ── Topic continuation ──
        if prev_turn_embedding is not None:
            turn_sim = cosine_sim(next_turn_embedding, prev_turn_embedding)
            topic_continued = turn_sim > self.topic_sim_thresh
            query_refined = turn_sim > self.refinement_sim_thresh
            schema_abandoned = turn_sim < self.abandon_sim_thresh
        else:
            topic_continued = None
            query_refined = None
            schema_abandoned = None

        # ── Depth progression ──
        depth_progressed = depth_at_t_plus_1 > depth_at_t + 0.05

        # ── Retrieval overlap ──
        if retrieved_ids_at_t and retrieved_ids_at_t_plus_1:
            set_t = set(retrieved_ids_at_t[:5])
            set_t1 = set(retrieved_ids_at_t_plus_1[:5])
            if set_t:
                retrieval_overlap = len(set_t & set_t1) / len(set_t)
            else:
                retrieval_overlap = None
        else:
            retrieval_overlap = None

        # ── Confidence calibration ──
        if bit is not None:
            # Was the engine's confidence level appropriate?
            predicted_confirm = confidence_at_t > 0.5
            was_correct = (predicted_confirm and bit == 1) or (
                not predicted_confirm and bit == 0
            )
        else:
            was_correct = None

        return FeedbackResult(
            turn_number=turn_number,
            bit=bit,
            continuous=continuous,
            topic_continued=topic_continued,
            depth_progressed=depth_progressed,
            query_refined=query_refined,
            schema_abandoned=schema_abandoned,
            retrieval_overlap=retrieval_overlap,
            confidence_at_prediction=round(confidence_at_t, 4),
            was_correct=was_correct,
        )


class FeedbackAccumulator:
    """
    Aggregates feedback across a full conversation to produce
    per-conversation summary statistics.
    """

    def __init__(self):
        self.results: List[FeedbackResult] = []

    def add(self, result: FeedbackResult) -> None:
        self.results.append(result)

    def summary(self) -> Dict:
        """Aggregate statistics across all turns with valid feedback."""
        valid = [r for r in self.results if r.bit is not None]
        if not valid:
            return {
                "total_turns": len(self.results),
                "feedback_turns": 0,
                "confirmation_rate": None,
                "mean_continuous": None,
                "calibration_accuracy": None,
            }

        bits = [r.bit for r in valid]
        continuous = [r.continuous for r in valid if r.continuous is not None]
        calibration = [r.was_correct for r in valid if r.was_correct is not None]

        # Confidence-weighted confirmation rate
        weighted_bits = []
        for r in valid:
            weight = r.confidence_at_prediction
            weighted_bits.append(r.bit * weight)

        return {
            "total_turns": len(self.results),
            "feedback_turns": len(valid),
            "confirmation_rate": round(sum(bits) / len(bits), 4),
            "mean_continuous": round(float(np.mean(continuous)), 4) if continuous else None,
            "std_continuous": round(float(np.std(continuous)), 4) if continuous else None,
            "calibration_accuracy": (
                round(sum(calibration) / len(calibration), 4)
                if calibration else None
            ),
            "weighted_confirmation": (
                round(sum(weighted_bits) / sum(
                    r.confidence_at_prediction for r in valid
                ), 4)
                if valid else None
            ),
            "topic_continuation_rate": self._rate(
                [r.topic_continued for r in valid]
            ),
            "schema_abandon_rate": self._rate(
                [r.schema_abandoned for r in valid]
            ),
            "depth_progression_rate": self._rate(
                [r.depth_progressed for r in valid]
            ),
        }

    @staticmethod
    def _rate(bools: List[Optional[bool]]) -> Optional[float]:
        valid = [b for b in bools if b is not None]
        if not valid:
            return None
        return round(sum(valid) / len(valid), 4)
