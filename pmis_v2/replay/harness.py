"""
harness.py — Replay Harness for PMIS V2 Session Tree Engine.

This is the main entry point. It hooks into the PMIS pipeline as a
passive observer, runs the session engine simulation on every turn,
computes feedback signals, and logs everything to JSONL.

Two modes of operation:
  1. LIVE MODE: Called from orchestrator.py on every turn during normal
     Claude Desktop usage. Observes real data in real-time.
  2. REPLAY MODE: Fed historical conversation data for offline analysis.

Integration: Add to orchestrator.py after retrieval, before composition.
See INTEGRATION.md for wiring instructions.
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict
import numpy as np

from .session_sim import (
    SimulatedSessionEngine,
    SessionHyperparams,
    SimTurnResult,
)
from .feedback import FeedbackComputer, FeedbackAccumulator, FeedbackResult

logger = logging.getLogger("pmis.replay")


class ReplayHarness:
    """
    Shadow-mode session tree engine with full logging and feedback.

    Usage (Live Mode):
        harness = ReplayHarness(log_dir="data/replay_logs")

        # In orchestrator.py, after retrieval:
        harness.observe_turn(
            conversation_id="conv_abc",
            turn_number=3,
            user_message="What pricing emails did I send?",
            turn_embedding=embedding,
            gamma_global=0.65,
            surprise_global=0.32,
            unbiased_retrieved=[
                {"id": "m1", "sc_id": "sc_01", "sc_name": "B2B Outreach",
                 "ctx_id": "ctx_05", "ctx_name": "Pricing",
                 "anc_id": "anc_12", "anc_name": "Email to JLL",
                 "score": 0.82, "embedding": np.array([...])},
                ...
            ],
        )

        # At end of conversation or periodically:
        summary = harness.get_conversation_summary("conv_abc")

    Usage (Replay Mode):
        harness = ReplayHarness(log_dir="data/replay_logs")
        for turn in historical_turns:
            harness.observe_turn(**turn)
        report = harness.generate_report()
    """

    def __init__(
        self,
        log_dir: str = "data/replay_logs",
        hp: Optional[SessionHyperparams] = None,
        enable_extended_feedback: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.hp = hp or SessionHyperparams()
        self.enable_extended_feedback = enable_extended_feedback

        # Per-conversation engines
        self._engines: Dict[str, SimulatedSessionEngine] = {}
        self._feedback_accumulators: Dict[str, FeedbackAccumulator] = {}
        self._feedback_computer = FeedbackComputer()

        # Per-conversation state for extended feedback
        self._prev_turn_data: Dict[str, Dict] = {}

        # Global stats
        self._total_turns = 0
        self._total_conversations = 0
        self._start_time = time.time()

        # Log file handles
        self._log_files: Dict[str, Path] = {}

        logger.info(
            f"Replay harness initialized. Log dir: {self.log_dir}"
        )

    def observe_turn(
        self,
        conversation_id: str,
        turn_number: int,
        user_message: str,
        turn_embedding: np.ndarray,
        gamma_global: float,
        surprise_global: float,
        unbiased_retrieved: List[Dict],
    ) -> SimTurnResult:
        """
        Observe one turn from the live pipeline. This is the main
        integration point — call this from orchestrator.py.

        Returns SimTurnResult for inspection (optional).
        """
        # Get or create engine for this conversation
        if conversation_id not in self._engines:
            self._engines[conversation_id] = SimulatedSessionEngine(self.hp)
            self._feedback_accumulators[conversation_id] = FeedbackAccumulator()
            self._total_conversations += 1
            logger.info(f"New conversation: {conversation_id}")

        engine = self._engines[conversation_id]
        fb_acc = self._feedback_accumulators[conversation_id]

        # ── Run session simulation ──
        result = engine.observe_turn(
            turn_embedding=turn_embedding,
            gamma_global=gamma_global,
            surprise_global=surprise_global,
            unbiased_retrieved=unbiased_retrieved,
            conversation_id=conversation_id,
        )

        # ── Compute extended feedback for PREVIOUS turn ──
        if self.enable_extended_feedback and conversation_id in self._prev_turn_data:
            prev = self._prev_turn_data[conversation_id]
            extended_fb = self._feedback_computer.compute(
                turn_number=prev["turn_number"],
                schema_embedding=prev.get("schema_embedding"),
                global_centroid=prev.get("global_centroid"),
                next_turn_embedding=turn_embedding,
                convergence_at_t=prev["convergence"],
                confidence_at_t=prev["confidence"],
                depth_at_t=prev["depth"],
                depth_at_t_plus_1=result.depth,
                prev_turn_embedding=prev.get("turn_embedding"),
                retrieved_ids_at_t=prev.get("retrieved_ids"),
                retrieved_ids_at_t_plus_1=[
                    m.get("id") for m in unbiased_retrieved
                ],
            )
            fb_acc.add(extended_fb)

            # Write extended feedback to log
            self._log_extended_feedback(conversation_id, extended_fb)

        # ── Store state for next turn's feedback ──
        best_tree = None
        if engine.trees:
            best_tree = max(engine.trees, key=lambda t: t.confidence)

        self._prev_turn_data[conversation_id] = {
            "turn_number": turn_number,
            "turn_embedding": turn_embedding.copy(),
            "schema_embedding": (
                best_tree.accumulated_embedding.copy()
                if best_tree and best_tree.accumulated_embedding is not None
                else None
            ),
            "global_centroid": engine._compute_global_centroid(),
            "convergence": result.convergence,
            "confidence": best_tree.confidence if best_tree else 0.0,
            "depth": result.depth,
            "retrieved_ids": [m.get("id") for m in unbiased_retrieved],
        }

        # ── Log turn ──
        self._log_turn(conversation_id, result, user_message)
        self._total_turns += 1

        return result

    def close_conversation(self, conversation_id: str) -> Optional[Dict]:
        """
        Called when a conversation ends. Produces final summary.
        Cleans up engine state but keeps logs.
        """
        if conversation_id not in self._engines:
            return None

        engine = self._engines[conversation_id]
        fb_acc = self._feedback_accumulators.get(conversation_id)

        summary = {
            "conversation_id": conversation_id,
            "total_turns": engine.turn_count,
            "final_convergence": (
                engine.convergence_history[-1]
                if engine.convergence_history else None
            ),
            "ever_converged": engine.is_converged or any(
                c > self.hp.theta_converged
                for c in engine.convergence_history
            ),
            "convergence_turn": next(
                (
                    i + 1
                    for i, c in enumerate(engine.convergence_history)
                    if c > self.hp.theta_converged
                ),
                None,
            ),
            "num_divergences": sum(
                1 for d in engine.divergence_history
                if d > self.hp.theta_diverge
            ),
            "max_frustration_stage": max(
                (r.frustration_stage for r in engine.results_history),
                default=0,
            ),
            "turns_hippocampus_led": sum(
                1 for r in engine.results_history if r.blend_weight < 0.3
            ),
            "turns_shared": sum(
                1 for r in engine.results_history
                if 0.3 <= r.blend_weight <= 0.6
            ),
            "turns_pfc_led": sum(
                1 for r in engine.results_history if r.blend_weight > 0.6
            ),
            "rerank_would_change_top": sum(
                1 for r in engine.results_history if r.top_memory_boosted
            ),
            "state_snapshot": engine.get_state_snapshot(),
            "feedback": fb_acc.summary() if fb_acc else None,
        }

        # Log summary
        self._log_summary(conversation_id, summary)

        # Cleanup
        del self._engines[conversation_id]
        if conversation_id in self._prev_turn_data:
            del self._prev_turn_data[conversation_id]

        return summary

    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict]:
        """Get summary without closing. For mid-conversation inspection."""
        if conversation_id not in self._engines:
            return None
        engine = self._engines[conversation_id]
        fb_acc = self._feedback_accumulators.get(conversation_id)

        return {
            "conversation_id": conversation_id,
            "turns_so_far": engine.turn_count,
            "is_converged": engine.is_converged,
            "current_convergence": (
                engine.convergence_history[-1]
                if engine.convergence_history else 0.0
            ),
            "blend_weight": (
                engine.results_history[-1].blend_weight
                if engine.results_history else 0.0
            ),
            "best_schema": (
                engine.results_history[-1].best_tree_sc
                if engine.results_history else None
            ),
            "num_candidates": len(engine.trees),
            "frustration_stage": engine.frustration.stage,
            "feedback_so_far": fb_acc.summary() if fb_acc else None,
        }

    def get_harness_status(self) -> Dict:
        """Global harness status. Expose this via MCP tool."""
        uptime = time.time() - self._start_time
        active = {
            cid: {
                "turns": eng.turn_count,
                "converged": eng.is_converged,
                "blend": (
                    eng.results_history[-1].blend_weight
                    if eng.results_history else 0.0
                ),
            }
            for cid, eng in self._engines.items()
        }

        return {
            "status": "running",
            "uptime_seconds": round(uptime, 1),
            "uptime_hours": round(uptime / 3600, 2),
            "total_turns_observed": self._total_turns,
            "total_conversations": self._total_conversations,
            "active_conversations": len(self._engines),
            "active_details": active,
            "log_dir": str(self.log_dir),
            "hyperparams": asdict(self.hp),
        }

    def generate_report(self) -> Dict:
        """
        Aggregate report across ALL conversations observed.
        Call after a full day of data collection.
        """
        all_summaries = []
        for cid in list(self._engines.keys()):
            summary = self.close_conversation(cid)
            if summary:
                all_summaries.append(summary)

        if not all_summaries:
            return {"error": "No conversations to report on."}

        total_turns = sum(s["total_turns"] for s in all_summaries)
        converged = [s for s in all_summaries if s["ever_converged"]]
        short = [s for s in all_summaries if s["total_turns"] <= 5]
        long_ = [s for s in all_summaries if s["total_turns"] > 10]

        # Aggregate feedback
        all_fb = [
            s["feedback"] for s in all_summaries
            if s.get("feedback") and s["feedback"].get("feedback_turns", 0) > 0
        ]

        report = {
            "collection_period": {
                "start": self._start_time,
                "end": time.time(),
                "duration_hours": round(
                    (time.time() - self._start_time) / 3600, 2
                ),
            },
            "conversations": {
                "total": len(all_summaries),
                "converged": len(converged),
                "convergence_rate": round(
                    len(converged) / len(all_summaries), 4
                ),
                "short_conversations_pct": round(
                    len(short) / len(all_summaries), 4
                ),
                "long_conversations_pct": round(
                    len(long_) / len(all_summaries), 4
                ),
            },
            "turns": {
                "total": total_turns,
                "mean_per_conversation": round(
                    total_turns / len(all_summaries), 1
                ),
                "median_per_conversation": float(np.median(
                    [s["total_turns"] for s in all_summaries]
                )),
            },
            "convergence": {
                "mean_convergence_turn": (
                    round(np.mean([
                        s["convergence_turn"]
                        for s in converged
                        if s["convergence_turn"] is not None
                    ]), 1)
                    if converged else None
                ),
                "pfc_led_turns_pct": round(
                    sum(s["turns_pfc_led"] for s in all_summaries)
                    / max(total_turns, 1),
                    4,
                ),
                "hippocampus_led_turns_pct": round(
                    sum(s["turns_hippocampus_led"] for s in all_summaries)
                    / max(total_turns, 1),
                    4,
                ),
            },
            "reranking_impact": {
                "turns_where_top_changed": sum(
                    s["rerank_would_change_top"] for s in all_summaries
                ),
                "pct_turns_top_changed": round(
                    sum(s["rerank_would_change_top"] for s in all_summaries)
                    / max(total_turns, 1),
                    4,
                ),
            },
            "frustration": {
                "conversations_with_frustration": sum(
                    1 for s in all_summaries if s["max_frustration_stage"] > 0
                ),
                "max_stage_reached": max(
                    (s["max_frustration_stage"] for s in all_summaries),
                    default=0,
                ),
            },
            "feedback_aggregate": self._aggregate_feedback(all_fb),
            # The critical question from the critique
            "short_conversation_analysis": {
                "count": len(short),
                "ever_converged": sum(
                    1 for s in short if s["ever_converged"]
                ),
                "convergence_rate": round(
                    sum(1 for s in short if s["ever_converged"])
                    / max(len(short), 1),
                    4,
                ),
                "mean_max_blend_weight": round(
                    np.mean([
                        max(
                            (r.blend_weight for r in
                             self._engines.get(s["conversation_id"],
                                SimulatedSessionEngine()).results_history),
                            default=0.0,
                        )
                        for s in short
                    ]),
                    4,
                ) if short else None,
                "verdict": (
                    "FATAL: Session engine provides no value for most conversations"
                    if len(short) > len(all_summaries) * 0.6
                    and sum(1 for s in short if s["ever_converged"]) < len(short) * 0.2
                    else "OK: Short conversations are minority or converge fast enough"
                ),
            },
            "per_conversation": all_summaries,
        }

        # Write report
        report_path = self.log_dir / "daily_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report written to {report_path}")

        return report

    def _aggregate_feedback(self, feedback_summaries: List[Dict]) -> Dict:
        """Aggregate feedback across all conversations."""
        if not feedback_summaries:
            return {"no_feedback_data": True}

        return {
            "total_feedback_turns": sum(
                f.get("feedback_turns", 0) for f in feedback_summaries
            ),
            "overall_confirmation_rate": round(
                np.mean([
                    f["confirmation_rate"]
                    for f in feedback_summaries
                    if f.get("confirmation_rate") is not None
                ]),
                4,
            ) if any(f.get("confirmation_rate") for f in feedback_summaries) else None,
            "overall_calibration_accuracy": round(
                np.mean([
                    f["calibration_accuracy"]
                    for f in feedback_summaries
                    if f.get("calibration_accuracy") is not None
                ]),
                4,
            ) if any(f.get("calibration_accuracy") for f in feedback_summaries) else None,
            "overall_schema_abandon_rate": round(
                np.mean([
                    f["schema_abandon_rate"]
                    for f in feedback_summaries
                    if f.get("schema_abandon_rate") is not None
                ]),
                4,
            ) if any(f.get("schema_abandon_rate") for f in feedback_summaries) else None,
        }

    # ─── Logging ────────────────────────────────────────────────────────

    def _get_log_path(self, conversation_id: str) -> Path:
        if conversation_id not in self._log_files:
            ts = time.strftime("%Y%m%d_%H%M%S")
            safe_id = conversation_id[:20].replace("/", "_")
            path = self.log_dir / f"turns_{safe_id}_{ts}.jsonl"
            self._log_files[conversation_id] = path
        return self._log_files[conversation_id]

    def _log_turn(
        self,
        conversation_id: str,
        result: SimTurnResult,
        user_message: str,
    ) -> None:
        path = self._get_log_path(conversation_id)
        entry = {
            "type": "turn",
            "conversation_id": conversation_id,
            "user_message_preview": user_message[:100],
            **asdict(result),
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _log_extended_feedback(
        self,
        conversation_id: str,
        fb: FeedbackResult,
    ) -> None:
        path = self._get_log_path(conversation_id)
        entry = {
            "type": "feedback",
            "conversation_id": conversation_id,
            **asdict(fb),
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _log_summary(
        self, conversation_id: str, summary: Dict
    ) -> None:
        path = self.log_dir / "summaries.jsonl"
        entry = {"type": "summary", **summary}
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
