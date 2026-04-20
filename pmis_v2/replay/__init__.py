"""
PMIS V2 Replay Harness — Shadow-mode session tree engine with feedback.

Modules:
    session_sim  — Pure math simulation of session engine
    feedback     — One-bit and extended feedback signals
    harness      — Main entry point, logging, MCP integration
    analyzer     — Post-hoc diagnostic analysis
"""

from .harness import ReplayHarness
from .session_sim import SimulatedSessionEngine, SessionHyperparams
from .feedback import FeedbackComputer, FeedbackAccumulator

__all__ = [
    "ReplayHarness",
    "SimulatedSessionEngine",
    "SessionHyperparams",
    "FeedbackComputer",
    "FeedbackAccumulator",
]
