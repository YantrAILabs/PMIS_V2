"""
Gamma (γ) Calculator for PMIS v2.

Computes the exploration-exploitation balance parameter.
γ = 0 → pure exploration (novel territory)
γ = 1 → pure exploitation (familiar territory)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GammaResult:
    gamma: float
    mode_label: str
    retrieval_narrow_weight: float
    retrieval_broad_weight: float
    confidence_instruction: str
    storage_instruction: str


def compute_gamma(
    effective_surprise: float,
    staleness: bool,
    hyperparams: Dict[str, Any],
) -> GammaResult:
    """
    γ = sigmoid(-effective_surprise × temperature + bias - staleness_penalty)
    """
    temperature = hyperparams.get("gamma_temperature", 3.0)
    bias = hyperparams.get("gamma_bias", 0.5)
    staleness_penalty = hyperparams.get("gamma_staleness_penalty", 1.5)

    z = -effective_surprise * temperature + bias
    if staleness:
        z -= staleness_penalty

    gamma = 1.0 / (1.0 + np.exp(-z))
    gamma = float(np.clip(gamma, 0.05, 0.95))

    # Derive mode, instructions, and weights
    if gamma > 0.7:
        mode = "ASSOCIATIVE"
        confidence = (
            "You have strong context. Be specific, reference prior work, "
            "push toward decisions. Minimal hedging."
        )
        storage = (
            "Update existing Anchor weights and recency. "
            "Do NOT create new Anchors unless genuinely new info emerges."
        )
    elif gamma > 0.4:
        mode = "BALANCED"
        confidence = (
            "You have partial context. Ground in what you know but actively "
            "probe the unfamiliar parts. Ask ONE high-value clarifying question."
        )
        storage = (
            "May create a tentative Anchor if new information emerges. "
            "Update existing Anchors if the info refines known territory."
        )
    else:
        mode = "PREDICTIVE"
        confidence = (
            "Novel territory. Explore openly. Surface possible connections "
            "across domains. Ask the question whose answer would MOST change "
            "your understanding."
        )
        storage = (
            "Create an orphan Anchor tagged for consolidation review. "
            "Flag this as a potential new Context seed if it develops."
        )

    return GammaResult(
        gamma=gamma,
        mode_label=mode,
        retrieval_narrow_weight=gamma,
        retrieval_broad_weight=1.0 - gamma,
        confidence_instruction=confidence,
        storage_instruction=storage,
    )
