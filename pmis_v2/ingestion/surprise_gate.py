"""
Surprise-Gated Storage Decisions for PMIS v2.

Given gamma (γ) and surprise for a turn, decides:
  - γ > 0.6 → Update existing Anchor weights (exploit mode)
  - γ < 0.4 → Create tentative orphan Anchor (explore mode)
  - 0.4-0.6 → Create if genuinely new info, skip if repetitive
  - Staleness → Flag Context for cross-pollination
"""

from typing import Optional, Dict, Any
from core.gamma import GammaResult
from core.surprise import SurpriseResult


class StorageDecision:
    CREATE = "create"         # Create new Anchor
    UPDATE = "update"         # Update existing Anchor weights
    SKIP = "skip"             # Don't store anything
    FLAG_STALE = "flag_stale" # Flag Context for review


def decide_storage(
    gamma_result: GammaResult,
    surprise_result: SurpriseResult,
    is_stale: bool,
    content_length: int,
    hyperparams: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Make the per-turn storage decision.

    Returns dict with:
        action: "create" | "update" | "skip" | "flag_stale"
        is_orphan: bool
        is_tentative: bool
        target_context_id: str (for updates)
        reason: str
    """
    gamma = gamma_result.gamma

    # Skip very short messages (greetings, acknowledgments)
    if content_length < 20:
        return {
            "action": StorageDecision.SKIP,
            "reason": "Message too short to store",
        }

    # Staleness override
    if is_stale:
        return {
            "action": StorageDecision.FLAG_STALE,
            "target_context_id": surprise_result.nearest_context_id,
            "reason": "Context is stale — flag for cross-pollination",
        }

    # Exploit mode: update existing
    if gamma > 0.6:
        if surprise_result.effective_surprise < hyperparams.get("surprise_low_threshold", 0.25):
            return {
                "action": StorageDecision.UPDATE,
                "target_context_id": surprise_result.nearest_context_id,
                "reason": "Low surprise, high gamma — update existing Anchor weights",
            }
        else:
            # Even in exploit mode, genuinely new info gets stored
            return {
                "action": StorageDecision.CREATE,
                "is_orphan": False,
                "is_tentative": False,
                "target_context_id": surprise_result.nearest_context_id,
                "reason": "High gamma but significant surprise — create Anchor under existing Context",
            }

    # Explore mode: create orphan
    if gamma < 0.4:
        return {
            "action": StorageDecision.CREATE,
            "is_orphan": surprise_result.is_orphan_territory,
            "is_tentative": True,
            "target_context_id": surprise_result.nearest_context_id if not surprise_result.is_orphan_territory else None,
            "reason": "Low gamma — create tentative orphan Anchor for review",
        }

    # Balanced mode: create if surprise is above threshold
    if surprise_result.effective_surprise > hyperparams.get("surprise_low_threshold", 0.25):
        return {
            "action": StorageDecision.CREATE,
            "is_orphan": surprise_result.is_orphan_territory,
            "is_tentative": False,
            "target_context_id": surprise_result.nearest_context_id,
            "reason": "Balanced mode with notable surprise — create Anchor",
        }
    else:
        return {
            "action": StorageDecision.UPDATE,
            "target_context_id": surprise_result.nearest_context_id,
            "reason": "Balanced mode, low surprise — update existing",
        }
