"""
System Prompt Composer for PMIS v2.

Generates the per-turn system prompt injection that tells Claude:
  - Current mode (Associative / Balanced / Predictive)
  - Gamma value and what it means
  - Retrieved memories ranked by relevance
  - Behavioral guidance (how to respond, what to ask)
  - Storage guidance (what to remember from this turn)
"""

from typing import List, Dict, Any, Optional
from core.gamma import GammaResult
from core.surprise import SurpriseResult


def compose_system_prompt(
    gamma_result: GammaResult,
    surprise_result: SurpriseResult,
    retrieved_memories: List[Dict[str, Any]],
    predictive_memories: List[Dict[str, Any]] = None,
    epistemic_questions: List[Dict[str, Any]] = None,
    active_tree: Optional[str] = None,
    session_turn_count: int = 0,
    is_stale: bool = False,
) -> str:
    """
    Generate the per-turn system prompt injection.
    This gets prepended to Claude's context for this turn.
    """
    memory_block = _format_memories(retrieved_memories)
    predictive_block = _format_predictive(predictive_memories)
    epistemic_block = _format_epistemic(epistemic_questions)
    guidance = _get_behavioral_guidance(gamma_result, surprise_result, is_stale)
    staleness_note = "\n⚠ STALENESS DETECTED: Proactively introduce cross-domain connections." if is_stale else ""

    prompt = f"""<memory_context>
CONVERSATION MODE: {gamma_result.mode_label} (γ = {gamma_result.gamma:.2f})
SURPRISE: {surprise_result.effective_surprise:.2f} (raw={surprise_result.raw_surprise:.2f}, precision={surprise_result.cluster_precision:.2f})
NEAREST CONTEXT: {surprise_result.nearest_context_name or "NONE"} (id={surprise_result.nearest_context_id or "—"})
ACTIVE TREE: {active_tree or "auto"}
TURN: {session_turn_count}{staleness_note}

RETRIEVED MEMORIES:
{memory_block}
{predictive_block}{epistemic_block}
BEHAVIORAL GUIDANCE:
{guidance}

STORAGE: {gamma_result.storage_instruction}

NOTE: Memory items are numbered [1], [2], etc. for THIS context only — these are ephemeral display indices, not persistent IDs. There are no reference codes like "PM-25", "MEM-3", or similar in this system. Never invent or cite such identifiers.
</memory_context>"""
    return prompt.strip()


def _format_memories(memories: List[Dict[str, Any]]) -> str:
    """Format retrieved memories for prompt injection."""
    if not memories:
        return "  (No relevant memories found. You are in uncharted territory.)"

    lines = []
    for i, mem in enumerate(memories[:8], 1):
        level = mem.get("level", "?")
        score = mem.get("final_score", 0)
        content = mem.get("content", "")[:250]
        source = mem.get("_source", "")
        era = mem.get("era", "")

        era_tag = f" [{era}]" if era else ""
        lines.append(f"  [{i}] [{level}|{source}]{era_tag} (score={score:.2f}) {content}")

    return "\n".join(lines)


def _format_predictive(memories: Optional[List[Dict[str, Any]]]) -> str:
    """Format predictive (what-comes-next) memories."""
    if not memories:
        return ""

    lines = ["\nPREDICTIVE (what typically follows):"]
    for i, mem in enumerate(memories[:3], 1):
        content = mem.get("content", "")[:150]
        depth = mem.get("_prediction_depth", "?")
        freq = mem.get("_prediction_frequency", "")
        freq_tag = f" (freq={freq})" if freq else ""
        lines.append(f"  →{i} [depth={depth}]{freq_tag} {content}")

    return "\n".join(lines) + "\n"


def _format_epistemic(questions: Optional[List[Dict[str, Any]]]) -> str:
    """Format epistemic questions for prompt injection."""
    if not questions:
        return ""

    lines = ["\nEPISTEMIC (ask ONE of these to disambiguate):"]
    for i, q in enumerate(questions[:3], 1):
        gain = q.get("information_gain", 0)
        question_text = q.get("question", "")
        lines.append(f"  ?{i} [info_gain={gain:.2f}] {question_text}")

    lines.append(
        "  → Pick the question with highest info_gain. Weave it naturally "
        "into your response — do not list it mechanically."
    )
    return "\n".join(lines) + "\n"


def _get_behavioral_guidance(
    gamma_result: GammaResult,
    surprise_result: SurpriseResult,
    is_stale: bool,
) -> str:
    """Generate specific behavioral instructions based on mode."""

    base = gamma_result.confidence_instruction

    # Add surprise-specific nuance
    if surprise_result.is_orphan_territory:
        base += (
            "\n  NOTE: This topic has NO matching Context in memory. "
            "Treat this as completely new territory. "
            "Ask questions that help determine which existing domain this might connect to, "
            "or confirm it's genuinely a new area."
        )
    elif surprise_result.effective_surprise > 0.6:
        base += (
            "\n  NOTE: High surprise against a known Context — something unexpected. "
            "Investigate whether this contradicts existing knowledge or extends it."
        )

    if is_stale:
        base += (
            "\n  STALENESS: The same Context has been active with low surprise for many turns. "
            "Proactively introduce connections to adjacent Contexts or unvisited Anchors. "
            "Suggest new angles the user hasn't explored."
        )

    return base


def compose_status_report(
    gamma_result: GammaResult,
    surprise_result: SurpriseResult,
    session_stats: Dict[str, Any],
) -> str:
    """
    Generate a human-readable status report for /memory status command.
    """
    return f"""Memory System Status
─────────────────────
Mode: {gamma_result.mode_label}
Gamma (γ): {gamma_result.gamma:.2f}
  → Narrow retrieval weight: {gamma_result.retrieval_narrow_weight:.0%}
  → Broad retrieval weight:  {gamma_result.retrieval_broad_weight:.0%}

Surprise:
  Raw: {surprise_result.raw_surprise:.2f}
  Cluster precision: {surprise_result.cluster_precision:.2f}
  Effective: {surprise_result.effective_surprise:.2f}
  Orphan territory: {"Yes" if surprise_result.is_orphan_territory else "No"}

Nearest Context: {surprise_result.nearest_context_name or "None"}

Session:
  Turns: {session_stats.get("turn_count", 0)}
  Avg gamma: {session_stats.get("avg_gamma", 0.5):.2f}
  Stored nodes: {session_stats.get("stored_count", 0)}
"""
