"""
Replay Harness for ProMe.

Replays historical sessions through ablated scoring pipelines to measure
the actual contribution of each component (semantic, hierarchy, temporal, precision).

Outputs:
  - Per-component contribution analysis
  - Rank displacement when each component is removed
  - Score variance attribution
  - Daily summary log

Usage:
    python3 pmis_v2/replay_harness.py                # Full analysis
    python3 pmis_v2/replay_harness.py --recent 20    # Last 20 turns only
    python3 pmis_v2/replay_harness.py --daily-log    # Append to daily log
"""

import sqlite3
import json
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "memory.db")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "replay")


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class RetrievedItem:
    """One retrieved memory from a historical turn."""
    turn_id: int
    node_id: str
    actual_rank: int
    final_score: float
    semantic_score: float
    hierarchy_score: float
    temporal_score: float
    precision_score: float
    source: str
    node_level: str


@dataclass
class TurnContext:
    """Metadata about a historical turn."""
    turn_id: int
    conversation_id: str
    gamma: float
    effective_surprise: float
    mode: str
    nearest_context: str
    active_tree: str
    timestamp: str
    items: List[RetrievedItem] = field(default_factory=list)


@dataclass
class AblationResult:
    """Result of scoring with one component removed or isolated."""
    turn_id: int
    ablation_name: str  # e.g. "no_hierarchy", "semantic_only"
    original_ranking: List[str]  # node_ids in original order
    ablated_ranking: List[str]   # node_ids in ablated order
    kendall_tau: float           # rank correlation
    top3_preserved: bool         # are top-3 identical?
    top1_preserved: bool         # is #1 the same?
    rank_displacements: List[int]  # per-item displacement


# ─────────────────────────────────────────────
# Scoring configurations for ablation
# ─────────────────────────────────────────────

SCORING_CONFIGS = {
    "full": {
        "semantic": 0.40, "hierarchy": 0.30,
        "temporal": 0.15, "precision": 0.15
    },
    "semantic_only": {
        "semantic": 1.00, "hierarchy": 0.00,
        "temporal": 0.00, "precision": 0.00
    },
    "no_hierarchy": {
        "semantic": 0.57, "hierarchy": 0.00,
        "temporal": 0.215, "precision": 0.215
    },
    "no_temporal": {
        "semantic": 0.47, "hierarchy": 0.35,
        "temporal": 0.00, "precision": 0.18
    },
    "no_precision": {
        "semantic": 0.47, "hierarchy": 0.35,
        "temporal": 0.18, "precision": 0.00
    },
    "semantic_hierarchy_only": {
        "semantic": 0.57, "hierarchy": 0.43,
        "temporal": 0.00, "precision": 0.00
    },
    "semantic_temporal_only": {
        "semantic": 0.73, "hierarchy": 0.00,
        "temporal": 0.27, "precision": 0.00
    },
    "simple_metadata": {
        # Proposed simplified model: semantic + flat metadata
        "semantic": 0.60, "hierarchy": 0.20,
        "temporal": 0.10, "precision": 0.10
    },
}


# ─────────────────────────────────────────────
# Core replay logic
# ─────────────────────────────────────────────

def load_historical_turns(db_path: str, limit: Optional[int] = None) -> List[TurnContext]:
    """Load all historical turns with their retrieved memories."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get turns that have retrieval data
    query = """
        SELECT DISTINCT ct.id as turn_id, ct.conversation_id, ct.gamma,
               ct.effective_surprise, ct.mode, ct.nearest_context_name,
               ct.active_tree, ct.timestamp
        FROM conversation_turns ct
        JOIN turn_retrieved_memories trm ON trm.turn_id = ct.id
        ORDER BY ct.id DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    turn_rows = cursor.fetchall()

    turns = []
    for tr in turn_rows:
        turn = TurnContext(
            turn_id=tr["turn_id"],
            conversation_id=tr["conversation_id"],
            gamma=tr["gamma"] or 0.5,
            effective_surprise=tr["effective_surprise"] or 0.3,
            mode=tr["mode"] or "BALANCED",
            nearest_context=tr["nearest_context_name"] or "",
            active_tree=tr["active_tree"] or "",
            timestamp=tr["timestamp"] or "",
        )

        # Load retrieved items for this turn
        cursor.execute("""
            SELECT memory_node_id, rank, final_score,
                   semantic_score, hierarchy_score, temporal_score,
                   precision_score, source, node_level
            FROM turn_retrieved_memories
            WHERE turn_id = ?
            ORDER BY rank ASC
        """, (tr["turn_id"],))

        for item_row in cursor.fetchall():
            turn.items.append(RetrievedItem(
                turn_id=tr["turn_id"],
                node_id=item_row["memory_node_id"],
                actual_rank=item_row["rank"],
                final_score=item_row["final_score"],
                semantic_score=item_row["semantic_score"],
                hierarchy_score=item_row["hierarchy_score"],
                temporal_score=item_row["temporal_score"],
                precision_score=item_row["precision_score"],
                source=item_row["source"] or "unknown",
                node_level=item_row["node_level"] or "ANC",
            ))

        if turn.items:
            turns.append(turn)

    conn.close()
    return turns


def compute_ablated_score(item: RetrievedItem, weights: Dict[str, float]) -> float:
    """Recompute score with different component weights."""
    return (
        weights["semantic"] * item.semantic_score +
        weights["hierarchy"] * item.hierarchy_score +
        weights["temporal"] * item.temporal_score +
        weights["precision"] * item.precision_score
    )


def kendall_tau(ranking_a: List[str], ranking_b: List[str]) -> float:
    """
    Compute Kendall tau rank correlation between two rankings.
    Returns value in [-1, 1] where 1 = identical, -1 = reversed.
    """
    n = len(ranking_a)
    if n <= 1:
        return 1.0

    # Create position maps
    pos_a = {item: i for i, item in enumerate(ranking_a)}
    pos_b = {item: i for i, item in enumerate(ranking_b)}

    # Only compare items in both rankings
    common = [item for item in ranking_a if item in pos_b]
    n = len(common)
    if n <= 1:
        return 1.0

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_order = pos_a[common[i]] - pos_a[common[j]]
            b_order = pos_b[common[i]] - pos_b[common[j]]
            if a_order * b_order > 0:
                concordant += 1
            elif a_order * b_order < 0:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def run_ablation(turn: TurnContext, config_name: str, weights: Dict[str, float]) -> AblationResult:
    """Run one ablation on one turn."""
    original_ranking = [item.node_id for item in turn.items]

    # Recompute scores with ablated weights
    rescored = [(item.node_id, compute_ablated_score(item, weights)) for item in turn.items]
    rescored.sort(key=lambda x: x[1], reverse=True)
    ablated_ranking = [node_id for node_id, _ in rescored]

    # Compute rank displacement for each item
    ablated_pos = {node_id: i for i, node_id in enumerate(ablated_ranking)}
    displacements = [abs(i - ablated_pos[node_id]) for i, node_id in enumerate(original_ranking)]

    tau = kendall_tau(original_ranking, ablated_ranking)

    return AblationResult(
        turn_id=turn.turn_id,
        ablation_name=config_name,
        original_ranking=original_ranking,
        ablated_ranking=ablated_ranking,
        kendall_tau=tau,
        top3_preserved=(original_ranking[:3] == ablated_ranking[:3]),
        top1_preserved=(original_ranking[0] == ablated_ranking[0]),
        rank_displacements=displacements,
    )


# ─────────────────────────────────────────────
# Analysis & reporting
# ─────────────────────────────────────────────

@dataclass
class ComponentReport:
    """Aggregated analysis of one scoring component's contribution."""
    component: str
    avg_score: float
    score_std: float
    score_range: Tuple[float, float]
    discriminative_power: float  # std/mean — how much it varies across candidates
    avg_kendall_tau_without: float  # how much ranking changes when removed
    top1_preservation_rate: float   # how often #1 stays #1 without this component
    top3_preservation_rate: float   # how often top-3 stays same
    avg_rank_displacement: float    # avg positions moved when component removed
    max_rank_displacement: float    # worst case displacement
    verdict: str


def analyze_component(
    component_name: str,
    turns: List[TurnContext],
    ablation_results: List[AblationResult],
) -> ComponentReport:
    """Deep analysis of one component."""

    # Extract scores for this component
    all_scores = []
    per_turn_stds = []

    for turn in turns:
        scores = [getattr(item, f"{component_name}_score") for item in turn.items]
        all_scores.extend(scores)
        if len(scores) > 1:
            per_turn_stds.append(np.std(scores))

    # Filter ablation results for this component's removal
    ablation_key = f"no_{component_name}"
    relevant_ablations = [r for r in ablation_results if r.ablation_name == ablation_key]

    if not relevant_ablations:
        # Use semantic_only as reference for semantic
        if component_name == "semantic":
            relevant_ablations = [r for r in ablation_results if r.ablation_name == "semantic_only"]

    avg_tau = np.mean([r.kendall_tau for r in relevant_ablations]) if relevant_ablations else 1.0
    top1_rate = np.mean([r.top1_preserved for r in relevant_ablations]) if relevant_ablations else 1.0
    top3_rate = np.mean([r.top3_preserved for r in relevant_ablations]) if relevant_ablations else 1.0
    all_displacements = []
    for r in relevant_ablations:
        all_displacements.extend(r.rank_displacements)
    avg_disp = np.mean(all_displacements) if all_displacements else 0.0
    max_disp = max(all_displacements) if all_displacements else 0

    # Discriminative power: avg within-turn std / overall mean
    avg_score = np.mean(all_scores) if all_scores else 0
    score_std = np.mean(per_turn_stds) if per_turn_stds else 0
    disc_power = score_std / avg_score if avg_score > 0 else 0

    # Verdict
    if avg_tau > 0.95 and top1_rate > 0.95:
        verdict = "NEGLIGIBLE — removing barely changes ranking"
    elif avg_tau > 0.85 and top1_rate > 0.80:
        verdict = "LOW IMPACT — minor reranking, top results stable"
    elif avg_tau > 0.60:
        verdict = "MODERATE IMPACT — affects mid-range rankings"
    elif avg_disp > 2.0:
        verdict = "HIGH IMPACT — significantly changes result order"
    else:
        verdict = "VARIABLE — impact depends on query type"

    return ComponentReport(
        component=component_name,
        avg_score=float(avg_score),
        score_std=float(score_std),
        score_range=(float(min(all_scores)) if all_scores else 0, float(max(all_scores)) if all_scores else 0),
        discriminative_power=float(disc_power),
        avg_kendall_tau_without=float(avg_tau),
        top1_preservation_rate=float(top1_rate),
        top3_preservation_rate=float(top3_rate),
        avg_rank_displacement=float(avg_disp),
        max_rank_displacement=float(max_disp),
        verdict=verdict,
    )


def compute_score_contribution(turns: List[TurnContext]) -> Dict[str, Dict]:
    """
    For each turn, compute what % of the final score variance is explained
    by each component. Like a poor man's ANOVA.
    """
    contributions = defaultdict(list)

    for turn in turns:
        if len(turn.items) < 3:
            continue

        # Get component score arrays
        sem = np.array([i.semantic_score for i in turn.items])
        hier = np.array([i.hierarchy_score for i in turn.items])
        temp = np.array([i.temporal_score for i in turn.items])
        prec = np.array([i.precision_score for i in turn.items])
        final = np.array([i.final_score for i in turn.items])

        # Weighted contribution to final score variance
        weights = {"semantic": 0.40, "hierarchy": 0.30, "temporal": 0.15, "precision": 0.15}
        total_var = np.var(final) if np.var(final) > 1e-10 else 1e-10

        for name, scores, w in [
            ("semantic", sem, 0.40),
            ("hierarchy", hier, 0.30),
            ("temporal", temp, 0.15),
            ("precision", prec, 0.15),
        ]:
            # Component's weighted variance contribution
            component_var = (w ** 2) * np.var(scores)
            pct = component_var / total_var
            contributions[name].append(min(pct, 1.0))

        # Also compute correlation between each component and final rank
        final_ranks = np.argsort(-final)  # descending
        for name, scores in [
            ("semantic", sem), ("hierarchy", hier),
            ("temporal", temp), ("precision", prec),
        ]:
            if np.std(scores) > 1e-10:
                component_ranks = np.argsort(-scores)
                tau = kendall_tau(
                    [str(x) for x in final_ranks],
                    [str(x) for x in component_ranks]
                )
                contributions[f"{name}_rank_corr"].append(tau)

    # Aggregate
    result = {}
    for key, values in contributions.items():
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


def surprise_analysis(turns: List[TurnContext]) -> Dict:
    """Analyze surprise/gamma/mode distributions."""
    gammas = [t.gamma for t in turns]
    surprises = [t.effective_surprise for t in turns]
    modes = defaultdict(int)
    for t in turns:
        modes[t.mode] += 1

    # How often does surprise vary significantly between turns?
    surprise_changes = [abs(surprises[i] - surprises[i-1]) for i in range(1, len(surprises))]

    return {
        "gamma": {
            "mean": float(np.mean(gammas)),
            "std": float(np.std(gammas)),
            "range": (float(np.min(gammas)), float(np.max(gammas))),
        },
        "surprise": {
            "mean": float(np.mean(surprises)),
            "std": float(np.std(surprises)),
            "range": (float(np.min(surprises)), float(np.max(surprises))),
        },
        "mode_distribution": dict(modes),
        "surprise_volatility": {
            "mean_change": float(np.mean(surprise_changes)) if surprise_changes else 0,
            "max_change": float(np.max(surprise_changes)) if surprise_changes else 0,
        },
        "gamma_stuck": float(np.std(gammas) < 0.05),  # 1.0 if gamma barely moves
    }


# ─────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────

def generate_report(
    turns: List[TurnContext],
    component_reports: List[ComponentReport],
    ablation_results: List[AblationResult],
    score_contributions: Dict,
    surprise_stats: Dict,
) -> str:
    """Generate human-readable analysis report."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"ProMe REPLAY HARNESS — COMPONENT ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Turns analyzed: {len(turns)}")
    lines.append(f"Total retrievals analyzed: {sum(len(t.items) for t in turns)}")
    lines.append("=" * 72)

    # Section 1: Component Scorecards
    lines.append("\n## 1. COMPONENT SCORECARDS\n")
    for cr in component_reports:
        lines.append(f"### {cr.component.upper()}")
        lines.append(f"  Avg score:           {cr.avg_score:.3f} (range: {cr.score_range[0]:.3f}–{cr.score_range[1]:.3f})")
        lines.append(f"  Within-turn StdDev:  {cr.score_std:.3f}")
        lines.append(f"  Discriminative power:{cr.discriminative_power:.3f} (higher = more useful for ranking)")
        lines.append(f"  Without this component:")
        lines.append(f"    Kendall tau:       {cr.avg_kendall_tau_without:.3f} (1.0 = no change, 0.0 = random)")
        lines.append(f"    Top-1 preserved:   {cr.top1_preservation_rate:.1%}")
        lines.append(f"    Top-3 preserved:   {cr.top3_preservation_rate:.1%}")
        lines.append(f"    Avg displacement:  {cr.avg_rank_displacement:.1f} positions")
        lines.append(f"    Max displacement:  {cr.max_rank_displacement:.0f} positions")
        lines.append(f"  ➤ VERDICT: {cr.verdict}")
        lines.append("")

    # Section 2: Score Variance Attribution
    lines.append("\n## 2. SCORE VARIANCE ATTRIBUTION")
    lines.append("  (What % of final score variance does each component explain?)\n")
    for comp in ["semantic", "hierarchy", "temporal", "precision"]:
        if comp in score_contributions:
            c = score_contributions[comp]
            lines.append(f"  {comp:12s}: {c['mean']:5.1%} avg ({c['min']:.1%}–{c['max']:.1%})")
    lines.append("")
    for comp in ["semantic", "hierarchy", "temporal", "precision"]:
        key = f"{comp}_rank_corr"
        if key in score_contributions:
            c = score_contributions[key]
            lines.append(f"  {comp:12s} rank correlation with final: τ={c['mean']:.3f} (std={c['std']:.3f})")

    # Section 3: Ablation Matrix
    lines.append("\n\n## 3. ABLATION MATRIX")
    lines.append(f"  {'Config':<25s} {'Kendall τ':>10s} {'Top1 kept':>10s} {'Top3 kept':>10s} {'Avg disp':>10s}")
    lines.append("  " + "-" * 67)

    config_stats = defaultdict(lambda: {"tau": [], "top1": [], "top3": [], "disp": []})
    for r in ablation_results:
        config_stats[r.ablation_name]["tau"].append(r.kendall_tau)
        config_stats[r.ablation_name]["top1"].append(r.top1_preserved)
        config_stats[r.ablation_name]["top3"].append(r.top3_preserved)
        config_stats[r.ablation_name]["disp"].extend(r.rank_displacements)

    for name in SCORING_CONFIGS:
        if name == "full":
            continue
        s = config_stats[name]
        if not s["tau"]:
            continue
        lines.append(
            f"  {name:<25s} "
            f"{np.mean(s['tau']):>10.3f} "
            f"{np.mean(s['top1']):>9.1%} "
            f"{np.mean(s['top3']):>9.1%} "
            f"{np.mean(s['disp']):>10.1f}"
        )

    # Section 4: Surprise & Gamma Analysis
    lines.append("\n\n## 4. SURPRISE & GAMMA DYNAMICS")
    sg = surprise_stats
    lines.append(f"  Gamma:    mean={sg['gamma']['mean']:.3f}  std={sg['gamma']['std']:.3f}  range={sg['gamma']['range']}")
    lines.append(f"  Surprise: mean={sg['surprise']['mean']:.3f}  std={sg['surprise']['std']:.3f}  range={sg['surprise']['range']}")
    lines.append(f"  Mode distribution: {sg['mode_distribution']}")
    lines.append(f"  Surprise volatility: avg Δ={sg['surprise_volatility']['mean_change']:.3f}, max Δ={sg['surprise_volatility']['max_change']:.3f}")
    if sg["gamma_stuck"]:
        lines.append(f"  ⚠️  GAMMA IS STUCK — std < 0.05, system is not adapting between explore/exploit")

    # Section 5: Recommendations
    lines.append("\n\n## 5. DATA-DRIVEN RECOMMENDATIONS\n")

    for cr in component_reports:
        if "NEGLIGIBLE" in cr.verdict:
            lines.append(f"  🔴 {cr.component.upper()}: Consider removing. "
                         f"Ranking barely changes without it (τ={cr.avg_kendall_tau_without:.3f})")
        elif "LOW IMPACT" in cr.verdict:
            lines.append(f"  🟡 {cr.component.upper()}: Low value. Could simplify to scalar heuristic. "
                         f"(top1 preserved {cr.top1_preservation_rate:.0%} without)")
        elif "HIGH IMPACT" in cr.verdict:
            lines.append(f"  🟢 {cr.component.upper()}: Essential. Removing causes avg {cr.avg_rank_displacement:.1f} "
                         f"position displacement")
        else:
            lines.append(f"  🔵 {cr.component.upper()}: {cr.verdict}")

    # Poincaré specific
    hier_report = next((cr for cr in component_reports if cr.component == "hierarchy"), None)
    if hier_report:
        lines.append(f"\n  📐 POINCARÉ ASSESSMENT:")
        if hier_report.discriminative_power < 0.10:
            lines.append(f"     Hierarchy scores have very low variance (disc_power={hier_report.discriminative_power:.3f}).")
            lines.append(f"     The Poincaré distance isn't discriminating between candidates.")
            lines.append(f"     A simple level+tree_match heuristic would likely perform identically.")
        elif hier_report.avg_kendall_tau_without > 0.90:
            lines.append(f"     Removing hierarchy barely changes ranking (τ={hier_report.avg_kendall_tau_without:.3f}).")
            lines.append(f"     The 30% weight is not earning its complexity cost (RSGD, 32d space, etc.)")
        else:
            lines.append(f"     Hierarchy IS contributing meaningfully (τ={hier_report.avg_kendall_tau_without:.3f}).")
            lines.append(f"     BUT — is the contribution from Poincaré geometry or just from norm bands?")
            lines.append(f"     → Run the 'simple_metadata' ablation to test flat heuristic replacement.")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_full_analysis(limit: Optional[int] = None) -> str:
    """Run complete replay analysis and return report."""

    print(f"Loading historical turns from {DB_PATH}...")
    turns = load_historical_turns(DB_PATH, limit=limit)
    print(f"Loaded {len(turns)} turns with {sum(len(t.items) for t in turns)} total retrievals")

    if not turns:
        return "No historical turns found. Run some sessions first."

    # Run all ablations
    print("Running ablations...")
    all_ablation_results = []
    for config_name, weights in SCORING_CONFIGS.items():
        if config_name == "full":
            continue
        for turn in turns:
            result = run_ablation(turn, config_name, weights)
            all_ablation_results.append(result)
    print(f"  {len(all_ablation_results)} ablation runs complete")

    # Analyze each component
    print("Analyzing components...")
    component_reports = []
    for component in ["semantic", "hierarchy", "temporal", "precision"]:
        report = analyze_component(component, turns, all_ablation_results)
        component_reports.append(report)

    # Score variance attribution
    print("Computing variance attribution...")
    score_contributions = compute_score_contribution(turns)

    # Surprise/gamma analysis
    print("Analyzing surprise/gamma dynamics...")
    surprise_stats = surprise_analysis(turns)

    # Generate report
    report = generate_report(
        turns, component_reports, all_ablation_results,
        score_contributions, surprise_stats
    )

    return report


def save_daily_log(report: str):
    """Save report to daily log file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"replay_{date_str}.txt")
    with open(log_path, "w") as f:
        f.write(report)
    print(f"Daily log saved to {log_path}")
    return log_path


def save_json_metrics(turns, component_reports, ablation_results, score_contributions, surprise_stats):
    """Save machine-readable metrics for trending."""
    os.makedirs(LOG_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    metrics_path = os.path.join(LOG_DIR, f"metrics_{date_str}.json")

    metrics = {
        "date": date_str,
        "turns_analyzed": len(turns),
        "total_retrievals": sum(len(t.items) for t in turns),
        "components": {},
        "score_contributions": score_contributions,
        "surprise_gamma": surprise_stats,
    }

    for cr in component_reports:
        metrics["components"][cr.component] = {
            "avg_score": cr.avg_score,
            "discriminative_power": cr.discriminative_power,
            "kendall_tau_without": cr.avg_kendall_tau_without,
            "top1_preservation": cr.top1_preservation_rate,
            "top3_preservation": cr.top3_preservation_rate,
            "avg_displacement": cr.avg_rank_displacement,
            "verdict": cr.verdict,
        }

    # Ablation summary
    config_stats = defaultdict(lambda: {"tau": [], "top1": [], "top3": []})
    for r in ablation_results:
        config_stats[r.ablation_name]["tau"].append(r.kendall_tau)
        config_stats[r.ablation_name]["top1"].append(r.top1_preserved)
        config_stats[r.ablation_name]["top3"].append(r.top3_preserved)

    metrics["ablations"] = {}
    for name, s in config_stats.items():
        metrics["ablations"][name] = {
            "avg_kendall_tau": float(np.mean(s["tau"])),
            "top1_rate": float(np.mean(s["top1"])),
            "top3_rate": float(np.mean(s["top3"])),
        }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"JSON metrics saved to {metrics_path}")
    return metrics_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ProMe Replay Harness")
    parser.add_argument("--recent", type=int, help="Only analyze last N turns")
    parser.add_argument("--daily-log", action="store_true", help="Save to daily log file")
    parser.add_argument("--json", action="store_true", help="Also save JSON metrics")
    parser.add_argument("--quiet", action="store_true", help="Only output file paths")
    args = parser.parse_args()

    report = run_full_analysis(limit=args.recent)

    if not args.quiet:
        print("\n")
        print(report)

    if args.daily_log or args.json:
        log_path = save_daily_log(report)

        if args.json:
            # Re-run to get structured data (or refactor to share)
            turns = load_historical_turns(DB_PATH, limit=args.recent)
            all_ablation_results = []
            for config_name, weights in SCORING_CONFIGS.items():
                if config_name == "full":
                    continue
                for turn in turns:
                    all_ablation_results.append(run_ablation(turn, config_name, weights))

            component_reports = [
                analyze_component(comp, turns, all_ablation_results)
                for comp in ["semantic", "hierarchy", "temporal", "precision"]
            ]
            score_contributions = compute_score_contribution(turns)
            surprise_stats = surprise_analysis(turns)
            save_json_metrics(turns, component_reports, all_ablation_results,
                              score_contributions, surprise_stats)
