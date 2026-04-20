"""
analyzer.py — Post-hoc analysis of replay harness data.

Run after a day of data collection to answer the critical questions
from the architecture critique:

1. SHORT CONVERSATION PROBLEM: What % of conversations are ≤5 turns?
   Does the session engine ever converge in time to help?

2. FEEDBACK LOOP: Is the schema direction usually confirmed or rejected?
   Is the engine's confidence calibrated?

3. RERANKING IMPACT: How often would schema boost actually change
   the top retrieval result? Is the boost strong enough?

4. FRUSTRATION FALSE POSITIVES: Does frustration detection fire on
   legitimate drilling or only actual frustration?

5. HYPERPARAMETER SENSITIVITY: Which parameters are near sigmoid
   inflection points (dangerous zones)?

Usage:
    python -m replay.analyzer data/replay_logs/
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from dataclasses import dataclass


@dataclass
class ConversationAnalysis:
    conversation_id: str
    total_turns: int
    turns: List[Dict]
    feedbacks: List[Dict]
    summary: Optional[Dict]


def load_logs(log_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load all JSONL log files from directory."""
    turns = []
    feedbacks = []
    summaries = []
    log_path = Path(log_dir)

    for f in log_path.glob("turns_*.jsonl"):
        with open(f) as fh:
            for line in fh:
                entry = json.loads(line.strip())
                if entry["type"] == "turn":
                    turns.append(entry)
                elif entry["type"] == "feedback":
                    feedbacks.append(entry)

    summary_file = log_path / "summaries.jsonl"
    if summary_file.exists():
        with open(summary_file) as fh:
            for line in fh:
                summaries.append(json.loads(line.strip()))

    return turns, feedbacks, summaries


def group_by_conversation(
    turns: List[Dict], feedbacks: List[Dict], summaries: List[Dict]
) -> Dict[str, ConversationAnalysis]:
    """Group data by conversation_id."""
    convos = defaultdict(lambda: {"turns": [], "feedbacks": [], "summary": None})

    for t in turns:
        cid = t["conversation_id"]
        convos[cid]["turns"].append(t)

    for f in feedbacks:
        cid = f["conversation_id"]
        convos[cid]["feedbacks"].append(f)

    for s in summaries:
        cid = s.get("conversation_id", "")
        if cid in convos:
            convos[cid]["summary"] = s

    result = {}
    for cid, data in convos.items():
        data["turns"].sort(key=lambda x: x.get("turn_number", 0))
        data["feedbacks"].sort(key=lambda x: x.get("turn_number", 0))
        result[cid] = ConversationAnalysis(
            conversation_id=cid,
            total_turns=len(data["turns"]),
            turns=data["turns"],
            feedbacks=data["feedbacks"],
            summary=data["summary"],
        )
    return result


# ─── Diagnostic 1: Short Conversation Problem ───────────────────────────

def diagnose_short_conversations(
    conversations: Dict[str, ConversationAnalysis]
) -> Dict:
    """
    THE critical question: does the architecture pay off for your
    actual conversation length distribution?
    """
    lengths = [c.total_turns for c in conversations.values()]
    if not lengths:
        return {"error": "No data"}

    bins = {
        "1-3 turns": [c for c in conversations.values() if c.total_turns <= 3],
        "4-5 turns": [c for c in conversations.values() if 4 <= c.total_turns <= 5],
        "6-10 turns": [c for c in conversations.values() if 6 <= c.total_turns <= 10],
        "11-20 turns": [c for c in conversations.values() if 11 <= c.total_turns <= 20],
        "21+ turns": [c for c in conversations.values() if c.total_turns > 20],
    }

    # For each bin, check: did the engine converge? How many turns of
    # PFC-led retrieval did they get?
    analysis = {}
    for label, convos in bins.items():
        if not convos:
            analysis[label] = {"count": 0}
            continue

        converged = sum(
            1 for c in convos
            if any(t.get("is_converged") for t in c.turns)
        )

        # Turns where blend_weight > 0.6 (PFC actually leading)
        pfc_turns = sum(
            sum(1 for t in c.turns if t.get("blend_weight", 0) > 0.6)
            for c in convos
        )
        total_turns = sum(c.total_turns for c in convos)

        # Mean turn at which convergence first happened
        convergence_turns = []
        for c in convos:
            for t in c.turns:
                if t.get("is_converged"):
                    convergence_turns.append(t["turn_number"])
                    break

        analysis[label] = {
            "count": len(convos),
            "pct_of_total": round(len(convos) / len(conversations), 4),
            "converged": converged,
            "convergence_rate": round(converged / len(convos), 4),
            "pfc_led_turns": pfc_turns,
            "pfc_led_pct": round(pfc_turns / max(total_turns, 1), 4),
            "mean_convergence_turn": (
                round(np.mean(convergence_turns), 1)
                if convergence_turns else None
            ),
        }

    # Verdict
    short_pct = (
        (len(bins["1-3 turns"]) + len(bins["4-5 turns"]))
        / len(conversations)
    )
    short_converged = sum(
        analysis[k]["converged"]
        for k in ["1-3 turns", "4-5 turns"]
    )
    short_total = len(bins["1-3 turns"]) + len(bins["4-5 turns"])

    verdict = "UNKNOWN"
    if short_pct > 0.6:
        if short_total > 0 and short_converged / short_total < 0.2:
            verdict = (
                "FATAL: >60% of conversations are ≤5 turns and "
                f"only {short_converged}/{short_total} converge. "
                "Session engine is dead weight for most usage."
            )
        else:
            verdict = (
                f"CAUTION: >60% short conversations but "
                f"{short_converged}/{short_total} converge. "
                "Consider fast-path bypass for 1-3 turn conversations."
            )
    else:
        verdict = (
            f"OK: Only {short_pct:.0%} of conversations are ≤5 turns. "
            "Architecture targets the right distribution."
        )

    return {
        "distribution": analysis,
        "summary": {
            "total_conversations": len(conversations),
            "median_length": float(np.median(lengths)),
            "mean_length": round(np.mean(lengths), 1),
            "short_pct": round(short_pct, 4),
        },
        "verdict": verdict,
    }


# ─── Diagnostic 2: Feedback Analysis ────────────────────────────────────

def diagnose_feedback(
    conversations: Dict[str, ConversationAnalysis]
) -> Dict:
    """
    Does the schema direction match what users actually do next?
    Is confidence calibrated?
    """
    all_fb = []
    for c in conversations.values():
        all_fb.extend(c.feedbacks)

    valid = [f for f in all_fb if f.get("bit") is not None]
    if not valid:
        return {"error": "No feedback data (need ≥2 turns with convergence)"}

    confirmed = sum(1 for f in valid if f["bit"] == 1)
    rejected = sum(1 for f in valid if f["bit"] == 0)

    # Calibration: bucket by confidence, check confirmation rate
    calibration_buckets = defaultdict(list)
    for f in valid:
        conf = f.get("confidence_at_prediction", 0)
        bucket = round(conf * 5) / 5  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        calibration_buckets[bucket].append(f["bit"])

    calibration = {}
    for bucket in sorted(calibration_buckets.keys()):
        bits = calibration_buckets[bucket]
        calibration[f"conf_{bucket:.1f}"] = {
            "count": len(bits),
            "confirmation_rate": round(sum(bits) / len(bits), 4),
            "expected_rate": round(bucket, 2),
            "gap": round(sum(bits) / len(bits) - bucket, 4),
        }

    # Continuous feedback distribution
    continuous = [
        f["continuous"] for f in valid
        if f.get("continuous") is not None
    ]

    return {
        "total_feedback_turns": len(valid),
        "confirmation_rate": round(confirmed / len(valid), 4),
        "rejection_rate": round(rejected / len(valid), 4),
        "mean_continuous": round(np.mean(continuous), 4) if continuous else None,
        "std_continuous": round(np.std(continuous), 4) if continuous else None,
        "calibration": calibration,
        "calibration_verdict": (
            "WELL_CALIBRATED"
            if all(abs(v["gap"]) < 0.15 for v in calibration.values())
            else "MISCALIBRATED"
        ),
        "schema_abandon_rate": round(
            sum(1 for f in valid if f.get("schema_abandoned")) / len(valid),
            4,
        ),
    }


# ─── Diagnostic 3: Reranking Impact ─────────────────────────────────────

def diagnose_reranking(
    conversations: Dict[str, ConversationAnalysis]
) -> Dict:
    """
    How often does schema boost actually change retrieval order?
    Is the boost strong enough to matter?
    """
    all_turns = []
    for c in conversations.values():
        all_turns.extend(c.turns)

    converged_turns = [
        t for t in all_turns if t.get("is_converged")
    ]

    if not converged_turns:
        return {"error": "No converged turns to analyze reranking"}

    top_changed = sum(1 for t in converged_turns if t.get("top_memory_boosted"))
    any_changed = sum(1 for t in converged_turns if t.get("rerank_changes", 0) > 0)
    max_boosts = [t.get("max_schema_boost", 0) for t in converged_turns]

    return {
        "converged_turns_analyzed": len(converged_turns),
        "top_result_changed": top_changed,
        "top_result_changed_pct": round(
            top_changed / len(converged_turns), 4
        ),
        "any_position_changed": any_changed,
        "any_position_changed_pct": round(
            any_changed / len(converged_turns), 4
        ),
        "mean_max_boost": round(np.mean(max_boosts), 4),
        "max_boost_p95": round(np.percentile(max_boosts, 95), 4),
        "verdict": (
            "TOO_WEAK: Schema boost rarely changes top result"
            if top_changed / max(len(converged_turns), 1) < 0.1
            else "EFFECTIVE: Schema boost meaningfully reshapes retrieval"
            if top_changed / max(len(converged_turns), 1) > 0.3
            else "MARGINAL: Schema boost has moderate impact"
        ),
    }


# ─── Diagnostic 4: Frustration False Positives ──────────────────────────

def diagnose_frustration(
    conversations: Dict[str, ConversationAnalysis]
) -> Dict:
    """
    Does frustration detection fire on legitimate drilling?
    Cross-reference with feedback: if frustration fires AND feedback
    confirms schema, it's a false positive.
    """
    all_turns = []
    for c in conversations.values():
        all_turns.extend(c.turns)

    frustrated_turns = [
        t for t in all_turns if t.get("frustration_stage", 0) > 0
    ]

    if not frustrated_turns:
        return {
            "total_frustration_events": 0,
            "verdict": "NO_DATA: Frustration never fired",
        }

    # Cross-reference with feedback
    # Build lookup: (conversation_id, turn_number) -> feedback
    fb_lookup = {}
    for c in conversations.values():
        for f in c.feedbacks:
            key = (c.conversation_id, f.get("turn_number"))
            fb_lookup[key] = f

    false_positives = 0
    true_positives = 0
    unknown = 0

    for t in frustrated_turns:
        key = (t["conversation_id"], t["turn_number"])
        fb = fb_lookup.get(key)
        if fb and fb.get("bit") is not None:
            if fb["bit"] == 1 and fb.get("topic_continued"):
                # User confirmed schema AND continued topic = false positive
                false_positives += 1
            else:
                true_positives += 1
        else:
            unknown += 1

    return {
        "total_frustration_events": len(frustrated_turns),
        "false_positives": false_positives,
        "true_positives": true_positives,
        "unknown": unknown,
        "false_positive_rate": round(
            false_positives / max(false_positives + true_positives, 1), 4
        ),
        "stage_distribution": {
            "stage_1": sum(1 for t in frustrated_turns if t["frustration_stage"] == 1),
            "stage_2": sum(1 for t in frustrated_turns if t["frustration_stage"] == 2),
            "stage_3": sum(1 for t in frustrated_turns if t["frustration_stage"] == 3),
        },
        "verdict": (
            "HIGH_FALSE_POSITIVE_RATE: Frustration detection needs tuning"
            if false_positives > true_positives
            else "ACCEPTABLE: Most frustration detections are genuine"
        ),
    }


# ─── Diagnostic 5: Hyperparameter Sensitivity ───────────────────────────

def diagnose_hyperparameter_sensitivity(
    conversations: Dict[str, ConversationAnalysis]
) -> Dict:
    """
    Check if key sigmoid inputs are clustering near inflection points
    (where small parameter changes would flip outputs).
    """
    all_turns = []
    for c in conversations.values():
        all_turns.extend(c.turns)

    if not all_turns:
        return {"error": "No data"}

    # Convergence values near the inflection (0.4-0.6 = danger zone)
    convergences = [t.get("convergence", 0) for t in all_turns]
    near_inflection = [c for c in convergences if 0.4 <= c <= 0.6]

    # Confidence gaps near the threshold
    gaps = [t.get("confidence_gap", 0) for t in all_turns]
    near_gap_thresh = [g for g in gaps if 0.2 <= g <= 0.3]  # near b_c=0.25

    # Blend weights — are we spending most time in the "shared" zone?
    blends = [t.get("blend_weight", 0) for t in all_turns]
    blend_zones = {
        "hippocampus_led (<0.3)": sum(1 for b in blends if b < 0.3),
        "shared (0.3-0.6)": sum(1 for b in blends if 0.3 <= b <= 0.6),
        "pfc_led (>0.6)": sum(1 for b in blends if b > 0.6),
    }

    return {
        "convergence_near_inflection_pct": round(
            len(near_inflection) / max(len(convergences), 1), 4
        ),
        "gap_near_threshold_pct": round(
            len(near_gap_thresh) / max(len(gaps), 1), 4
        ),
        "blend_weight_distribution": {
            k: {
                "count": v,
                "pct": round(v / max(len(blends), 1), 4),
            }
            for k, v in blend_zones.items()
        },
        "gamma_session_stats": {
            "mean": round(np.mean([t.get("gamma_session", 0.5) for t in all_turns]), 4),
            "std": round(np.std([t.get("gamma_session", 0.5) for t in all_turns]), 4),
        },
        "verdict": (
            "UNSTABLE: >30% of convergence values near sigmoid inflection"
            if len(near_inflection) / max(len(convergences), 1) > 0.3
            else "STABLE: Most values are in confident regions of the sigmoid"
        ),
    }


# ─── Full Diagnostic Report ─────────────────────────────────────────────

def run_full_diagnostic(log_dir: str) -> Dict:
    """Run all diagnostics and produce a complete report."""
    turns, feedbacks, summaries = load_logs(log_dir)
    conversations = group_by_conversation(turns, feedbacks, summaries)

    report = {
        "data_loaded": {
            "turns": len(turns),
            "feedbacks": len(feedbacks),
            "summaries": len(summaries),
            "conversations": len(conversations),
        },
        "1_short_conversation_problem": diagnose_short_conversations(conversations),
        "2_feedback_analysis": diagnose_feedback(conversations),
        "3_reranking_impact": diagnose_reranking(conversations),
        "4_frustration_false_positives": diagnose_frustration(conversations),
        "5_hyperparameter_sensitivity": diagnose_hyperparameter_sensitivity(conversations),
    }

    # Overall verdict
    verdicts = []
    for key in ["1_short_conversation_problem", "2_feedback_analysis",
                 "3_reranking_impact", "4_frustration_false_positives",
                 "5_hyperparameter_sensitivity"]:
        v = report[key].get("verdict", "")
        if v.startswith("FATAL"):
            verdicts.append(("FATAL", key, v))
        elif "FALSE_POSITIVE" in v or "MISCALIBRATED" in v or "TOO_WEAK" in v:
            verdicts.append(("WARNING", key, v))
        else:
            verdicts.append(("OK", key, v))

    report["overall_verdict"] = {
        "fatals": [v for v in verdicts if v[0] == "FATAL"],
        "warnings": [v for v in verdicts if v[0] == "WARNING"],
        "ok": [v for v in verdicts if v[0] == "OK"],
        "recommendation": (
            "DO NOT PROCEED: Fatal issues found. Address before implementation."
            if any(v[0] == "FATAL" for v in verdicts)
            else "PROCEED WITH CAUTION: Warnings found. Tune parameters."
            if any(v[0] == "WARNING" for v in verdicts)
            else "GREEN LIGHT: All diagnostics pass."
        ),
    }

    # Save
    output_path = Path(log_dir) / "diagnostic_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Diagnostic report written to {output_path}")

    return report


# ─── CLI ─────────────────────────────────────────────────────────────────

def print_report(report: Dict, indent: int = 0) -> None:
    """Pretty-print report to terminal."""
    prefix = "  " * indent
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_report(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, tuple):
                    print(f"{prefix}  [{item[0]}] {item[1]}: {item[2]}")
                else:
                    print(f"{prefix}  - {item}")
        else:
            print(f"{prefix}{key}: {value}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m replay.analyzer <log_dir>")
        sys.exit(1)

    log_dir = sys.argv[1]
    if not os.path.isdir(log_dir):
        print(f"Error: {log_dir} is not a directory")
        sys.exit(1)

    report = run_full_diagnostic(log_dir)
    print("\n" + "=" * 60)
    print("PMIS V2 SESSION TREE ENGINE — DIAGNOSTIC REPORT")
    print("=" * 60 + "\n")
    print_report(report)
