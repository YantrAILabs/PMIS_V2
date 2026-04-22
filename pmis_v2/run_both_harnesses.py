"""
Unified Replay Runner — Runs BOTH harnesses on historical data.

Harness 1: Retrieval Component Ablation (replay_harness.py)
  → Measures contribution of semantic, hierarchy, temporal, precision

Harness 2: Session Tree Engine Simulation (replay/)
  → Measures convergence, schema prediction, reranking impact, feedback

Both run on the SAME historical turns from conversation_turns + turn_retrieved_memories.

Usage:
    python3 pmis_v2/run_both_harnesses.py
    python3 pmis_v2/run_both_harnesses.py --recent 30
    python3 pmis_v2/run_both_harnesses.py --daily-log
"""

import sqlite3
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "memory.db")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "unified_replay")


def load_replay_data(db_path: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Load historical turns with EVERYTHING both harnesses need:
    - Retrieval scores (for ablation harness)
    - Embeddings + parent/tree structure (for session engine harness)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build a lookup: node_id → {sc_id, sc_name, ctx_id, ctx_name, embedding}
    print("Building node hierarchy lookup...")
    node_lookup = {}

    # Load all non-deleted nodes
    cursor.execute("""
        SELECT id, level, parent_ids, tree_ids, content
        FROM memory_nodes WHERE is_deleted = 0
    """)
    for row in cursor.fetchall():
        node_lookup[row["id"]] = {
            "id": row["id"],
            "level": row["level"],
            "parent_ids": row["parent_ids"],
            "tree_ids": row["tree_ids"],
            "content": row["content"] or "",
        }

    # Load relations for parent-child resolution
    cursor.execute("""
        SELECT source_id, target_id, relation_type
        FROM relations WHERE relation_type = 'child_of'
    """)
    child_to_parent = {}
    for row in cursor.fetchall():
        child_to_parent[row["source_id"]] = row["target_id"]

    # Build SC lookup for each node (walk up tree)
    def find_ancestor(node_id, target_level):
        visited = set()
        current = node_id
        while current and current not in visited:
            visited.add(current)
            node = node_lookup.get(current)
            if node and node["level"] == target_level:
                return current, node.get("content", "")[:60]
            current = child_to_parent.get(current)
        return None, ""

    # Load embeddings
    print("Loading embeddings...")
    embeddings = {}
    cursor.execute("SELECT node_id, euclidean FROM embeddings")
    for row in cursor.fetchall():
        if row["euclidean"]:
            embeddings[row["node_id"]] = np.frombuffer(
                row["euclidean"], dtype=np.float32
            ).copy()

    # Load tree names
    tree_names = {}
    cursor.execute("SELECT tree_id, name FROM trees")
    for row in cursor.fetchall():
        tree_names[row["tree_id"]] = row["name"]

    # Load turns ordered by conversation then turn number
    query = """
        SELECT ct.id as turn_id, ct.conversation_id, ct.turn_number,
               ct.gamma, ct.effective_surprise, ct.mode,
               ct.nearest_context_id, ct.nearest_context_name,
               ct.active_tree, ct.timestamp, ct.raw_surprise,
               ct.cluster_precision
        FROM conversation_turns ct
        WHERE ct.id IN (SELECT DISTINCT turn_id FROM turn_retrieved_memories)
        ORDER BY ct.conversation_id, ct.turn_number
    """
    if limit:
        query = f"""
            SELECT ct.id as turn_id, ct.conversation_id, ct.turn_number,
                   ct.gamma, ct.effective_surprise, ct.mode,
                   ct.nearest_context_id, ct.nearest_context_name,
                   ct.active_tree, ct.timestamp, ct.raw_surprise,
                   ct.cluster_precision
            FROM conversation_turns ct
            WHERE ct.id IN (SELECT DISTINCT turn_id FROM turn_retrieved_memories)
            ORDER BY ct.id DESC
            LIMIT {limit}
        """

    cursor.execute(query)
    turn_rows = cursor.fetchall()

    print(f"Loading {len(turn_rows)} turns...")
    turns = []
    for tr in turn_rows:
        # Load retrieved memories for this turn
        cursor.execute("""
            SELECT memory_node_id, rank, final_score,
                   semantic_score, hierarchy_score, temporal_score,
                   precision_score, source, node_level, content_preview
            FROM turn_retrieved_memories
            WHERE turn_id = ?
            ORDER BY rank ASC
        """, (tr["turn_id"],))

        retrieved = []
        for item in cursor.fetchall():
            node_id = item["memory_node_id"]

            # Resolve hierarchy for session engine
            sc_id, sc_name = find_ancestor(node_id, "SC")
            ctx_id, ctx_name = find_ancestor(node_id, "CTX")

            retrieved.append({
                # For ablation harness
                "node_id": node_id,
                "rank": item["rank"],
                "final_score": item["final_score"],
                "semantic_score": item["semantic_score"],
                "hierarchy_score": item["hierarchy_score"],
                "temporal_score": item["temporal_score"],
                "precision_score": item["precision_score"],
                "source": item["source"],
                "node_level": item["node_level"],
                "content_preview": item["content_preview"] or "",
                # For session engine harness
                "id": node_id,
                "sc_id": sc_id,
                "sc_name": sc_name or "",
                "ctx_id": ctx_id,
                "ctx_name": ctx_name or "",
                "anc_id": node_id if item["node_level"] == "ANC" else None,
                "anc_name": (item["content_preview"] or "")[:60],
                "score": item["final_score"],
                "embedding": embeddings.get(node_id),
            })

        # Get query embedding — use nearest context's embedding as proxy
        # (actual query embeddings aren't stored, but nearest context is)
        query_embedding = None
        nearest_ctx = tr["nearest_context_id"]
        if nearest_ctx and nearest_ctx in embeddings:
            query_embedding = embeddings[nearest_ctx]
        elif retrieved and retrieved[0]["embedding"] is not None:
            # Fallback: use top-1 retrieved memory's embedding
            query_embedding = retrieved[0]["embedding"]

        # Get query text from access_log
        query_text = ""
        cursor.execute("""
            SELECT query_text FROM access_log
            WHERE node_id = ? AND query_text IS NOT NULL
            ORDER BY accessed_at DESC LIMIT 1
        """, (retrieved[0]["node_id"],) if retrieved else ("",))
        qt_row = cursor.fetchone()
        if qt_row:
            query_text = qt_row["query_text"]

        turns.append({
            "turn_id": tr["turn_id"],
            "conversation_id": tr["conversation_id"],
            "turn_number": tr["turn_number"] or 1,
            "gamma": tr["gamma"] or 0.5,
            "effective_surprise": tr["effective_surprise"] or 0.3,
            "raw_surprise": tr["raw_surprise"] or 0.3,
            "cluster_precision": tr["cluster_precision"] or 0.5,
            "mode": tr["mode"] or "BALANCED",
            "nearest_context_id": nearest_ctx,
            "nearest_context_name": tr["nearest_context_name"] or "",
            "active_tree": tr["active_tree"] or "",
            "timestamp": tr["timestamp"] or "",
            "query_text": query_text,
            "query_embedding": query_embedding,
            "retrieved": retrieved,
        })

    conn.close()
    print(f"Loaded {len(turns)} turns across {len(set(t['conversation_id'] for t in turns))} conversations")
    return turns


def run_ablation_harness(turns: List[Dict]) -> Dict:
    """Run the retrieval component ablation analysis."""
    from replay_harness import (
        load_historical_turns, run_ablation, analyze_component,
        compute_score_contribution, surprise_analysis,
        generate_report, SCORING_CONFIGS, TurnContext, RetrievedItem,
    )

    # Convert our format to replay_harness format
    turn_contexts = []
    for t in turns:
        tc = TurnContext(
            turn_id=t["turn_id"],
            conversation_id=t["conversation_id"],
            gamma=t["gamma"],
            effective_surprise=t["effective_surprise"],
            mode=t["mode"],
            nearest_context=t["nearest_context_name"],
            active_tree=t["active_tree"],
            timestamp=t["timestamp"],
        )
        for item in t["retrieved"]:
            tc.items.append(RetrievedItem(
                turn_id=t["turn_id"],
                node_id=item["node_id"],
                actual_rank=item["rank"],
                final_score=item["final_score"],
                semantic_score=item["semantic_score"],
                hierarchy_score=item["hierarchy_score"],
                temporal_score=item["temporal_score"],
                precision_score=item["precision_score"],
                source=item["source"] or "unknown",
                node_level=item["node_level"] or "ANC",
            ))
        if tc.items:
            turn_contexts.append(tc)

    # Run ablations
    all_ablation_results = []
    for config_name, weights in SCORING_CONFIGS.items():
        if config_name == "full":
            continue
        for tc in turn_contexts:
            all_ablation_results.append(run_ablation(tc, config_name, weights))

    # Analyze
    component_reports = [
        analyze_component(comp, turn_contexts, all_ablation_results)
        for comp in ["semantic", "hierarchy", "temporal", "precision"]
    ]
    score_contributions = compute_score_contribution(turn_contexts)
    surprise_stats = surprise_analysis(turn_contexts)
    report = generate_report(
        turn_contexts, component_reports, all_ablation_results,
        score_contributions, surprise_stats
    )

    return {
        "report_text": report,
        "component_reports": component_reports,
        "score_contributions": score_contributions,
        "surprise_stats": surprise_stats,
        "turn_count": len(turn_contexts),
    }


def run_session_engine_harness(turns: List[Dict]) -> Dict:
    """Run the session tree engine simulation."""
    from replay.harness import ReplayHarness
    from replay.session_sim import SessionHyperparams

    log_dir = os.path.join(LOG_DIR, "session_engine")
    harness = ReplayHarness(
        log_dir=log_dir,
        hp=SessionHyperparams(),
        enable_extended_feedback=True,
    )

    # Group turns by conversation
    conversations = defaultdict(list)
    for t in turns:
        conversations[t["conversation_id"]].append(t)

    # Sort each conversation by turn number
    for cid in conversations:
        conversations[cid].sort(key=lambda x: x["turn_number"])

    skipped = 0
    processed = 0

    for cid, conv_turns in conversations.items():
        for t in conv_turns:
            if t["query_embedding"] is None:
                skipped += 1
                continue

            # Filter retrieved to those with embeddings for session engine
            retrieved_for_session = []
            for item in t["retrieved"]:
                entry = {
                    "id": item["id"],
                    "sc_id": item["sc_id"],
                    "sc_name": item["sc_name"],
                    "ctx_id": item["ctx_id"],
                    "ctx_name": item["ctx_name"],
                    "anc_id": item["anc_id"],
                    "anc_name": item["anc_name"],
                    "score": item["score"],
                }
                if item.get("embedding") is not None:
                    entry["embedding"] = item["embedding"]
                retrieved_for_session.append(entry)

            try:
                harness.observe_turn(
                    conversation_id=cid,
                    turn_number=t["turn_number"],
                    user_message=t["query_text"] or f"Turn {t['turn_number']}",
                    turn_embedding=t["query_embedding"],
                    gamma_global=t["gamma"],
                    surprise_global=t["effective_surprise"],
                    unbiased_retrieved=retrieved_for_session,
                )
                processed += 1
            except Exception as e:
                skipped += 1
                if processed < 3:  # Show first few errors
                    print(f"  Session engine error on {cid}/{t['turn_number']}: {e}")

    print(f"  Session engine: processed={processed}, skipped={skipped}")

    # Generate report
    report = harness.generate_report()

    # Run diagnostic analyzer if we have data
    diagnostic = None
    try:
        from replay.analyzer import DiagnosticAnalyzer
        analyzer = DiagnosticAnalyzer(log_dir)
        diagnostic = analyzer.run_full_diagnostic()
    except Exception as e:
        print(f"  Diagnostic analyzer: {e}")
        # Try running analyzer standalone
        try:
            from replay.analyzer import load_logs, analyze_conversations
            turn_logs, fb_logs, summary_logs = load_logs(log_dir)
            diagnostic = {"turns_logged": len(turn_logs), "feedbacks": len(fb_logs)}
        except Exception as e2:
            diagnostic = {"error": str(e2)}

    return {
        "report": report,
        "diagnostic": diagnostic,
        "processed": processed,
        "skipped": skipped,
        "conversations": len(conversations),
    }


def generate_unified_report(
    ablation_results: Dict,
    session_results: Dict,
    turns: List[Dict],
) -> str:
    """Combine both harness results into one actionable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("ProMe UNIFIED HARNESS REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Data: {len(turns)} turns, "
                 f"{len(set(t['conversation_id'] for t in turns))} conversations")
    lines.append("=" * 80)

    # ── PART A: Retrieval Component Ablation ──
    lines.append("\n" + "─" * 80)
    lines.append("PART A: RETRIEVAL ENGINE — COMPONENT ABLATION")
    lines.append("─" * 80)
    lines.append(ablation_results["report_text"])

    # ── PART B: Session Engine Simulation ──
    lines.append("\n" + "─" * 80)
    lines.append("PART B: SESSION TREE ENGINE — SHADOW SIMULATION")
    lines.append("─" * 80)

    sr = session_results.get("report", {})
    if isinstance(sr, dict) and "error" not in sr:
        lines.append(f"\nConversations: {sr.get('conversations', {}).get('total', 'N/A')}")
        lines.append(f"  Converged: {sr.get('conversations', {}).get('converged', 'N/A')} "
                     f"({sr.get('conversations', {}).get('convergence_rate', 0):.1%})")
        lines.append(f"  Short (≤5 turns): {sr.get('conversations', {}).get('short_conversations_pct', 0):.1%}")

        turns_info = sr.get("turns", {})
        lines.append(f"\nTurns: {turns_info.get('total', 'N/A')} total, "
                     f"{turns_info.get('mean_per_conversation', 0):.1f} avg/conv")

        conv_info = sr.get("convergence", {})
        lines.append(f"\nConvergence:")
        lines.append(f"  Mean convergence turn: {conv_info.get('mean_convergence_turn', 'N/A')}")
        lines.append(f"  PFC-led turns: {conv_info.get('pfc_led_turns_pct', 0):.1%}")
        lines.append(f"  Hippocampus-led turns: {conv_info.get('hippocampus_led_turns_pct', 0):.1%}")

        rerank = sr.get("reranking_impact", {})
        lines.append(f"\nReranking Impact:")
        lines.append(f"  Turns where schema boost changed top result: "
                     f"{rerank.get('turns_where_top_changed', 0)} "
                     f"({rerank.get('pct_turns_top_changed', 0):.1%})")

        frust = sr.get("frustration", {})
        lines.append(f"\nFrustration:")
        lines.append(f"  Conversations with frustration: {frust.get('conversations_with_frustration', 0)}")
        lines.append(f"  Max stage reached: {frust.get('max_stage_reached', 0)}")

        fb = sr.get("feedback_aggregate", {})
        if not fb.get("no_feedback_data"):
            lines.append(f"\nFeedback:")
            lines.append(f"  Total feedback turns: {fb.get('total_feedback_turns', 0)}")
            lines.append(f"  Confirmation rate: {fb.get('overall_confirmation_rate', 'N/A')}")
            lines.append(f"  Calibration accuracy: {fb.get('overall_calibration_accuracy', 'N/A')}")
            lines.append(f"  Schema abandon rate: {fb.get('overall_schema_abandon_rate', 'N/A')}")

        # Short conversation analysis
        sca = sr.get("short_conversation_analysis", {})
        if sca:
            lines.append(f"\nShort Conversation Analysis:")
            lines.append(f"  Count: {sca.get('count', 0)}")
            lines.append(f"  Ever converged: {sca.get('ever_converged', 0)}")
            lines.append(f"  Convergence rate: {sca.get('convergence_rate', 0):.1%}")
            lines.append(f"  VERDICT: {sca.get('verdict', 'N/A')}")
    else:
        lines.append(f"\nSession engine report: {sr}")

    lines.append(f"\nProcessing stats: {session_results['processed']} turns processed, "
                 f"{session_results['skipped']} skipped")

    # ── PART C: Cross-Harness Insights ──
    lines.append("\n" + "─" * 80)
    lines.append("PART C: CROSS-HARNESS INSIGHTS")
    lines.append("─" * 80)

    # Compare gamma dynamics
    ss = ablation_results.get("surprise_stats", {})
    if ss:
        gamma_std = ss.get("gamma", {}).get("std", 0)
        gamma_range = ss.get("gamma", {}).get("range", (0, 0))
        surprise_std = ss.get("surprise", {}).get("std", 0)

        lines.append(f"\n1. GAMMA DYNAMICS:")
        lines.append(f"   Retrieval engine γ: std={gamma_std:.3f}, range={gamma_range}")
        if gamma_std < 0.06:
            lines.append(f"   ⚠️  Gamma barely moves. The explore/exploit dial is stuck.")
            lines.append(f"   → Session engine's dual-gamma could help IF it diverges from global gamma.")
            # Check if session engine's gamma differs
            if isinstance(sr, dict) and sr.get("convergence", {}).get("pfc_led_turns_pct", 0) > 0.1:
                lines.append(f"   ✓ Session engine DOES differentiate — "
                             f"{sr['convergence']['pfc_led_turns_pct']:.1%} turns go PFC-led.")
            else:
                lines.append(f"   ✗ Session engine also stays hippocampus-led. Dual-gamma adds nothing.")

        lines.append(f"\n2. RETRIEVAL vs SESSION ENGINE VALUE:")
        # If hierarchy is weak but session convergence works, session engine replaces hierarchy
        hier_report = next(
            (cr for cr in ablation_results.get("component_reports", []) if cr.component == "hierarchy"),
            None
        )
        if hier_report and isinstance(sr, dict):
            conv_rate = sr.get("conversations", {}).get("convergence_rate", 0)
            lines.append(f"   Hierarchy discriminative power: {hier_report.discriminative_power:.3f}")
            lines.append(f"   Session convergence rate: {conv_rate:.1%}")
            if hier_report.discriminative_power < 0.15 and conv_rate > 0.3:
                lines.append(f"   → Session engine's schema boost could REPLACE static hierarchy scoring.")
                lines.append(f"      Schema boost is dynamic (learned per conversation) vs fixed Poincaré norms.")
            elif hier_report.discriminative_power < 0.15 and conv_rate < 0.2:
                lines.append(f"   → Both hierarchy and session engine are weak. Neither is discriminating.")
                lines.append(f"      Consider: simpler level+tree heuristic instead of either.")

        lines.append(f"\n3. ACTIONABLE NEXT STEPS:")
        lines.append(f"   Based on combined data:")

        # Decision tree
        if isinstance(sr, dict):
            confirm_rate = (sr.get("feedback_aggregate", {}).get("overall_confirmation_rate") or 0)
            rerank_pct = sr.get("reranking_impact", {}).get("pct_turns_top_changed", 0)
            conv_rate = sr.get("conversations", {}).get("convergence_rate", 0)

            if confirm_rate > 0.6 and rerank_pct > 0.10:
                lines.append(f"   ✅ Session engine is VALIDATED — schema predictions confirmed {confirm_rate:.0%}, "
                             f"reranking changes top {rerank_pct:.0%}.")
                lines.append(f"   → Proceed to Stage A implementation.")
            elif confirm_rate > 0.5 and conv_rate > 0.3:
                lines.append(f"   🟡 Session engine is PROMISING but boost is too weak.")
                lines.append(f"   → Increase sc_boost/ctx_boost/anc_boost weights and re-run.")
            elif conv_rate < 0.2:
                lines.append(f"   🔴 Session engine rarely converges ({conv_rate:.0%}).")
                lines.append(f"   → Lower theta_converged threshold or increase T_c temperature.")
            else:
                lines.append(f"   🟡 Mixed signals — collect more data.")

        # Ablation-driven simplification
        lines.append(f"\n   From retrieval ablation:")
        for cr in ablation_results.get("component_reports", []):
            if "NEGLIGIBLE" in cr.verdict:
                lines.append(f"   🔴 DROP {cr.component.upper()} — adds no ranking value.")
            elif cr.discriminative_power < 0.10:
                lines.append(f"   🟡 SIMPLIFY {cr.component.upper()} — low variance, replace with scalar.")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified ProMe Harness Runner")
    parser.add_argument("--recent", type=int, help="Only analyze last N turns")
    parser.add_argument("--daily-log", action="store_true", help="Save to daily log")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    # Load data
    turns = load_replay_data(DB_PATH, limit=args.recent)
    if not turns:
        print("No historical turns found.")
        return

    # Run Harness 1: Ablation
    print("\n" + "=" * 60)
    print("RUNNING HARNESS 1: Retrieval Component Ablation")
    print("=" * 60)
    t0 = time.time()
    ablation_results = run_ablation_harness(turns)
    t1 = time.time()
    print(f"  Ablation complete in {t1-t0:.1f}s")

    # Run Harness 2: Session Engine
    print("\n" + "=" * 60)
    print("RUNNING HARNESS 2: Session Tree Engine Simulation")
    print("=" * 60)
    t2 = time.time()
    session_results = run_session_engine_harness(turns)
    t3 = time.time()
    print(f"  Session engine complete in {t3-t2:.1f}s")

    # Generate unified report
    print("\n" + "=" * 60)
    print("GENERATING UNIFIED REPORT")
    print("=" * 60)
    report = generate_unified_report(ablation_results, session_results, turns)
    print(report)

    # Save
    if args.daily_log:
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = os.path.join(LOG_DIR, f"unified_{date_str}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nSaved to {report_path}")

        # Save JSON metrics
        metrics = {
            "date": date_str,
            "turns_analyzed": len(turns),
            "conversations": len(set(t["conversation_id"] for t in turns)),
            "ablation": {
                "components": {
                    cr.component: {
                        "discriminative_power": cr.discriminative_power,
                        "kendall_tau_without": cr.avg_kendall_tau_without,
                        "top1_preservation": cr.top1_preservation_rate,
                        "verdict": cr.verdict,
                    }
                    for cr in ablation_results.get("component_reports", [])
                },
            },
            "session_engine": {
                k: v for k, v in session_results.get("report", {}).items()
                if k not in ("per_conversation",)
            } if isinstance(session_results.get("report"), dict) else {},
        }
        metrics_path = os.path.join(LOG_DIR, f"metrics_{date_str}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
