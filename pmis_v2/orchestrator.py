"""
PMIS v2 Orchestrator.

The single entry point that ties everything together.
Called on every conversation turn by Claude Desktop hooks.

Full per-turn pipeline:
  1. Embed incoming message
  2. Resolve active tree
  3. Compute surprise
  4. Compute gamma
  5. Retrieve memories (γ-weighted blend)
  6. Retrieve predictive memories (sequence-based)
  7. Compose system prompt
  8. Execute storage decision (via ingestion pipeline)
  9. Log turn to session state
  10. Return prompt injection + metadata
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from core import config
from core.session_state import SessionState
from core.gamma import GammaResult
from core.surprise import SurpriseResult, detect_staleness

from db.manager import DBManager
from ingestion.embedder import Embedder
from ingestion.pipeline import IngestionPipeline
from retrieval.engine import RetrievalEngine
from retrieval.tree_resolver import TreeResolver
from retrieval.predictive import PredictiveRetriever
from retrieval.epistemic import EpistemicScorer
from claude_integration.prompt_composer import compose_system_prompt, compose_status_report
from core.diagnostics import DiagnosticCapture


class OrchestratorResult:
    """Everything the orchestrator returns for a single turn."""
    def __init__(self):
        self.system_prompt: str = ""
        self.gamma_result: Optional[GammaResult] = None
        self.surprise_result: Optional[SurpriseResult] = None
        self.retrieved_memories: List[Dict[str, Any]] = []
        self.predictive_memories: List[Dict[str, Any]] = []
        self.epistemic_questions: List[Dict[str, Any]] = []
        self.storage_action: str = "skip"
        self.stored_node_id: Optional[str] = None
        self.active_tree: Optional[str] = None
        self.is_stale: bool = False
        self.turn_number: int = 0


class Orchestrator:
    """
    Main controller for the PMIS v2 memory system.
    One instance per application lifetime. Manages sessions internally.
    """

    def __init__(self, db_path: str = "data/memory.db", config_path: str = None):
        # Load config
        self.hp = config.load_config(config_path) if config_path else config.get_all()

        # P1a: Initialize ChromaDB ANN index
        from db.chroma_store import ChromaStore
        chroma_dir = str(Path(db_path).parent / "chroma")
        chroma_store = ChromaStore(persist_dir=chroma_dir)

        # Initialize components (DB gets ChromaDB attached)
        self.db = DBManager(db_path, chroma_store=chroma_store)
        self.embedder = Embedder(hyperparams=self.hp)

        # P2b: Check embedding model consistency on startup
        model_name = self.embedder.get_model_name()
        self.db.check_embedding_model_consistency(model_name)

        # P1a: If ChromaDB is empty but SQLite has data, rebuild index
        if chroma_store.enabled and chroma_store.count() == 0 and self.db.count_nodes() > 0:
            print("[Orchestrator] ChromaDB empty but SQLite has data. Rebuilding ANN index...")
            chroma_store.rebuild_from_db(self.db)

        self.ingestion = IngestionPipeline(self.db, self.embedder, self.hp)
        self.retrieval = RetrievalEngine(self.db, self.hp)
        self.tree_resolver = TreeResolver(self.db)
        self.predictive = PredictiveRetriever(self.db)
        self.epistemic = EpistemicScorer(self.db, self.hp)

        # Session management
        self._sessions: Dict[str, SessionState] = {}

    def get_or_create_session(self, conversation_id: str = None) -> SessionState:
        """Get existing session or create new one."""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())[:8]

        if conversation_id not in self._sessions:
            buffer_size = self.hp.get("session_surprise_buffer_size", 50)
            self._sessions[conversation_id] = SessionState(
                conversation_id=conversation_id,
                buffer_size=buffer_size,
            )
        return self._sessions[conversation_id]

    def process_turn(
        self,
        content: str,
        conversation_id: str = None,
        role: str = "user",
    ) -> OrchestratorResult:
        """
        Process a single conversation turn through the full pipeline.
        This is THE main method — called on every user message.
        """
        result = OrchestratorResult()
        session = self.get_or_create_session(conversation_id)
        result.turn_number = session.turn_counter + 1

        # --- 1. INGEST (embed + surprise + gamma + storage decision) ---
        ingestion_result = self.ingestion.process_turn(
            content=content,
            session=session,
            role=role,
            conversation_id=session.conversation_id,
        )

        result.gamma_result = ingestion_result.gamma_result
        result.surprise_result = ingestion_result.surprise_result
        result.storage_action = ingestion_result.action
        result.stored_node_id = ingestion_result.node_id
        result.is_stale = ingestion_result.is_stale

        # --- 2. RESOLVE TREE ---
        query_embedding = session.last_user_embedding
        if query_embedding is not None:
            tree_id = self.tree_resolver.resolve(
                query_text=content,
                query_embedding=query_embedding,
                session=session,
            )
            if tree_id:
                session.set_active_tree(tree_id)
            result.active_tree = session.active_tree_id

        # --- 3. RETRIEVE MEMORIES ---
        if query_embedding is not None and ingestion_result.gamma_result:
            gamma = ingestion_result.gamma_result.gamma
            result.retrieved_memories = self.retrieval.retrieve(
                query_embedding=query_embedding,
                gamma=gamma,
                effective_surprise=ingestion_result.surprise_result.effective_surprise,
                tree_context=session.active_tree_id,
                top_k=self.hp.get("retrieval_max_results", 10),
            )

        # --- 3b. EPISTEMIC QUESTIONS (ambiguity detection) ---
        if query_embedding is not None and ingestion_result.gamma_result:
            result.epistemic_questions = self.epistemic.score(
                retrieved_memories=result.retrieved_memories,
                query_embedding=query_embedding,
                gamma=ingestion_result.gamma_result.gamma,
                is_orphan_territory=ingestion_result.surprise_result.is_orphan_territory,
            )

        # --- 4. PREDICTIVE RETRIEVAL ---
        if session.active_context_id:
            result.predictive_memories = self.predictive.predict_from_context(
                context_id=session.active_context_id,
                max_results=3,
            )
        elif result.stored_node_id:
            result.predictive_memories = self.predictive.predict_next(
                current_node_id=result.stored_node_id,
                depth=2,
                max_results=3,
            )
        elif session.last_stored_node_id:
            result.predictive_memories = self.predictive.predict_next(
                current_node_id=session.last_stored_node_id,
                depth=2,
                max_results=3,
            )

        # --- 5. COMPOSE SYSTEM PROMPT ---
        result.system_prompt = compose_system_prompt(
            gamma_result=ingestion_result.gamma_result,
            surprise_result=ingestion_result.surprise_result,
            retrieved_memories=result.retrieved_memories,
            predictive_memories=result.predictive_memories,
            epistemic_questions=result.epistemic_questions,
            active_tree=session.active_tree_id,
            session_turn_count=result.turn_number,
            is_stale=result.is_stale,
        )

        # --- 6. LOG TURN (rich: all reasoning data) ---
        self.db.log_turn({
            "conversation_id": session.conversation_id,
            "turn_number": result.turn_number,
            "role": role,
            "node_id": result.stored_node_id,
            "gamma": ingestion_result.gamma_result.gamma,
            "effective_surprise": ingestion_result.surprise_result.effective_surprise,
            "mode": ingestion_result.gamma_result.mode_label,
            # Rich fields
            "raw_surprise": ingestion_result.surprise_result.raw_surprise,
            "cluster_precision": ingestion_result.surprise_result.cluster_precision,
            "nearest_context_id": ingestion_result.surprise_result.nearest_context_id,
            "nearest_context_name": ingestion_result.surprise_result.nearest_context_name,
            "active_tree": result.active_tree,
            "is_stale": 1 if result.is_stale else 0,
            "storage_action": result.storage_action,
            "system_prompt": result.system_prompt,
            # Detail lists
            "retrieved_memories": [
                {
                    "memory_node_id": m["id"],
                    "rank": i + 1,
                    "final_score": m.get("final_score"),
                    "semantic_score": m.get("semantic_score"),
                    "hierarchy_score": m.get("hierarchy_score"),
                    "temporal_score": m.get("temporal_score"),
                    "precision_score": m.get("precision_score"),
                    "source": m.get("_source", "broad"),
                    "content_preview": m.get("content", "")[:150],
                    "node_level": m.get("level"),
                }
                for i, m in enumerate(result.retrieved_memories)
            ],
            "epistemic_questions": [
                {
                    "question_text": q.get("question", ""),
                    "information_gain": q.get("information_gain"),
                    "parent_context_id": q.get("parent_context_id"),
                    "parent_context_name": q.get("parent_context_name"),
                    "anchor_id": q.get("anchor_id"),
                    "anchor_content": q.get("anchor_content", "")[:150],
                }
                for q in (result.epistemic_questions or [])
            ],
            "predictive_memories": [
                {
                    "memory_node_id": p.get("id"),
                    "content_preview": p.get("content", "")[:150],
                    "prediction_depth": p.get("_prediction_depth"),
                    "prediction_frequency": p.get("_prediction_frequency"),
                }
                for p in (result.predictive_memories or [])
            ],
        })

        # --- 7. DIAGNOSTICS (end-to-end instrumentation) ---
        try:
            diag = DiagnosticCapture(
                conversation_id=session.conversation_id,
                turn_number=result.turn_number,
            )

            # Embedding stage
            diag.mark_embedding_done(
                model=self.hp.get("local_embedding_model", "nomic-embed-text")
                if self.hp.get("use_local", True)
                else self.hp.get("embedding_model", "text-embedding-3-small"),
                dim=len(query_embedding) if query_embedding is not None else 0,
            )

            # Surprise stage (Phase 1: includes precision components + K-nearest metadata)
            if ingestion_result.surprise_result:
                sr = ingestion_result.surprise_result
                diag.mark_surprise(
                    raw_surprise=sr.raw_surprise,
                    cluster_precision=sr.cluster_precision,
                    effective_surprise=sr.effective_surprise,
                    is_orphan_territory=getattr(sr, "is_orphan_territory", False),
                    anchor_factor=getattr(sr, "precision_anchor_factor", 0.0),
                    recency_factor=getattr(sr, "precision_recency_factor", 0.0),
                    consistency_factor=getattr(sr, "precision_consistency_factor", 0.0),
                )
                diag.mark_nearest_context(
                    ctx_id=sr.nearest_context_id or "",
                    ctx_name=sr.nearest_context_name or "",
                    distance=sr.nearest_distance if hasattr(sr, "nearest_distance") else sr.raw_surprise,
                    second_distance=getattr(sr, "second_nearest_distance", 0.0),
                    contexts_searched=getattr(sr, "contexts_searched", 0),
                )

            # Gamma stage (Phase 1: includes raw gamma + session boost)
            if ingestion_result.gamma_result:
                gr = ingestion_result.gamma_result
                diag.mark_gamma(
                    gamma_final=gr.gamma,
                    mode=gr.mode_label,
                    input_surprise=ingestion_result.surprise_result.effective_surprise if ingestion_result.surprise_result else 0.0,
                    temperature=self.hp.get("gamma_temperature", 6.0),
                    bias=self.hp.get("gamma_bias", 0.3),
                    gamma_raw=getattr(gr, "gamma_raw", 0.0),
                    session_boost=getattr(gr, "session_boost", 0.0),
                    override_active=(
                        getattr(session, "gamma_override", None) is not None
                        and getattr(session, "gamma_override_turns_remaining", 0) > 0
                    ),
                )

            # Tree resolution
            diag.mark_tree_resolution(
                tree_id=result.active_tree or "",
            )

            # Retrieval results
            diag.mark_retrieval_params(
                narrow_k=max(3, int(10 * 1.5 * (ingestion_result.gamma_result.gamma if ingestion_result.gamma_result else 0.5))),
                narrow_threshold=self.hp.get("retrieval_narrow_threshold", 0.82),
                broad_k=max(3, int(10 * 1.5 * (1.0 - (ingestion_result.gamma_result.gamma if ingestion_result.gamma_result else 0.5)))),
                broad_threshold=self.hp.get("retrieval_broad_threshold", 0.45),
            )
            diag.mark_retrieval_results(result.retrieved_memories)

            # Storage
            diag.mark_storage(
                action=result.storage_action or "skip",
                node_id=result.stored_node_id or "",
            )

            # Epistemic & predictive
            diag.mark_epistemic(result.epistemic_questions or [])
            diag.mark_predictive(result.predictive_memories or [])

            # Finalize with session state
            row = diag.finalize(session)
            self.db.log_diagnostics(row)

        except Exception as e:
            # Diagnostics should NEVER break the pipeline
            import logging
            logging.getLogger("pmis.diagnostics").warning(f"Diagnostic capture failed: {e}")

        return result

    def handle_command(self, command: str, conversation_id: str = None) -> str:
        """
        Handle /memory slash commands.
        Returns a human-readable response string.
        """
        session = self.get_or_create_session(conversation_id)
        parts = command.strip().lower().split()

        if not parts:
            return "Usage: /memory [status|explore|exploit|orphans|tree <name>|consolidate]"

        cmd = parts[0]

        if cmd == "status":
            if not session.gamma_history:
                return "No turns processed yet in this session."
            # Use last gamma/surprise
            from core.gamma import compute_gamma
            from core.surprise import SurpriseResult
            last_gamma = GammaResult(
                gamma=session.gamma_history[-1],
                mode_label="UNKNOWN", retrieval_narrow_weight=0,
                retrieval_broad_weight=0, confidence_instruction="",
                storage_instruction="",
            )
            last_surprise = SurpriseResult(
                raw_surprise=session.surprise_history[-1] if session.surprise_history else 0,
                cluster_precision=0.5, effective_surprise=session.surprise_history[-1] if session.surprise_history else 0,
                nearest_context_id=session.active_context_id,
                nearest_context_name="", nearest_distance=0, is_orphan_territory=False,
            )
            return compose_status_report(last_gamma, last_surprise, {
                "turn_count": session.turn_counter,
                "avg_gamma": session.avg_gamma,
                "stored_count": len(session.stored_node_ids),
            })

        elif cmd == "explore":
            # Force exploration mode for next 5 turns
            session.gamma_override = 0.2
            session.gamma_override_turns_remaining = 5
            return "Exploration mode forced (γ=0.2) for next 5 turns."

        elif cmd == "exploit":
            session.gamma_override = 0.9
            session.gamma_override_turns_remaining = 5
            return "Exploitation mode forced (γ=0.9) for next 5 turns."

        elif cmd == "orphans":
            orphans = self.db.get_orphan_nodes()
            if not orphans:
                return "No orphan Anchors found."
            lines = [f"Orphan Anchors ({len(orphans)}):"]
            for o in orphans[:20]:
                lines.append(f"  • [{o['id'][:8]}] {o['content'][:100]}")
            return "\n".join(lines)

        elif cmd == "tree" and len(parts) > 1:
            tree_name = " ".join(parts[1:])
            trees = self.db.get_all_trees()
            match = next((t for t in trees if tree_name in t["name"].lower()), None)
            if match:
                session.set_active_tree(match["tree_id"])
                return f"Active tree set to: {match['name']} ({match['tree_id']})"
            return f"Tree '{tree_name}' not found. Available: {[t['name'] for t in trees]}"

        elif cmd == "consolidate":
            from consolidation.nightly import NightlyConsolidation
            engine = NightlyConsolidation(self.db, self.hp)
            results = engine.run()
            total = sum(len(v) for v in results.values())
            return f"Consolidation complete. {total} actions: {dict((k, len(v)) for k, v in results.items())}"

        elif cmd == "surprise":
            if not session.surprise_history:
                return "No surprise history yet."
            hist = session.surprise_history[-10:]
            lines = ["Recent surprise history:"]
            for i, s in enumerate(hist):
                bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
                lines.append(f"  Turn -{len(hist)-i}: {bar} {s:.2f}")
            return "\n".join(lines)

        else:
            return "Unknown command. Usage: /memory [status|explore|exploit|orphans|tree <name>|consolidate|surprise]"

    def close_session(self, conversation_id: str):
        """
        Clean up when a conversation ends.
        Create co-occurrence edges and release session.
        Turn logs are already persisted per-turn in process_turn() step 6.
        """
        if conversation_id in self._sessions:
            session = self._sessions[conversation_id]

            # Create co-occurrence edges
            if len(session.stored_node_ids) > 1:
                from ingestion.sequence_linker import link_co_occurrence
                link_co_occurrence(self.db, session.stored_node_ids, conversation_id)

            del self._sessions[conversation_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        return {
            "total_nodes": self.db.count_nodes(),
            "super_contexts": self.db.count_nodes("SC"),
            "contexts": self.db.count_nodes("CTX"),
            "anchors": self.db.count_nodes("ANC"),
            "orphans": len(self.db.get_orphan_nodes()),
            "trees": len(self.db.get_all_trees()),
            "active_sessions": len(self._sessions),
        }
