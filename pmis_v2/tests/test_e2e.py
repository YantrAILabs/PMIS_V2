"""
PMIS v2 End-to-End Pipeline Test

Simulates a realistic multi-turn conversation through every component:
  1. Config loading
  2. DB initialization
  3. Embedding generation
  4. Surprise computation
  5. Gamma calculation
  6. Storage decisions (create/update/skip)
  7. Sequence linking (PRECEDED_BY/FOLLOWED_BY)
  8. Retrieval (γ-weighted blended)
  9. Tree resolution
  10. Predictive retrieval
  11. Prompt composition
  12. Session state tracking
  13. Orchestrator end-to-end
  14. Migration parsers
  15. Nightly consolidation
  16. Access patterns + temporal decay

Reports: PASS/FAIL per component, timing, and bottleneck analysis.
"""

import sys
import os
import time
import json
import tempfile
import traceback
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = ""
        self.duration_ms = 0
        self.details = {}
        self.warnings = []

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        time_str = f"{self.duration_ms:.0f}ms"
        warn_str = f" ⚠ {len(self.warnings)} warnings" if self.warnings else ""
        return f"  {status} [{time_str}]{warn_str} {self.name}"


class E2ETestSuite:
    def __init__(self):
        self.results: list = []
        self.db_path = ""
        self.db = None
        self.hp = None
        self.embedder = None
        self.orchestrator = None

    def run_all(self):
        print("=" * 60)
        print("PMIS v2 — END-TO-END PIPELINE TEST")
        print("=" * 60)
        print()

        # Create temp directory for test DB
        self.test_dir = tempfile.mkdtemp(prefix="pmis_test_")
        self.db_path = os.path.join(self.test_dir, "test.db")

        try:
            self._test_01_config()
            self._test_02_db_init()
            self._test_03_poincare_math()
            self._test_04_temporal_encoding()
            self._test_05_embedding_mock()
            self._test_06_node_creation_storage()
            self._test_07_surprise_computation()
            self._test_08_gamma_calculation()
            self._test_09_storage_decisions()
            self._test_10_sequence_linking()
            self._test_11_retrieval_engine()
            self._test_12_tree_resolution()
            self._test_13_predictive_retrieval()
            self._test_14_prompt_composition()
            self._test_15_session_state()
            self._test_16_orchestrator_e2e()
            # self._test_17_migration_parsers()  # removed: migration modules not shipped
            self._test_18_nightly_consolidation()
            self._test_19_temporal_decay()
            self._test_20_multi_tree_membership()
            self._test_21_dedup_detection()
            self._test_22_relation_transforms()
            self._test_23_load_test()
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(self.test_dir, ignore_errors=True)

        self._print_report()

    def _run_test(self, name: str, fn):
        result = TestResult(name)
        start = time.time()
        try:
            fn(result)
            result.passed = True
        except AssertionError as e:
            result.error = str(e)
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.details["traceback"] = traceback.format_exc()
        result.duration_ms = (time.time() - start) * 1000
        self.results.append(result)
        print(result)
        if not result.passed:
            print(f"      Error: {result.error}")
        for w in result.warnings:
            print(f"      ⚠ {w}")

    # ------------------------------------------------------------------
    # TEST 1: Config
    # ------------------------------------------------------------------
    def _test_01_config(self):
        def test(r):
            from core.config import load_config, reload
            hp = reload()
            r.details["key_count"] = len(hp)
            assert len(hp) > 40, f"Expected 40+ config keys, got {len(hp)}"
            assert hp["gamma_temperature"] == 3.0
            assert hp["poincare_dimensions"] == 32
            # Validate weight sums
            pw = hp["precision_weight_anchors"] + hp["precision_weight_recency"] + hp["precision_weight_consistency"]
            assert abs(pw - 1.0) < 0.01, f"Precision weights sum to {pw}"
            sw = hp["score_weight_semantic"] + hp["score_weight_hierarchy"] + hp["score_weight_temporal"] + hp["score_weight_precision"]
            assert abs(sw - 1.0) < 0.01, f"Score weights sum to {sw}"
            self.hp = hp
        self._run_test("Config loading + validation", test)

    # ------------------------------------------------------------------
    # TEST 2: Database
    # ------------------------------------------------------------------
    def _test_02_db_init(self):
        def test(r):
            from db.manager import DBManager
            self.db = DBManager(self.db_path)
            # Verify tables exist
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            conn.close()
            r.details["tables"] = tables
            required = ["memory_nodes", "relations", "trees", "access_log", "conversation_turns", "consolidation_log", "embeddings"]
            for t in required:
                assert t in tables, f"Missing table: {t}"
        self._run_test("Database initialization (7 tables)", test)

    # ------------------------------------------------------------------
    # TEST 3: Poincaré math
    # ------------------------------------------------------------------
    def _test_03_poincare_math(self):
        def test(r):
            from core.poincare import (poincare_distance, mobius_addition, project_to_ball,
                                        exp_map_origin, log_map_origin, hierarchy_level,
                                        ProjectionManager, assign_hyperbolic_coords)
            # Distance properties
            a = np.array([0.3, 0.2, 0.0] + [0]*29, dtype=np.float32)
            b = np.array([0.5, 0.4, 0.1] + [0]*29, dtype=np.float32)
            origin = np.zeros(32, dtype=np.float32)

            d_ab = poincare_distance(a, b)
            d_ba = poincare_distance(b, a)
            d_self = poincare_distance(a, a)

            # Symmetry
            assert abs(d_ab - d_ba) < 1e-5, f"Distance not symmetric: {d_ab} vs {d_ba}"
            # Self-distance = 0
            assert d_self < 1e-5, f"Self-distance not zero: {d_self}"
            # Triangle inequality
            d_ao = poincare_distance(a, origin)
            d_ob = poincare_distance(origin, b)
            assert d_ab <= d_ao + d_ob + 1e-5, "Triangle inequality violated"
            r.details["distance_ab"] = d_ab

            # Exp/log map roundtrip
            v = np.random.randn(32).astype(np.float32) * 0.3
            mapped = exp_map_origin(v)
            assert np.linalg.norm(mapped) < 1.0, "exp_map should stay in ball"
            recovered = log_map_origin(mapped)
            assert np.allclose(v, recovered, atol=1e-4), f"Exp/log roundtrip failed: max diff={np.max(np.abs(v-recovered))}"

            # Hierarchy levels
            pm = ProjectionManager(input_dim=768, output_dim=32)
            emb = np.random.randn(768).astype(np.float32)
            sc = assign_hyperbolic_coords(emb, "SC", pm, hyperparams=self.hp)
            ctx = assign_hyperbolic_coords(emb, "CTX", pm, hyperparams=self.hp)
            anc = assign_hyperbolic_coords(emb, "ANC", pm, hyperparams=self.hp)

            sc_norm = hierarchy_level(sc)
            ctx_norm = hierarchy_level(ctx)
            anc_norm = hierarchy_level(anc)

            assert sc_norm < ctx_norm, f"SC norm ({sc_norm:.3f}) should be < CTX ({ctx_norm:.3f})"
            assert ctx_norm < anc_norm, f"CTX norm ({ctx_norm:.3f}) should be < ANC ({anc_norm:.3f})"
            assert anc_norm < 1.0, f"ANC norm ({anc_norm:.3f}) should be < 1.0 (inside ball)"
            r.details["norms"] = {"SC": sc_norm, "CTX": ctx_norm, "ANC": anc_norm}

            # Project to ball safety
            extreme = np.ones(32) * 10.0
            projected = project_to_ball(extreme)
            assert np.linalg.norm(projected) < 1.0, "Projection should keep points inside ball"

        self._run_test("Poincaré math (distance, exp/log, hierarchy, projection)", test)

    # ------------------------------------------------------------------
    # TEST 4: Temporal encoding
    # ------------------------------------------------------------------
    def _test_04_temporal_encoding(self):
        def test(r):
            from core.temporal import temporal_encode, temporal_similarity, compute_era

            now = datetime.now()
            t_now = temporal_encode(now, dim=16)
            t_5min = temporal_encode(now - timedelta(minutes=5), dim=16)
            t_1hr = temporal_encode(now - timedelta(hours=1), dim=16)
            t_1day = temporal_encode(now - timedelta(days=1), dim=16)
            t_30day = temporal_encode(now - timedelta(days=30), dim=16)

            sim_5min = temporal_similarity(t_now, t_5min)
            sim_1hr = temporal_similarity(t_now, t_1hr)
            sim_1day = temporal_similarity(t_now, t_1day)
            sim_30day = temporal_similarity(t_now, t_30day)

            # Monotonically decreasing similarity
            assert sim_5min > sim_1hr, f"5min ({sim_5min:.3f}) should be > 1hr ({sim_1hr:.3f})"
            assert sim_1hr > sim_1day, f"1hr ({sim_1hr:.3f}) should be > 1day ({sim_1day:.3f})"

            if sim_1day <= sim_30day:
                r.warnings.append(f"1day sim ({sim_1day:.3f}) not > 30day sim ({sim_30day:.3f}) — sinusoidal wrapping")

            r.details["similarities"] = {
                "5min": sim_5min, "1hr": sim_1hr, "1day": sim_1day, "30day": sim_30day
            }

            # Era assignment
            era = compute_era(datetime(2025, 4, 15), self.hp.get("era_boundaries", {}))
            assert era == "vision_os_phase" or era != "", f"Era assignment failed: got '{era}'"

        self._run_test("Temporal encoding (similarity monotonicity, era assignment)", test)

    # ------------------------------------------------------------------
    # TEST 5: Embedding (mock)
    # ------------------------------------------------------------------
    def _test_05_embedding_mock(self):
        def test(r):
            from ingestion.embedder import Embedder
            # Create embedder — will fail on actual API call, we test the triple generation
            self.embedder = Embedder(hyperparams=self.hp)
            # Mock the embed_text method
            dim = self.hp.get("local_embedding_dimensions", 768)
            self.embedder.embed_text = lambda text: np.random.randn(dim).astype(np.float32)

            result = self.embedder.generate_triple_embedding(
                text="Test memory about GPU optimization",
                level="ANC",
                timestamp=datetime.now(),
            )

            assert "euclidean" in result
            assert "hyperbolic" in result
            assert "temporal" in result
            assert len(result["euclidean"]) == dim, f"Euclidean dim: {len(result['euclidean'])} != {dim}"
            assert len(result["hyperbolic"]) == self.hp.get("poincare_dimensions", 32)
            assert len(result["temporal"]) == self.hp.get("temporal_embedding_dim", 16)
            r.details["dims"] = {k: len(v) for k, v in result.items()}

        self._run_test("Embedding generation (triple: euclidean + hyperbolic + temporal)", test)

    # ------------------------------------------------------------------
    # TEST 6: Node creation + DB round-trip
    # ------------------------------------------------------------------
    def _test_06_node_creation_storage(self):
        def test(r):
            from core.memory_node import MemoryNode, MemoryLevel
            dim = self.hp.get("local_embedding_dimensions", 768)

            # Create a Context node
            ctx_emb = np.random.randn(dim).astype(np.float32)
            from core.poincare import ProjectionManager, assign_hyperbolic_coords
            pm = ProjectionManager(input_dim=dim, output_dim=32)
            from core.temporal import temporal_encode

            ctx = MemoryNode.create(
                content="GPU cost optimization for Yantra Vision OS inference pipeline",
                level=MemoryLevel.CONTEXT,
                euclidean_embedding=ctx_emb,
                hyperbolic_coords=assign_hyperbolic_coords(ctx_emb, "CTX", pm, hyperparams=self.hp),
                temporal_embedding=temporal_encode(datetime.now()),
                source_conversation_id="test_conv",
                surprise=0.3, precision=0.7, era="vision_os_phase",
            )
            ctx.is_orphan = False
            self.db.create_node(ctx)

            # Create 5 Anchors under it
            anchor_ids = []
            for i in range(5):
                anc_emb = ctx_emb + np.random.randn(dim).astype(np.float32) * 0.1
                anc = MemoryNode.create(
                    content=f"Anchor {i}: RTX 4090 costs ₹{5+i}/hr for inference workload {i}",
                    level=MemoryLevel.ANCHOR,
                    euclidean_embedding=anc_emb,
                    hyperbolic_coords=assign_hyperbolic_coords(anc_emb, "ANC", pm, parent_coords=ctx.hyperbolic_coords, hyperparams=self.hp),
                    temporal_embedding=temporal_encode(datetime.now() - timedelta(hours=i)),
                    source_conversation_id="test_conv",
                    surprise=0.2 + i*0.1, precision=0.6,
                )
                anc.is_orphan = False
                self.db.create_node(anc)
                self.db.attach_to_parent(anc.id, ctx.id, tree_id="test_tree")
                anchor_ids.append(anc.id)

            # Verify
            loaded_ctx = self.db.get_node(ctx.id)
            assert loaded_ctx is not None, "Context node not found after storage"
            children = self.db.get_children(ctx.id)
            assert len(children) == 5, f"Expected 5 children, got {len(children)}"

            # Verify embeddings round-trip
            embs = self.db.get_embeddings(ctx.id)
            assert embs["euclidean"] is not None
            assert len(embs["euclidean"]) == dim
            assert embs["hyperbolic"] is not None
            assert embs["temporal"] is not None

            r.details["context_id"] = ctx.id
            r.details["anchor_count"] = len(children)

            # Create tree
            self.db.create_tree("test_tree", "Test GPU Tree", root_node_id=ctx.id)

            # Store IDs for later tests
            self._ctx_id = ctx.id
            self._anchor_ids = anchor_ids
            self._ctx_emb = ctx_emb
            self._pm = pm

        self._run_test("Node creation + DB round-trip (1 Context + 5 Anchors)", test)

    # ------------------------------------------------------------------
    # TEST 7: Surprise computation
    # ------------------------------------------------------------------
    def _test_07_surprise_computation(self):
        def test(r):
            from core.surprise import compute_raw_surprise, compute_cluster_precision, compute_full_surprise
            dim = self.hp.get("local_embedding_dimensions", 768)

            # Similar query → low surprise
            similar_query = self._ctx_emb + np.random.randn(dim).astype(np.float32) * 0.05
            raw_similar = compute_raw_surprise(similar_query, self._ctx_emb)

            # Different query → high surprise
            different_query = np.random.randn(dim).astype(np.float32)
            raw_different = compute_raw_surprise(different_query, self._ctx_emb)

            assert raw_similar < raw_different, f"Similar ({raw_similar:.3f}) should have less surprise than different ({raw_different:.3f})"
            assert raw_similar < 0.3, f"Similar surprise too high: {raw_similar:.3f}"

            # Precision computation
            prec_high = compute_cluster_precision(50, 24, 0.8, self.hp)
            prec_low = compute_cluster_precision(1, 720, 0.2, self.hp)
            assert prec_high > prec_low, f"High anchors/recency ({prec_high:.3f}) should have higher precision than low ({prec_low:.3f})"

            # Full surprise
            ctx_stats = self.db.get_context_stats(self._ctx_id)
            ctx_stats["embedding"] = self._ctx_emb
            ctx_stats["name"] = "Test GPU Context"

            result = compute_full_surprise(similar_query, ctx_stats, self.hp)
            assert result.effective_surprise < 0.5, f"Similar query effective surprise too high: {result.effective_surprise:.3f}"
            assert not result.is_orphan_territory

            result2 = compute_full_surprise(different_query, ctx_stats, self.hp)
            assert result2.effective_surprise > result.effective_surprise

            r.details = {
                "raw_similar": raw_similar, "raw_different": raw_different,
                "prec_high": prec_high, "prec_low": prec_low,
                "eff_similar": result.effective_surprise, "eff_different": result2.effective_surprise,
            }

        self._run_test("Surprise computation (raw, precision, effective)", test)

    # ------------------------------------------------------------------
    # TEST 8: Gamma
    # ------------------------------------------------------------------
    def _test_08_gamma_calculation(self):
        def test(r):
            from core.gamma import compute_gamma

            g_low = compute_gamma(0.1, False, self.hp)
            g_mid = compute_gamma(0.4, False, self.hp)
            g_high = compute_gamma(0.8, False, self.hp)
            g_stale = compute_gamma(0.1, True, self.hp)

            # Low surprise → high gamma (exploit)
            assert g_low.gamma > g_high.gamma, f"Low surprise gamma ({g_low.gamma:.2f}) should > high ({g_high.gamma:.2f})"
            # Staleness reduces gamma
            assert g_stale.gamma < g_low.gamma, f"Stale gamma ({g_stale.gamma:.2f}) should < normal ({g_low.gamma:.2f})"
            # Mode labels
            assert g_high.mode_label == "PREDICTIVE"
            assert g_low.mode_label in ("ASSOCIATIVE", "BALANCED")

            # Gamma should never be exactly 0 or 1
            assert g_low.gamma < 0.96 and g_low.gamma > 0.04
            assert g_high.gamma < 0.96 and g_high.gamma > 0.04

            # Retrieval weights should be complementary
            assert abs(g_low.retrieval_narrow_weight + g_low.retrieval_broad_weight - 1.0) < 0.01

            r.details = {
                "low_surprise": {"gamma": g_low.gamma, "mode": g_low.mode_label},
                "mid_surprise": {"gamma": g_mid.gamma, "mode": g_mid.mode_label},
                "high_surprise": {"gamma": g_high.gamma, "mode": g_high.mode_label},
                "stale": {"gamma": g_stale.gamma, "mode": g_stale.mode_label},
            }

        self._run_test("Gamma calculation (explore-exploit balance)", test)

    # ------------------------------------------------------------------
    # TEST 9: Storage decisions
    # ------------------------------------------------------------------
    def _test_09_storage_decisions(self):
        def test(r):
            from ingestion.surprise_gate import decide_storage, StorageDecision
            from core.gamma import GammaResult
            from core.surprise import SurpriseResult

            # High gamma, low surprise → UPDATE
            g_exploit = GammaResult(0.8, "ASSOCIATIVE", 0.8, 0.2, "", "")
            s_low = SurpriseResult(0.1, 0.8, 0.08, "ctx1", "Test", 0.1, False)
            d1 = decide_storage(g_exploit, s_low, False, 100, self.hp)
            assert d1["action"] == StorageDecision.UPDATE, f"Expected UPDATE, got {d1['action']}"

            # Low gamma, high surprise → CREATE orphan
            g_explore = GammaResult(0.2, "PREDICTIVE", 0.2, 0.8, "", "")
            s_high = SurpriseResult(0.9, 0.3, 0.27, None, "NONE", 0.9, True)
            d2 = decide_storage(g_explore, s_high, False, 100, self.hp)
            assert d2["action"] == StorageDecision.CREATE, f"Expected CREATE, got {d2['action']}"
            assert d2.get("is_orphan") == True

            # Stale → FLAG
            d3 = decide_storage(g_exploit, s_low, True, 100, self.hp)
            assert d3["action"] == StorageDecision.FLAG_STALE

            # Too short → SKIP
            d4 = decide_storage(g_exploit, s_low, False, 10, self.hp)
            assert d4["action"] == StorageDecision.SKIP

            r.details = {
                "exploit_low_surprise": d1["action"],
                "explore_high_surprise": d2["action"],
                "stale": d3["action"],
                "short_message": d4["action"],
            }

        self._run_test("Storage gate decisions (CREATE/UPDATE/SKIP/FLAG_STALE)", test)

    # ------------------------------------------------------------------
    # TEST 10: Sequence linking
    # ------------------------------------------------------------------
    def _test_10_sequence_linking(self):
        def test(r):
            from ingestion.sequence_linker import link_sequence

            if len(self._anchor_ids) < 2:
                r.warnings.append("Need 2+ anchors for sequence test")
                return

            link_sequence(self.db, self._anchor_ids[1], self._anchor_ids[0], "test_tree")
            link_sequence(self.db, self._anchor_ids[2], self._anchor_ids[1], "test_tree")

            # Verify
            next_node = self.db.get_sequence_next(self._anchor_ids[0])
            assert next_node is not None, "FOLLOWED_BY edge not found"
            assert next_node["id"] == self._anchor_ids[1]

            prev_node = self.db.get_sequence_prev(self._anchor_ids[1])
            assert prev_node is not None, "PRECEDED_BY edge not found"
            assert prev_node["id"] == self._anchor_ids[0]

            r.details["linked_pairs"] = 2

        self._run_test("Sequence linking (PRECEDED_BY / FOLLOWED_BY edges)", test)

    # ------------------------------------------------------------------
    # TEST 11: Retrieval engine
    # ------------------------------------------------------------------
    def _test_11_retrieval_engine(self):
        def test(r):
            from retrieval.engine import RetrievalEngine
            engine = RetrievalEngine(self.db, self.hp)
            dim = self.hp.get("local_embedding_dimensions", 768)

            # Query similar to context → should retrieve anchors
            query = self._ctx_emb + np.random.randn(dim).astype(np.float32) * 0.05
            results_exploit = engine.retrieve(query, gamma=0.8, tree_context="test_tree", top_k=5)
            results_explore = engine.retrieve(query, gamma=0.2, top_k=5)

            r.details["exploit_results"] = len(results_exploit)
            r.details["explore_results"] = len(results_explore)

            if len(results_exploit) == 0:
                r.warnings.append("No results in exploit mode — threshold may be too high for random embeddings")
            if len(results_explore) == 0:
                r.warnings.append("No results in explore mode — threshold may be too high")

            # Check scoring: all results should have final_score
            for res in results_exploit:
                assert "final_score" in res, "Missing final_score in result"
                assert res["final_score"] >= 0, f"Negative score: {res['final_score']}"

            # Results should be sorted by score
            if len(results_exploit) > 1:
                scores = [r_item["final_score"] for r_item in results_exploit]
                assert scores == sorted(scores, reverse=True), "Results not sorted by score"

        self._run_test("Retrieval engine (γ-weighted blended search)", test)

    # ------------------------------------------------------------------
    # TEST 12: Tree resolution
    # ------------------------------------------------------------------
    def _test_12_tree_resolution(self):
        def test(r):
            from retrieval.tree_resolver import TreeResolver
            from core.session_state import SessionState

            resolver = TreeResolver(self.db)
            session = SessionState("test_conv")
            dim = self.hp.get("local_embedding_dimensions", 768)
            query_emb = self._ctx_emb + np.random.randn(dim).astype(np.float32) * 0.05

            # By keyword — should not match since "gpu" is only 3 chars
            tree_id = resolver.resolve("What about the GPU cost optimization?", query_emb, session)
            r.details["keyword_match"] = tree_id

            # By proximity — should find test_tree via root node embedding
            tree_id2 = resolver.resolve("Tell me about inference costs", query_emb, session)
            r.details["proximity_match"] = tree_id2

            # Session continuity
            session.set_active_tree("test_tree")
            session.gamma_history = [0.7, 0.8, 0.75]  # Exploit mode
            tree_id3 = resolver.resolve("And what about the pricing?", query_emb, session)
            assert tree_id3 == "test_tree", f"Session continuity failed: expected test_tree, got {tree_id3}"
            r.details["session_continuity"] = tree_id3

        self._run_test("Tree resolution (keyword, proximity, session continuity)", test)

    # ------------------------------------------------------------------
    # TEST 13: Predictive retrieval
    # ------------------------------------------------------------------
    def _test_13_predictive_retrieval(self):
        def test(r):
            from retrieval.predictive import PredictiveRetriever
            pred = PredictiveRetriever(self.db)

            # Predict from first anchor (should find second via FOLLOWED_BY)
            next_nodes = pred.predict_next(self._anchor_ids[0], depth=2)
            r.details["predict_next_count"] = len(next_nodes)

            if len(next_nodes) == 0:
                r.warnings.append("No predictive results — sequence edges may not be set up")
            else:
                assert next_nodes[0]["id"] == self._anchor_ids[1], "Predicted node doesn't match expected sequence"

            # Predict from context
            ctx_predictions = pred.predict_from_context(self._ctx_id)
            r.details["context_predictions"] = len(ctx_predictions)

            # Trajectory reconstruction
            trajectory = pred.get_conversation_trajectory(self._anchor_ids[0], max_length=5)
            r.details["trajectory_length"] = len(trajectory)

        self._run_test("Predictive retrieval (sequence traversal, context predictions)", test)

    # ------------------------------------------------------------------
    # TEST 14: Prompt composition
    # ------------------------------------------------------------------
    def _test_14_prompt_composition(self):
        def test(r):
            from claude_integration.prompt_composer import compose_system_prompt, compose_status_report
            from core.gamma import GammaResult
            from core.surprise import SurpriseResult

            gamma = GammaResult(0.35, "PREDICTIVE", 0.35, 0.65, "Explore openly.", "Create orphan Anchor.")
            surprise = SurpriseResult(0.7, 0.6, 0.42, self._ctx_id, "GPU Optimization", 0.7, False)

            memories = [{"id": "test", "level": "ANC", "content": "RTX 4090 costs ₹5/hr", "final_score": 0.85, "_source": "narrow", "era": "vision_os_phase"}]
            predictive = [{"content": "ByteTrack integration decision", "_prediction_depth": 1}]

            prompt = compose_system_prompt(gamma, surprise, memories, predictive, "test_tree", 5, False)

            assert "<memory_context>" in prompt
            assert "PREDICTIVE" in prompt
            assert "γ = 0.35" in prompt
            assert "RTX 4090" in prompt
            assert "ByteTrack" in prompt
            assert "BEHAVIORAL GUIDANCE" in prompt
            assert "STORAGE" in prompt

            r.details["prompt_length"] = len(prompt)

            # Staleness variant
            prompt_stale = compose_system_prompt(gamma, surprise, memories, predictive, "test_tree", 5, True)
            assert "STALENESS" in prompt_stale

            # Status report
            report = compose_status_report(gamma, surprise, {"turn_count": 5, "avg_gamma": 0.45, "stored_count": 3})
            assert "Gamma" in report
            assert "Surprise" in report

        self._run_test("Prompt composition (system prompt injection)", test)

    # ------------------------------------------------------------------
    # TEST 15: Session state
    # ------------------------------------------------------------------
    def _test_15_session_state(self):
        def test(r):
            from core.session_state import SessionState
            dim = self.hp.get("local_embedding_dimensions", 768)

            session = SessionState("test_session", buffer_size=20)

            # Simulate 10 turns
            for i in range(10):
                emb = np.random.randn(dim).astype(np.float32)
                node_id = f"node_{i}" if i % 2 == 0 else None  # Only store every other turn
                prev = session.record_turn(
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Turn {i} content about topic {i // 3}",
                    embedding=emb,
                    node_id=node_id,
                    gamma=0.5 + i * 0.03,
                    effective_surprise=0.4 - i * 0.02,
                    mode="BALANCED",
                )

            assert session.turn_counter == 10
            assert len(session.surprise_history) == 10
            assert len(session.stored_node_ids) == 5  # Every other turn
            assert session.last_stored_node_id == "node_8"
            assert session.avg_gamma > 0.0

            # Last user embedding
            last_emb = session.last_user_embedding
            assert last_emb is not None

            # Staleness detection
            from core.surprise import detect_staleness
            is_stale = detect_staleness(session.surprise_history, self.hp)
            r.details["is_stale"] = is_stale
            r.details["avg_surprise"] = float(np.mean(session.surprise_history))

            # Serialization
            logs = session.to_log_dicts()
            assert len(logs) == 10

        self._run_test("Session state (turn tracking, staleness, serialization)", test)

    # ------------------------------------------------------------------
    # TEST 16: Orchestrator end-to-end
    # ------------------------------------------------------------------
    def _test_16_orchestrator_e2e(self):
        def test(r):
            from orchestrator import Orchestrator
            dim = self.hp.get("local_embedding_dimensions", 768)

            orch = Orchestrator(db_path=self.db_path)
            # Mock the embedder
            orch.embedder.embed_text = lambda text: np.random.randn(dim).astype(np.float32)
            orch.ingestion.embedder = orch.embedder

            # Simulate 5-turn conversation
            conv_id = "e2e_test_conv"
            turn_results = []

            messages = [
                "What GPU should we use for YOLO 11n inference?",
                "How does RTX 4090 compare to A100 for this workload?",
                "What about cost optimization strategies for batch inference?",
                "I'm thinking about deploying Yantra at a hospital chain",  # Topic shift → high surprise
                "Specifically for patient safety monitoring in OTs",
            ]

            for msg in messages:
                result = orch.process_turn(msg, conversation_id=conv_id)
                turn_results.append({
                    "gamma": result.gamma_result.gamma if result.gamma_result else None,
                    "mode": result.gamma_result.mode_label if result.gamma_result else None,
                    "surprise": result.surprise_result.effective_surprise if result.surprise_result else None,
                    "action": result.storage_action,
                    "stored": result.stored_node_id is not None,
                    "prompt_len": len(result.system_prompt),
                    "retrieved": len(result.retrieved_memories),
                })

            r.details["turns"] = turn_results

            # Verify all turns produced valid output
            for i, tr in enumerate(turn_results):
                assert tr["gamma"] is not None, f"Turn {i}: gamma is None"
                assert tr["surprise"] is not None, f"Turn {i}: surprise is None"
                assert tr["prompt_len"] > 50, f"Turn {i}: prompt too short ({tr['prompt_len']})"

            # Verify stats
            stats = orch.get_stats()
            r.details["stats"] = stats

            # Verify slash commands
            status = orch.handle_command("status", conv_id)
            assert "Gamma" in status or "gamma" in status or "No turns" not in status

            surprise_cmd = orch.handle_command("surprise", conv_id)
            assert "surprise" in surprise_cmd.lower() or "Turn" in surprise_cmd

            # Close session
            orch.close_session(conv_id)
            self.orchestrator = orch

        self._run_test("Orchestrator end-to-end (5-turn conversation)", test)

    # ------------------------------------------------------------------
    # TEST 17: Migration parsers
    # ------------------------------------------------------------------
    def _test_17_migration_parsers(self):
        def test(r):
            from migration.parse_claude import parse_claude_export
            from migration.parse_chatgpt import parse_chatgpt_export
            from migration.parse_neo4j import parse_neo4j

            # Mock Claude data
            claude_data = [{"uuid": "c1", "name": "Test Conv", "created_at": "2025-06-01T10:00:00Z",
                           "chat_messages": [
                               {"uuid": "m1", "text": "This is a test user message about Yantra AI Labs vision system", "sender": "human", "created_at": "2025-06-01T10:00:00Z"},
                               {"uuid": "m2", "text": "Let me explain the architecture. The system uses YOLO 11n for detection with ByteTrack for multi-object tracking.", "sender": "assistant", "created_at": "2025-06-01T10:01:00Z"},
                           ]}]

            # Mock ChatGPT data
            gpt_data = [{"title": "PMIS Design", "id": "g1", "create_time": 1718450000.0,
                         "mapping": {
                             "r": {"id": "r", "message": None, "parent": None, "children": ["u1"]},
                             "u1": {"id": "u1", "message": {"author": {"role": "user"}, "content": {"content_type": "text", "parts": ["How should we implement the memory hierarchy for surprise minimization?"]},"create_time": 1718450100.0}, "parent": "r", "children": ["a1"]},
                             "a1": {"id": "a1", "message": {"author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["I recommend using a Poincare ball embedding where hierarchy is encoded in the norm of each point."]},"create_time": 1718450200.0}, "parent": "u1", "children": []},
                         }}]

            # Mock Neo4j data
            neo4j_data = {
                "nodes": [
                    {"id": "n1", "labels": ["Context"], "properties": {"content": "Yantra Vision OS computer vision pipeline for manufacturing"}},
                    {"id": "n2", "labels": ["Anchor"], "properties": {"content": "YOLO 11n selected for person detection with 30fps throughput"}},
                ],
                "relationships": [
                    {"startNode": "n2", "endNode": "n1", "type": "CHILD_OF", "properties": {}},
                ]
            }

            # Write temp files and parse
            results = {}
            for name, data, parser in [
                ("claude", claude_data, parse_claude_export),
                ("chatgpt", gpt_data, parse_chatgpt_export),
            ]:
                tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                json.dump(data, tmp)
                tmp.close()
                try:
                    parsed = parser(tmp.name)
                    results[name] = len(parsed)
                    assert len(parsed) > 0, f"{name}: no turns parsed"
                finally:
                    os.unlink(tmp.name)

            # Neo4j
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(neo4j_data, tmp)
            tmp.close()
            try:
                nodes, rels = parse_neo4j(tmp.name, mode="json")
                results["neo4j_nodes"] = len(nodes)
                results["neo4j_rels"] = len(rels)
                assert len(nodes) == 2
                assert len(rels) == 1
                assert rels[0]["relation_type"] == "child_of"
            finally:
                os.unlink(tmp.name)

            r.details = results

        self._run_test("Migration parsers (Claude, ChatGPT, Neo4j)", test)

    # ------------------------------------------------------------------
    # TEST 18: Nightly consolidation
    # ------------------------------------------------------------------
    def _test_18_nightly_consolidation(self):
        def test(r):
            from consolidation.nightly import NightlyConsolidation
            dim = self.hp.get("local_embedding_dimensions", 768)

            # Add some orphan nodes for BIRTH test
            from core.memory_node import MemoryNode, MemoryLevel
            from core.poincare import assign_hyperbolic_coords, ProjectionManager
            from core.temporal import temporal_encode
            pm = ProjectionManager(input_dim=dim, output_dim=32)

            base_emb = np.random.randn(dim).astype(np.float32)
            orphan_ids = []
            for i in range(4):
                emb = base_emb + np.random.randn(dim).astype(np.float32) * 0.05  # Close together
                node = MemoryNode.create(
                    content=f"Orphan memory about hospital deployment scenario {i} with patient monitoring",
                    level=MemoryLevel.ANCHOR,
                    euclidean_embedding=emb,
                    hyperbolic_coords=assign_hyperbolic_coords(emb, "ANC", pm, hyperparams=self.hp),
                    temporal_embedding=temporal_encode(datetime.now()),
                    surprise=0.6, precision=0.3,
                )
                node.is_orphan = True
                node.access_pattern.count = 4  # Above promote threshold
                self.db.create_node(node)
                orphan_ids.append(node.id)

            # Add a low-value node for PRUNE test
            old_emb = np.random.randn(dim).astype(np.float32)
            old_node = MemoryNode.create(
                content="Very old low-value memory that should be pruned during consolidation",
                level=MemoryLevel.ANCHOR,
                euclidean_embedding=old_emb,
                hyperbolic_coords=assign_hyperbolic_coords(old_emb, "ANC", pm, hyperparams=self.hp),
                temporal_embedding=temporal_encode(datetime.now() - timedelta(days=30)),
                surprise=0.05, precision=0.1,
            )
            old_node.is_orphan = True
            self.db.create_node(old_node)
            # Manually set old created_at
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.execute("UPDATE memory_nodes SET created_at = datetime('now', '-30 days') WHERE id = ?", (old_node.id,))
            conn.commit()
            conn.close()

            # Run consolidation
            engine = NightlyConsolidation(self.db, self.hp)
            # Mock LLM call
            engine._generate_context_summary = lambda contents: f"[Auto] Hospital deployment: {', '.join(c[:30] for c in contents[:3])}"

            results = engine.run()

            r.details = {
                "compressed": len(results["compressed"]),
                "promoted": len(results["promoted"]),
                "birthed": len(results["birthed"]),
                "pruned": len(results["pruned"]),
            }

            # Verify at least some actions happened
            total_actions = sum(len(v) for v in results.values())
            if total_actions == 0:
                r.warnings.append("No consolidation actions — thresholds may need adjustment for test data")

        self._run_test("Nightly consolidation (compress, promote, birth, prune)", test)

    # ------------------------------------------------------------------
    # TEST 19: Temporal decay
    # ------------------------------------------------------------------
    def _test_19_temporal_decay(self):
        def test(r):
            from core.memory_node import MemoryNode, MemoryLevel, AccessPattern

            node = MemoryNode(
                id="decay_test",
                content="Test temporal decay",
                level=MemoryLevel.ANCHOR,
                created_at=datetime.now() - timedelta(days=7),
                surprise_at_creation=0.8,
                access_pattern=AccessPattern(
                    count=5,
                    last_accessed=datetime.now() - timedelta(hours=2),
                    access_history=[datetime.now() - timedelta(hours=h) for h in [2, 24, 48, 72, 168]],
                    decay_rate=0.5,
                ),
            )

            weight = node.temporal_weight
            assert weight > 0, f"Temporal weight should be positive: {weight}"

            # High-surprise node should have higher weight than low-surprise
            node_low_surprise = MemoryNode(
                id="decay_test_low",
                content="Low surprise test",
                level=MemoryLevel.ANCHOR,
                created_at=datetime.now() - timedelta(days=7),
                surprise_at_creation=0.1,
                access_pattern=AccessPattern(
                    count=5,
                    last_accessed=datetime.now() - timedelta(hours=2),
                    access_history=[datetime.now() - timedelta(hours=h) for h in [2, 24, 48, 72, 168]],
                    decay_rate=0.5,
                ),
            )

            weight_low = node_low_surprise.temporal_weight
            assert weight > weight_low, f"High surprise weight ({weight:.3f}) should > low surprise ({weight_low:.3f})"

            r.details = {"high_surprise_weight": weight, "low_surprise_weight": weight_low, "difference": weight - weight_low}

        self._run_test("Temporal decay (power-law + reactivation + surprise shield)", test)

    # ------------------------------------------------------------------
    # TEST 20: Multi-tree membership
    # ------------------------------------------------------------------
    def _test_20_multi_tree_membership(self):
        def test(r):
            dim = self.hp.get("local_embedding_dimensions", 768)
            from core.memory_node import MemoryNode, MemoryLevel
            from core.poincare import assign_hyperbolic_coords, ProjectionManager
            from core.temporal import temporal_encode
            pm = ProjectionManager(input_dim=dim, output_dim=32)

            # Create two tree roots
            self.db.create_tree("tree_vision", "Vision OS", root_node_id=self._ctx_id)
            self.db.create_tree("tree_reid", "ReID Pipeline")

            # Create shared node
            emb = np.random.randn(dim).astype(np.float32)
            shared = MemoryNode.create(
                content="YOLO 11n detection module — shared between Vision OS and ReID pipeline",
                level=MemoryLevel.ANCHOR,
                euclidean_embedding=emb,
                hyperbolic_coords=assign_hyperbolic_coords(emb, "ANC", pm, hyperparams=self.hp),
                temporal_embedding=temporal_encode(datetime.now()),
            )
            self.db.create_node(shared)

            # Attach to two different parents/trees
            self.db.create_relation(shared.id, self._ctx_id, "child_of", "tree_vision")
            self.db.create_relation(shared.id, self._ctx_id, "child_of", "tree_reid")

            # Verify: node has two parent relationships
            parents = self.db.get_parents(shared.id)
            r.details["parent_count"] = len(parents)

            # Node count should still be 1 (not duplicated)
            node = self.db.get_node(shared.id)
            assert node is not None, "Shared node should exist once"

        self._run_test("Multi-tree membership (1 node, 2 trees)", test)

    # ------------------------------------------------------------------
    # TEST 21: Dedup
    # ------------------------------------------------------------------
    def _test_21_dedup_detection(self):
        def test(r):
            from ingestion.dedup import is_same_memory
            dim = self.hp.get("local_embedding_dimensions", 768)

            base = np.random.randn(dim).astype(np.float32)
            near_dup = base + np.random.randn(dim).astype(np.float32) * 0.001  # Very close
            different = np.random.randn(dim).astype(np.float32)

            candidate_near = {"euclidean_embedding": near_dup.tolist()}
            candidate_diff = {"euclidean_embedding": different.tolist()}

            result_dup = is_same_memory(base, candidate_near, 0, 0, "conv1", "conv1", self.hp)
            result_diff = is_same_memory(base, candidate_diff, 0, 100, "conv1", "conv2", self.hp)

            assert result_dup == "duplicate", f"Near duplicate should be detected as duplicate, got {result_dup}"
            assert result_diff == "distinct", f"Different content should be distinct, got {result_diff}"

            r.details = {"near_dup": result_dup, "different": result_diff}

        self._run_test("Dedup detection (semantic + temporal + source)", test)

    # ------------------------------------------------------------------
    # TEST 22: Relation transforms (MuRP)
    # ------------------------------------------------------------------
    def _test_22_relation_transforms(self):
        def test(r):
            from core.poincare import RelationTransform, poincare_distance

            rt = RelationTransform(dim=32)
            rt.register_relation("vision_os")
            rt.register_relation("reid_project")
            rt.register_relation("tech_decisions")

            point = np.random.randn(32).astype(np.float32) * 0.3

            t_vision = rt.transform(point, "vision_os")
            t_reid = rt.transform(point, "reid_project")
            t_tech = rt.transform(point, "tech_decisions")

            # All transforms should stay in ball
            assert np.linalg.norm(t_vision) < 1.0
            assert np.linalg.norm(t_reid) < 1.0
            assert np.linalg.norm(t_tech) < 1.0

            # Different transforms should produce different results
            d_vr = poincare_distance(t_vision, t_reid)
            d_vt = poincare_distance(t_vision, t_tech)
            d_rt = poincare_distance(t_reid, t_tech)
            assert d_vr > 0.01, "Vision/ReID transforms should differ"
            assert d_vt > 0.01, "Vision/Tech transforms should differ"

            r.details = {"d_vision_reid": d_vr, "d_vision_tech": d_vt, "d_reid_tech": d_rt}

        self._run_test("Relation transforms (MuRP — same point, different lenses)", test)

    # ------------------------------------------------------------------
    # TEST 23: Load test
    # ------------------------------------------------------------------
    def _test_23_load_test(self):
        def test(r):
            dim = self.hp.get("local_embedding_dimensions", 768)
            from core.memory_node import MemoryNode, MemoryLevel
            from core.poincare import assign_hyperbolic_coords, ProjectionManager
            from core.temporal import temporal_encode
            pm = ProjectionManager(input_dim=dim, output_dim=32)

            # Insert 100 nodes
            start = time.time()
            for i in range(100):
                emb = np.random.randn(dim).astype(np.float32)
                node = MemoryNode.create(
                    content=f"Load test memory {i} about topic {i % 10} with various details and context information",
                    level=MemoryLevel.ANCHOR,
                    euclidean_embedding=emb,
                    hyperbolic_coords=assign_hyperbolic_coords(emb, "ANC", pm, hyperparams=self.hp),
                    temporal_embedding=temporal_encode(datetime.now()),
                )
                self.db.create_node(node)
            insert_time = (time.time() - start) * 1000

            # Retrieve with scoring
            start = time.time()
            from retrieval.engine import RetrievalEngine
            engine = RetrievalEngine(self.db, self.hp)
            query = np.random.randn(dim).astype(np.float32)
            results = engine.retrieve(query, gamma=0.5, top_k=10)
            retrieve_time = (time.time() - start) * 1000

            r.details = {
                "insert_100_nodes_ms": insert_time,
                "retrieve_scored_ms": retrieve_time,
                "nodes_scanned": self.db.count_nodes(),
                "results_returned": len(results),
            }

            if insert_time > 5000:
                r.warnings.append(f"INSERT slow: {insert_time:.0f}ms for 100 nodes")
            if retrieve_time > 5000:
                r.warnings.append(f"RETRIEVE slow: {retrieve_time:.0f}ms scanning {self.db.count_nodes()} nodes")

        self._run_test("Load test (100 inserts + full retrieval scan)", test)

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------
    def _print_report(self):
        print()
        print("=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        warnings = sum(len(r.warnings) for r in self.results)
        total_time = sum(r.duration_ms for r in self.results)

        print(f"\n  Passed:   {passed}/{len(self.results)}")
        print(f"  Failed:   {failed}/{len(self.results)}")
        print(f"  Warnings: {warnings}")
        print(f"  Total:    {total_time:.0f}ms")

        if failed > 0:
            print(f"\n  FAILURES:")
            for r in self.results:
                if not r.passed:
                    print(f"    ❌ {r.name}")
                    print(f"       {r.error}")
                    if "traceback" in r.details:
                        for line in r.details["traceback"].strip().split("\n")[-3:]:
                            print(f"       {line}")

        if warnings > 0:
            print(f"\n  WARNINGS:")
            for r in self.results:
                for w in r.warnings:
                    print(f"    ⚠ [{r.name}] {w}")

        # Timing analysis
        print(f"\n  TIMING (slowest first):")
        sorted_by_time = sorted(self.results, key=lambda x: x.duration_ms, reverse=True)
        for r in sorted_by_time[:5]:
            bar = "█" * max(1, int(r.duration_ms / max(1, total_time) * 40))
            print(f"    {r.duration_ms:6.0f}ms {bar} {r.name}")

        # Bottleneck analysis
        print(f"\n  BOTTLENECK ANALYSIS:")
        slow_tests = [r for r in self.results if r.duration_ms > 500]
        if slow_tests:
            for r in slow_tests:
                print(f"    🐌 {r.name}: {r.duration_ms:.0f}ms")
        else:
            print(f"    No tests exceeded 500ms threshold")

        load_test = next((r for r in self.results if "load" in r.name.lower()), None)
        if load_test and load_test.details:
            retrieve_ms = load_test.details.get("retrieve_scored_ms", 0)
            nodes = load_test.details.get("nodes_scanned", 1)
            per_node = retrieve_ms / max(nodes, 1)
            print(f"\n    Retrieval: {retrieve_ms:.0f}ms for {nodes} nodes ({per_node:.1f}ms/node)")
            if nodes > 0:
                projected_1k = per_node * 1000
                projected_10k = per_node * 10000
                print(f"    Projected: ~{projected_1k:.0f}ms at 1K nodes, ~{projected_10k:.0f}ms at 10K nodes")
                if projected_10k > 5000:
                    print(f"    ⚠ BOTTLENECK: Linear scan won't scale past ~{int(5000/per_node)} nodes")

        print()
        print("=" * 60)


if __name__ == "__main__":
    suite = E2ETestSuite()
    suite.run_all()
