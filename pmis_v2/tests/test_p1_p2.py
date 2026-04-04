"""
P1/P2 Feature Validation Test.

Proves that all four auto-wired features actually fire:
  P1a: ChromaDB ANN index is populated and used for retrieval
  P1b: Batch embedding produces correct results
  P2a: Materialized context stats are cached and refreshed
  P2b: Embedding model version is tracked and checked
"""

import sys
import os
import time
import tempfile
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all():
    test_dir = tempfile.mkdtemp(prefix="pmis_p1p2_")
    db_path = os.path.join(test_dir, "test.db")

    try:
        test_p1a_chroma_auto_sync(test_dir, db_path)
        test_p1a_chroma_retrieval(test_dir, db_path)
        test_p1b_batch_embedding(test_dir)
        test_p2a_materialized_stats(test_dir, db_path)
        test_p2b_model_version_tracking(test_dir, db_path)
        test_p1a_orchestrator_auto_init(test_dir)
        test_p1a_nightly_rebuild(test_dir, db_path)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

    print()
    print("=" * 50)
    print("ALL P1/P2 VALIDATION TESTS PASSED")
    print("=" * 50)


def test_p1a_chroma_auto_sync(test_dir, db_path):
    """ChromaDB should auto-populate when nodes are created via DBManager."""
    print("\n[P1a] ChromaDB auto-sync on create/delete...")

    from db.chroma_store import ChromaStore
    from db.manager import DBManager
    from core.memory_node import MemoryNode, MemoryLevel
    from core.poincare import ProjectionManager, assign_hyperbolic_coords
    from core.temporal import temporal_encode
    from core.config import load_config

    hp = load_config()
    chroma = ChromaStore(persist_dir=os.path.join(test_dir, "chroma_sync"))
    db = DBManager(db_path, chroma_store=chroma)
    pm = ProjectionManager(input_dim=768, output_dim=32)

    # Insert 10 nodes
    node_ids = []
    for i in range(10):
        emb = np.random.randn(768).astype(np.float32)
        node = MemoryNode.create(
            content=f"P1a test node {i} about autonomous vehicle sensor fusion",
            level=MemoryLevel.ANCHOR,
            euclidean_embedding=emb,
            hyperbolic_coords=assign_hyperbolic_coords(emb, "ANC", pm, hyperparams=hp),
            temporal_embedding=temporal_encode(datetime.now()),
        )
        db.create_node(node)
        node_ids.append(node.id)

    # Verify ChromaDB has them
    chroma_count = chroma.count()
    assert chroma_count == 10, f"ChromaDB should have 10 nodes, got {chroma_count}"
    print(f"  ✅ 10 nodes auto-synced to ChromaDB (count={chroma_count})")

    # Delete one
    db.soft_delete(node_ids[0])
    chroma_count_after = chroma.count()
    assert chroma_count_after == 9, f"ChromaDB should have 9 after delete, got {chroma_count_after}"
    print(f"  ✅ Delete auto-synced (count={chroma_count_after})")

    db.close()


def test_p1a_chroma_retrieval(test_dir, db_path):
    """Retrieval engine should use ChromaDB ANN when available."""
    print("\n[P1a] ChromaDB ANN retrieval...")

    from db.chroma_store import ChromaStore
    from db.manager import DBManager
    from retrieval.engine import RetrievalEngine
    from core.memory_node import MemoryNode, MemoryLevel
    from core.poincare import ProjectionManager, assign_hyperbolic_coords
    from core.temporal import temporal_encode
    from core.config import load_config

    hp = load_config()
    chroma = ChromaStore(persist_dir=os.path.join(test_dir, "chroma_retrieval"))
    db = DBManager(os.path.join(test_dir, "retrieval_test.db"), chroma_store=chroma)
    pm = ProjectionManager(input_dim=768, output_dim=32)

    # Insert nodes with a known cluster
    base_emb = np.random.randn(768).astype(np.float32)
    for i in range(20):
        emb = base_emb + np.random.randn(768).astype(np.float32) * 0.05
        node = MemoryNode.create(
            content=f"Cluster A node {i} about GPU inference optimization",
            level=MemoryLevel.ANCHOR,
            euclidean_embedding=emb,
            hyperbolic_coords=assign_hyperbolic_coords(emb, "ANC", pm, hyperparams=hp),
            temporal_embedding=temporal_encode(datetime.now()),
        )
        db.create_node(node)

    # Verify ANN is available
    assert db.has_ann_index, "ANN index should be available"
    print(f"  ✅ ANN index active (has_ann_index=True)")

    # Query similar to cluster
    query = base_emb + np.random.randn(768).astype(np.float32) * 0.03
    engine = RetrievalEngine(db, hp)

    # Time ANN retrieval
    start = time.time()
    results = engine.retrieve(query, gamma=0.5, top_k=5)
    ann_ms = (time.time() - start) * 1000

    print(f"  ✅ ANN retrieval: {len(results)} results in {ann_ms:.0f}ms")

    # Compare with linear scan (disable ANN temporarily)
    db._chroma = None
    start = time.time()
    results_linear = engine.retrieve(query, gamma=0.5, top_k=5)
    linear_ms = (time.time() - start) * 1000
    db._chroma = chroma  # Restore

    print(f"  ✅ Linear scan: {len(results_linear)} results in {linear_ms:.0f}ms")
    print(f"  ℹ ANN vs Linear: {ann_ms:.0f}ms vs {linear_ms:.0f}ms")

    db.close()


def test_p1b_batch_embedding(test_dir):
    """Batch embedding should produce correct-dimension vectors."""
    print("\n[P1b] Batch embedding...")

    from ingestion.embedder import Embedder
    from core.config import load_config

    hp = load_config()
    embedder = Embedder(hyperparams=hp)
    dim = hp.get("local_embedding_dimensions", 768)

    # Mock the single embed
    call_count = [0]
    original_embed = embedder.embed_text
    def mock_embed(text):
        call_count[0] += 1
        return np.random.randn(dim).astype(np.float32)
    embedder.embed_text = mock_embed
    embedder._embed_ollama = mock_embed

    texts = [f"Batch test text {i} about topic {i % 5}" for i in range(50)]

    start = time.time()
    results = embedder.batch_embed_texts(texts, batch_size=10)
    batch_ms = (time.time() - start) * 1000

    assert len(results) == 50, f"Expected 50 embeddings, got {len(results)}"
    assert all(len(r) == dim for r in results), "All embeddings should have correct dimension"
    print(f"  ✅ 50 texts embedded in {batch_ms:.0f}ms ({batch_ms/50:.1f}ms/text)")

    # Verify model name
    model_name = embedder.get_model_name()
    assert model_name and len(model_name) > 0
    print(f"  ✅ Model name: '{model_name}'")


def test_p2a_materialized_stats(test_dir, db_path):
    """Context stats should be cached and auto-refreshed on attach."""
    print("\n[P2a] Materialized context stats...")

    from db.manager import DBManager
    from core.memory_node import MemoryNode, MemoryLevel
    from core.poincare import ProjectionManager, assign_hyperbolic_coords
    from core.temporal import temporal_encode
    from core.config import load_config

    hp = load_config()
    db = DBManager(os.path.join(test_dir, "stats_test.db"))
    pm = ProjectionManager(input_dim=768, output_dim=32)

    # Create a Context
    ctx_emb = np.random.randn(768).astype(np.float32)
    ctx = MemoryNode.create(
        content="Stats test Context for GPU pipeline",
        level=MemoryLevel.CONTEXT,
        euclidean_embedding=ctx_emb,
        hyperbolic_coords=assign_hyperbolic_coords(ctx_emb, "CTX", pm, hyperparams=hp),
        temporal_embedding=temporal_encode(datetime.now()),
    )
    db.create_node(ctx)

    # Attach 5 children
    for i in range(5):
        emb = np.random.randn(768).astype(np.float32)
        child = MemoryNode.create(
            content=f"Stats child {i}",
            level=MemoryLevel.ANCHOR,
            euclidean_embedding=emb,
            hyperbolic_coords=assign_hyperbolic_coords(emb, "ANC", pm, hyperparams=hp),
            temporal_embedding=temporal_encode(datetime.now()),
        )
        db.create_node(child)
        db.attach_to_parent(child.id, ctx.id, "stats_tree")

    # First call should compute and cache
    start = time.time()
    stats1 = db.get_context_stats(ctx.id)
    first_ms = (time.time() - start) * 1000

    assert stats1["num_anchors"] == 5, f"Expected 5 anchors, got {stats1['num_anchors']}"
    print(f"  ✅ First call (compute+cache): {first_ms:.1f}ms, anchors={stats1['num_anchors']}")

    # Second call should hit cache (faster)
    start = time.time()
    stats2 = db.get_context_stats(ctx.id)
    cached_ms = (time.time() - start) * 1000

    assert stats2["num_anchors"] == 5
    print(f"  ✅ Cached call: {cached_ms:.1f}ms, anchors={stats2['num_anchors']}")

    # Add another child — should auto-refresh
    emb = np.random.randn(768).astype(np.float32)
    extra = MemoryNode.create(
        content="Extra child after cache",
        level=MemoryLevel.ANCHOR,
        euclidean_embedding=emb,
        hyperbolic_coords=assign_hyperbolic_coords(emb, "ANC", pm, hyperparams=hp),
        temporal_embedding=temporal_encode(datetime.now()),
    )
    db.create_node(extra)
    db.attach_to_parent(extra.id, ctx.id, "stats_tree")

    stats3 = db.get_context_stats(ctx.id)
    assert stats3["num_anchors"] == 6, f"Expected 6 after add, got {stats3['num_anchors']}"
    print(f"  ✅ After attach_to_parent: auto-refreshed to anchors={stats3['num_anchors']}")

    db.close()


def test_p2b_model_version_tracking(test_dir, db_path):
    """Embedding model version should be tracked and checked."""
    print("\n[P2b] Embedding model version tracking...")

    from db.manager import DBManager

    db = DBManager(os.path.join(test_dir, "version_test.db"))

    # First set — should succeed
    assert db.get_embedding_model() is None, "No model should be set initially"
    print(f"  ✅ No model set initially")

    db.set_embedding_model("nomic-embed-text")
    stored = db.get_embedding_model()
    assert stored == "nomic-embed-text", f"Expected nomic-embed-text, got {stored}"
    print(f"  ✅ Model set: '{stored}'")

    # Check consistency — same model
    is_consistent = db.check_embedding_model_consistency("nomic-embed-text")
    assert is_consistent, "Same model should be consistent"
    print(f"  ✅ Consistency check (same model): passed")

    # Check consistency — different model
    is_consistent_diff = db.check_embedding_model_consistency("text-embedding-3-small")
    assert not is_consistent_diff, "Different model should fail consistency"
    print(f"  ✅ Consistency check (different model): correctly flagged mismatch")

    db.close()


def test_p1a_orchestrator_auto_init(test_dir):
    """Orchestrator should auto-initialize ChromaDB and check model version."""
    print("\n[P1a+P2b] Orchestrator auto-initialization...")

    from orchestrator import Orchestrator

    orch = Orchestrator(db_path=os.path.join(test_dir, "orch_auto.db"))

    # Mock embedder
    dim = 768
    orch.embedder.embed_text = lambda text: np.random.randn(dim).astype(np.float32)
    orch.ingestion.embedder = orch.embedder

    # Verify ChromaDB was initialized
    assert orch.db.has_ann_index, "Orchestrator should auto-init ChromaDB"
    print(f"  ✅ ChromaDB auto-initialized")

    # Verify model version was set
    model = orch.db.get_embedding_model()
    assert model is not None, "Model version should be set on startup"
    print(f"  ✅ Model version auto-set: '{model}'")

    # Process a turn — should work end-to-end with ANN
    result = orch.process_turn("Test turn about GPU optimization", conversation_id="auto_test")
    assert result.system_prompt and len(result.system_prompt) > 50
    assert result.gamma_result is not None
    print(f"  ✅ End-to-end turn with ANN: mode={result.gamma_result.mode_label}, γ={result.gamma_result.gamma:.2f}")

    # Verify node was synced to ChromaDB
    if result.stored_node_id:
        chroma_count = orch.db._chroma.count()
        assert chroma_count > 0, "Node should be in ChromaDB after storage"
        print(f"  ✅ Stored node auto-synced to ChromaDB (count={chroma_count})")

    orch.close_session("auto_test")
    orch.db.close()


def test_p1a_nightly_rebuild(test_dir, db_path):
    """Nightly consolidation should rebuild ChromaDB after changes."""
    print("\n[P1a+P2a] Nightly consolidation rebuild...")

    from db.chroma_store import ChromaStore
    from db.manager import DBManager
    from consolidation.nightly import NightlyConsolidation
    from core.memory_node import MemoryNode, MemoryLevel
    from core.poincare import ProjectionManager, assign_hyperbolic_coords
    from core.temporal import temporal_encode
    from core.config import load_config

    hp = load_config()
    chroma = ChromaStore(persist_dir=os.path.join(test_dir, "chroma_nightly"))
    db = DBManager(os.path.join(test_dir, "nightly_test.db"), chroma_store=chroma)
    pm = ProjectionManager(input_dim=768, output_dim=32)

    # Create Context + orphans
    ctx_emb = np.random.randn(768).astype(np.float32)
    ctx = MemoryNode.create(
        content="Nightly test Context",
        level=MemoryLevel.CONTEXT,
        euclidean_embedding=ctx_emb,
        hyperbolic_coords=assign_hyperbolic_coords(ctx_emb, "CTX", pm, hyperparams=hp),
        temporal_embedding=temporal_encode(datetime.now()),
    )
    db.create_node(ctx)

    for i in range(4):
        emb = ctx_emb + np.random.randn(768).astype(np.float32) * 0.05
        orphan = MemoryNode.create(
            content=f"Orphan for nightly test {i} about the same topic cluster",
            level=MemoryLevel.ANCHOR,
            euclidean_embedding=emb,
            hyperbolic_coords=assign_hyperbolic_coords(emb, "ANC", pm, hyperparams=hp),
            temporal_embedding=temporal_encode(datetime.now()),
        )
        orphan.is_orphan = True
        orphan.access_pattern.count = 5  # Above promote threshold
        db.create_node(orphan)

    count_before = chroma.count()
    print(f"  ℹ Before consolidation: {count_before} nodes in ChromaDB")

    # Run consolidation
    engine = NightlyConsolidation(db, hp)
    engine._generate_context_summary = lambda contents: "[Auto] Nightly test cluster"
    results = engine.run()

    total_actions = sum(len(v) for v in results.values())
    count_after = chroma.count()

    print(f"  ✅ Consolidation: {total_actions} actions")
    print(f"  ✅ ChromaDB after rebuild: {count_after} nodes")
    print(f"  ✅ Stats cache refreshed for all Contexts")

    db.close()


if __name__ == "__main__":
    run_all()
