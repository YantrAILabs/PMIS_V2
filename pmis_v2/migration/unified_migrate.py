"""
Unified Migration Orchestrator for PMIS v2.

Merges data from three sources into one PMIS v2 database:
  1. Claude.ai exported conversations (JSON)
  2. ChatGPT exported conversations (JSON/ZIP)
  3. Neo4j graph database (Bolt/JSON/CSV)

Pipeline:
  Source → Parser → Unified Candidates → Dedup → Embed → Assign Coords →
  Create Nodes → Create Relations → Create Trees → Save

Usage:
  python -m migration.unified_migrate \
    --claude ./claude_export.json \
    --chatgpt ./chatgpt_export.zip \
    --neo4j ./neo4j_export.json \
    --db ./data/memory.db
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory_node import MemoryNode, MemoryLevel
from core.poincare import ProjectionManager, assign_hyperbolic_coords
from core.temporal import temporal_encode, compute_era
from core import config
from db.manager import DBManager
from ingestion.embedder import Embedder

from migration.parse_claude import parse_claude_export
from migration.parse_chatgpt import parse_chatgpt_export
from migration.parse_neo4j import parse_neo4j


class MigrationStats:
    def __init__(self):
        self.source_counts = {"claude": 0, "chatgpt": 0, "neo4j_nodes": 0, "neo4j_rels": 0}
        self.candidates_total = 0
        self.deduped = 0
        self.embedded = 0
        self.stored = 0
        self.relations_created = 0
        self.trees_created = 0
        self.errors = 0
        self.skipped_short = 0
        self.skipped_dup = 0

    def report(self) -> str:
        return f"""
Migration Complete
══════════════════════════════════════
Sources:
  Claude conversations:  {self.source_counts['claude']} turns
  ChatGPT conversations: {self.source_counts['chatgpt']} turns
  Neo4j nodes:           {self.source_counts['neo4j_nodes']}
  Neo4j relationships:   {self.source_counts['neo4j_rels']}

Processing:
  Total candidates:      {self.candidates_total}
  Skipped (too short):   {self.skipped_short}
  Skipped (duplicate):   {self.skipped_dup}
  Errors:                {self.errors}

Results:
  Nodes stored:          {self.stored}
  Relations created:     {self.relations_created}
  Trees created:         {self.trees_created}
══════════════════════════════════════"""


def migrate(
    claude_path: Optional[str] = None,
    chatgpt_path: Optional[str] = None,
    neo4j_source: Optional[str] = None,
    neo4j_mode: str = "auto",
    neo4j_user: str = "neo4j",
    neo4j_pass: str = "",
    db_path: str = "data/memory.db",
    config_path: Optional[str] = None,
    batch_size: int = 20,
    skip_embedding: bool = False,
) -> MigrationStats:
    """
    Run the full migration from all sources into PMIS v2.
    """
    hp = config.load_config(config_path) if config_path else config.get_all()
    stats = MigrationStats()

    # Initialize components
    db = DBManager(db_path)
    embedder = None if skip_embedding else Embedder(hyperparams=hp)

    input_dim = hp.get("local_embedding_dimensions", 768) if hp.get("use_local", True) else hp.get("embedding_dimensions", 1536)
    projection = ProjectionManager(input_dim=input_dim, output_dim=hp.get("poincare_dimensions", 32))

    # ──────────────────────────────────────────
    # PHASE 1: PARSE ALL SOURCES
    # ──────────────────────────────────────────
    print("\n[Phase 1] Parsing sources...")

    all_candidates: List[Dict[str, Any]] = []
    neo4j_relations: List[Dict[str, Any]] = []
    neo4j_id_map: Dict[str, str] = {}  # Maps neo4j_id → pmis_id

    # 1a. Claude
    if claude_path:
        print(f"  Parsing Claude: {claude_path}")
        try:
            claude_candidates = parse_claude_export(claude_path)
            stats.source_counts["claude"] = len(claude_candidates)
            all_candidates.extend(claude_candidates)
            print(f"    Found {len(claude_candidates)} turns")
        except Exception as e:
            print(f"    ERROR: {e}")
            stats.errors += 1

    # 1b. ChatGPT
    if chatgpt_path:
        print(f"  Parsing ChatGPT: {chatgpt_path}")
        try:
            gpt_candidates = parse_chatgpt_export(chatgpt_path)
            stats.source_counts["chatgpt"] = len(gpt_candidates)
            all_candidates.extend(gpt_candidates)
            print(f"    Found {len(gpt_candidates)} turns")
        except Exception as e:
            print(f"    ERROR: {e}")
            stats.errors += 1

    # 1c. Neo4j
    if neo4j_source:
        print(f"  Parsing Neo4j: {neo4j_source} (mode={neo4j_mode})")
        try:
            neo_nodes, neo_rels = parse_neo4j(
                neo4j_source, mode=neo4j_mode,
                username=neo4j_user, password=neo4j_pass,
            )
            stats.source_counts["neo4j_nodes"] = len(neo_nodes)
            stats.source_counts["neo4j_rels"] = len(neo_rels)
            all_candidates.extend(neo_nodes)
            neo4j_relations = neo_rels
            print(f"    Found {len(neo_nodes)} nodes, {len(neo_rels)} relationships")
        except Exception as e:
            print(f"    ERROR: {e}")
            stats.errors += 1

    stats.candidates_total = len(all_candidates)
    print(f"\n  Total candidates: {stats.candidates_total}")

    if not all_candidates:
        print("  No candidates to migrate. Done.")
        return stats

    # ──────────────────────────────────────────
    # PHASE 2: DEDUP + FILTER
    # ──────────────────────────────────────────
    print("\n[Phase 2] Deduplicating...")

    seen_hashes = set()
    filtered = []

    for c in all_candidates:
        content = c.get("content", "")

        # Skip too-short content
        if len(content.strip()) < 30:
            stats.skipped_short += 1
            continue

        # Dedup by content hash
        content_hash = MemoryNode.generate_id(content)
        if content_hash in seen_hashes:
            stats.skipped_dup += 1
            continue
        seen_hashes.add(content_hash)

        c["_pmis_id"] = content_hash
        filtered.append(c)

    print(f"  After dedup: {len(filtered)} candidates (skipped {stats.skipped_short} short, {stats.skipped_dup} dups)")

    # ──────────────────────────────────────────
    # PHASE 3: EMBED + STORE
    # ──────────────────────────────────────────
    print(f"\n[Phase 3] Embedding and storing ({len(filtered)} nodes)...")

    # P1a: Initialize ChromaDB for auto-sync during migration
    from db.chroma_store import ChromaStore
    chroma_dir = str(Path(db_path).parent / "chroma")
    chroma = ChromaStore(persist_dir=chroma_dir)
    db.set_chroma(chroma)

    # P2b: Record embedding model version
    if embedder and not skip_embedding:
        db.set_embedding_model(embedder.get_model_name())

    # P1b: Pre-compute embeddings in batches for speed
    all_contents = [c["content"] for c in filtered]
    all_embeddings = []

    if embedder and not skip_embedding:
        print(f"  Batch embedding {len(all_contents)} texts...")
        batch_start = time.time()
        all_embeddings = embedder.batch_embed_texts(all_contents, batch_size=20)
        batch_ms = (time.time() - batch_start) * 1000
        print(f"  Batch embedding done in {batch_ms:.0f}ms ({batch_ms/max(len(all_contents),1):.1f}ms/text)")
    else:
        all_embeddings = [np.random.randn(input_dim).astype(np.float32) * 0.01 for _ in filtered]

    # Track conversation → tree mapping
    conv_trees: Dict[str, str] = {}

    for i, candidate in enumerate(filtered):
        try:
            content = candidate["content"]
            level_str = candidate.get("level_hint", "ANC")
            created_at = candidate.get("created_at") or datetime.now()
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = datetime.now()

            pmis_id = candidate["_pmis_id"]

            # Use pre-computed embedding
            euclidean = all_embeddings[i]

            hyperbolic = assign_hyperbolic_coords(
                euclidean_embedding=euclidean,
                level=level_str,
                projection_manager=projection,
                hyperparams=hp,
            )
            temporal = temporal_encode(created_at, dim=hp.get("temporal_embedding_dim", 16))
            era = compute_era(created_at, hp.get("era_boundaries", {}))

            # Determine level enum
            level_map = {"SC": MemoryLevel.SUPER_CONTEXT, "CTX": MemoryLevel.CONTEXT, "ANC": MemoryLevel.ANCHOR}
            level = level_map.get(level_str, MemoryLevel.ANCHOR)

            # Create node
            node = MemoryNode(
                id=pmis_id,
                content=content,
                source_conversation_id=candidate.get("conversation_id", ""),
                euclidean_embedding=euclidean,
                hyperbolic_coords=hyperbolic,
                temporal_embedding=temporal,
                level=level,
                precision=0.5,
                surprise_at_creation=0.3,  # Neutral default for migrated data
                created_at=created_at,
                last_modified=created_at,
                era=era,
                is_orphan=(level == MemoryLevel.ANCHOR),
                is_tentative=False,
            )

            db.create_node(node)
            stats.stored += 1

            # Track neo4j ID mapping
            if candidate.get("neo4j_id"):
                neo4j_id_map[candidate["neo4j_id"]] = pmis_id

            # Create tree from conversation
            conv_id = candidate.get("conversation_id", "")
            source = candidate.get("source", "unknown")
            if conv_id and conv_id not in conv_trees:
                conv_name = candidate.get("conversation_name", conv_id)
                tree_id = f"{source}_{conv_id[:16]}"
                db.create_tree(tree_id, name=conv_name, description=f"Migrated from {source}")
                conv_trees[conv_id] = tree_id
                stats.trees_created += 1

            # Create sequence link to previous message in same conversation
            prev_msg_id = candidate.get("prev_message_id")
            if prev_msg_id:
                # Look up pmis_id of prev message
                prev_pmis = None
                for prev_c in filtered[:i]:
                    if prev_c.get("message_id") == prev_msg_id:
                        prev_pmis = prev_c.get("_pmis_id")
                        break
                if prev_pmis and prev_pmis != pmis_id:
                    tree_id = conv_trees.get(conv_id, "default")
                    db.create_relation(pmis_id, prev_pmis, "preceded_by", tree_id)
                    db.create_relation(prev_pmis, pmis_id, "followed_by", tree_id)
                    stats.relations_created += 2

            if (i + 1) % batch_size == 0:
                print(f"    Processed {i + 1}/{len(filtered)}")

        except Exception as e:
            print(f"    ERROR on item {i}: {e}")
            stats.errors += 1

    # ──────────────────────────────────────────
    # PHASE 4: IMPORT NEO4J RELATIONSHIPS
    # ──────────────────────────────────────────
    if neo4j_relations:
        print(f"\n[Phase 4] Importing {len(neo4j_relations)} Neo4j relationships...")

        for rel in neo4j_relations:
            try:
                source_neo = rel.get("source_neo4j_id")
                target_neo = rel.get("target_neo4j_id")

                source_pmis = neo4j_id_map.get(source_neo)
                target_pmis = neo4j_id_map.get(target_neo)

                if source_pmis and target_pmis and source_pmis != target_pmis:
                    db.create_relation(
                        source_id=source_pmis,
                        target_id=target_pmis,
                        relation_type=rel.get("relation_type", "related_to"),
                        tree_id=rel.get("tree_id", "neo4j_import"),
                        weight=rel.get("weight", 1.0),
                    )
                    stats.relations_created += 1
            except Exception as e:
                stats.errors += 1

        print(f"    Imported {stats.relations_created} relations")

    # ──────────────────────────────────────────
    # PHASE 5: SAVE PROJECTION MATRIX
    # ──────────────────────────────────────────
    proj_path = str(Path(db_path).parent / "projection_matrix.npy")
    projection.save(proj_path)
    print(f"\n[Phase 5] Projection matrix saved to {proj_path}")

    print(stats.report())
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PMIS v2 Migration")
    parser.add_argument("--claude", help="Path to Claude.ai exported JSON")
    parser.add_argument("--chatgpt", help="Path to ChatGPT exported JSON or ZIP")
    parser.add_argument("--neo4j", help="Path to Neo4j export (JSON/CSV) or 'bolt://...'")
    parser.add_argument("--neo4j-mode", default="auto", choices=["auto", "json", "csv", "bolt"])
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="")
    parser.add_argument("--db", default="data/memory.db", help="Output database path")
    parser.add_argument("--config", default=None, help="Path to hyperparameters.yaml")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding (use random vectors)")
    args = parser.parse_args()

    if not any([args.claude, args.chatgpt, args.neo4j]):
        parser.error("At least one source required: --claude, --chatgpt, or --neo4j")

    migrate(
        claude_path=args.claude,
        chatgpt_path=args.chatgpt,
        neo4j_source=args.neo4j,
        neo4j_mode=args.neo4j_mode,
        neo4j_user=args.neo4j_user,
        neo4j_pass=args.neo4j_pass,
        db_path=args.db,
        config_path=args.config,
        batch_size=args.batch_size,
        skip_embedding=args.skip_embedding,
    )
