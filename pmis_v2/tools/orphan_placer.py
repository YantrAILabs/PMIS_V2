"""
Orphan Placement Tool v2 — finds truly disconnected nodes, prunes
low-value activity notes, and proposes where to attach remaining
nodes in the connected tree.

Logic (v2 — ancestor walk):
  1. BFS from SC roots → find all truly disconnected nodes
  2. Identify & soft-delete low-value activity notes (auto-context noise)
  3. For each remaining disconnected node, find top-5 semantic neighbors
     that ARE connected to the tree
  4. For each neighbor, walk UP through ancestors (parent, grandparent, etc.)
  5. Score each ancestor as potential parent:
     sim(orphan, ancestor_content) × depth_plausibility × level_compat
  6. Propose the highest-scoring ancestor as the attachment point

Usage:
  python3 -m tools.orphan_placer diagnose       # report only
  python3 -m tools.orphan_placer prune          # identify activity notes to prune
  python3 -m tools.orphan_placer prune --commit # soft-delete them
  python3 -m tools.orphan_placer propose      # show proposed placements
  python3 -m tools.orphan_placer execute      # actually attach (dry-run first)
"""

import sqlite3
import numpy as np
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class PlacementProposal:
    node_id: str
    node_level: str
    node_content: str
    proposed_parent_id: str
    proposed_parent_content: str
    proposed_parent_level: str
    proposed_depth: int
    confidence: float  # 0-1
    nearest_connected_id: str
    nearest_connected_similarity: float
    reasoning: str


ACTIVITY_PATTERNS = [
    "During this segment",
    "During this work segment",
    "In this segment",
    "In this 10-second",
    "In this 20-second",
    "In this 30-second",
    "the worker was",
    "the user was actively",
    "the human worker",
    "Viewing a blank page",
    "Activity pattern (",
]


def is_activity_note(content: str) -> bool:
    """Detect low-value auto-context activity observations."""
    if not content:
        return False
    cl = content.lower()
    return any(p.lower() in cl for p in ACTIVITY_PATTERNS)


class OrphanPlacer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def find_disconnected(self) -> Tuple[set, dict, int]:
        """
        BFS from SC roots. Returns:
          (disconnected_ids, depth_map_of_connected, max_depth)
        """
        edges = self.conn.execute(
            "SELECT source_id, target_id FROM relations WHERE relation_type='child_of'"
        ).fetchall()
        children = defaultdict(set)
        for s, t in edges:
            children[t].add(s)

        all_ids = {r[0] for r in self.conn.execute(
            "SELECT id FROM memory_nodes WHERE is_deleted=0"
        ).fetchall()}
        sc_ids = {r[0] for r in self.conn.execute(
            "SELECT id FROM memory_nodes WHERE is_deleted=0 AND level='SC'"
        ).fetchall()}

        # BFS from roots
        connected = set()
        depth_map = {}
        for sc in sc_ids:
            q = deque([(sc, 0)])
            while q:
                n, d = q.popleft()
                if n in connected:
                    continue
                connected.add(n)
                depth_map[n] = d
                for c in children.get(n, set()):
                    q.append((c, d + 1))

        disconnected = all_ids - connected
        max_depth = max(depth_map.values()) if depth_map else 0
        return disconnected, depth_map, max_depth

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load all euclidean embeddings into memory."""
        rows = self.conn.execute(
            "SELECT node_id, euclidean FROM embeddings WHERE euclidean IS NOT NULL"
        ).fetchall()
        embs = {}
        for r in rows:
            embs[r["node_id"]] = np.frombuffer(r["euclidean"], dtype=np.float32).copy()
        return embs

    def _get_node_info(self, node_id: str) -> Optional[Dict]:
        r = self.conn.execute(
            "SELECT id, level, content FROM memory_nodes WHERE id=? AND is_deleted=0",
            (node_id,)
        ).fetchone()
        return dict(r) if r else None

    def _get_parent_of(self, node_id: str) -> Optional[str]:
        """Get the parent (target of child_of edge from this node)."""
        r = self.conn.execute(
            "SELECT target_id FROM relations WHERE source_id=? AND relation_type='child_of' LIMIT 1",
            (node_id,)
        ).fetchone()
        return r["target_id"] if r else None

    def prune_activity_notes(self, dry_run: bool = True) -> Dict:
        """
        Soft-delete low-value auto-context activity observations.
        These are "During this segment, the worker was..." notes that
        add noise to the tree without reusable insight.
        """
        rows = self.conn.execute(
            "SELECT id, content, level, access_count FROM memory_nodes WHERE is_deleted=0"
        ).fetchall()

        to_prune = []
        for r in rows:
            content = r["content"] or ""
            if is_activity_note(content):
                # Don't prune if it's been accessed frequently (user found it useful)
                if (r["access_count"] or 0) <= 3:
                    to_prune.append({
                        "id": r["id"],
                        "level": r["level"],
                        "content": content[:80],
                        "access_count": r["access_count"] or 0,
                    })

        print(f"=== Activity Note Pruning {'(DRY RUN)' if dry_run else ''} ===")
        print(f"Identified: {len(to_prune)} low-value activity notes (access_count ≤ 3)")
        by_level = defaultdict(int)
        for p in to_prune:
            by_level[p["level"]] += 1
        for lvl, cnt in sorted(by_level.items()):
            print(f"  {lvl}: {cnt}")

        if not dry_run and to_prune:
            for p in to_prune:
                self.conn.execute(
                    "UPDATE memory_nodes SET is_deleted=1 WHERE id=?", (p["id"],)
                )
            self.conn.commit()
            print(f"Soft-deleted {len(to_prune)} nodes.")

        return {"pruned": len(to_prune), "dry_run": dry_run, "by_level": dict(by_level)}

    def propose_placements(self, max_proposals: int = 500) -> List[PlacementProposal]:
        """
        V2 ancestor-walk placement.
        For each disconnected node:
          1. Find top-5 connected semantic neighbors
          2. Walk up each neighbor's ancestor chain
          3. Score each ancestor as potential parent
          4. Propose the best one
        """
        disconnected, depth_map, max_depth = self.find_disconnected()
        connected_ids = set(depth_map.keys())
        embs = self._load_embeddings()

        # Build matrix of connected embeddings for fast search
        connected_list = [nid for nid in connected_ids if nid in embs]
        if not connected_list:
            return []
        connected_matrix = np.array([embs[nid] for nid in connected_list])
        norms_c = np.linalg.norm(connected_matrix, axis=1, keepdims=True) + 1e-10
        connected_normed = connected_matrix / norms_c

        # Filter disconnected: skip activity notes and deleted
        disc_with_emb = []
        for nid in disconnected:
            if nid not in embs:
                continue
            info = self._get_node_info(nid)
            if not info:
                continue
            if is_activity_note(info.get("content", "")):
                continue  # skip activity notes — should be pruned, not placed
            disc_with_emb.append(nid)

        proposals = []
        for disc_id in disc_with_emb[:max_proposals]:
            disc_info = self._get_node_info(disc_id)
            if not disc_info:
                continue

            disc_emb = embs[disc_id]
            disc_normed = disc_emb / (np.linalg.norm(disc_emb) + 1e-10)

            # Top-5 connected semantic neighbors
            sims = connected_normed @ disc_normed
            top5_idxs = np.argsort(-sims)[:5]

            # For each neighbor, walk up ancestors and score each as potential parent
            best_proposal = None
            best_score = -1

            for tidx in top5_idxs:
                neighbor_id = connected_list[tidx]
                neighbor_sim = float(sims[tidx])

                # Walk up: neighbor → parent → grandparent → ...
                cur = neighbor_id
                ancestors = [(cur, depth_map.get(cur, max_depth))]
                for _ in range(5):  # max 5 levels up
                    parent = self._get_parent_of(cur)
                    if not parent or parent not in connected_ids:
                        break
                    ancestors.append((parent, depth_map.get(parent, 0)))
                    cur = parent

                # Also include the neighbor itself as a candidate parent
                for anc_id, anc_depth in ancestors:
                    anc_info = self._get_node_info(anc_id)
                    if not anc_info:
                        continue

                    # Score this ancestor as parent
                    # 1. Semantic match between orphan and ancestor content
                    if anc_id in embs:
                        anc_normed = embs[anc_id] / (np.linalg.norm(embs[anc_id]) + 1e-10)
                        anc_sim = float(disc_normed @ anc_normed)
                    else:
                        anc_sim = neighbor_sim * 0.5  # fallback

                    # 2. Level compatibility
                    level_compat = 1.0
                    child_level = disc_info["level"]
                    parent_level = anc_info["level"]
                    if child_level == "ANC" and parent_level == "CTX":
                        level_compat = 1.0  # ideal: ANC under CTX
                    elif child_level == "ANC" and parent_level == "SC":
                        level_compat = 0.5  # ok but skip CTX level
                    elif child_level == "ANC" and parent_level == "ANC":
                        level_compat = 0.2  # ANC under ANC is unusual
                    elif child_level == "CTX" and parent_level == "SC":
                        level_compat = 1.0  # ideal: CTX under SC
                    elif child_level == "CTX" and parent_level == "CTX":
                        level_compat = 0.8  # ok: sub-CTX
                    elif child_level == "CTX" and parent_level == "ANC":
                        level_compat = 0.05  # wrong
                    elif child_level == "SC":
                        level_compat = 0.01  # SCs shouldn't be attached

                    # 3. Depth plausibility: prefer moderate depth, not too shallow/deep
                    proposed_depth = anc_depth + 1
                    depth_score = max(0, 1.0 - abs(proposed_depth - 3.0) / 5.0)

                    # Combined score
                    score = (anc_sim * 0.5 + level_compat * 0.3 + depth_score * 0.2)

                    if score > best_score:
                        best_score = score
                        best_proposal = PlacementProposal(
                            node_id=disc_id,
                            node_level=disc_info["level"],
                            node_content=disc_info["content"][:80] if disc_info["content"] else "",
                            proposed_parent_id=anc_id,
                            proposed_parent_content=anc_info["content"][:80] if anc_info["content"] else "",
                            proposed_parent_level=anc_info["level"],
                            proposed_depth=proposed_depth,
                            confidence=score,
                            nearest_connected_id=connected_list[top5_idxs[0]],
                            nearest_connected_similarity=float(sims[top5_idxs[0]]),
                            reasoning=(
                                f"ancestor_sim={anc_sim:.3f}, "
                                f"level_compat={level_compat:.1f}, "
                                f"depth_score={depth_score:.2f}, "
                                f"via neighbor sim={neighbor_sim:.3f}"
                            ),
                        )

            if best_proposal:
                proposals.append(best_proposal)

        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals

    def diagnose(self):
        """Print a full diagnostic report."""
        disconnected, depth_map, max_depth = self.find_disconnected()
        connected = set(depth_map.keys())

        all_nodes = dict(self.conn.execute(
            "SELECT id, level FROM memory_nodes WHERE is_deleted=0"
        ).fetchall())

        print(f"=== Orphan Placement Diagnostic ===")
        print(f"Total nodes:       {len(all_nodes)}")
        print(f"Connected to SCs:  {len(connected)} ({100*len(connected)/len(all_nodes):.1f}%)")
        print(f"Disconnected:      {len(disconnected)} ({100*len(disconnected)/len(all_nodes):.1f}%)")
        print(f"Max tree depth:    {max_depth}")

        # By level
        print(f"\n--- Disconnected by level ---")
        for lvl in ["SC", "CTX", "ANC"]:
            lvl_total = sum(1 for v in all_nodes.values() if v == lvl)
            lvl_disc = sum(1 for nid in disconnected if all_nodes.get(nid) == lvl)
            print(f"  {lvl}: {lvl_disc} / {lvl_total} disconnected")

        # Flagged vs actual
        flagged = self.conn.execute(
            "SELECT COUNT(*) FROM memory_nodes WHERE is_orphan=1 AND is_deleted=0"
        ).fetchone()[0]
        print(f"\n--- Flag accuracy ---")
        print(f"  is_orphan=1 flag:  {flagged}")
        print(f"  Actually disconnected: {len(disconnected)}")
        print(f"  Flag misses: {len(disconnected) - flagged} nodes ({100*(len(disconnected)-flagged)/max(len(disconnected),1):.1f}%)")

    def show_proposals(self, n: int = 20):
        """Show top-N placement proposals with confidence."""
        proposals = self.propose_placements(max_proposals=500)

        # Confidence distribution
        confs = [p.confidence for p in proposals]
        if not confs:
            print("No proposals generated.")
            return

        print(f"=== Placement Proposals ({len(proposals)} total) ===\n")
        print(f"Confidence distribution:")
        for lo, hi in [(0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]:
            count = sum(1 for c in confs if lo <= c < hi)
            pct = 100 * count / len(confs)
            bar = "█" * int(pct / 2)
            print(f"  [{lo:.1f}-{hi:.1f}): {bar} {count} ({pct:.0f}%)")

        print(f"\n--- Top {n} proposals ---")
        for i, p in enumerate(proposals[:n]):
            print(f"\n  [{i+1}] {p.node_level} '{p.node_content[:60]}'")
            print(f"      → under {p.proposed_parent_level} '{p.proposed_parent_content[:60]}'")
            print(f"      depth={p.proposed_depth}, conf={p.confidence:.3f}, sim={p.nearest_connected_similarity:.3f}")

        # Training parameter suggestion
        high_conf = sum(1 for c in confs if c >= 0.6)
        print(f"\n--- Batch execution parameter ---")
        print(f"  Proposals with confidence ≥ 0.6: {high_conf}")
        print(f"  Proposals with confidence ≥ 0.5: {sum(1 for c in confs if c >= 0.5)}")
        print(f"  Proposals with confidence ≥ 0.4: {sum(1 for c in confs if c >= 0.4)}")
        print(f"  Recommended threshold: 0.6 (attach {high_conf} nodes)")
        print(f"  Conservative threshold: 0.7 (attach {sum(1 for c in confs if c>=0.7)} nodes)")

    def execute(self, min_confidence: float = 0.6, dry_run: bool = True):
        """Attach disconnected nodes at or above confidence threshold."""
        proposals = self.propose_placements(max_proposals=5000)
        eligible = [p for p in proposals if p.confidence >= min_confidence]

        action = "DRY RUN" if dry_run else "EXECUTING"
        print(f"=== {action}: Attaching {len(eligible)} nodes (conf ≥ {min_confidence}) ===")

        attached = 0
        for p in eligible:
            if not dry_run:
                # Create child_of edge
                self.conn.execute(
                    "INSERT OR IGNORE INTO relations (source_id, target_id, relation_type) VALUES (?, ?, 'child_of')",
                    (p.node_id, p.proposed_parent_id)
                )
                # Update is_orphan flag
                self.conn.execute(
                    "UPDATE memory_nodes SET is_orphan=0 WHERE id=?",
                    (p.node_id,)
                )
                attached += 1

        if not dry_run:
            self.conn.commit()
            print(f"Attached {attached} nodes.")
        else:
            print(f"Would attach {len(eligible)} nodes. Run with dry_run=False to execute.")

        return eligible


if __name__ == "__main__":
    import os
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "memory.db")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "diagnose"
    placer = OrphanPlacer(db_path)

    if cmd == "diagnose":
        placer.diagnose()
    elif cmd == "prune":
        dry = "--commit" not in sys.argv
        placer.prune_activity_notes(dry_run=dry)
    elif cmd == "propose":
        placer.show_proposals(n=30)
    elif cmd == "execute":
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
        dry = "--commit" not in sys.argv
        placer.execute(min_confidence=threshold, dry_run=dry)
    else:
        print(f"Unknown command: {cmd}. Use: diagnose, prune, propose, execute")
