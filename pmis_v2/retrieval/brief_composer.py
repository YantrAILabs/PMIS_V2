"""
Brief Composer — Phase 2.

When a work_session is bound to a deliverable, surface two buckets of prior
memory that help the user start:

  Bucket A — "Claude can do this"
      Anchors describing repeatable task patterns. Heuristic filter on imperative
      verbs + automation markers. These become candidates for the Phase 4
      memory-driven automation harness.

  Bucket B — "You've done this before — look here"
      Top-K anchors by semantic proximity to the deliverable, regardless of
      whether they look automatable. Pointers to prior thinking, not instructions.

No LLM call. Pure retrieval. Fast (<100ms typical). The `/api/work/brief`
endpoint consumes this directly.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("pmis.brief_composer")

# ─── Automatability heuristics ────────────────────────────────────────
# Verbs that suggest a repeatable mechanical task Claude can script.
_AUTO_VERBS = {
    "generate", "script", "refactor", "format", "extract", "build", "deploy",
    "run", "compile", "fetch", "parse", "transform", "convert", "validate",
    "test", "lint", "migrate", "sync", "export", "import", "render", "scrape",
    "summarize", "translate", "classify", "cluster", "embed", "diff", "patch",
    "rename", "reformat", "bundle", "minify", "package", "query", "ingest",
}
# Markers hinting at repeat / boilerplate / scripted nature
_AUTO_MARKERS = [
    r"\bscript\b", r"\bpipeline\b", r"\bharness\b", r"\bbatch\b",
    r"\bcli\b", r"\btest\b", r"\bboilerplate\b", r"\brepetit",
    r"\bautomat", r"\bregex\b", r"\btemplate\b", r"\bhelper\b",
    r"\bworkflow\b", r"\breport\b",
]
_AUTO_MARKER_RE = re.compile("|".join(_AUTO_MARKERS), re.IGNORECASE)


@dataclass
class BriefItem:
    node_id: str
    content: str
    preview: str
    level: str
    similarity: float
    automatability: float
    is_automatable: bool
    created_at: str
    value_score: float = 0.0
    value_feedback: float = 0.0
    redflag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class BriefComposer:
    def __init__(self, db, embedder, hyperparams: Optional[Dict[str, Any]] = None):
        self.db = db
        self.embedder = embedder
        self.hp = hyperparams or {}
        self.top_k_similar = int(self.hp.get("brief_top_k_similar", 5))
        self.top_k_claude = int(self.hp.get("brief_top_k_claude", 3))
        self.min_similarity = float(self.hp.get("brief_min_similarity", 0.35))
        self.auto_threshold = float(self.hp.get("brief_auto_threshold", 0.5))
        self.preview_chars = int(self.hp.get("brief_preview_chars", 240))
        # Candidate pool cap — keep search fast
        self.candidate_pool = int(self.hp.get("brief_candidate_pool", 200))
        # Phase 3 — blend in value_score when ranking
        self.value_blend = float(self.hp.get("value_brief_blend", 0.35))
        self.redflag_threshold = float(self.hp.get("value_feedback_redflag", -0.3))

    # ------------------------------------------------------------------

    def compose(self, deliverable_id: str) -> Dict[str, Any]:
        """Return {"deliverable": {...}, "claude_can_do": [...],
                    "you_did_before": [...], "query_text": str}."""
        deliv = self.db.get_deliverable(deliverable_id)
        if not deliv:
            return {"error": "deliverable_not_found", "deliverable_id": deliverable_id}

        query_text = self._deliverable_query_text(deliv)
        if not query_text:
            return {
                "deliverable": _deliv_summary(deliv),
                "claude_can_do": [],
                "you_did_before": [],
                "query_text": "",
                "note": "empty query — deliverable has no name/description",
            }

        try:
            query_emb = self.embedder.embed_text(query_text)
        except Exception as e:
            logger.warning("Embed failed: %s", e)
            return {
                "deliverable": _deliv_summary(deliv),
                "claude_can_do": [],
                "you_did_before": [],
                "query_text": query_text,
                "note": f"embedding_failed: {e}",
            }

        # Fast ANN path when chroma is available; fall back to linear scan
        # when ANN errors (ChromaDB broken index) or returns empty.
        candidates: List[Tuple[str, str, str, Optional[np.ndarray]]] = []
        if self.db.has_ann_index:
            try:
                hits = self.db.ann_query(query_emb, n_results=self.candidate_pool, level_filter="ANC")
            except Exception as e:
                logger.warning("ANN query failed, falling back to linear scan: %s", e)
                hits = []
            for hit in hits:
                nid = hit.get("id")
                node = self.db.get_node(nid) if nid else None
                if not node:
                    continue
                emb = self.db.get_embeddings(nid).get("euclidean")
                candidates.append((nid, node.get("content", ""), node.get("created_at", ""), emb))
        if not candidates:
            candidates = self._linear_scan_candidates()

        # Batch lookup value_score columns for all candidate ids
        value_map = self._fetch_value_map([c[0] for c in candidates])

        # Score and rank — drop candidates with strongly negative feedback
        # (value_feedback <= redflag_threshold) to keep Bucket B clean.
        ranked: List[BriefItem] = []
        for nid, content, created_at, emb in candidates:
            sim = _cosine(query_emb, emb)
            if sim < self.min_similarity:
                continue
            vinfo = value_map.get(nid, (0.0, 0.0))
            vscore, vfb = vinfo
            redflag = vfb <= self.redflag_threshold
            if redflag:
                # Known pitfall — exclude from positive suggestion buckets.
                # Phase 5 wiki rendering will surface these as warnings.
                continue
            auto = _automatability_score(content)
            ranked.append(BriefItem(
                node_id=nid,
                content=content,
                preview=(content or "")[: self.preview_chars],
                level="ANC",
                similarity=sim,
                automatability=auto,
                is_automatable=auto >= self.auto_threshold,
                created_at=created_at,
                value_score=vscore,
                value_feedback=vfb,
                redflag=redflag,
            ))

        # Bucket A: blend (similarity × automatability) with value_score
        def _bucket_a_key(b: BriefItem) -> float:
            base = b.similarity * (0.5 + 0.5 * b.automatability)
            return (1 - self.value_blend) * base + self.value_blend * b.value_score

        bucket_a_pool = [b for b in ranked if b.is_automatable]
        bucket_a_pool.sort(key=_bucket_a_key, reverse=True)
        bucket_a = bucket_a_pool[: self.top_k_claude]

        # Bucket B: blend similarity with value_score (no automatability filter)
        def _bucket_b_key(b: BriefItem) -> float:
            return (1 - self.value_blend) * b.similarity + self.value_blend * b.value_score

        seen_a = {b.node_id for b in bucket_a}
        bucket_b_pool = sorted(
            (b for b in ranked if b.node_id not in seen_a),
            key=_bucket_b_key,
            reverse=True,
        )
        bucket_b = bucket_b_pool[: self.top_k_similar]

        return {
            "deliverable": _deliv_summary(deliv),
            "query_text": query_text,
            "claude_can_do": [b.to_dict() for b in bucket_a],
            "you_did_before": [b.to_dict() for b in bucket_b],
            "candidates_scored": len(ranked),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _deliverable_query_text(self, deliv: Dict[str, Any]) -> str:
        parts = [
            deliv.get("name") or "",
            deliv.get("description") or "",
            deliv.get("project_name") or "",
        ]
        return ". ".join(p.strip() for p in parts if p and p.strip())

    def _fetch_value_map(self, node_ids: List[str]) -> Dict[str, Tuple[float, float]]:
        """Return {node_id: (value_score, value_feedback)} for the given ids.
        One query, chunked to avoid SQLite's SQLITE_MAX_VARIABLE_NUMBER."""
        if not node_ids:
            return {}
        result: Dict[str, Tuple[float, float]] = {}
        CHUNK = 500
        with self.db._connect() as conn:
            for i in range(0, len(node_ids), CHUNK):
                chunk = node_ids[i:i + CHUNK]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"""SELECT id, value_score, value_feedback
                        FROM memory_nodes WHERE id IN ({placeholders})""",
                    chunk,
                ).fetchall()
                for r in rows:
                    result[r["id"]] = (
                        float(r["value_score"] or 0.0),
                        float(r["value_feedback"] or 0.0),
                    )
        return result

    def _linear_scan_candidates(
        self,
    ) -> List[Tuple[str, str, str, Optional[np.ndarray]]]:
        rows: List[Tuple[str, str, str, Optional[np.ndarray]]] = []
        with self.db._connect() as conn:
            sql_rows = conn.execute(
                """SELECT id, content, created_at FROM memory_nodes
                   WHERE level = 'ANC' AND is_deleted = 0
                   ORDER BY last_modified DESC LIMIT ?""",
                (self.candidate_pool,),
            ).fetchall()
        for r in sql_rows:
            nid = r["id"]
            emb = self.db.get_embeddings(nid).get("euclidean")
            rows.append((nid, r["content"] or "", r["created_at"] or "", emb))
        return rows


# ─── module-level helpers ─────────────────────────────────────────────

def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    if getattr(a, "shape", None) != getattr(b, "shape", None):
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / (na * nb))))


def _automatability_score(content: str) -> float:
    """Rough score in [0,1]. Sums verb + marker hits into a sigmoid-ish scale."""
    if not content:
        return 0.0
    lc = content.lower()
    verb_hits = sum(1 for v in _AUTO_VERBS if re.search(rf"\b{v}\b", lc))
    marker_hits = len(_AUTO_MARKER_RE.findall(content))
    total = verb_hits + 0.7 * marker_hits
    # Saturating curve: 0 hits → 0; 1 → 0.4; 3 → 0.75; 5+ → ~0.9
    return float(1 - np.exp(-total / 2.5))


def _deliv_summary(deliv: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": deliv.get("id"),
        "name": deliv.get("name"),
        "project_id": deliv.get("project_id"),
        "project_name": deliv.get("project_name") or "",
        "description": deliv.get("description") or "",
        "status": deliv.get("status"),
    }
