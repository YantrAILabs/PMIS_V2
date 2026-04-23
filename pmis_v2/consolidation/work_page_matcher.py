"""Dream's auto-match pass over untagged work_pages (step 4).

Runs after HGCN has been retrained for the night. Gated by the bootstrap
cold-start rule: do nothing until the user has manually confirmed at least
`dream_auto_match_min_confirmed` tags (default 20). Below the threshold
HGCN's feedback edges haven't seen enough signal to beat plain similarity,
so we leave pages untagged for the user to tag in the Unassigned lane.

Output is always `tag_state='proposed'`, never `'confirmed'`. The user's
morning review is what flips a proposal to a training signal — this
prevents Dream's guesses from self-confirming into HGCN.

Scoring (v1): cosine similarity on page-summary embeddings vs. a composed
deliverable-context text. Keep it simple until we have signal to justify
the full triangulated scoring from `retrieval/project_matcher.py`.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("pmis.consolidation.work_page_matcher")


def run_work_page_matching(
    db,
    hyperparams: Dict,
    user_id: str = "local",
    force: bool = False,
    since_date: Optional[str] = None,
) -> Dict:
    """Gated auto-match pass. Returns a summary dict describing what happened."""
    min_confirmed = int(hyperparams.get("dream_auto_match_min_confirmed", 20))
    score_floor = float(hyperparams.get("dream_proposal_score_threshold", 0.55))
    ttl_days = int(hyperparams.get("dream_proposal_ttl_days", 3))

    confirmed = db.count_confirmed_page_tags(user_id=user_id)
    gate_open = force or (confirmed >= min_confirmed)

    expired = db.expire_stale_proposals(max_age_days=ttl_days, user_id=user_id)

    if not gate_open:
        logger.info(
            "bootstrap gate CLOSED: %d confirmed tags < %d required; skipping",
            confirmed, min_confirmed,
        )
        return {
            "status": "gated",
            "gated_reason": "bootstrap",
            "confirmed_tags": confirmed,
            "threshold": min_confirmed,
            "expired_proposals": expired,
            "proposals_written": 0,
        }

    pages = db.list_untagged_pages_for_matching(
        user_id=user_id, since_date=since_date
    )
    if not pages:
        return {
            "status": "ok",
            "confirmed_tags": confirmed,
            "threshold": min_confirmed,
            "forced": force,
            "expired_proposals": expired,
            "pages_scored": 0,
            "proposals_written": 0,
        }

    deliverables = _load_deliverable_catalog()
    if not deliverables:
        return {
            "status": "no_catalog",
            "reason": "no projects found in goals.yaml",
            "confirmed_tags": confirmed,
            "expired_proposals": expired,
            "proposals_written": 0,
        }

    from ingestion.embedder import Embedder
    embedder = Embedder(hyperparams=hyperparams)

    catalog_embeddings = _embed_deliverables(embedder, deliverables)
    if not catalog_embeddings:
        return {
            "status": "embed_failed",
            "reason": "could not embed any deliverable in catalog",
            "proposals_written": 0,
        }

    proposals = 0
    skipped_low_score = 0

    for page in pages:
        page_emb = page.get("embedding")
        if page_emb is None:
            # summary got re-embedded on append? fall back to on-the-fly.
            try:
                raw = embedder.embed_text(page.get("summary") or "")
                page_emb = np.asarray(raw, dtype=np.float32) if raw is not None else None
            except Exception:
                page_emb = None
        if page_emb is None:
            continue

        best = _best_match(page_emb, catalog_embeddings)
        if not best:
            continue
        project_id, deliverable_id, score = best

        if score < score_floor:
            skipped_low_score += 1
            continue

        db.set_work_page_proposal(
            page_id=page["id"],
            project_id=project_id,
            deliverable_id=deliverable_id,
            score=score,
            user_id=user_id,
        )
        proposals += 1
        logger.info(
            "proposed: page=%s → project=%s deliv=%s score=%.3f",
            page["id"], project_id, deliverable_id or "—", score,
        )

    return {
        "status": "ok",
        "confirmed_tags": confirmed,
        "threshold": min_confirmed,
        "forced": force,
        "expired_proposals": expired,
        "pages_scored": len(pages),
        "proposals_written": proposals,
        "skipped_low_score": skipped_low_score,
        "score_floor": score_floor,
    }


def _load_deliverable_catalog() -> List[Dict]:
    """Flatten goals.yaml into a list of {project_id, deliverable_id, text}.

    Text fuses project title + deliverable name + match_patterns so the
    embedding captures both the project scope and the specific deliverable.
    When a project has no deliverables, we still emit one row for it.
    """
    import os
    from pathlib import Path
    import yaml

    pt_root = Path.home() / "Desktop" / "memory" / "productivity-tracker"
    goals_path = pt_root / "config" / "goals.yaml"
    deliv_path = pt_root / "config" / "deliverables.yaml"

    if not goals_path.exists():
        return []

    try:
        with open(goals_path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        return []

    deliv_ix: Dict[str, Dict] = {}
    if deliv_path.exists():
        try:
            with open(deliv_path) as f:
                d_raw = yaml.safe_load(f) or {}
            for d in d_raw.get("deliverables", []) or []:
                deliv_ix[d["id"]] = {
                    "name": d.get("name", d["id"]),
                    "supercontext": d.get("supercontext", ""),
                }
        except Exception:
            pass

    catalog: List[Dict] = []
    for goal in raw.get("goals", []) or []:
        for project in goal.get("projects", []) or []:
            proj_id = project.get("id") or ""
            proj_title = project.get("title", proj_id)
            patterns = project.get("match_patterns", []) or []
            deliv_ids = project.get("deliverables", []) or []

            if not deliv_ids:
                text = _compose_catalog_text(proj_title, None, patterns)
                catalog.append({
                    "project_id": proj_id, "deliverable_id": "", "text": text,
                })
                continue

            for did in deliv_ids:
                dname = deliv_ix.get(did, {}).get("name", did)
                text = _compose_catalog_text(proj_title, dname, patterns)
                catalog.append({
                    "project_id": proj_id, "deliverable_id": did, "text": text,
                })

    return catalog


def _compose_catalog_text(project_title: str, deliverable_name: Optional[str],
                          patterns: List[str]) -> str:
    parts = [project_title]
    if deliverable_name:
        parts.append(deliverable_name)
    if patterns:
        parts.append("keywords: " + ", ".join(patterns[:6]))
    return " | ".join(parts)


def _embed_deliverables(embedder, catalog: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for row in catalog:
        try:
            e = embedder.embed_text(row["text"])
        except Exception:
            e = None
        if e is None:
            continue
        row["embedding"] = np.asarray(e, dtype=np.float32)
        out.append(row)
    return out


def _best_match(page_emb: np.ndarray,
                catalog_embs: List[Dict]) -> Optional[tuple]:
    best_score = -1.0
    best_proj = None
    best_deliv = None
    for row in catalog_embs:
        d = row["embedding"]
        score = float(np.dot(page_emb, d) / (
            (np.linalg.norm(page_emb) * np.linalg.norm(d)) + 1e-8
        ))
        if score > best_score:
            best_score = score
            best_proj = row["project_id"]
            best_deliv = row["deliverable_id"]
    if best_proj is None:
        return None
    return best_proj, best_deliv or "", best_score
