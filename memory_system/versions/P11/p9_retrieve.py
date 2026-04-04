#!/usr/bin/env python3
"""
P9+ Retrieval Engine for PMIS
==============================
Drop this file into desktop/memory/scripts/ alongside memory.py

Zero external dependencies — uses only Python stdlib.
Optionally uses sklearn for TF-IDF vectors (auto-detected).

Provides: p9_retrieve(conn, query) → same output format as existing cmd_retrieve
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict

# Import from sibling memory.py
sys.path.insert(0, os.path.dirname(__file__))
import memory as mem

# Try optional sklearn for vectors
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# =============================================================================
# STOP WORDS & TOKENIZER
# =============================================================================

_STOP = frozenset({
    "the","a","an","is","are","was","were","be","been","for","and","or",
    "but","in","on","at","to","of","with","how","what","which","do","does",
    "can","should","i","my","our","that","this","it","by","from","has","have",
    "had","not","will","about","need","help","me","now","also","its","we",
    "these","more","into","using","used","use","get","set","new","like",
    "just","very","so","if","than","then","been","being","too","some"
})

def _tok(text):
    return [w.lower().strip(".,;:!?()[]{}\"'/-") for w in text.split()
            if len(w) > 2 and w.lower().strip(".,;:!?()[]{}\"'/-") not in _STOP]


# =============================================================================
# BM25 ENGINE (pure Python, zero deps)
# =============================================================================

class BM25:
    def __init__(self, k1=1.5, b=0.75, **kwargs):
        self.k1, self.b = k1, b
        self.docs = []
        self.tf = []
        self.idf = {}
        self.avgdl = 1
        self.N = 0

    def build(self, documents):
        self.docs = [_tok(d) for d in documents]
        self.N = len(self.docs)
        if self.N == 0:
            return
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        df = Counter()
        self.tf = []
        for d in self.docs:
            tf = Counter(d)
            self.tf.append(tf)
            for t in set(d):
                df[t] += 1
        self.idf = {t: math.log((self.N - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def score(self, query_tokens, doc_idx):
        s = 0.0
        dl = len(self.docs[doc_idx])
        tf = self.tf[doc_idx]
        for t in query_tokens:
            if t not in self.idf:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


# =============================================================================
# TAG ENGINE (pure Python, zero deps)
# =============================================================================

class TagEngine:
    def __init__(self):
        self.anchor_tags = {}  # anchor_id -> set of tags

    def build(self, conn):
        """Build tag index from nodes table. Auto-generates tags from title words."""
        rows = conn.execute(
            "SELECT id, title, content FROM nodes WHERE type='anchor'"
        ).fetchall()
        for r in rows:
            tags = set()
            # Use title words as implicit tags
            for w in _tok(r["title"]):
                tags.add(w)
            # Use content words (first 10 significant words)
            if r["content"]:
                for w in _tok(r["content"])[:10]:
                    tags.add(w)
            self.anchor_tags[r["id"]] = tags

    def score_anchors(self, query_tags):
        """Returns {anchor_id: jaccard_score}"""
        scores = {}
        for aid, atags in self.anchor_tags.items():
            overlap = len(query_tags & atags)
            if overlap == 0:
                continue
            union = len(query_tags | atags)
            scores[aid] = overlap / union
        return scores


# =============================================================================
# VECTOR ENGINE (optional, needs sklearn)
# =============================================================================

class VectorEngine:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.active = False

    def build(self, texts):
        if not HAS_SKLEARN or not texts:
            return
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=500, stop_words='english',
                ngram_range=(1, 2), sublinear_tf=True
            )
            self.matrix = self.vectorizer.fit_transform(texts).toarray()
            self.active = True
        except Exception:
            self.active = False

    def search(self, query):
        if not self.active:
            return []
        qv = self.vectorizer.transform([query]).toarray()
        return cosine_similarity(qv, self.matrix)[0].tolist()


# =============================================================================
# P11 EXPERIMENTS: SEMANTIC UNDERSTANDING
# =============================================================================

# C1: Synonym expansion map (domain-specific)
SYNONYMS = {
    "email": ["message", "outreach", "mail", "correspondence"],
    "camera": ["cctv", "surveillance", "cam", "dvr"],
    "factory": ["manufacturing", "industrial", "plant", "workshop"],
    "store": ["shop", "kirana", "retail", "outlet"],
    "hospital": ["medical", "healthcare", "clinic", "patient"],
    "construction": ["building", "site", "infrastructure"],
    "security": ["safety", "protection", "surveillance", "guard"],
    "deploy": ["install", "setup", "rollout", "launch"],
    "cost": ["price", "expense", "budget", "pricing"],
    "gpu": ["compute", "inference", "accelerator"],
    "pipeline": ["workflow", "process", "architecture"],
    "lead": ["prospect", "contact", "target"],
    "demo": ["demonstration", "showcase", "preview"],
    "dashboard": ["panel", "console", "monitor", "ui"],
    "detection": ["recognition", "identification", "sensing"],
    "alert": ["notification", "warning", "alarm"],
    "model": ["algorithm", "network", "ai"],
    "training": ["learning", "fine-tuning", "teaching"],
    "accuracy": ["precision", "reliability", "performance"],
    "residential": ["society", "apartment", "housing", "colony"],
}

# C1: Reverse synonym map for lookup
_REVERSE_SYNONYMS = {}
for key, syns in SYNONYMS.items():
    for s in syns:
        if s not in _REVERSE_SYNONYMS:
            _REVERSE_SYNONYMS[s] = []
        _REVERSE_SYNONYMS[s].append(key)
    _REVERSE_SYNONYMS[key] = syns


def _expand_synonyms(tokens):
    """C1: Expand query tokens with domain synonyms."""
    expanded = set(tokens)
    for t in tokens:
        if t in SYNONYMS:
            expanded.update(SYNONYMS[t])
        if t in _REVERSE_SYNONYMS:
            expanded.update(_REVERSE_SYNONYMS[t])
    return expanded


# C12: Morphological stemming (lightweight suffix stripping)
_SUFFIXES = [
    ("ation", ""), ("tion", ""), ("sion", ""), ("ment", ""),
    ("ness", ""), ("ity", ""), ("ing", ""), ("ed", ""),
    ("ly", ""), ("er", ""), ("est", ""), ("ies", "y"),
    ("es", ""), ("s", ""),
]

def _stem(word):
    """C12: Lightweight suffix stripping for English morphology."""
    if len(word) < 5:
        return word
    for suffix, replacement in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)] + replacement
    return word


def _stem_tokens(tokens):
    """C12: Stem a list of tokens and return both original + stemmed."""
    result = set(tokens)
    for t in tokens:
        stemmed = _stem(t)
        if stemmed != t and len(stemmed) >= 3:
            result.add(stemmed)
    return result


# C9: Compound query splitting
_SPLIT_PREPS = {"for", "about", "regarding", "on", "with", "to", "from", "in", "at", "of"}

def _split_compound_query(query):
    """C9: Split compound queries at prepositions.
    'cold email for construction safety' → ['cold email', 'construction safety']
    Returns list of sub-queries, or [query] if no split possible."""
    words = query.lower().split()
    if len(words) < 4:
        return [query]

    splits = []
    current = []
    for w in words:
        if w in _SPLIT_PREPS and len(current) >= 2:
            splits.append(" ".join(current))
            current = []
        else:
            current.append(w)
    if current:
        splits.append(" ".join(current))

    # Only split if we got 2+ meaningful sub-queries
    splits = [s for s in splits if len(s.split()) >= 2]
    return splits if len(splits) >= 2 else [query]


# C8: Abbreviation expansion
ABBREVIATIONS = {
    "ppe": "personal protective equipment",
    "anpr": "automatic number plate recognition",
    "cctv": "closed circuit television camera",
    "rtsp": "real time streaming protocol",
    "gpu": "graphics processing unit",
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "vlm": "vision language model",
    "sku": "stock keeping unit",
    "fmcg": "fast moving consumer goods",
    "roi": "return on investment",
    "gtm": "go to market",
    "bts": "behind the scenes",
    "ir": "infrared",
    "isp": "internet service provider",
}


def _expand_abbreviations(tokens):
    """C8: Expand known abbreviations into full forms."""
    expanded = set(tokens)
    for t in tokens:
        if t in ABBREVIATIONS:
            expanded.update(_tok(ABBREVIATIONS[t]))
    return expanded


# E6: SC description enrichment (computed once per retrieval)
def _enrich_sc_descriptions(conn, nodes_by_id, children_map):
    """E6: Auto-generate richer SC descriptions from top anchor titles."""
    for sc_id, sc in nodes_by_id.items():
        if sc["type"] != "super_context":
            continue
        if sc.get("description") and len(sc["description"]) > 30:
            continue  # already has a good description

        # Gather top anchor titles from children
        titles = []
        for ctx_id in children_map.get(sc_id, []):
            ctx = nodes_by_id.get(ctx_id)
            if ctx:
                titles.append(ctx["title"])
            for anc_id in children_map.get(ctx_id, []):
                anc = nodes_by_id.get(anc_id)
                if anc and anc.get("weight", 0) >= 0.5:
                    titles.append(anc["title"])
        if titles:
            sc["description"] = " | ".join(titles[:5])


# =============================================================================
# EXACT-MATCH BYPASS (P10: short query fast path)
# =============================================================================

def _exact_match_bypass(query, nodes_by_id, node_sc_map):
    """Short-circuit for very short or exact-match queries (1-3 tokens).
    Returns {sc_id: hit_score} or None if query is too long."""
    tokens = _tok(query)
    raw_lower = query.lower().strip()

    if len(tokens) > 3 or len(raw_lower) < 2:
        return None

    type_weights = {"anchor": 2.0, "context": 1.5, "super_context": 1.0}
    direct_hits = defaultdict(float)

    for nid, node in nodes_by_id.items():
        title_lower = node["title"].lower()
        content_lower = (node.get("content") or "").lower()

        # Check if raw query appears as substring in title or content
        if raw_lower in title_lower or raw_lower in content_lower:
            sc_id = node_sc_map.get(nid)
            if sc_id:
                tw = type_weights.get(node["type"], 1.0)
                # Exact title match gets extra bonus
                if raw_lower == title_lower or raw_lower in title_lower.split():
                    tw *= 1.5
                direct_hits[sc_id] += tw

    return dict(direct_hits) if direct_hits else None


# =============================================================================
# ADAPTIVE WEIGHTS (based on memory size)
# =============================================================================

def _get_weights(n_anchors):
    if n_anchors < 500:
        return 0.40, 0.35, 0.25  # tag, bm25, vec
    elif n_anchors < 5000:
        return 0.25, 0.45, 0.30
    else:
        return 0.10, 0.50, 0.40


# =============================================================================
# P9+ RETRIEVE — replaces cmd_retrieve in memory.py
# =============================================================================

def p9_retrieve(conn, query, top_k=3):
    """
    P9+ Triple Fusion retrieval.
    Returns the SAME JSON format as existing cmd_retrieve.
    """

    # Load all nodes
    all_nodes = [dict(r) for r in conn.execute("SELECT * FROM nodes").fetchall()]
    if not all_nodes:
        print(json.dumps({"query": query, "memories": [], "total_matches": 0}))
        return

    nodes_by_id = {n["id"]: n for n in all_nodes}
    sc_nodes = [n for n in all_nodes if n["type"] == "super_context"]
    n_anchors = sum(1 for n in all_nodes if n["type"] == "anchor")

    # Build parent/child maps
    edges = conn.execute("SELECT * FROM edges WHERE type='parent_child'").fetchall()
    children = defaultdict(list)
    parent_of = {}
    for e in edges:
        children[e["src"]].append(e["tgt"])
        parent_of[e["tgt"]] = e["src"]

    # Transfer edges
    transfers = defaultdict(list)
    for e in conn.execute("SELECT * FROM edges WHERE type='transfer'").fetchall():
        transfers[e["src"]].append((e["tgt"], e["weight"]))

    def find_sc(nid):
        n = nodes_by_id.get(nid)
        if not n:
            return None
        if n["type"] == "super_context":
            return nid
        cur, vis = nid, set()
        while cur in parent_of and cur not in vis:
            vis.add(cur)
            cur = parent_of[cur]
        p = nodes_by_id.get(cur)
        return cur if p and p["type"] == "super_context" else None

    # Build node texts (enriched with path)
    node_order = []
    node_sc_map = {}
    texts = []
    for n in all_nodes:
        node_order.append(n["id"])
        sc_id = find_sc(n["id"])
        node_sc_map[n["id"]] = sc_id

        # Enriched text: path + title + content snippet
        t = n["title"]
        if n.get("parent_id") or n["id"] in parent_of:
            pid = parent_of.get(n["id"])
            if pid and pid in nodes_by_id:
                t = nodes_by_id[pid]["title"] + " " + t
                sc_id2 = find_sc(pid)
                if sc_id2 and sc_id2 != pid and sc_id2 in nodes_by_id:
                    t = nodes_by_id[sc_id2]["title"] + " " + t
        if n.get("content") and n["content"] != n["title"]:
            t += " " + n["content"][:100]
        texts.append(t)

    # ── Build indices ──
    bm25 = BM25()
    bm25.build(texts)

    tags = TagEngine()
    tags.build(conn)

    vectors = VectorEngine()
    vectors.build(texts)

    # ── Query expansion: prior queries only, NOT SC titles ──
    expanded = query
    # (In Claude Desktop, conversation history would be passed here.
    #  For CLI, we just use the raw query.)

    query_tokens = _tok(expanded)
    query_tags = set(query_tokens)

    # Also add content words from query for tag matching
    for w in _tok(query):
        query_tags.add(w)

    tw = {"anchor": 1.5, "context": 1.2, "super_context": 0.8}
    w_tag, w_bm25, w_vec = _get_weights(n_anchors)

    # ── Tag search ──
    tag_anchor_scores = tags.score_anchors(query_tags)
    tag_sc = defaultdict(float)
    for aid, jscore in tag_anchor_scores.items():
        sc = node_sc_map.get(aid)
        if sc:
            node = nodes_by_id[aid]
            ew = mem.compute_evidence_weight(conn, aid)
            tag_sc[sc] += jscore * ew * tw.get("anchor", 1.5)

    # ── BM25 search (current query 60% + expanded 40%) ──
    current_tokens = _tok(query)
    bm25_sc = defaultdict(float)
    for i, nid in enumerate(node_order):
        sc = node_sc_map.get(nid)
        if not sc:
            continue
        # Current query emphasis (60/40 split)
        s_current = bm25.score(current_tokens, i)
        s_expanded = bm25.score(query_tokens, i)
        s = 0.6 * s_current + 0.4 * s_expanded

        node = nodes_by_id[nid]
        t_weight = tw.get(node["type"], 1.0)
        n_weight = node.get("weight", 0.5)
        if node["type"] == "anchor":
            n_weight = mem.compute_evidence_weight(conn, nid)
        bm25_sc[sc] += s * t_weight * n_weight

    # ── Vector search (optional) ──
    vec_sc = defaultdict(float)
    if vectors.active:
        sims = vectors.search(expanded)
        for i, sim in enumerate(sims):
            if sim < 0.04:
                continue
            nid = node_order[i]
            sc = node_sc_map.get(nid)
            if not sc:
                continue
            node = nodes_by_id[nid]
            t_weight = tw.get(node["type"], 1.0)
            n_weight = node.get("weight", 0.5)
            if node["type"] == "anchor":
                n_weight = mem.compute_evidence_weight(conn, nid)
            vec_sc[sc] += sim * t_weight * n_weight

    # ── Triple merge (adaptive weights + RRF) ──
    all_scs = set(list(tag_sc.keys()) + list(bm25_sc.keys()) + list(vec_sc.keys()))
    mt = max(tag_sc.values()) if tag_sc else 1
    mb = max(bm25_sc.values()) if bm25_sc else 1
    mv = max(vec_sc.values()) if vec_sc else 1

    # Phase A: weighted raw blend
    raw_scores = {}
    for sid in all_scs:
        tn = tag_sc.get(sid, 0) / mt if mt > 0 else 0
        bn = bm25_sc.get(sid, 0) / mb if mb > 0 else 0
        vn = vec_sc.get(sid, 0) / mv if mv > 0 else 0
        raw_scores[sid] = w_tag * tn + w_bm25 * bn + w_vec * vn

    # Phase B: RRF
    K = 60
    rrf = defaultdict(float)
    for rank, (sid, _) in enumerate(sorted(tag_sc.items(), key=lambda x: x[1], reverse=True)):
        rrf[sid] += 1 / (K + rank + 1)
    for rank, (sid, _) in enumerate(sorted(bm25_sc.items(), key=lambda x: x[1], reverse=True)):
        rrf[sid] += 1 / (K + rank + 1)
    for rank, (sid, _) in enumerate(sorted(vec_sc.items(), key=lambda x: x[1], reverse=True)):
        rrf[sid] += 1 / (K + rank + 1)

    merged = {}
    for sid in all_scs:
        merged[sid] = 0.70 * raw_scores.get(sid, 0) + 0.30 * rrf.get(sid, 0) * 100

    # ── Boosts (temporal + mode + quality) ──
    for sid in merged:
        sc = nodes_by_id.get(sid)
        if not sc:
            continue
        # Temporal
        t_boost = mem.temporal_boost(sc.get("memory_stage", "impulse"))

        # Mode
        query_mode = mem.infer_mode_vector(query, [])
        try:
            node_mode = json.loads(sc.get("mode_vector") or "{}")
        except (json.JSONDecodeError, TypeError):
            node_mode = {}
        m_boost = 0.5 + mem.mode_similarity(query_mode, node_mode) if node_mode else 1.0

        # Quality (tree-level: best context quality)
        best_ctx_q = 0.5
        for ctx_id in children.get(sid, []):
            anc_weights = []
            for anc_id in children.get(ctx_id, []):
                ew = mem.compute_evidence_weight(conn, anc_id)
                anc_weights.append(ew)
            if anc_weights:
                ctx_q = sum(anc_weights) / len(anc_weights)
                best_ctx_q = max(best_ctx_q, ctx_q)
        quality_bonus = best_ctx_q * 0.25

        # Feedback bonus
        fb_bonus = 0
        sc_quality = sc.get("quality", 0)
        if sc_quality > 0:
            fb_bonus = (sc_quality / 5.0) * 0.15

        merged[sid] = (merged[sid] + quality_bonus + fb_bonus) * t_boost * m_boost

    # ── Transfer edges ──
    for sid, s in list(merged.items()):
        for partner_id, tw_val in transfers.get(sid, []):
            if partner_id not in merged:
                merged[partner_id] = s * 0.4 * tw_val

    # ── Build output (SAME FORMAT as existing cmd_retrieve) ──
    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    context_pack = {"query": query, "memories": [], "total_matches": len(ranked)}

    for score, (sc_id, sc_score) in enumerate(ranked[:top_k]):
        sc = nodes_by_id.get(sc_id)
        if not sc:
            continue

        winning = mem._get_winning_structure(conn, sc_id)

        memory = {
            "super_context": sc["title"],
            "sc_id": sc_id,
            "relevance": round(sc_score, 2),
            "quality": sc.get("quality", 0),
            "uses": sc.get("use_count", 0),
            "stage": sc.get("memory_stage", "impulse"),
            "temporal_boost": mem.temporal_boost(sc.get("memory_stage", "impulse")),
            "winning_score": winning.get("score", 0) if winning else 0,
            "retrieval_method": "p9_triple_fusion",
            "merge_weights": {"tag": w_tag, "bm25": w_bm25, "vec": w_vec},
            "contexts": []
        }

        # Decisions (same as existing)
        dec_rows = conn.execute(
            "SELECT decision, confidence, reversible FROM decisions WHERE sc_id=? AND confidence > 0.3 ORDER BY confidence DESC",
            (sc_id,)
        ).fetchall()
        if dec_rows:
            memory["decisions"] = [{"decision": d["decision"], "confidence": round(d["confidence"], 2), "locked": not d["reversible"]} for d in dec_rows]
            memory["convergence"] = mem.get_convergence_state(conn, sc_id)["convergence"]

        # Winning anchors
        winning_anchors = set()
        if winning and winning.get("snapshot"):
            for ctx_name, anc_list in winning["snapshot"].items():
                for a in anc_list:
                    winning_anchors.add(a.lower() if isinstance(a, str) else "")

        # Build context trees (sorted by evidence quality)
        ctx_entries = []
        for ctx in mem.get_children(conn, sc_id):
            ctx_entry = {
                "context": ctx["title"],
                "weight": round(ctx.get("edge_weight", ctx.get("weight", 0.5)), 2),
                "stage": ctx.get("memory_stage", "impulse"),
                "anchors": []
            }

            anc_evidence_weights = []
            for anc in mem.get_children(conn, ctx["id"]):
                ev_weight = mem.compute_evidence_weight(conn, anc["id"])
                in_winning = anc["title"].lower() in winning_anchors
                adp_val = anc.get("discrimination_power", 0.0)

                anc_entry = {
                    "title": anc["title"],
                    "content": anc.get("content", ""),
                    "weight": ev_weight,
                    "stage": anc.get("memory_stage", "impulse"),
                    "in_winning_structure": in_winning,
                    "discrimination_power": adp_val,
                    "variant_sensitive": adp_val > 1.0,
                }

                # ADP execution params (same as existing)
                if adp_val > 1.0:
                    best = conn.execute("""
                        SELECT ta.execution_params, t.score FROM task_anchors ta
                        JOIN tasks t ON t.id = ta.task_id
                        WHERE ta.anchor_id = ? AND t.score > 0
                        ORDER BY t.score DESC LIMIT 1
                    """, (anc["id"],)).fetchone()
                    if best:
                        try:
                            anc_entry["best_execution"] = {"params": json.loads(best["execution_params"] or "{}"), "score": best["score"]}
                        except (json.JSONDecodeError, TypeError):
                            pass

                ctx_entry["anchors"].append(anc_entry)
                anc_evidence_weights.append(ev_weight)

            # Sort: winning first, then by evidence weight
            ctx_entry["anchors"].sort(
                key=lambda a: (a["in_winning_structure"], a["weight"]),
                reverse=True
            )

            # Context quality from evidence
            ctx_entry["quality"] = round(sum(anc_evidence_weights) / len(anc_evidence_weights), 3) if anc_evidence_weights else 0.5
            ctx_entries.append(ctx_entry)

        # Sort contexts by quality (best tree first)
        ctx_entries.sort(key=lambda c: c["quality"], reverse=True)
        memory["contexts"] = ctx_entries
        context_pack["memories"].append(memory)

    # Transfer knowledge (same as existing)
    seen_sc_ids = {m["sc_id"] for m in context_pack["memories"]}
    for mem_entry in list(context_pack["memories"]):
        transfer_rows = conn.execute("""
            SELECT e.tgt as tgt_id, e.weight as strength, n.title as tgt_title
            FROM edges e JOIN nodes n ON n.id = e.tgt
            WHERE e.src = ? AND e.type = 'transfer'
            ORDER BY e.weight DESC LIMIT 2
        """, (mem_entry["sc_id"],)).fetchall()

        for t in transfer_rows:
            if t["tgt_id"] not in seen_sc_ids:
                tgt_sc = conn.execute("SELECT * FROM nodes WHERE id=?", (t["tgt_id"],)).fetchone()
                if tgt_sc:
                    transfer_mem = {
                        "super_context": tgt_sc["title"],
                        "sc_id": tgt_sc["id"],
                        "transferred_from": mem_entry["super_context"],
                        "transfer_strength": t["strength"],
                        "relevance": round(mem_entry["relevance"] * 0.6, 2),
                        "quality": tgt_sc["quality"],
                        "contexts": []
                    }
                    for ctx in mem.get_children(conn, tgt_sc["id"]):
                        ctx_entry = {"context": ctx["title"], "weight": round(ctx.get("weight", 0.5), 2), "anchors": []}
                        for anc in mem.get_children(conn, ctx["id"])[:5]:
                            ctx_entry["anchors"].append({
                                "title": anc["title"], "content": anc.get("content", ""),
                                "weight": round(anc.get("weight", 0.5) * 0.6, 2),
                                "transferred": True
                            })
                        if ctx_entry["anchors"]:
                            transfer_mem["contexts"].append(ctx_entry)
                    if transfer_mem["contexts"]:
                        context_pack["memories"].append(transfer_mem)
                        seen_sc_ids.add(t["tgt_id"])

    # Record retrieval (same as existing)
    mem._record_retrieval(conn, context_pack)

    print(json.dumps(context_pack, indent=2))


# =============================================================================
# SESSION CONTINUITY ENGINE
# =============================================================================

class SessionEngine:
    """Tracks conversation context and detects divergence."""

    def __init__(self, decay=0.85, divergence_threshold=0.35):
        self.decay = decay
        self.divergence_threshold = divergence_threshold
        self.session_vector = Counter()  # running word-frequency vector
        self.query_history = []
        self.divergence_log = []

    def update(self, query):
        """Add query to session. Returns (continuity_score, diverged)."""
        tokens = _tok(query)
        query_vec = Counter(tokens)

        if not self.session_vector:
            # First query — initialize
            self.session_vector = Counter(query_vec)
            self.query_history.append(query)
            return 1.0, False

        # Cosine similarity between session vector and new query
        similarity = self._cosine(self.session_vector, query_vec)

        # Decay old context, blend in new
        decayed = Counter()
        for k, v in self.session_vector.items():
            decayed[k] = v * self.decay
        for k, v in query_vec.items():
            decayed[k] = decayed.get(k, 0) + v * (1 - self.decay)
        self.session_vector = decayed

        self.query_history.append(query)

        diverged = similarity < self.divergence_threshold
        if diverged:
            self.divergence_log.append({
                "at_query": len(self.query_history),
                "similarity": round(similarity, 4),
                "from": self.query_history[-2] if len(self.query_history) > 1 else "",
                "to": query,
            })
            # P10: Auto-break — reset context to new query only
            self.session_vector = Counter(query_vec)

        return round(similarity, 4), diverged

    def accumulate_results(self, memories):
        """P10: Feed retrieved results back to build richer context vector."""
        for m in memories:
            for w in _tok(m.get("super_context", "")):
                self.session_vector[w] += 0.5
            for ctx in m.get("contexts", []):
                for w in _tok(ctx.get("context", "")):
                    self.session_vector[w] += 0.3
                for anc in ctx.get("anchors", [])[:3]:  # top 3 anchors only
                    for w in _tok(anc.get("title", "")):
                        self.session_vector[w] += 0.2

    def get_context_boost(self, sc_title, anchor_titles):
        """Boost SCs/anchors that align with session context."""
        if not self.session_vector:
            return 1.0
        sc_tokens = Counter(_tok(sc_title))
        for t in anchor_titles:
            sc_tokens.update(_tok(t))
        return max(0.5, min(2.0, 0.5 + self._cosine(self.session_vector, sc_tokens)))

    def _cosine(self, a, b):
        if not a or not b:
            return 0.0
        keys = set(list(a.keys()) + list(b.keys()))
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        mag_a = sum(v**2 for v in a.values()) ** 0.5
        mag_b = sum(v**2 for v in b.values()) ** 0.5
        if mag_a < 0.001 or mag_b < 0.001:
            return 0.0
        return dot / (mag_a * mag_b)

    def export_state(self):
        return {
            "queries": len(self.query_history),
            "divergences": len(self.divergence_log),
            "divergence_log": self.divergence_log,
            "top_context_words": [w for w, _ in self.session_vector.most_common(10)],
        }


# Global session engine (persists across calls within same process)
_session = SessionEngine()


# =============================================================================
# GOAL VECTOR ENGINE
# =============================================================================

# Goal categories for memory alignment
GOAL_TEMPLATES = {
    "build": {"create": 0.9, "deliver": 0.7, "learn": 0.2, "decide": 0.3, "optimize": 0.4},
    "design": {"create": 0.8, "deliver": 0.5, "learn": 0.3, "decide": 0.4, "optimize": 0.3},
    "research": {"create": 0.2, "deliver": 0.1, "learn": 0.9, "decide": 0.5, "optimize": 0.3},
    "fix": {"create": 0.1, "deliver": 0.8, "learn": 0.3, "decide": 0.2, "optimize": 0.7},
    "optimize": {"create": 0.1, "deliver": 0.4, "learn": 0.5, "decide": 0.3, "optimize": 0.9},
    "plan": {"create": 0.3, "deliver": 0.2, "learn": 0.4, "decide": 0.9, "optimize": 0.3},
    "write": {"create": 0.8, "deliver": 0.6, "learn": 0.2, "decide": 0.3, "optimize": 0.2},
    "analyze": {"create": 0.1, "deliver": 0.3, "learn": 0.8, "decide": 0.6, "optimize": 0.5},
    "sell": {"create": 0.3, "deliver": 0.8, "learn": 0.1, "decide": 0.4, "optimize": 0.6},
    "deploy": {"create": 0.2, "deliver": 0.9, "learn": 0.1, "decide": 0.3, "optimize": 0.5},
}

DEFAULT_GOAL = {"create": 0.5, "deliver": 0.5, "learn": 0.5, "decide": 0.5, "optimize": 0.5}


def infer_goal_vector(query):
    """Infer what the user is trying to achieve from the query."""
    q = query.lower()
    for keyword, template in GOAL_TEMPLATES.items():
        if keyword in q:
            return dict(template)
    # Check for implicit goals
    if any(w in q for w in ["how", "what", "why", "understand", "learn", "explain"]):
        return dict(GOAL_TEMPLATES["research"])
    if any(w in q for w in ["create", "make", "add", "implement", "new"]):
        return dict(GOAL_TEMPLATES["build"])
    if any(w in q for w in ["improve", "better", "faster", "reduce"]):
        return dict(GOAL_TEMPLATES["optimize"])
    if any(w in q for w in ["send", "email", "outreach", "pitch"]):
        return dict(GOAL_TEMPLATES["sell"])
    return dict(DEFAULT_GOAL)


def goal_alignment(query_goal, sc_mode_vector):
    """How well does the SC's domain align with the user's current goal?"""
    if not query_goal or not sc_mode_vector:
        return 0.5
    # Map mode dimensions to goal dimensions heuristically
    mode_to_goal = {
        "work": "deliver", "creative": "create", "learning": "learn",
        "social": "deliver", "travel": "deliver", "home": "deliver",
    }
    mapped = {}
    for mk, mv in sc_mode_vector.items():
        gk = mode_to_goal.get(mk, "deliver")
        mapped[gk] = mapped.get(gk, 0) + mv

    # Cosine between goal and mapped SC
    keys = set(list(query_goal.keys()) + list(mapped.keys()))
    dot = sum(query_goal.get(k, 0) * mapped.get(k, 0) for k in keys)
    mag_a = sum(v**2 for v in query_goal.values()) ** 0.5
    mag_b = sum(v**2 for v in mapped.values()) ** 0.5
    if mag_a < 0.001 or mag_b < 0.001:
        return 0.5
    return round(dot / (mag_a * mag_b), 4)


# =============================================================================
# PARAMETERIZED P9+ RETRIEVE
# =============================================================================

def p9_retrieve_parameterized(conn, query, params=None, session=None, top_k=3):
    """
    P9+ with tunable parameters, session continuity, and goal alignment.
    params: dict of tunable parameters (from autoresearch)
    session: SessionEngine instance (for continuity tracking)
    Returns same JSON format as p9_retrieve.
    """
    if params is None:
        params = {}

    # Extract params with defaults
    p_k1 = params.get("bm25_k1", 1.5)
    p_b = params.get("bm25_b", 0.75)
    p_tag_w = params.get("tag_weight", 0.40)
    p_bm25_w = params.get("bm25_weight", 0.35)
    p_vec_w = params.get("vec_weight", 0.25)
    p_tag_boost = params.get("tag_jaccard_boost", 1.0)
    p_tag_partial = params.get("tag_partial_weight", 0.3)
    p_boost_impulse = params.get("boost_impulse", 0.8)
    p_boost_active = params.get("boost_active", 1.2)
    p_boost_established = params.get("boost_established", 1.5)
    p_boost_fading = params.get("boost_fading", 0.5)
    p_anchor_cw = params.get("anchor_content_weight", 0.5)
    p_goal_w = params.get("goal_weight_in_ranking", 0.15)
    p_max_results = int(params.get("max_results", top_k))

    stage_boosts = {
        "impulse": p_boost_impulse, "active": p_boost_active,
        "established": p_boost_established, "fading": p_boost_fading,
    }

    # Session continuity
    if session is None:
        session = _session
    continuity_score, diverged = session.update(query)

    # Goal vector
    query_goal = infer_goal_vector(query)

    # ── Load all nodes ──
    all_nodes = [dict(r) for r in conn.execute("SELECT * FROM nodes").fetchall()]
    if not all_nodes:
        result = {"query": query, "memories": [], "total_matches": 0,
                  "session": session.export_state(), "goal": query_goal}
        print(json.dumps(result, indent=2))
        return result

    nodes_by_id = {n["id"]: n for n in all_nodes}
    sc_nodes = [n for n in all_nodes if n["type"] == "super_context"]

    edges = conn.execute("SELECT * FROM edges WHERE type='parent_child'").fetchall()
    children_map = defaultdict(list)
    parent_of = {}
    for e in edges:
        children_map[e["src"]].append(e["tgt"])
        parent_of[e["tgt"]] = e["src"]

    def find_sc(nid):
        n = nodes_by_id.get(nid)
        if not n: return None
        if n["type"] == "super_context": return nid
        cur, vis = nid, set()
        while cur in parent_of and cur not in vis:
            vis.add(cur); cur = parent_of[cur]
        p = nodes_by_id.get(cur)
        return cur if p and p["type"] == "super_context" else None

    # Build texts
    node_order = []
    node_sc_map = {}
    texts = []
    for n in all_nodes:
        node_order.append(n["id"])
        sc_id = find_sc(n["id"])
        node_sc_map[n["id"]] = sc_id
        t = n["title"]
        pid = parent_of.get(n["id"])
        if pid and pid in nodes_by_id:
            t = nodes_by_id[pid]["title"] + " " + t
            sc_id2 = find_sc(pid)
            if sc_id2 and sc_id2 != pid and sc_id2 in nodes_by_id:
                t = nodes_by_id[sc_id2]["title"] + " " + t
        if n.get("content") and n["content"] != n["title"]:
            content_limit = int(100 * p_anchor_cw / 0.5)  # Scale with param
            t += " " + n["content"][:content_limit]
        texts.append(t)

    # ── P10: Exact-match bypass for short queries ──
    exact_hits = _exact_match_bypass(query, nodes_by_id, node_sc_map)

    # ── Build indices with PARAMS ──
    bm25 = BM25(k1=p_k1, b=p_b)
    bm25.build(texts)

    tag_engine = TagEngine()
    tag_engine.build(conn)

    vec_engine = VectorEngine()
    vec_engine.build(texts)

    # ── P11: Enhanced query preprocessing ──
    raw_tokens = _tok(query)
    # C12: Morphological stemming
    stemmed_tokens = _stem_tokens(raw_tokens)
    # C1: Synonym expansion
    expanded_tokens = _expand_synonyms(stemmed_tokens)
    # C8: Abbreviation expansion
    expanded_tokens = _expand_abbreviations(expanded_tokens)

    query_tokens = list(expanded_tokens)
    query_tags = expanded_tokens

    # C9: Compound query splitting for cross-domain
    sub_queries = _split_compound_query(query)

    # E6: Enrich SC descriptions for better BM25 matching
    _enrich_sc_descriptions(conn, nodes_by_id, children_map)

    tw = {"anchor": 1.5, "context": 1.2, "super_context": 0.8}

    # ── Tag search with PARAM boost (using expanded tags) ──
    tag_anchor_scores = tag_engine.score_anchors(query_tags)
    tag_sc = defaultdict(float)
    for aid, jscore in tag_anchor_scores.items():
        sc = node_sc_map.get(aid)
        if sc:
            ew = mem.compute_evidence_weight(conn, aid)
            tag_sc[sc] += jscore * p_tag_boost * ew * tw.get("anchor", 1.5)

    # ── BM25 search (C9: also score sub-queries for cross-domain) ──
    bm25_sc = defaultdict(float)
    # Score with expanded tokens + sub-query tokens
    all_bm25_tokens = [query_tokens]
    for sq in sub_queries:
        sq_tokens = list(_expand_synonyms(_stem_tokens(_tok(sq))))
        if sq_tokens != query_tokens:
            all_bm25_tokens.append(sq_tokens)

    for i, nid in enumerate(node_order):
        sc = node_sc_map.get(nid)
        if not sc: continue
        # Take max score across main query and sub-queries
        s = max(bm25.score(toks, i) for toks in all_bm25_tokens)
        if s <= 0: continue
        node = nodes_by_id[nid]
        t_weight = tw.get(node["type"], 1.0)
        n_weight = node.get("weight", 0.5)
        if node["type"] == "anchor":
            n_weight = mem.compute_evidence_weight(conn, nid)
        bm25_sc[sc] += s * t_weight * n_weight

    # ── Vector search ──
    vec_sc = defaultdict(float)
    if vec_engine.active:
        sims = vec_engine.search(query)
        for i, sim in enumerate(sims):
            if sim < 0.04: continue
            nid = node_order[i]
            sc = node_sc_map.get(nid)
            if not sc: continue
            node = nodes_by_id[nid]
            t_weight = tw.get(node["type"], 1.0)
            n_weight = node.get("weight", 0.5)
            if node["type"] == "anchor":
                n_weight = mem.compute_evidence_weight(conn, nid)
            vec_sc[sc] += sim * t_weight * n_weight

    # ── Triple merge with PARAM weights ──
    all_scs = set(list(tag_sc.keys()) + list(bm25_sc.keys()) + list(vec_sc.keys()))
    mt = max(tag_sc.values()) if tag_sc else 1
    mb = max(bm25_sc.values()) if bm25_sc else 1
    mv = max(vec_sc.values()) if vec_sc else 1

    raw_scores = {}
    for sid in all_scs:
        tn = tag_sc.get(sid, 0) / mt if mt > 0 else 0
        bn = bm25_sc.get(sid, 0) / mb if mb > 0 else 0
        vn = vec_sc.get(sid, 0) / mv if mv > 0 else 0
        raw_scores[sid] = p_tag_w * tn + p_bm25_w * bn + p_vec_w * vn

    K = 60
    rrf = defaultdict(float)
    for rank, (sid, _) in enumerate(sorted(tag_sc.items(), key=lambda x: x[1], reverse=True)):
        rrf[sid] += 1 / (K + rank + 1)
    for rank, (sid, _) in enumerate(sorted(bm25_sc.items(), key=lambda x: x[1], reverse=True)):
        rrf[sid] += 1 / (K + rank + 1)
    for rank, (sid, _) in enumerate(sorted(vec_sc.items(), key=lambda x: x[1], reverse=True)):
        rrf[sid] += 1 / (K + rank + 1)

    merged = {}
    # P11 A15: SC size normalization — prevent large SCs from dominating
    sc_sizes = {}
    for sid in all_scs:
        n_anchors = sum(len(children_map.get(ctx_id, [])) for ctx_id in children_map.get(sid, []))
        sc_sizes[sid] = max(n_anchors, 1)

    for sid in all_scs:
        raw = 0.70 * raw_scores.get(sid, 0) + 0.30 * rrf.get(sid, 0) * 100
        # Normalize by sqrt of anchor count to dampen large-SC advantage
        size_norm = math.sqrt(sc_sizes.get(sid, 1))
        if size_norm > 3:  # only normalize SCs with >9 anchors
            raw = raw / (size_norm / 3)
        merged[sid] = raw

    # ── P10: Blend exact-match hits into merged scores ──
    if exact_hits:
        for sid, ex_score in exact_hits.items():
            merged[sid] = merged.get(sid, 0) + ex_score * 2.0  # strong boost for direct hits
            if sid not in all_scs:
                all_scs.add(sid)

    # ── Boosts: temporal (PARAMETERIZED) + mode + quality + goal + session ──
    for sid in merged:
        sc = nodes_by_id.get(sid)
        if not sc: continue

        # Temporal boost (parameterized)
        t_boost = stage_boosts.get(sc.get("memory_stage", "impulse"), 1.0)

        # Mode boost
        query_mode = mem.infer_mode_vector(query, [])
        try:
            node_mode = json.loads(sc.get("mode_vector") or "{}")
        except (json.JSONDecodeError, TypeError):
            node_mode = {}
        m_boost = 0.5 + mem.mode_similarity(query_mode, node_mode) if node_mode else 1.0

        # Quality
        best_ctx_q = 0.5
        for ctx_id in children_map.get(sid, []):
            anc_weights = []
            for anc_id in children_map.get(ctx_id, []):
                ew = mem.compute_evidence_weight(conn, anc_id)
                anc_weights.append(ew)
            if anc_weights:
                best_ctx_q = max(best_ctx_q, sum(anc_weights) / len(anc_weights))

        # Goal alignment (NEW)
        try:
            sc_mode = json.loads(sc.get("mode_vector") or "{}")
        except (json.JSONDecodeError, TypeError):
            sc_mode = {}
        g_align = goal_alignment(query_goal, sc_mode)
        goal_bonus = g_align * p_goal_w

        # Session continuity boost (NEW)
        anchor_titles = []
        for ctx_id in children_map.get(sid, []):
            for anc_id in children_map.get(ctx_id, []):
                if anc_id in nodes_by_id:
                    anchor_titles.append(nodes_by_id[anc_id]["title"])
        session_boost = session.get_context_boost(sc["title"], anchor_titles)

        # Feedback reinforcement (P10: multiplicative, scales with evidence depth)
        fb_multiplier = 1.0
        sc_quality = sc.get("quality", 0)
        if sc_quality > 0:
            n_scored = conn.execute(
                "SELECT COUNT(*) c FROM tasks WHERE sc_id=? AND score > 0 AND completed IS NOT NULL",
                (sid,)
            ).fetchone()["c"]
            if n_scored > 0:
                confidence = min(1.0, n_scored / 5.0)  # ramp over 5 scored tasks
                p_fb_amp = params.get("feedback_amplifier", 1.0)
                p_quality_floor = params.get("quality_floor", 0.3)
                fb_multiplier = p_quality_floor + (sc_quality / 5.0) * p_fb_amp * confidence

        merged[sid] = (merged[sid] + best_ctx_q * 0.25 + goal_bonus) * t_boost * m_boost * session_boost * fb_multiplier

    # ── Context tightening for multi-turn (P10) ──
    if len(session.query_history) > 1 and not diverged:
        p_tighten = params.get("tighten_rate", 0.15)
        p_min_overlap = params.get("tighten_min_overlap", 0.2)
        ctx_words = set(session.session_vector.keys())
        if ctx_words:
            for sid in list(merged.keys()):
                sc = nodes_by_id.get(sid)
                if not sc:
                    continue
                sc_words = set(_tok(sc["title"]))
                for ctx_id in children_map.get(sid, []):
                    if ctx_id in nodes_by_id:
                        sc_words.update(_tok(nodes_by_id[ctx_id]["title"]))
                    for anc_id in children_map.get(ctx_id, []):
                        if anc_id in nodes_by_id:
                            sc_words.update(_tok(nodes_by_id[anc_id]["title"])[:3])
                overlap = len(sc_words & ctx_words) / max(len(sc_words), 1)
                if overlap < p_min_overlap:
                    merged[sid] *= (1 - p_tighten)

    # ── Transfer edges ──
    transfers_map = defaultdict(list)
    for e in conn.execute("SELECT * FROM edges WHERE type='transfer'").fetchall():
        transfers_map[e["src"]].append((e["tgt"], e["weight"]))
    for sid, s in list(merged.items()):
        for partner_id, tw_val in transfers_map.get(sid, []):
            if partner_id not in merged:
                merged[partner_id] = s * 0.4 * tw_val

    # ── Build output ──
    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    context_pack = {
        "query": query,
        "memories": [],
        "total_matches": len(ranked),
        "retrieval_method": "p9_parameterized",
        "session": {
            "continuity": continuity_score,
            "diverged": diverged,
            "query_count": len(session.query_history),
            "context_words": [w for w, _ in session.session_vector.most_common(5)],
        },
        "goal": query_goal,
    }

    for score_idx, (sc_id, sc_score) in enumerate(ranked[:p_max_results]):
        sc = nodes_by_id.get(sc_id)
        if not sc: continue

        winning = mem._get_winning_structure(conn, sc_id)
        winning_anchors = set()
        if winning and winning.get("snapshot"):
            for ctx_name, anc_list in winning["snapshot"].items():
                for a in anc_list:
                    winning_anchors.add(a.lower() if isinstance(a, str) else "")

        memory_entry = {
            "super_context": sc["title"],
            "sc_id": sc_id,
            "relevance": round(sc_score, 2),
            "quality": sc.get("quality", 0),
            "uses": sc.get("use_count", 0),
            "stage": sc.get("memory_stage", "impulse"),
            "temporal_boost": stage_boosts.get(sc.get("memory_stage", "impulse"), 1.0),
            "winning_score": winning.get("score", 0) if winning else 0,
            "retrieval_method": "p9_parameterized",
            "merge_weights": {"tag": p_tag_w, "bm25": p_bm25_w, "vec": p_vec_w},
            "contexts": []
        }

        # Decisions
        dec_rows = conn.execute(
            "SELECT decision, confidence, reversible FROM decisions WHERE sc_id=? AND confidence > 0.3 ORDER BY confidence DESC",
            (sc_id,)
        ).fetchall()
        if dec_rows:
            memory_entry["decisions"] = [{"decision": d["decision"], "confidence": round(d["confidence"], 2), "locked": not d["reversible"]} for d in dec_rows]
            memory_entry["convergence"] = mem.get_convergence_state(conn, sc_id)["convergence"]

        # Contexts & anchors
        for ctx in mem.get_children(conn, sc_id):
            ctx_entry = {
                "context": ctx["title"],
                "weight": round(ctx.get("edge_weight", ctx.get("weight", 0.5)), 2),
                "stage": ctx.get("memory_stage", "impulse"),
                "anchors": []
            }
            for anc in mem.get_children(conn, ctx["id"]):
                ev_weight = mem.compute_evidence_weight(conn, anc["id"])
                in_winning = anc["title"].lower() in winning_anchors
                ctx_entry["anchors"].append({
                    "title": anc["title"],
                    "content": anc.get("content", ""),
                    "weight": ev_weight,
                    "stage": anc.get("memory_stage", "impulse"),
                    "in_winning_structure": in_winning,
                    "discrimination_power": anc.get("discrimination_power", 0.0),
                    "variant_sensitive": anc.get("discrimination_power", 0.0) > 1.0,
                })
            ctx_entry["anchors"].sort(key=lambda a: (a["in_winning_structure"], a["weight"]), reverse=True)
            memory_entry["contexts"].append(ctx_entry)

        context_pack["memories"].append(memory_entry)

    mem._record_retrieval(conn, context_pack)

    # P10: Accumulate results back into session context for next turn
    session.accumulate_results(context_pack.get("memories", []))

    print(json.dumps(context_pack, indent=2))
    return context_pack


# =============================================================================
# CLI — can be called directly for testing
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "retrieve":
        conn = mem.get_db()
        try:
            p9_retrieve(conn, sys.argv[2])
        finally:
            conn.close()
    else:
        print("Usage: python3 scripts/p9_retrieve.py retrieve \"your query here\"")
        print("Or import and call: from p9_retrieve import p9_retrieve")
