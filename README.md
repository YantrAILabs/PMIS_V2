# PMIS

### A personal operating system for work that ships.

Surprise-gated memory + project tracking + work sessions + goal-driven learning, in one stack. Every anchor lives under a project; every project has a goal; every goal's success signal reweights memory via a nightly HGCN pass. Storage isn't passive — PMIS *learns which memories predict which deliverables ship*.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#)

---

## The layers

```
   Goals           ─► success signal
      │            
   Projects        ─► scoped work containers
      │
   Deliverables   ─► concrete things you ship
      │
   Work sessions  ─► live activity tracking + pre-work briefs
      │
   Memory graph   ─► surprise-gated anchors in a hyperbolic tree
      │
   HGCN (nightly) ─► outcome-driven weight update
```

Most memory libraries stop at the bottom layer. Most PM tools stop at the top two. PMIS does all five, and they feed back into each other.

---

## What makes it different

| | Typical memory library | Typical PM tool | **PMIS** |
|---|---|---|---|
| Storage | Everything | — | Surprise-gated (Free Energy Principle) |
| Structure | Flat vector store | Flat DB | Hyperbolic hierarchy with cell division |
| Retrieval | Fixed top-k | Keyword | γ-blended explore/exploit, per-query |
| Weights | Static | None | Goal-signal → HGCN nightly update |
| Capture | Manual | Manual | Passive screen-pulse feeds memory |
| PM | — | Tasks + deadlines | Projects / deliverables / goals linked to memory |
| Learning | None | None | Value score per node; thumbs retrain weights |

Seven mechanisms. No competitor combines more than three.

---

## Quickstart

### 🧑‍💻 For developers

```bash
git clone https://github.com/YantrAILabs/PMIS_V2.git && cd PMIS_V2
pip install -e .
cp .env.template .env  # edit for OPENAI_API_KEY if you're not using Ollama
```

```python
import asyncio, pmis

async def main():
    # Memory layer — the 5-verb substrate
    nid = await pmis.ingest("CISOs respond to threat language 3x over ROI")
    await pmis.attach(nid)
    hits = await pmis.retrieve("cold outreach learnings", mode="predictive", k=5)
    await pmis.consolidate()         # nightly: HGCN + compress + promote + prune

    # Sugar
    await pmis.remember("I learned X")
    answers = await pmis.ask("what did I learn about X?")

asyncio.run(main())
```

Full walkthrough: [examples/quickstart.py](examples/quickstart.py).

### 📒 For productivity users (macOS + Claude Desktop)

```bash
./install.sh              # installs tracker daemon + Claude Desktop MCP config
./start.sh                # launches PMIS servers on :8100 and :8200
```

Then in Claude Desktop:

- *"What am I working on right now?"* → live work-session status
- *"Brief me before I start on the Vision OS launch"* → Claude-can-do / You-did-before pre-work brief
- *"Remember that CISOs respond to threat language"* → surprise-gated memory write
- *"Thumbs-up that match"* → reweights memory via goal signal
- *"Show the review queue"* → opens `/wiki/review`

Setup guide: [examples/claude_desktop_mcp.md](examples/claude_desktop_mcp.md).

---

## The 5-verb memory API

| Verb | What it does | CLI |
|---|---|---|
| `ingest(text)` | Embed + surprise-gate. Returns node_id or None. | `pmis ingest "text"` |
| `attach(node_id=...)` | Promote orphan into the hyperbolic hierarchy. | `pmis attach --node <id>` |
| `retrieve(query, mode=...)` | γ-blended search. | `pmis retrieve "query" --mode predictive` |
| `consolidate()` | Nightly: HGCN training · compress · promote · birth · prune · cell-divide. | `pmis consolidate` |
| `delete(node_id=..., all=False)` | Soft-delete or reset. | `pmis delete --node <id>` |

Plus `remember(text)` = `ingest` + `attach`, and `ask(query)` = `retrieve` with auto mode.

All async. See [pmis/api.py](pmis/api.py).

---

## The PM layer (REST)

The memory engine is one half; PMIS also exposes a full work-management surface:

| Endpoint | Purpose |
|---|---|
| `POST /api/pm/projects` · `/api/pm/deliverables` · `/api/pm/goals` | CRUD for the project tree |
| `POST /api/pm/quick-add` | Natural-language "add this as a project / deliverable / goal" |
| `POST /api/work/start` · `/confirm` · `/end` | Live work-session lifecycle |
| `GET /api/work/brief` | Pre-work brief: Claude-can-do + You-did-before |
| `POST /api/work/compose-problem` | Meta-LLM composer that writes a `problem_statement.md` for a harness |
| `GET /api/work/current` · `/deliverables` · `/sessions` | Read work state |
| `POST /api/match/{id}/thumbs` | Thumbs-up/down on activity↔deliverable matches |
| `GET /api/review/pending` · `POST /api/review/{id}/confirm|reject` | Review queue for pending anchors |
| `POST /api/value/recompute` · `GET /api/node/{id}/value` | Outcome-driven value score |
| `POST /api/project/{id}/consolidate-{preview,day}` | Per-project manual consolidation |

Full interactive spec at `http://localhost:8100/docs`.

---

## Servers & dashboards

PMIS runs **two** FastAPI servers side by side:

| Port | Server | What's on it |
|---|---|---|
| **8100** | [pmis_v2/server.py](pmis_v2/server.py) | Public API + `/wiki/*` (Goals, Review, Productivity, per-node pages) |
| **8200** | [pmis_v2/health_dashboard.py](pmis_v2/health_dashboard.py) | Health + Feedback + Diagnostics + Lint + one-click actions (train-hgcn, consolidate, cell-divide, reindex, regen-wiki) |

The **Wiki** is LLM-generated prose rather than a dashboard — each node has a readable page (with a backend data panel one click away). Goals, Review, and Productivity are rich server-rendered pages.

---

## Architecture

```
 User message / screen pulse
      │
      ▼
 Embed (Euclidean 768d + Hyperbolic 16d + Temporal)
      │
      ▼
 Surprise  ──►  γ  ──►  Retrieve (γ-blended)  ──►  Compose
                           │
                           ▼
                     Storage gate
                           │
                           ▼
         ┌──── SQLite + ChromaDB ANN ────┐
         │                               │
         ▼                               ▼
     Projects                     Anchors (tree)
     Deliverables                  │
     Work sessions   ──match──►    │
     Goals ──success signal──►     │
         │                         │
         └─────► HGCN nightly ◄────┘
                  (co-retrieval pairs,
                   16d Poincaré,
                   value-score update)
```

Three embeddings per node — **about** (semantic), **belong** (hierarchy, 16d Poincaré), **when** (temporal). Retrieval scores them 40/30/15/15; HGCN nightly refines the hyperbolic positions using co-retrieval signals and goal-achievement outcomes.

### Design references

- Friston — Free Energy Principle
- Nickel & Kiela (2017) — Poincaré embeddings for hierarchies (proved 5d works for 130k-node WordNet)
- Chami et al. (2019) — Hyperbolic Graph Convolutional Networks
- Local design doc: [pmis_v2/REDESIGN.md](pmis_v2/REDESIGN.md)

---

## Configuration

All 50+ knobs live in [pmis_v2/hyperparameters.yaml](pmis_v2/hyperparameters.yaml). Edit + restart, or use the feedback/lint page on `:8200` which has one-click retrain. Environment — see [.env.template](.env.template).

---

## Project layout

```
pmis/                        public API package (5 async verbs + CLI)
pmis_v2/                     core engine + PM layer
├── orchestrator.py          per-turn pipeline
├── core/                    surprise · gamma · poincare · hgcn · co_retrieval
├── ingestion/               embedder · pipeline · surprise-gate · dedup
├── retrieval/               γ-blended engine · predictive · epistemic
├── consolidation/           nightly 5-pass + HGCN training
├── claude_integration/      prompt + brief + meta-LLM composers
├── server.py                FastAPI app — API + /wiki/* (:8100)
├── health_dashboard.py      Health / Feedback / Diagnostics / Lint (:8200)
├── wiki_renderer.py         LLM-generated wiki pages
├── cli.py                   internal CLI (Claude Desktop hooks)
├── db/schema.sql            SQLite schema (projects, deliverables, goals, ...)
└── tests/                   test_e2e.py, test_p1_p2.py
productivity-tracker/        macOS screen-pulse daemon
install.sh · start.sh        one-command macOS install + launch
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Be decent — [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{pmis_2026,
  title = {PMIS: A Personal Operating System for Work That Ships},
  author = {YantrAI Labs},
  year = {2026},
  url = {https://github.com/YantrAILabs/PMIS_V2}
}
```
