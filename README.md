# PMIS

### Memory that knows what you actually ship.

A personal OS — surprise-minimizing memory + project tracking, bound together. Every anchor lives under a project, every project has a goal, and the goal's success signal flows back to reweight the memory. Storage isn't passive — it's learning which memories predict which projects ship.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#)

---

## What makes it different

| | Most memory libraries | **PMIS** |
|---|---|---|
| Storage | Everything | Surprise-gated (Free Energy Principle) |
| Structure | Flat vector store | Hyperbolic hierarchy — tree geometry *is* the embedding space |
| Retrieval | Fixed top-k | γ-blended explore/exploit, per-query |
| Weights | Static | Outcome-driven — project success reweights memories via HGCN |
| Capture | Manual only | Passive screen-pulse + idle-detect feeds memory automatically |

Five mechanisms nobody else combines. Cognee owns #2 partially; the rest is open ground.

---

## Quickstart

### 🧑‍💻 For developers

```bash
pip install -e .
export OPENAI_API_KEY=sk-...
```

```python
import asyncio, pmis

async def main():
    # Low-level 5-verb API
    node_id = await pmis.ingest("CISOs respond to threat language 3x over ROI")
    await pmis.attach(node_id, project="B2B Cold Outreach")
    hits = await pmis.retrieve("cold outreach learnings", mode="predictive", k=5)
    await pmis.consolidate()

    # High-level sugar
    await pmis.remember("I learned X", project="Q2 pipeline")
    answers = await pmis.ask("what did I learn about X?")

asyncio.run(main())
```

Full walkthrough: [examples/quickstart.py](examples/quickstart.py).

### 📒 For productivity users (Claude Desktop)

```bash
./install.sh          # macOS installer — tracker daemon + Claude Desktop MCP config
```

Then in Claude Desktop:

- *"What am I working on right now?"* → answered from PMIS pulse log
- *"Remember that CISOs respond to threat language"* → stored with surprise gate
- *"What did I learn about cold outreach?"* → γ-blended retrieval
- *"How's the Vision OS dashboard coming along?"* → deliverable progress

Setup guide: [examples/claude_desktop_mcp.md](examples/claude_desktop_mcp.md).

---

## The 5-verb API

| Verb | What it does | CLI |
|---|---|---|
| `ingest(text)` | Embed + surprise-gate. Returns node_id or None. | `pmis ingest "text"` |
| `attach(node_id, project=...)` | Promote orphan into SC/CTX/ANC hierarchy. | `pmis attach --node <id>` |
| `retrieve(query, mode=..., k=...)` | γ-blended search. | `pmis retrieve "query" --mode predictive` |
| `consolidate()` | 5-pass: compress · promote · birth · prune · HGCN. | `pmis consolidate` |
| `delete(node_id=..., all=False)` | Soft-delete or reset. | `pmis delete --node <id>` |

Plus two convenience wrappers:

- `remember(text, project=...)` = `ingest` + `attach` in one call
- `ask(query, k=...)` = `retrieve` with automatic mode

All are async. See [pmis/api.py](pmis/api.py) for full signatures.

---

## Dashboard

```bash
python3 pmis_v2/server.py    # binds :8100
open http://localhost:8100
```

Three-section sidebar:

- **Actions** — run the 5 verbs from a form
- **Work** — projects, goals, deliverables, pulse log
- **Mind** — graph, timeline, stats, hyperparameters

Power-user dashboard with conversation log + live hyperparameter sliders: `/legacy`.

REST: every verb is also a `POST /api/{verb}` endpoint — see `/docs` (Swagger).

---

## Architecture

```
User message
    │
    ▼
Embed  ──────────────►  Surprise  ──────►  γ (explore/exploit)
(Euclidean 768d            (effective =           │
 + Hyperbolic 32d           raw × precision)      ▼
 + Temporal 16d)                           Retrieve
                                           (γ-blended: narrow + broad)
                                                │
                                                ▼
                           Storage  ◄─────  Compose response
                           decision         (with retrieved context)
                           (surprise-gate)
                                │
                                ▼
                           SQLite + ChromaDB ANN
                                │
                                ▼
                           Nightly consolidation
                           (compress · promote · birth · prune · HGCN)
```

Three embeddings per node: **about** (semantic), **belong** (hierarchy), **when** (temporal). Retrieval scores them with learned weights; consolidation updates hyperbolic coords via RSGD (default) or HGCN (experimental).

Theory references: Friston — Free Energy Principle; Nickel & Kiela — Poincaré embeddings.

---

## Configuration

All 50+ knobs live in [pmis_v2/hyperparameters.yaml](pmis_v2/hyperparameters.yaml). Key ones:

- `surprise_low_threshold` — below = update existing; above = new anchor
- `gamma_bias` — explore/exploit baseline
- `score_weight_{semantic,hierarchy,temporal,precision}` — retrieval weights
- `prune_min_age_days`, `birth_min_orphans` — consolidation gates

Edit the YAML and restart, or use the live sliders at `/legacy`.

Environment variables — see [.env.template](.env.template).

---

## Project layout

```
pmis/                       public API package (5 verbs + CLI)
pmis_v2/                    core engine
├── orchestrator.py         per-turn pipeline controller
├── core/                   surprise, gamma, Poincaré ball, RSGD
├── ingestion/              embedder, surprise-gate, dedup
├── retrieval/              γ-blended search, predictive, epistemic
├── consolidation/          nightly 5-pass
├── claude_integration/     prompt composer for Claude Desktop
├── server.py               FastAPI app + REST + dashboards
├── cli.py                  internal CLI (Claude Desktop hooks)
└── tests/                  test_e2e.py, test_p1_p2.py
productivity-tracker/       macOS screen-pulse daemon
install.sh                  one-command macOS installer
```

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, test commands, and the PR checklist. By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

If PMIS informs research:

```bibtex
@software{pmis_2026,
  title        = {PMIS: Personal Memory Intelligence System},
  author       = {PMIS contributors},
  year         = {2026},
  url          = {https://github.com/yourorg/pmis}
}
```

Underlying ideas:

- Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience.
- Nickel, M. & Kiela, D. (2017). *Poincaré embeddings for learning hierarchical representations.* NeurIPS.
- Chami, I., Ying, R., Ré, C., & Leskovec, J. (2019). *Hyperbolic graph convolutional neural networks.* NeurIPS.
