"""PMIS quickstart — the 5-verb API in 40 lines.

Run:   python examples/quickstart.py
Needs: `pip install -e .` from the repo root, plus Ollama running locally
       with `nomic-embed-text` pulled. No API keys required by default.
"""

import asyncio
import pmis


async def main():
    print(f"pmis version: {pmis.__version__}")
    print(f"exports:     {pmis.__all__}\n")

    # ── 1. Ingest: embed + surprise-gate ────────────────────────────────
    # On an empty or familiar store, low-surprise turns may be "gated" — PMIS
    # decides they aren't novel enough to warrant a new anchor. That's a
    # feature, not a bug (Free Energy Principle).
    text = "CISOs respond to threat-intel language 3x over ROI framing."
    node_id = await pmis.ingest(text)
    print(f"ingest  → node_id={node_id}  (None means surprise-gated)")

    # ── 2. Attach: promote orphan into the hierarchy ───────────────────
    if node_id:
        att = await pmis.attach(node_id)
        print(f"attach  → {att}")

    # ── 3. Retrieve: γ-blended search ──────────────────────────────────
    hits = await pmis.retrieve("cold outreach", mode="predictive", k=3)
    print(f"retrieve → {len(hits)} hit(s):")
    for h in hits:
        print(f"  score={h['score']:.3f} lvl={h['level']} | {(h['content'] or '')[:70]}")

    # ── 4. Consolidate: nightly 5-pass (run manually here) ─────────────
    res = await pmis.consolidate()
    print(f"consolidate → {res}")

    # ── Sugar: remember = ingest + attach in one call ──────────────────
    r = await pmis.remember("Deliverables ship faster when work is time-boxed.")
    print(f"remember → {r}")

    # ── Sugar: ask = retrieve with auto mode ───────────────────────────
    answers = await pmis.ask("time-boxing", k=2)
    print(f"ask      → {len(answers)} hit(s)")


if __name__ == "__main__":
    asyncio.run(main())
