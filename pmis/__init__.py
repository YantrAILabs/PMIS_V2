"""PMIS — Personal Memory Intelligence System.

A surprise-minimizing memory engine with hyperbolic hierarchy and
outcome-driven weight evolution. Public API is 5 async verbs plus
two convenience wrappers.

Quickstart:

    import asyncio, pmis

    async def main():
        node_id = await pmis.ingest("CISOs respond to threat language")
        await pmis.attach(node_id, project="B2B Cold Outreach")
        hits = await pmis.retrieve("cold outreach", mode="predictive", k=5)
        await pmis.consolidate()

    asyncio.run(main())

Sugar:

    await pmis.remember("I learned X", project="Q2 pipeline")
    answers = await pmis.ask("what did I learn?")
"""

from pmis.api import (
    ingest,
    attach,
    retrieve,
    consolidate,
    delete,
    remember,
    ask,
)

__all__ = [
    "ingest",
    "attach",
    "retrieve",
    "consolidate",
    "delete",
    "remember",
    "ask",
]

__version__ = "0.1.0"
