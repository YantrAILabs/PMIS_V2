"""PMIS command-line interface.

Top-level verbs mirror the Python API:

    pmis ingest "text"
    pmis attach [--node ID] [--project NAME]
    pmis retrieve "query" [--mode {auto,associative,balanced,predictive}] [--k N]
    pmis consolidate
    pmis delete [--node ID | --all]

Sugar:

    pmis remember "text" [--project NAME]
    pmis ask "query" [--k N]

Diagnostics:

    pmis stats
    pmis status
    pmis dashboard

Back-compat aliases (delegate to pmis_v2/cli.py unchanged):

    pmis session {begin,store,rate,end,log-response} ...
    pmis browse
    pmis orphans
    pmis command {explore,exploit,surprise}

Add --json to any command for machine-readable output.
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_PMIS_V2_CLI = Path(__file__).resolve().parent.parent / "pmis_v2" / "cli.py"


def _run(coro):
    return asyncio.run(coro)


def _dump(obj: Any):
    print(json.dumps(obj, default=str, ensure_ascii=False))


def _preview(text: str, n: int = 70) -> str:
    return (text or "").replace("\n", " ")[:n]


# ---------------------------------------------------------------------------
# 5 verbs + sugar
# ---------------------------------------------------------------------------


def _cmd_ingest(args):
    import pmis
    nid = _run(pmis.ingest(args.text))
    if args.json:
        _dump({"node_id": nid, "stored": nid is not None})
    elif nid:
        print(f"✓ ingested  {nid}")
    else:
        print("• surprise-gated (not stored)")


def _cmd_attach(args):
    import pmis
    res = _run(pmis.attach(args.node, project=args.project))
    if args.json:
        _dump(res)
        return
    if res.get("attached"):
        print(f"✓ {res['node_id']} → {res['parent_id']}  ({_preview(res.get('parent_preview',''))})")
    else:
        print(f"• not attached: {res.get('reason','unknown')}")


def _cmd_retrieve(args):
    import pmis
    hits = _run(pmis.retrieve(args.query, mode=args.mode, k=args.k))
    if args.json:
        _dump(hits)
        return
    if not hits:
        print("• no hits")
        return
    print(f"{'#':>2}  {'lvl':<4} {'score':>6}  content")
    for i, h in enumerate(hits):
        print(f"{i:>2}  {str(h['level']):<4} {h['score']:>6.3f}  {_preview(h['content'])}")


def _cmd_consolidate(args):
    import pmis
    res = _run(pmis.consolidate())
    if args.json:
        _dump(res)
        return
    for k, v in res.items():
        n = len(v) if isinstance(v, list) else v
        print(f"{k:<16} {n}")


def _cmd_delete(args):
    import pmis
    if not args.node and not args.all:
        print("error: give --node ID or --all", file=sys.stderr)
        sys.exit(2)
    res = _run(pmis.delete(args.node, all=args.all))
    if args.json:
        _dump(res)
    else:
        print(json.dumps(res, default=str))


def _cmd_remember(args):
    import pmis
    res = _run(pmis.remember(args.text, project=args.project))
    if args.json:
        _dump(res)
    elif res.get("stored"):
        print(f"✓ remembered  {res['node_id']}  → {res.get('parent_id') or '(orphan)'}")
    else:
        print(f"• skipped: {res.get('reason','unknown')}")


def _cmd_ask(args):
    import pmis
    hits = _run(pmis.ask(args.query, k=args.k))
    if args.json:
        _dump(hits)
        return
    if not hits:
        print("• no hits")
        return
    for i, h in enumerate(hits):
        print(f"[{i}] {h['score']:.3f}  {_preview(h['content'], 80)}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _cmd_stats(args):
    _delegate(["stats"])


def _cmd_status(args):
    _delegate(["status"])


def _cmd_dashboard(args):
    print("Start the dashboard with:")
    print("    python3 pmis_v2/server.py")
    print("Then open:  http://localhost:8100")


# ---------------------------------------------------------------------------
# Back-compat aliases → delegate to pmis_v2/cli.py
# ---------------------------------------------------------------------------


def _delegate(argv: List[str]) -> None:
    """Invoke pmis_v2/cli.py with the given argv; stream its stdout."""
    try:
        subprocess.run([sys.executable, str(_PMIS_V2_CLI), *argv], check=False)
    except FileNotFoundError:
        print(f"error: pmis_v2 CLI not found at {_PMIS_V2_CLI}", file=sys.stderr)
        sys.exit(2)


def _cmd_session(args):
    argv = ["session", args.subcommand, *args.rest]
    _delegate(argv)


def _cmd_browse(args):
    _delegate(["browse"])


def _cmd_orphans(args):
    _delegate(["orphans"])


def _cmd_command(args):
    _delegate(["command", args.mode])


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pmis",
        description="PMIS — Personal Memory Intelligence System.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quickstart:\n"
            "  pmis remember 'CISOs respond to threat language 3x over ROI'\n"
            "  pmis ask 'cold outreach learnings'\n"
            "  pmis stats\n"
            "\n"
            "See `pmis <verb> --help` for per-command options."
        ),
    )
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")

    sub = p.add_subparsers(dest="cmd", metavar="<verb>")

    # --- 5 verbs ---
    pi = sub.add_parser("ingest", help="embed + surprise-gate a new memory")
    pi.add_argument("text", help="text to ingest")
    pi.set_defaults(func=_cmd_ingest)

    pa = sub.add_parser("attach", help="attach an orphan to its nearest context")
    pa.add_argument("--node", help="node id (default: most recent orphan)")
    pa.add_argument("--project", help="project name to bind to")
    pa.set_defaults(func=_cmd_attach)

    pr = sub.add_parser("retrieve", help="γ-blended retrieval")
    pr.add_argument("query")
    pr.add_argument("--mode", choices=["auto", "associative", "balanced", "predictive"], default="auto")
    pr.add_argument("--k", type=int, default=8)
    pr.set_defaults(func=_cmd_retrieve)

    pc = sub.add_parser("consolidate", help="run nightly 5-pass consolidation")
    pc.set_defaults(func=_cmd_consolidate)

    pd = sub.add_parser("delete", help="soft-delete node or reset store")
    pd.add_argument("--node", help="node id")
    pd.add_argument("--all", action="store_true", help="reset entire store")
    pd.set_defaults(func=_cmd_delete)

    # --- sugar ---
    pm = sub.add_parser("remember", help="ingest + attach in one call")
    pm.add_argument("text")
    pm.add_argument("--project")
    pm.set_defaults(func=_cmd_remember)

    pk = sub.add_parser("ask", help="retrieve with auto mode")
    pk.add_argument("query")
    pk.add_argument("--k", type=int, default=5)
    pk.set_defaults(func=_cmd_ask)

    # --- diagnostics ---
    ps = sub.add_parser("stats", help="memory counts + activity")
    ps.set_defaults(func=_cmd_stats)

    pst = sub.add_parser("status", help="current γ / mode / active tree")
    pst.set_defaults(func=_cmd_status)

    pdb = sub.add_parser("dashboard", help="print dashboard launch info")
    pdb.set_defaults(func=_cmd_dashboard)

    # --- back-compat aliases ---
    pses = sub.add_parser(
        "session",
        help="[back-compat] delegate to pmis_v2/cli.py session …",
    )
    pses.add_argument("subcommand", choices=["begin", "store", "rate", "end", "log-response"])
    pses.add_argument("rest", nargs=argparse.REMAINDER)
    pses.set_defaults(func=_cmd_session)

    pbr = sub.add_parser("browse", help="[back-compat] tree browser")
    pbr.set_defaults(func=_cmd_browse)

    por = sub.add_parser("orphans", help="[back-compat] list orphan anchors")
    por.set_defaults(func=_cmd_orphans)

    pcm = sub.add_parser("command", help="[back-compat] explore/exploit/surprise")
    pcm.add_argument("mode", choices=["explore", "exploit", "surprise"])
    pcm.set_defaults(func=_cmd_command)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
