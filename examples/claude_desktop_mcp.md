# Claude Desktop + PMIS (MCP setup)

PMIS ships with an MCP server that lets Claude Desktop query your live memory and PM state directly. After this guide, you can ask Claude things like *"what did I learn about cold outreach last week?"* or *"what am I working on right now?"* and get answers grounded in your own history.

## Prerequisites

- macOS (the installer currently targets Darwin; Linux support is straightforward but untested).
- Python 3.10+.
- [Ollama](https://ollama.com/) running locally with `nomic-embed-text` pulled:
  ```bash
  ollama pull nomic-embed-text
  ```
- Claude Desktop installed.

## 1. Install PMIS

From the repo root:

```bash
./install.sh
```

That script creates a `.venv`, installs dependencies, sets up the launchd agent for the productivity tracker, and writes the MCP entry into your Claude Desktop config.

## 2. Start the servers

```bash
./start.sh
# or manually:
python3 pmis_v2/server.py &          # :8100 — API + /wiki
python3 pmis_v2/health_dashboard.py & # :8200 — Health + Feedback
```

Open http://localhost:8100/wiki/ to confirm PMIS is live, and http://localhost:8200/ for the health dashboard.

## 3. Verify Claude Desktop sees the MCP server

Restart Claude Desktop after running `install.sh`. A new PMIS icon should appear in the compose box (the usual MCP indicator). If it doesn't, check:

```bash
cat "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
```

You should see a `"pmis"` entry pointing at `pmis_v2/mcp_server.py`.

## 4. Try it

In Claude Desktop, ask:

- *"What am I working on right now?"* — hits `/api/work/current` under the hood.
- *"Brief me on the cold outreach deliverable before I start."* — hits `/api/work/brief`, returns the Claude-can-do + You-did-before buckets.
- *"Remember that CISOs respond to threat-intel language 3x more than ROI framing."* — hits the ingest + attach path; may be surprise-gated if similar content is already stored.
- *"What did I learn about email copywriting last week?"* — γ-blended retrieval across your last 7 days of anchors.
- *"Show the review queue."* — opens the `/wiki/review` page.

## Troubleshooting

- **No PMIS icon in Claude Desktop** — `install.sh` didn't write the config. Re-run it, or paste the snippet from `pmis_v2/claude_mcp_config.json` into `~/Library/Application Support/Claude/claude_desktop_config.json` manually.
- **`Error finding id` in logs** — your ChromaDB index is out of sync with SQLite. Run `curl -X POST http://127.0.0.1:8200/api/action/reindex` or, for a full reset, `rm -rf pmis_v2/data/chroma && ./start.sh`.
- **Server on :8100 won't start** — port collision. `lsof -i :8100` to find the old process, `kill <pid>`, retry.
