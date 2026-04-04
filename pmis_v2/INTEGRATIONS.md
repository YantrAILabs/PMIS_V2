# PMIS V2 — Cross-Platform Integration Guide

Your memory system runs as a persistent HTTP server on port 8100. Any AI platform that can make HTTP calls or connect via MCP can share the same memory graph.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │     PMIS Server (port 8100)       │
                    │  ┌─────────────────────────────┐  │
                    │  │  memory_nodes (source of     │  │
                    │  │  truth) + ChromaDB ANN       │  │
                    │  └─────────────────────────────┘  │
                    │         ▲           ▲              │
                    │    Retrieve      Store             │
                    │         │           │              │
                    │  ┌──────┴───────────┴──────────┐  │
                    │  │     Orchestrator Pipeline     │  │
                    │  │  embed → surprise → gamma →   │  │
                    │  │  retrieve → compose → store   │  │
                    │  └──────────────────────────────┘  │
                    │         ▲                          │
                    │    /api/turn  /api/webhook         │
                    └─────┬──────────┬──────────────────┘
                          │          │
          ┌───────────────┼──────────┼───────────────┐
          │               │          │               │
    Claude Code     Claude Web   OpenAI GPT     Cursor/Other
    (hook→HTTP)     (MCP/SSE)   (Actions API)   (webhook)
          │               │          │               │
          └───────┬───────┴──────────┴───────┬───────┘
                  │                          │
           platform_memories          platform_memories
           (read-only audit)          (read-only audit)
```

## Universal Rules (any platform)

**Rule 1: Call PMIS before responding.**
On every user message, call `POST /api/webhook` with the user's text. Inject the returned `memory_context` into the system prompt.

**Rule 2: Store after learning.**
When a conversation produces a new insight, call `POST /api/store` with structured SC > CTX > ANC hierarchy.

**Rule 3: End sessions cleanly.**
When conversation ends, call `POST /api/session/end`.

## Quick Start (5 minutes)

```bash
# 1. Ensure server is running
curl http://localhost:8100/health

# 2. Generate API key
python3 pmis_v2/auth.py create <platform-name>

# 3. On each user message, call:
curl -X POST http://localhost:8100/api/webhook \
  -H "Content-Type: application/json" \
  -H "X-API-Key: pmis-<platform>-<token>" \
  -d '{"message": "user text here", "platform": "<name>"}'

# 4. Inject returned memory_context into your system prompt
# 5. Done — all platforms share the same memory graph
```

## Platform-Specific Guides

### Claude Code (CLI) — Already Connected

The `UserPromptSubmit` hook in `.claude/hooks/pmis-session-begin.py` calls the server automatically on every prompt. It tries HTTP first (~50ms), falls back to CLI subprocess (~2-3s) if the server is down.

No additional setup needed.

### Claude Web (claude.ai) — MCP Server

Claude Web supports MCP servers via SSE transport.

1. Start the PMIS server: `python3 pmis_v2/agent.py start`
2. Start the MCP server: `python3 pmis_v2/mcp_server.py --transport sse --port 8101`
3. Start a tunnel: `ngrok http 8101` (get public URL)
4. In claude.ai: Settings → MCP Servers → Add → Enter `<tunnel-url>/sse`
5. In chat, the tools `memory_turn`, `memory_store`, `memory_browse`, `memory_stats` appear

### Claude Desktop App — MCP Server (Stdio)

1. Copy the config from `pmis_v2/claude_mcp_config.json`
2. Add it to your Claude Desktop config file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
3. Restart Claude Desktop
4. The `pmis-memory` tools appear in conversations

### OpenAI Custom GPT — Actions API

1. Start a tunnel: `ngrok http 8100` (get public URL)
2. Generate API key: `python3 pmis_v2/auth.py create openai-gpt`
3. In ChatGPT: Create a GPT → Configure → Actions → Import from URL
4. Enter: `<tunnel-url>/openapi-actions.json`
5. Set authentication: API Key → Header: `X-API-Key` → Enter your key
6. Add to GPT instructions:
   > On every user message, call processMemoryTurn first with the user's message. Use the returned memory_context to inform your response. Store key learnings when the conversation produces new insights.

### Cursor IDE — Local HTTP

1. Generate API key: `python3 pmis_v2/auth.py create cursor`
2. Add to `.cursorrules` in your project:
   ```
   Before responding, call http://localhost:8100/api/webhook with:
   POST body: {"message": "<user message>", "platform": "cursor"}
   Header: X-API-Key: <your-key>
   Use the returned memory_context to inform your response.
   ```

### Any Other Platform — Universal Webhook

```
POST http://localhost:8100/api/webhook
Headers: Content-Type: application/json, X-API-Key: <your-key>
Body: {"message": "user text", "platform": "your-platform-name"}

Response:
{
  "memory_context": "<system prompt injection>",
  "mode": "BALANCED",
  "gamma": 0.5,
  "surprise": 0.3,
  "retrieved_count": 5,
  "stored": true,
  "platform": "your-platform-name"
}
```

## Data Flow

1. Platform sends message → `POST /api/webhook` or `/api/turn`
2. Server logs raw input to `platform_memories` (audit only, non-blocking)
3. Orchestrator runs full pipeline: embed → surprise → gamma → retrieve → compose → store
4. All retrieval uses `memory_nodes` + ChromaDB (main memory, single source of truth)
5. Storage decision creates/updates nodes in `memory_nodes` (main memory)
6. `platform_memories` entry updated with merge status (merged/skipped)
7. Memory context returned to calling platform

**Cross-platform knowledge is automatic** — what you learn in ChatGPT is retrievable in Claude and vice versa.

## Server Management

```bash
python3 pmis_v2/agent.py start     # Start server (launchd)
python3 pmis_v2/agent.py stop      # Stop server
python3 pmis_v2/agent.py restart   # Restart
python3 pmis_v2/agent.py status    # Full status report
python3 pmis_v2/agent.py connect <platform>  # Setup wizard
```

## Dashboard

- **Main**: http://localhost:8100
- **Integrations**: http://localhost:8100/integrations
- **API Docs**: http://localhost:8100/docs

## API Key Management

```bash
python3 pmis_v2/auth.py create <platform>   # Generate key
python3 pmis_v2/auth.py list                # List all keys
python3 pmis_v2/auth.py revoke <platform>   # Revoke keys
```

Or use the dashboard: http://localhost:8100/integrations → API Keys section.
