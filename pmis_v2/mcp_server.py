#!/usr/bin/env python3
"""
PMIS V2 MCP Server

Exposes memory tools via the Model Context Protocol for:
- Claude Web (claude.ai) via SSE/streamable-http transport
- Claude Desktop App via stdio transport

Each tool calls the PMIS HTTP server at localhost:8100 internally.

Usage:
    # Stdio transport (Claude Desktop):
    python3 pmis_v2/mcp_server.py

    # SSE transport (Claude Web, needs public URL via tunnel):
    python3 pmis_v2/mcp_server.py --transport sse --port 8101
"""

import sys
import json
import argparse
import urllib.request
import urllib.error

PMIS_URL = "http://localhost:8100"


def _call_pmis(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Call the PMIS HTTP server."""
    url = f"{PMIS_URL}{endpoint}"
    body = json.dumps(data).encode() if data else None
    headers = {"Content-Type": "application/json"} if data else {}

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


# ─── Tool handlers ───

def memory_turn(message: str, conversation_id: str = "", platform: str = "claude-web") -> str:
    """Process a conversation turn and retrieve relevant memories.
    Call this on EVERY user message to inject memory context."""
    result = _call_pmis("/api/webhook", "POST", {
        "message": message,
        "platform": platform,
        "conversation_id": conversation_id or None,
    })
    if "error" in result:
        return f"Memory server error: {result['error']}"
    return result.get("memory_context", "No memories retrieved.")


def memory_store(super_context: str, contexts: list = None, summary: str = "") -> str:
    """Store a new memory with structured hierarchy: Super Context > Context > Anchor."""
    result = _call_pmis("/api/store", "POST", {
        "super_context": super_context,
        "contexts": contexts or [],
        "summary": summary,
    })
    if "error" in result:
        return f"Store error: {result['error']}"
    return f"Stored: {result.get('stored', False)}, node_id: {result.get('stored_node_id', 'none')}"


def memory_browse() -> str:
    """Browse all super contexts and their hierarchy."""
    result = _call_pmis("/api/browse")
    if "error" in result:
        return f"Browse error: {result['error']}"

    lines = []
    for sc in result.get("super_contexts", []):
        lines.append(f"\n[SC] {sc['title']}")
        for ctx in sc.get("contexts", []):
            lines.append(f"  [CTX] {ctx['title']} ({ctx['anchor_count']} anchors)")
    return "\n".join(lines) if lines else "No memories stored yet."


def memory_stats() -> str:
    """Get memory system statistics."""
    result = _call_pmis("/api/stats")
    if "error" in result:
        return f"Stats error: {result['error']}"

    return (
        f"Total nodes: {result.get('total_nodes', 0)}\n"
        f"Super Contexts: {result.get('super_contexts', 0)}\n"
        f"Contexts: {result.get('contexts', 0)}\n"
        f"Anchors: {result.get('anchors', 0)}\n"
        f"Orphans: {result.get('orphans', 0)}\n"
        f"Trees: {result.get('trees', 0)}"
    )


# ─── MCP Protocol Implementation ───

TOOLS = {
    "memory_turn": {
        "description": "Process a conversation turn and retrieve relevant memories. Call this on EVERY user message to inject memory context into your response.",
        "inputSchema": {
            "type": "object",
            "required": ["message"],
            "properties": {
                "message": {"type": "string", "description": "The user's message text"},
                "conversation_id": {"type": "string", "description": "Optional conversation ID for session continuity", "default": ""},
                "platform": {"type": "string", "description": "Platform identifier", "default": "claude-web"},
            }
        },
        "handler": memory_turn,
    },
    "memory_store": {
        "description": "Store a new memory with structured hierarchy: Super Context > Context > Anchor. Use when the conversation produces a new insight worth remembering.",
        "inputSchema": {
            "type": "object",
            "required": ["super_context"],
            "properties": {
                "super_context": {"type": "string", "description": "Broad work domain name (e.g., 'B2B Cold Outreach')"},
                "contexts": {"type": "array", "description": "List of context objects with title, weight, and anchors", "default": []},
                "summary": {"type": "string", "description": "Brief summary", "default": ""},
            }
        },
        "handler": memory_store,
    },
    "memory_browse": {
        "description": "Browse all super contexts and their hierarchy to see what's stored in memory.",
        "inputSchema": {"type": "object", "properties": {}},
        "handler": memory_browse,
    },
    "memory_stats": {
        "description": "Get memory system statistics (node counts, trees, orphans).",
        "inputSchema": {"type": "object", "properties": {}},
        "handler": memory_stats,
    },
}


def _handle_jsonrpc(request: dict) -> dict:
    """Handle a JSON-RPC 2.0 request."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "pmis-memory",
                    "version": "2.0.0",
                },
            }
        }

    elif method == "notifications/initialized":
        return None  # No response for notifications

    elif method == "tools/list":
        tools_list = []
        for name, tool in TOOLS.items():
            tools_list.append({
                "name": name,
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            })
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools_list}
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        tool = TOOLS.get(tool_name)
        if not tool:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                }
            }

        try:
            result_text = tool["handler"](**arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                    "isError": True,
                }
            }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }


def run_stdio():
    """Run MCP server via stdio transport (for Claude Desktop)."""
    sys.stderr.write("[PMIS MCP] Starting stdio transport...\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = _handle_jsonrpc(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


def run_sse(port: int = 8101):
    """Run MCP server via SSE transport (for Claude Web)."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        import asyncio
    except ImportError:
        print("FastAPI/uvicorn required for SSE transport. pip install fastapi uvicorn")
        sys.exit(1)

    mcp_app = FastAPI(title="PMIS MCP Server (SSE)")
    mcp_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @mcp_app.get("/sse")
    async def sse_endpoint(request: Request):
        async def event_stream():
            # Send endpoint info
            yield f"event: endpoint\ndata: /message\n\n"
            # Keep alive
            while True:
                await asyncio.sleep(30)
                yield f"event: ping\ndata: {{}}\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @mcp_app.post("/sse")
    async def sse_post_endpoint(request: Request):
        """Streamable-HTTP transport: Claude Web may POST JSON-RPC directly to /sse."""
        body = await request.json()
        response = _handle_jsonrpc(body)
        return JSONResponse(response) if response else JSONResponse({})

    @mcp_app.post("/message")
    async def message_endpoint(request: Request):
        body = await request.json()
        response = _handle_jsonrpc(body)
        return JSONResponse(response) if response else JSONResponse({})

    sys.stderr.write(f"[PMIS MCP] Starting SSE transport on port {port}...\n")
    uvicorn.run(mcp_app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PMIS V2 MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8101)
    args = parser.parse_args()

    if args.transport == "sse":
        run_sse(args.port)
    else:
        run_stdio()
