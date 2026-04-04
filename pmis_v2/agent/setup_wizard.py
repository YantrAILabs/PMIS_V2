"""
Platform Setup Wizard

Interactive CLI and API-driven setup for connecting platforms.
Each platform has a step-by-step guide and test verification.
"""

from typing import Dict, List, Optional
from auth import create_key


# ─── Platform definitions ───

PLATFORM_CONFIGS = {
    "claude-code": {
        "name": "Claude Code (CLI)",
        "icon": "terminal",
        "transport": "local_hook",
        "needs_tunnel": False,
        "auto_detected": True,
        "steps": [
            {"title": "Hook Already Installed", "description": "The UserPromptSubmit hook is already configured in .claude/settings.local.json. It calls the PMIS server on every prompt.", "action": "none"},
            {"title": "Verify Server Running", "description": "The hook calls http://localhost:8100/api/turn. Ensure the PMIS server is running.", "action": "test", "test_url": "/health"},
        ],
    },
    "claude-web": {
        "name": "Claude Web (claude.ai)",
        "icon": "globe",
        "transport": "mcp_sse",
        "needs_tunnel": True,
        "auto_detected": False,
        "steps": [
            {"title": "Start Tunnel", "description": "Claude Web needs a public HTTPS URL. Start ngrok or cloudflare tunnel to expose port 8101.", "action": "start_tunnel"},
            {"title": "Start MCP Server", "description": "Run: python3 pmis_v2/mcp_server.py --transport sse --port 8101", "action": "start_mcp"},
            {"title": "Add MCP Server in Claude", "description": "Go to claude.ai → Settings → MCP Servers → Add Server → Enter your tunnel URL + /sse", "action": "manual"},
            {"title": "Test Connection", "description": "Send a test message in Claude Web. The memory_turn tool should appear.", "action": "test", "test_url": "/health"},
        ],
    },
    "claude-desktop": {
        "name": "Claude Desktop App",
        "icon": "monitor",
        "transport": "mcp_stdio",
        "needs_tunnel": False,
        "auto_detected": False,
        "steps": [
            {"title": "Copy MCP Config", "description": "Add the PMIS MCP server to your Claude Desktop config. See claude_mcp_config.json for the config block to add.", "action": "copy_config"},
            {"title": "Restart Claude Desktop", "description": "Restart the Claude Desktop app to pick up the new MCP server.", "action": "manual"},
            {"title": "Test Connection", "description": "Open a chat in Claude Desktop. The pmis-memory tools should appear.", "action": "test", "test_url": "/health"},
        ],
    },
    "openai-gpt": {
        "name": "OpenAI Custom GPT",
        "icon": "brain",
        "transport": "openapi_actions",
        "needs_tunnel": True,
        "auto_detected": False,
        "steps": [
            {"title": "Start Tunnel", "description": "ChatGPT Actions need a public HTTPS URL. Start ngrok or cloudflare tunnel to expose port 8100.", "action": "start_tunnel"},
            {"title": "Generate API Key", "description": "Create an API key for OpenAI authentication.", "action": "generate_key"},
            {"title": "Create Custom GPT", "description": "Go to chat.openai.com → Explore → Create a GPT → Configure → Actions → Import from URL", "action": "manual"},
            {"title": "Import OpenAPI Schema", "description": "Enter your tunnel URL + /openapi-actions.json as the import URL.", "action": "manual"},
            {"title": "Set Authentication", "description": "In Actions config: Authentication → API Key → Header Name: X-API-Key → Enter your key.", "action": "manual"},
            {"title": "Add Instructions", "description": "In GPT instructions, add: 'On every user message, call processMemoryTurn first. Use the returned memory_context to inform your response.'", "action": "manual"},
            {"title": "Test Connection", "description": "Send a test message. The GPT should call the memory endpoint.", "action": "test", "test_url": "/health"},
        ],
    },
    "cursor": {
        "name": "Cursor IDE",
        "icon": "code",
        "transport": "local_http",
        "needs_tunnel": False,
        "auto_detected": False,
        "steps": [
            {"title": "Generate API Key", "description": "Create an API key for Cursor.", "action": "generate_key"},
            {"title": "Add to .cursorrules", "description": "Add PMIS webhook call to your .cursorrules file for automatic memory injection.", "action": "copy_config"},
            {"title": "Test Connection", "description": "Open Cursor in a project. Start a conversation and check if memory context appears.", "action": "test", "test_url": "/health"},
        ],
    },
    "custom": {
        "name": "Custom Platform",
        "icon": "puzzle",
        "transport": "webhook",
        "needs_tunnel": False,
        "auto_detected": False,
        "steps": [
            {"title": "Generate API Key", "description": "Create an API key for your platform.", "action": "generate_key"},
            {"title": "Configure Webhook", "description": "Call POST http://localhost:8100/api/webhook with {\"message\": \"...\", \"platform\": \"...\"}  and X-API-Key header on each user message.", "action": "manual"},
            {"title": "Test Connection", "description": "Send a test webhook call.", "action": "test", "test_url": "/health"},
        ],
    },
}


def get_platform_config(platform_id: str) -> Optional[Dict]:
    return PLATFORM_CONFIGS.get(platform_id)


def list_available_platforms() -> List[Dict]:
    result = []
    for pid, config in PLATFORM_CONFIGS.items():
        result.append({
            "id": pid,
            "name": config["name"],
            "icon": config["icon"],
            "transport": config["transport"],
            "needs_tunnel": config["needs_tunnel"],
            "auto_detected": config["auto_detected"],
            "step_count": len(config["steps"]),
        })
    return result


def get_setup_steps(platform_id: str) -> List[Dict]:
    config = PLATFORM_CONFIGS.get(platform_id)
    if not config:
        return []
    return config["steps"]


def generate_platform_key(platform_id: str) -> str:
    return create_key(platform_id)


# ─── CLI wizard ───

def run_cli_wizard(platform_id: str):
    """Interactive CLI setup wizard for a platform."""
    config = PLATFORM_CONFIGS.get(platform_id)
    if not config:
        print(f"Unknown platform: {platform_id}")
        print(f"Available: {', '.join(PLATFORM_CONFIGS.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"  Setting up: {config['name']}")
    print(f"  Transport: {config['transport']}")
    print(f"  Needs tunnel: {'Yes' if config['needs_tunnel'] else 'No'}")
    print(f"{'='*60}\n")

    for i, step in enumerate(config["steps"], 1):
        print(f"  Step {i}/{len(config['steps'])}: {step['title']}")
        print(f"  {step['description']}")

        if step["action"] == "generate_key":
            key = create_key(platform_id)
            print(f"\n  API Key: {key}")
            print(f"  Save this key — it won't be shown again.\n")
        elif step["action"] == "test":
            import urllib.request
            try:
                with urllib.request.urlopen("http://localhost:8100/health", timeout=3) as resp:
                    if resp.status == 200:
                        print(f"  Test: PASSED (server is healthy)\n")
                    else:
                        print(f"  Test: FAILED (status {resp.status})\n")
            except Exception as e:
                print(f"  Test: FAILED ({e})\n")
        else:
            print()

        if i < len(config["steps"]):
            input("  Press Enter to continue...")

    print(f"\n  Setup complete for {config['name']}!")
    print(f"  The platform is now registered and ready to use.\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        run_cli_wizard(sys.argv[1])
    else:
        print("Available platforms:")
        for p in list_available_platforms():
            print(f"  {p['id']:20s} {p['name']}")
        print("\nUsage: python3 agent/setup_wizard.py <platform-id>")
