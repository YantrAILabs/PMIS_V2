"""
mcp_tools.py — MCP tool definitions for the replay harness.

Register these tools in your MCP server to expose replay harness
capabilities to Claude Desktop.

Tools:
  replay_status      — Check harness health and stats
  replay_summary     — Get current conversation's session analysis
  replay_diagnostic  — Run full diagnostic report on collected data
  replay_feedback    — View feedback signal for current conversation
"""

from typing import Optional
from .harness import ReplayHarness
from .analyzer import run_full_diagnostic


def create_replay_tools(harness: ReplayHarness) -> list:
    """
    Returns a list of tool definitions for your MCP server.

    Integration example (in your mcp_server.py):

        from replay import ReplayHarness
        from replay.mcp_tools import create_replay_tools

        harness = ReplayHarness(log_dir="data/replay_logs")
        tools = create_replay_tools(harness)
        for tool in tools:
            mcp_server.register_tool(**tool)
    """

    def replay_status() -> dict:
        """Check replay harness health and collection stats."""
        return harness.get_harness_status()

    def replay_summary(conversation_id: Optional[str] = None) -> dict:
        """
        Get session tree analysis for a conversation.
        If no conversation_id provided, shows all active.
        """
        if conversation_id:
            summary = harness.get_conversation_summary(conversation_id)
            if summary:
                return summary
            return {"error": f"No active session for {conversation_id}"}

        # Return all active
        active = {}
        for cid in harness._engines:
            active[cid] = harness.get_conversation_summary(cid)
        return {"active_conversations": active}

    def replay_feedback(conversation_id: str) -> dict:
        """View feedback signals for a conversation."""
        if conversation_id not in harness._feedback_accumulators:
            return {"error": f"No feedback data for {conversation_id}"}
        return harness._feedback_accumulators[conversation_id].summary()

    def replay_diagnostic() -> dict:
        """
        Run full diagnostic report on all collected data.
        Call this after a day of data collection.
        """
        return run_full_diagnostic(str(harness.log_dir))

    def replay_close(conversation_id: str) -> dict:
        """Close a conversation and generate its final summary."""
        summary = harness.close_conversation(conversation_id)
        if summary:
            return summary
        return {"error": f"No active session for {conversation_id}"}

    return [
        {
            "name": "replay_status",
            "description": (
                "Check the replay harness status: uptime, total turns "
                "observed, active conversations, collection stats."
            ),
            "handler": replay_status,
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "replay_summary",
            "description": (
                "Get the session tree analysis for a specific conversation "
                "or all active conversations. Shows convergence state, "
                "blend weight, schema prediction, and feedback signals."
            ),
            "handler": replay_summary,
            "input_schema": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "Conversation ID (optional, shows all if omitted)",
                    },
                },
            },
        },
        {
            "name": "replay_feedback",
            "description": (
                "View the one-bit feedback signal analysis for a conversation. "
                "Shows confirmation rate, calibration accuracy, and whether "
                "the session engine's schema predictions match user behavior."
            ),
            "handler": replay_feedback,
            "input_schema": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "Conversation ID",
                    },
                },
                "required": ["conversation_id"],
            },
        },
        {
            "name": "replay_diagnostic",
            "description": (
                "Run the full diagnostic report on all collected replay data. "
                "Analyzes: short conversation problem, feedback calibration, "
                "reranking impact, frustration false positives, and "
                "hyperparameter sensitivity. Call after a full day of data."
            ),
            "handler": replay_diagnostic,
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "replay_close",
            "description": (
                "Close a conversation's replay session and generate "
                "its final summary with feedback aggregation."
            ),
            "handler": replay_close,
            "input_schema": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "Conversation ID to close",
                    },
                },
                "required": ["conversation_id"],
            },
        },
    ]
