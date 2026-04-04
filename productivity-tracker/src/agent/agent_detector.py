"""
Agent detector — determines if an autonomous agent is running on the user's behalf.

This is the core Human vs Agent distinction:
- Human: user is in the driver's seat (even if chatting with ChatGPT, that's human work)
- Agent: an autonomous process is executing tasks without continuous user input

Detection is process-level: scan for known agent process signatures.
"""

import logging
import subprocess
import re
from dataclasses import dataclass

logger = logging.getLogger("tracker.agent_detector")


@dataclass
class AgentProcess:
    name: str
    pid: int
    cmdline: str


# Known agent process signatures
# Each entry: (process_name_pattern, cmdline_pattern, description)
AGENT_SIGNATURES = [
    # Claude Code — CLI agent that autonomously writes code
    {
        "name_pattern": r"(claude|anthropic)",
        "cmd_pattern": r"claude\s+(code|--agent|run)",
        "label": "Claude Code",
    },
    # Cursor agent mode — autonomous coding in Cursor IDE
    {
        "name_pattern": r"cursor",
        "cmd_pattern": r"(--agent|agent-mode|composer)",
        "label": "Cursor Agent",
    },
    # Aider — AI pair programming agent
    {
        "name_pattern": r"aider",
        "cmd_pattern": r"aider",
        "label": "Aider",
    },
    # GitHub Copilot Workspace
    {
        "name_pattern": r"copilot",
        "cmd_pattern": r"copilot.*(workspace|agent)",
        "label": "Copilot Workspace",
    },
    # Devin / SWE-agent / OpenHands / similar
    {
        "name_pattern": r"(devin|swe.agent|openhands)",
        "cmd_pattern": r"(devin|swe.agent|openhands)",
        "label": "SWE Agent",
    },
    # Generic: any process with 'agent' in a coding context
    {
        "name_pattern": r"agent",
        "cmd_pattern": r"(coding|code|dev|build|test).*agent",
        "label": "Generic Agent",
    },
    # MCP-driven automation (look for MCP client processes running tasks)
    {
        "name_pattern": r"mcp",
        "cmd_pattern": r"mcp.*(run|execute|task)",
        "label": "MCP Automation",
    },
]


class AgentDetector:
    """Detects whether an autonomous agent is currently running."""

    def __init__(self):
        self._last_result = False
        self._last_agent: str | None = None
        self._custom_signatures: list[dict] = []

    def add_custom_signature(self, name_pattern: str, cmd_pattern: str, label: str):
        """Add a custom agent signature to detect."""
        self._custom_signatures.append({
            "name_pattern": name_pattern,
            "cmd_pattern": cmd_pattern,
            "label": label,
        })

    def is_agent_active(self) -> bool:
        """
        Check if any known agent process is currently running.
        Returns True if an agent is detected, False otherwise.
        """
        try:
            processes = self._get_running_processes()
            all_signatures = self._custom_signatures + AGENT_SIGNATURES

            for proc in processes:
                for sig in all_signatures:
                    name_match = re.search(
                        sig["name_pattern"], proc.name, re.IGNORECASE
                    )
                    cmd_match = re.search(
                        sig["cmd_pattern"], proc.cmdline, re.IGNORECASE
                    )
                    if name_match or cmd_match:
                        if not self._last_result or self._last_agent != sig["label"]:
                            logger.info(f"Agent detected: {sig['label']} (PID {proc.pid})")
                        self._last_result = True
                        self._last_agent = sig["label"]
                        return True

            if self._last_result:
                logger.info("Agent no longer detected, switching to human.")
            self._last_result = False
            self._last_agent = None
            return False

        except Exception as e:
            logger.error(f"Agent detection failed: {e}")
            return False

    def get_active_agent_name(self) -> str | None:
        """Return the name of the currently active agent, or None."""
        return self._last_agent if self._last_result else None

    def _get_running_processes(self) -> list[AgentProcess]:
        """Get list of running processes with their command lines."""
        try:
            # ps with full command line
            result = subprocess.run(
                ["ps", "-eo", "pid,comm,args"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            processes = []
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.strip().split(None, 2)
                if len(parts) >= 2:
                    pid = int(parts[0]) if parts[0].isdigit() else 0
                    name = parts[1]
                    cmdline = parts[2] if len(parts) > 2 else ""
                    processes.append(AgentProcess(name=name, pid=pid, cmdline=cmdline))
            return processes
        except Exception as e:
            logger.error(f"Process listing failed: {e}")
            return []

    @staticmethod
    def is_agent_window(window_info: dict) -> bool:
        """
        Secondary check: is the active window itself an agent interface?
        This supplements process detection for web-based agents.
        """
        title = window_info.get("title", "").lower()
        agent_window_hints = [
            "cursor composer",
            "claude code",
            "aider",
            "devin",
            "copilot workspace",
            "agent mode",
        ]
        return any(hint in title for hint in agent_window_hints)
