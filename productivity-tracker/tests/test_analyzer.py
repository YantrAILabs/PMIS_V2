"""Tests for the agent detector."""

import pytest
from unittest.mock import patch, MagicMock
from src.agent.agent_detector import AgentDetector, AgentProcess


@pytest.fixture
def detector():
    return AgentDetector()


class TestAgentDetector:
    def test_no_agents_running(self, detector):
        with patch.object(detector, "_get_running_processes", return_value=[
            AgentProcess(name="python3", pid=100, cmdline="python3 app.py"),
            AgentProcess(name="Chrome", pid=200, cmdline="/Applications/Chrome.app"),
        ]):
            assert not detector.is_agent_active()

    def test_claude_code_detected(self, detector):
        with patch.object(detector, "_get_running_processes", return_value=[
            AgentProcess(name="node", pid=100, cmdline="claude code run fix-auth"),
        ]):
            assert detector.is_agent_active()
            assert detector.get_active_agent_name() == "Claude Code"

    def test_cursor_agent_detected(self, detector):
        with patch.object(detector, "_get_running_processes", return_value=[
            AgentProcess(name="cursor", pid=100, cmdline="cursor --agent-mode"),
        ]):
            assert detector.is_agent_active()
            assert detector.get_active_agent_name() == "Cursor Agent"

    def test_aider_detected(self, detector):
        with patch.object(detector, "_get_running_processes", return_value=[
            AgentProcess(name="aider", pid=100, cmdline="aider --model gpt-4"),
        ]):
            assert detector.is_agent_active()

    def test_chatgpt_is_NOT_agent(self, detector):
        """Chatting with ChatGPT = human work, NOT agent."""
        with patch.object(detector, "_get_running_processes", return_value=[
            AgentProcess(name="Chrome", pid=200, cmdline="/Applications/Chrome.app"),
        ]):
            assert not detector.is_agent_active()

    def test_agent_window_detection(self):
        assert AgentDetector.is_agent_window({"title": "Claude Code — fixing auth"})
        assert AgentDetector.is_agent_window({"title": "Cursor Composer"})
        assert not AgentDetector.is_agent_window({"title": "ChatGPT - Chrome"})
        assert not AgentDetector.is_agent_window({"title": "VS Code"})

    def test_custom_signature(self, detector):
        detector.add_custom_signature(
            name_pattern=r"my-agent",
            cmd_pattern=r"my-agent",
            label="Custom Agent",
        )
        with patch.object(detector, "_get_running_processes", return_value=[
            AgentProcess(name="my-agent", pid=100, cmdline="my-agent run"),
        ]):
            assert detector.is_agent_active()
            assert detector.get_active_agent_name() == "Custom Agent"
