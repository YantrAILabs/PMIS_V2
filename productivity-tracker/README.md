# Productivity Tracker

A Mac-native productivity tracking system powered by ChatGPT Vision and PMIS v2 memory architecture. Tracks what you're working on, whether you or an agent did the work, and matches it against your deliverables.

Managed entirely through Claude Desktop via MCP.

## Quick start

```bash
# 1. Clone and install
cd productivity-tracker
chmod +x scripts/install.sh
./scripts/install.sh

# 2. Set up API keys
cp .env.example .env
# Edit .env with your OpenAI API key

# 3. Configure Claude Desktop
# Copy the snippet from claude_desktop_config.json into your 
# Claude Desktop config at ~/Library/Application Support/Claude/claude_desktop_config.json

# 4. Start tracking
./scripts/start_tracker.sh
```

## Usage via Claude Desktop

Once the MCP server is running, talk to Claude Desktop:

- *"What am I working on right now?"* → `get_status`
- *"Show me today's summary"* → `get_daily_summary`
- *"How's the Vision OS dashboard coming along?"* → `get_deliverable_progress`
- *"Pause tracking"* → `pause_tracking`
- *"What didn't contribute to any deliverable this week?"* → `get_contribution_report`

## Configuration

- `config/settings.yaml` — intervals, thresholds, model settings
- `config/deliverables.yaml` — define your deliverables manually
- `config/privacy.yaml` — exclude apps/windows from tracking

## Architecture

See [PLAN.md](./PLAN.md) for full architecture documentation.

## Requirements

- macOS 12+ (Monterey or later)
- Python 3.11+
- OpenAI API key
- Claude Desktop (for MCP interaction)
- Accessibility permissions (System Settings → Privacy → Accessibility)
- Screen Recording permissions (System Settings → Privacy → Screen Recording)
