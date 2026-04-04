# Productivity Tracker — Claude Desktop Skill

## What this skill does

This skill interfaces with the Productivity Tracker system running on the user's Mac M2. The tracker captures screen activity, classifies work into a SC → Context → Anchor hierarchy, detects autonomous agent activity, and matches work against assigned deliverables.

## Available MCP tools

All interaction happens through the `productivity-tracker` MCP server. The following tools are available:

### Status & control
- `get_status` — Current tracking state, active window, running segment
- `pause_tracking` — Pause screenshot capture
- `resume_tracking` — Resume capture

### Analysis
- `get_daily_summary(target_date?)` — Today's time breakdown by SC/Context/Anchor with human/agent split
- `get_deliverable_progress(deliverable_name?)` — Progress per deliverable with time allocation
- `get_contribution_report(days_back=7)` — What contributed to deliverables vs what didn't
- `get_trends(days_back=30)` — Daily totals, agent leverage ratios, patterns

### Management
- `add_deliverable(name, supercontext, contexts, owner?, deadline?)` — Add a deliverable
- `complete_deliverable(deliverable_id)` — Mark as complete, propagate contributions
- `search_productivity_memory(query, days_back=7)` — Semantic search across productivity data

## Key concepts

### Human vs Agent
- **Human**: User is actively directing work (includes chatting with AI, coding, browsing)
- **Agent**: An autonomous process is running without user input (Claude Code, Cursor agent, Aider)

### Memory hierarchy
- **Super Context (SC)**: High-level objective (e.g., "Product Development")
- **Context**: Specific work area (e.g., "Frontend Development")
- **Anchor**: Precise task (e.g., "Building chart components")
- Time rolls up: Anchor → Context → SC

### Matching
- Deliverables (from YAML or Asana/Notion) are matched against daily work
- Exact SC match first, then semantic embedding similarity
- Unmatched work = potential wasted effort

## Example interactions

User: "What am I working on right now?"
→ Call `get_status`

User: "Show me today's time breakdown"
→ Call `get_daily_summary`

User: "How much time did the agent spend on Vision OS this week?"
→ Call `get_deliverable_progress("Vision OS")`

User: "What work didn't contribute to anything this week?"
→ Call `get_contribution_report(days_back=7)`

User: "Add a deliverable for the GTM strategy"
→ Call `add_deliverable("GTM Strategy Doc", "Sales", "Research, Outreach, Collateral")`

User: "How has my agent usage trended this month?"
→ Call `get_trends(days_back=30)`
