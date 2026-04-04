import { useState, useEffect } from "react";
import { api } from "../api";
import type { DailyHierarchy, SCData } from "../types";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

const COLORS = ["#8b5cf6", "#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#ec4899", "#06b6d4", "#84cc16"];

function TimeBar({ human, agent, max }: { human: number; agent: number; max: number }) {
  const total = human + agent;
  const pctH = max > 0 ? (human / max) * 100 : 0;
  const pctA = max > 0 ? (agent / max) * 100 : 0;
  return (
    <>
      <div className="time-bar">
        <div className="time-bar-human" style={{ width: `${pctH}%` }} />
        <div className="time-bar-agent" style={{ width: `${pctA}%` }} />
      </div>
      <span className="time-bar-value">{total.toFixed(0)} min</span>
    </>
  );
}

function SCRow({ name, sc, maxMins }: { name: string; sc: SCData; maxMins: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="tree-sc">
      <div
        className="time-bar-container"
        style={{ cursor: "pointer" }}
        onClick={() => setExpanded(!expanded)}
      >
        <span className="time-bar-label">
          <span style={{ display: "inline-block", width: 16, fontSize: 10, color: "var(--text-dim)", transition: "transform 0.15s", transform: expanded ? "rotate(90deg)" : "none" }}>
            &#9654;
          </span>
          {name}
        </span>
        <TimeBar human={sc.human_mins} agent={sc.agent_mins} max={maxMins} />
      </div>

      {expanded && Object.entries(sc.contexts).map(([ctxName, ctx]) => (
        <div key={ctxName} className="tree-ctx">
          <div className="time-bar-container">
            <span className="time-bar-label">{ctxName}</span>
            <TimeBar human={ctx.human_mins} agent={ctx.agent_mins} max={maxMins} />
          </div>
          {Object.entries(ctx.anchors).map(([anchorName, anchor]) => (
            <div key={anchorName} className="tree-anchor">
              <div className="time-bar-container">
                <span className="time-bar-label">
                  {anchor.contributed && <span style={{ color: "var(--contributed)", marginRight: 4 }}>&#9679;</span>}
                  {anchorName}
                </span>
                <TimeBar human={anchor.human_mins} agent={anchor.agent_mins} max={maxMins} />
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 12px", fontSize: 13 }}>
      <div style={{ fontWeight: 600 }}>{d.name}</div>
      <div style={{ color: "var(--text-dim)" }}>{d.value.toFixed(1)} min</div>
    </div>
  );
}

export default function DailySummary({ date }: { date: string }) {
  const [hierarchy, setHierarchy] = useState<DailyHierarchy | null>(null);

  useEffect(() => {
    api.daily(date).then((r) => setHierarchy(r.hierarchy)).catch(console.error);
  }, [date]);

  if (!hierarchy) return <div className="empty">Loading...</div>;

  const entries = Object.entries(hierarchy);
  if (entries.length === 0) return <div className="empty">No data for {date}</div>;

  const maxMins = Math.max(...entries.map(([, sc]) => sc.time_mins), 1);
  const totalMins = entries.reduce((s, [, sc]) => s + sc.time_mins, 0);

  // SC pie data
  const scPieData = entries
    .map(([name, sc]) => ({ name, value: sc.time_mins }))
    .sort((a, b) => b.value - a.value);

  // Top 5 contexts
  const allContexts: { name: string; sc: string; value: number }[] = [];
  for (const [scName, sc] of entries) {
    for (const [ctxName, ctx] of Object.entries(sc.contexts)) {
      allContexts.push({ name: ctxName, sc: scName, value: ctx.time_mins });
    }
  }
  const top5Contexts = allContexts.sort((a, b) => b.value - a.value).slice(0, 5);

  // Top 5 anchors
  const allAnchors: { name: string; ctx: string; value: number }[] = [];
  for (const [, sc] of entries) {
    for (const [ctxName, ctx] of Object.entries(sc.contexts)) {
      for (const [anchorName, anchor] of Object.entries(ctx.anchors)) {
        allAnchors.push({ name: anchorName, ctx: ctxName, value: anchor.time_mins });
      }
    }
  }
  const top5Anchors = allAnchors.sort((a, b) => b.value - a.value).slice(0, 5);

  return (
    <div>
      {/* Summary pills */}
      <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
        <div className="status-pill"><strong>{totalMins.toFixed(0)}</strong> min total</div>
        <div className="status-pill"><strong>{entries.length}</strong> supercontexts</div>
        <div className="status-pill"><strong>{allAnchors.length}</strong> tasks</div>
      </div>

      <div className="legend" style={{ marginBottom: 16 }}>
        <div className="legend-item"><div className="legend-dot" style={{ background: "var(--human)" }} /> Human</div>
        <div className="legend-item"><div className="legend-dot" style={{ background: "var(--agent)" }} /> Agent</div>
      </div>

      {/* Collapsible SC tree */}
      {entries
        .sort(([, a], [, b]) => b.time_mins - a.time_mins)
        .map(([scName, sc]) => (
          <SCRow key={scName} name={scName} sc={sc} maxMins={maxMins} />
        ))}

      {/* Charts section */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginTop: 28 }}>
        {/* SC Pie Chart */}
        <div className="card">
          <h3>Time by Super Context</h3>
          <div style={{ height: 260, marginTop: 8 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={scPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={90}
                  dataKey="value"
                  paddingAngle={2}
                >
                  {scPieData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 12px", justifyContent: "center", fontSize: 11 }}>
              {scPieData.map((d, i) => (
                <span key={d.name} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: COLORS[i % COLORS.length], display: "inline-block" }} />
                  {d.name}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Top 5 Contexts Bar */}
        <div className="card">
          <h3>Top 5 Contexts</h3>
          <div style={{ height: 260, marginTop: 8 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={top5Contexts} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis type="number" stroke="var(--text-dim)" fontSize={11} unit="m" />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={120}
                  stroke="var(--text-dim)"
                  fontSize={11}
                  tick={{ fill: "var(--text)" }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {top5Contexts.map((_, i) => (
                    <Cell key={i} fill={COLORS[(i + 2) % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top 5 Anchors Bar */}
        <div className="card">
          <h3>Top 5 Tasks</h3>
          <div style={{ height: 260, marginTop: 8 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={top5Anchors} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis type="number" stroke="var(--text-dim)" fontSize={11} unit="m" />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={120}
                  stroke="var(--text-dim)"
                  fontSize={11}
                  tick={{ fill: "var(--text)" }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {top5Anchors.map((_, i) => (
                    <Cell key={i} fill={COLORS[(i + 4) % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
