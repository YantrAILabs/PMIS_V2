import { useState, useEffect } from "react";
import { api } from "../api";
import type { DeliverablesResponse } from "../types";

export default function Deliverables({ date }: { date: string }) {
  const [data, setData] = useState<DeliverablesResponse | null>(null);

  useEffect(() => {
    api.deliverables(date).then(setData).catch(console.error);
  }, [date]);

  if (!data) return <div className="empty">Loading...</div>;
  if (data.deliverables.length === 0 && data.unmatched.length === 0)
    return <div className="empty">No data for {date}</div>;

  return (
    <div>
      <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
        <div className="status-pill">
          <strong>{data.total_tracked_mins.toFixed(0)}</strong> min tracked
        </div>
        <div className="status-pill">
          <strong>{data.total_matched_mins.toFixed(0)}</strong> min matched
        </div>
        <div className="status-pill">
          <strong>{data.total_tracked_mins > 0 ? ((data.total_matched_mins / data.total_tracked_mins) * 100).toFixed(0) : 0}%</strong> utilization
        </div>
      </div>

      {data.deliverables.map((d) => (
        <div key={d.id} className="deliverable-card">
          <div className="deliverable-header">
            <h3>{d.name}</h3>
            <span className={`leverage-badge ${d.agent_leverage_pct > 20 ? "high" : "low"}`}>
              {d.agent_leverage_pct > 0 ? `${d.agent_leverage_pct.toFixed(0)}% agent` : "100% human"}
            </span>
          </div>
          <div className="deliverable-meta">
            {d.supercontext} &middot; {d.total_mins.toFixed(0)} min total
            &middot; {d.human_mins.toFixed(0)} human / {d.agent_mins.toFixed(0)} agent
          </div>

          <div style={{ marginTop: 8 }}>
            {d.contexts.map((ctx) => (
              <div key={ctx.name} style={{ marginBottom: 6 }}>
                <div style={{ fontSize: 13, marginBottom: 4, fontWeight: 500 }}>{ctx.name}</div>
                {ctx.anchors.map((a) => {
                  const pct = d.total_mins > 0 ? (a.time_mins / d.total_mins) * 100 : 0;
                  return (
                    <div key={a.name} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                      <span style={{ fontSize: 12, color: "var(--text-dim)", minWidth: 160 }}>{a.name}</span>
                      <div className="time-bar" style={{ maxWidth: 300 }}>
                        <div className="time-bar-human" style={{ width: `${(a.human_mins / (d.total_mins || 1)) * 100}%` }} />
                        <div className="time-bar-agent" style={{ width: `${(a.agent_mins / (d.total_mins || 1)) * 100}%` }} />
                      </div>
                      <span style={{ fontSize: 11, color: "var(--text-dim)" }}>{a.time_mins.toFixed(0)}m</span>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      ))}

      {data.unmatched.length > 0 && (
        <div className="card" style={{ borderColor: "rgba(239,68,68,0.3)" }}>
          <h3 style={{ color: "var(--unmatched)" }}>Unmatched Work</h3>
          <div style={{ marginTop: 8 }}>
            {data.unmatched.map((u) => (
              <div key={u.id} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid var(--border)", fontSize: 13 }}>
                <span>{u.supercontext} &gt; {u.context} &gt; {u.anchor}</span>
                <span style={{ color: "var(--text-dim)" }}>{u.time_mins.toFixed(0)} min</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
