import { useState, useEffect } from "react";
import { api } from "../api";
import type { TrendDay } from "../types";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

export default function Trends() {
  const [data, setData] = useState<TrendDay[] | null>(null);

  useEffect(() => {
    api.trends(30).then((r) => setData(r.totals)).catch(console.error);
  }, []);

  if (!data) return <div className="empty">Loading...</div>;
  if (data.length === 0) return <div className="empty">No trend data yet. Run the tracker for a few days.</div>;

  const chartData = data.map((d) => ({
    date: d.date.slice(5), // MM-DD
    Human: d.human,
    Agent: d.agent,
    Total: d.total,
  }));

  return (
    <div>
      <div className="card">
        <h3>Daily Tracked Time (minutes)</h3>
        <div style={{ marginTop: 16, height: 360 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" stroke="var(--text-dim)" fontSize={12} />
              <YAxis stroke="var(--text-dim)" fontSize={12} />
              <Tooltip
                contentStyle={{
                  background: "var(--bg-card)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  color: "var(--text)",
                }}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="Human"
                stackId="1"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.4}
              />
              <Area
                type="monotone"
                dataKey="Agent"
                stackId="1"
                stroke="#22c55e"
                fill="#22c55e"
                fillOpacity={0.4}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {data.length > 1 && (
        <div className="card">
          <h3>Agent Leverage Over Time</h3>
          <div style={{ marginTop: 16, height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartData.map((d) => ({
                  ...d,
                  "Agent %": d.Total > 0 ? Math.round((d.Agent / d.Total) * 100) : 0,
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="date" stroke="var(--text-dim)" fontSize={12} />
                <YAxis stroke="var(--text-dim)" fontSize={12} unit="%" />
                <Tooltip
                  contentStyle={{
                    background: "var(--bg-card)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    color: "var(--text)",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="Agent %"
                  stroke="#22c55e"
                  fill="#22c55e"
                  fillOpacity={0.2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
