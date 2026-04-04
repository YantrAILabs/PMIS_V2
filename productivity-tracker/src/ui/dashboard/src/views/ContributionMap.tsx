import { useState, useEffect } from "react";
import { api } from "../api";
import type { DailyHierarchy } from "../types";

interface AnchorItem {
  sc: string;
  ctx: string;
  name: string;
  time_mins: number;
  contributed: boolean;
  deliverable_id: string | null;
}

export default function ContributionMap({ date }: { date: string }) {
  const [hierarchy, setHierarchy] = useState<DailyHierarchy | null>(null);

  useEffect(() => {
    api.daily(date).then((r) => setHierarchy(r.hierarchy)).catch(console.error);
  }, [date]);

  if (!hierarchy) return <div className="empty">Loading...</div>;

  // Flatten hierarchy to anchor list
  const anchors: AnchorItem[] = [];
  for (const [scName, sc] of Object.entries(hierarchy)) {
    for (const [ctxName, ctx] of Object.entries(sc.contexts)) {
      for (const [anchorName, anchor] of Object.entries(ctx.anchors)) {
        anchors.push({
          sc: scName,
          ctx: ctxName,
          name: anchorName,
          time_mins: anchor.time_mins,
          contributed: anchor.contributed,
          deliverable_id: anchor.deliverable_id,
        });
      }
    }
  }

  if (anchors.length === 0) return <div className="empty">No data for {date}</div>;

  const contributed = anchors.filter((a) => a.contributed);
  const unmatched = anchors.filter((a) => !a.contributed);
  const totalMins = anchors.reduce((s, a) => s + a.time_mins, 0);
  const contribMins = contributed.reduce((s, a) => s + a.time_mins, 0);

  return (
    <div>
      <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
        <div className="status-pill">
          <strong>{contributed.length}</strong> contributed
        </div>
        <div className="status-pill" style={{ borderColor: "rgba(239,68,68,0.3)" }}>
          <strong>{unmatched.length}</strong> unmatched
        </div>
        <div className="status-pill">
          <strong>{totalMins > 0 ? ((contribMins / totalMins) * 100).toFixed(0) : 0}%</strong> delivery rate
        </div>
      </div>

      <div className="contribution-grid">
        {anchors.map((a, i) => (
          <div
            key={i}
            className={`contribution-cell ${a.contributed ? "contributed" : "unmatched"}`}
          >
            <div className="anchor-name">{a.name}</div>
            <div className="anchor-meta">
              {a.sc} &gt; {a.ctx} &middot; {a.time_mins.toFixed(0)}m
              {a.deliverable_id && <>{" \u00B7 "}{a.deliverable_id}</>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
