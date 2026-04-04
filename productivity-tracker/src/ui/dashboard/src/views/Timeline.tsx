import { useState, useEffect } from "react";
import { api } from "../api";
import type { Segment } from "../types";

export default function Timeline({ date }: { date: string }) {
  const [segments, setSegments] = useState<Segment[] | null>(null);

  useEffect(() => {
    api.segments(date).then((r) => setSegments(r.segments)).catch(console.error);
  }, [date]);

  if (!segments) return <div className="empty">Loading...</div>;
  if (segments.length === 0) return <div className="empty">No segments for {date}</div>;

  // Find the time range of the day
  const starts = segments.map((s) => new Date(s.timestamp_start).getTime());
  const ends = segments.map((s) => new Date(s.timestamp_end).getTime());
  const dayStart = Math.min(...starts);
  const dayEnd = Math.max(...ends);
  const daySpan = dayEnd - dayStart || 1;

  const formatTime = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div>
      <div className="legend">
        <div className="legend-item"><div className="legend-dot" style={{ background: "var(--human)" }} /> Human</div>
        <div className="legend-item"><div className="legend-dot" style={{ background: "var(--agent)" }} /> Agent</div>
      </div>

      <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 12 }}>
        {formatTime(segments[0].timestamp_start)} &mdash; {formatTime(segments[segments.length - 1].timestamp_end)}
      </div>

      {segments.map((seg) => {
        const start = new Date(seg.timestamp_start).getTime();
        const end = new Date(seg.timestamp_end).getTime();
        const left = ((start - dayStart) / daySpan) * 100;
        const width = Math.max(((end - start) / daySpan) * 100, 0.5);

        return (
          <div key={seg.target_segment_id} className="timeline-row">
            <span className="timeline-label" title={seg.anchor}>{seg.anchor}</span>
            <div className="timeline-track">
              <div
                className={`timeline-block ${seg.worker}`}
                style={{ left: `${left}%`, width: `${width}%` }}
                title={`${seg.anchor} (${seg.worker}) — ${Math.round(seg.target_segment_length_secs / 60)}m`}
              >
                {width > 8 ? `${Math.round(seg.target_segment_length_secs / 60)}m` : ""}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
