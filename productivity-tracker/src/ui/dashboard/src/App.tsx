import { useState, useEffect } from "react";
import { api } from "./api";
import type { StatusResponse } from "./types";
import DailySummary from "./views/DailySummary";
import Deliverables from "./views/Deliverables";
import Timeline from "./views/Timeline";
import ContributionMap from "./views/ContributionMap";
import Trends from "./views/Trends";
import { format } from "date-fns";

const TABS = ["Daily Summary", "Deliverables", "Timeline", "Contributions", "Trends"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Daily Summary");
  const [date, setDate] = useState(format(new Date(), "yyyy-MM-dd"));
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    api.status().then(setStatus).catch(console.error);
  }, []);

  return (
    <div className="app">
      <div className="header">
        <h1>Productivity Tracker</h1>
        <div className="header-right">
          {status && (
            <div className="status-pill">
              <strong>{status.segments}</strong> segments
              <strong>{status.total_mins}</strong> min tracked
            </div>
          )}
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
          />
        </div>
      </div>

      <div className="tabs">
        {TABS.map((t) => (
          <button
            key={t}
            className={`tab ${t === tab ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "Daily Summary" && <DailySummary date={date} />}
      {tab === "Deliverables" && <Deliverables date={date} />}
      {tab === "Timeline" && <Timeline date={date} />}
      {tab === "Contributions" && <ContributionMap date={date} />}
      {tab === "Trends" && <Trends />}
    </div>
  );
}
