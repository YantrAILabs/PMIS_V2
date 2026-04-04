import type {
  StatusResponse,
  DailyResponse,
  DeliverablesResponse,
  SegmentsResponse,
  TrendsResponse,
} from "./types";

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  status: () => get<StatusResponse>("/api/status"),
  daily: (date: string) => get<DailyResponse>(`/api/daily/${date}`),
  deliverables: (date: string) => get<DeliverablesResponse>(`/api/deliverables/${date}`),
  segments: (date: string) => get<SegmentsResponse>(`/api/segments/${date}`),
  trends: (days = 30) => get<TrendsResponse>(`/api/trends?days=${days}`),
};
