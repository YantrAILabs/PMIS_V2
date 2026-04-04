export interface StatusResponse {
  segments: number;
  total_mins: number;
}

export interface AnchorData {
  time_mins: number;
  human_mins: number;
  agent_mins: number;
  contributed: boolean;
  deliverable_id: string | null;
}

export interface ContextData {
  time_mins: number;
  human_mins: number;
  agent_mins: number;
  anchors: Record<string, AnchorData>;
}

export interface SCData {
  time_mins: number;
  human_mins: number;
  agent_mins: number;
  contexts: Record<string, ContextData>;
}

export type DailyHierarchy = Record<string, SCData>;

export interface DailyResponse {
  date: string;
  hierarchy: DailyHierarchy;
}

export interface DeliverableAnchor {
  name: string;
  time_mins: number;
  human_mins: number;
  agent_mins: number;
}

export interface DeliverableContext {
  name: string;
  time_mins: number;
  human_mins: number;
  agent_mins: number;
  anchors: DeliverableAnchor[];
}

export interface Deliverable {
  id: string;
  name: string;
  supercontext: string;
  total_mins: number;
  human_mins: number;
  agent_mins: number;
  agent_leverage_pct: number;
  contexts: DeliverableContext[];
}

export interface UnmatchedEntry {
  id: string;
  supercontext: string;
  context: string;
  anchor: string;
  time_mins: number;
  human_mins: number;
  agent_mins: number;
}

export interface DeliverablesResponse {
  date: string;
  deliverables: Deliverable[];
  unmatched: UnmatchedEntry[];
  total_tracked_mins: number;
  total_matched_mins: number;
}

export interface Segment {
  target_segment_id: string;
  timestamp_start: string;
  timestamp_end: string;
  window_name: string;
  platform: string;
  medium: string;
  context: string;
  supercontext: string;
  anchor: string;
  target_segment_length_secs: number;
  worker: string;
  detailed_summary: string;
}

export interface SegmentsResponse {
  segments: Segment[];
}

export interface TrendDay {
  date: string;
  total: number;
  human: number;
  agent: number;
}

export interface TrendsResponse {
  totals: TrendDay[];
}
