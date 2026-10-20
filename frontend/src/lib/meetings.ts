// The meeting browser's filters and sort, shared by the browser (grid), the detail screen (Previous and Next in
// the same order) and the export link. State lives in the URL query, so every view of it agrees.

import type { MeetingSummary } from "./api";

export interface Filters {
  room: "all" | "small" | "large";
  split: "all" | "train" | "dev" | "test";
  minSpeakers: number;
  remote: boolean;
  presenters: boolean;
  wall: boolean;
  sort: "id" | "speakers" | "rt60" | "snr";
}

export const DEFAULT_FILTERS: Filters = { room: "all", split: "all", minSpeakers: 2, remote: false, presenters: false, wall: false, sort: "id" };

export function readFilters(q: URLSearchParams): Filters {
  const pick = <T extends string>(k: string, allowed: readonly T[], d: T): T => {
    const v = q.get(k) as T | null;
    return v && allowed.includes(v) ? v : d;
  };
  const n = Number(q.get("min"));
  return {
    room: pick("room", ["all", "small", "large"] as const, "all"),
    split: pick("split", ["all", "train", "dev", "test"] as const, "all"),
    minSpeakers: Number.isFinite(n) && n >= 2 && n <= 20 ? Math.round(n) : 2,
    remote: q.get("remote") === "1",
    presenters: q.get("presenters") === "1",
    wall: q.get("wall") === "1",
    sort: pick("sort", ["id", "speakers", "rt60", "snr"] as const, "id"),
  };
}

export function writeFilters(f: Filters): URLSearchParams {
  const q = new URLSearchParams();
  if (f.room !== "all") q.set("room", f.room);
  if (f.split !== "all") q.set("split", f.split);
  if (f.minSpeakers > 2) q.set("min", String(f.minSpeakers));
  if (f.remote) q.set("remote", "1");
  if (f.presenters) q.set("presenters", "1");
  if (f.wall) q.set("wall", "1");
  if (f.sort !== "id") q.set("sort", f.sort);
  return q;
}

export const kinds = (m: MeetingSummary) => new Set(m.plan?.people.map((p) => p.position_type) ?? []);

/** Measured RT60 when stage 7 ran, else the sampled target. */
export function rt60(m: MeetingSummary): number | null {
  const v = m.manifest?.acoustic?.rt60_measured;
  return typeof v === "number" ? v : (m.plan?.room.rt60_target ?? null);
}

/** Realised SNR when stage 7 ran, else the sampled value. */
export function snr(m: MeetingSummary): number | null {
  const v = m.manifest?.acoustic?.snr_db;
  return typeof v === "number" ? v : (m.plan?.acoustics.snr_db ?? null);
}

/** Same rules as the export route (viewer/api.py _matches). */
export function matches(m: MeetingSummary, f: Filters): boolean {
  const p = m.plan;
  if (!p) return false;
  const k = kinds(m);
  return (
    (f.room === "all" || p.room_class === f.room) &&
    (f.split === "all" || p.split === f.split) &&
    p.n_speakers >= f.minSpeakers &&
    (!f.remote || k.has("remote")) &&
    (!f.presenters || k.has("presenter")) &&
    (!f.wall || k.has("wall"))
  );
}

export function sortMeetings(ms: MeetingSummary[], sort: Filters["sort"]): MeetingSummary[] {
  const key: Record<Filters["sort"], (m: MeetingSummary) => number> = {
    id: () => 0,
    speakers: (m) => -(m.plan?.n_speakers ?? 0),
    rt60: (m) => -(rt60(m) ?? -1),
    snr: (m) => -(snr(m) ?? -999),
  };
  return [...ms].sort((a, b) => key[sort](a) - key[sort](b) || a.meeting_id.localeCompare(b.meeting_id));
}

export function visible(ms: MeetingSummary[], f: Filters): MeetingSummary[] {
  return sortMeetings(ms.filter((m) => matches(m, f)), f.sort);
}

/** "9 at table · 2 at wall (by choice) · 1 presenter · 2 remote" */
export function positionsLine(m: MeetingSummary): string {
  const people = m.plan?.people ?? [];
  const n = (t: string) => people.filter((p) => p.position_type === t).length;
  const parts = [`${n("table")} at table`];
  const wall = people.filter((p) => p.position_type === "wall");
  if (wall.length) {
    const reasons = [...new Set(wall.map((p) => (p.wall_reason === "by_choice" ? "by choice" : "overflow")))];
    parts.push(`${wall.length} at wall (${reasons.join(", ")})`);
  }
  if (n("presenter")) parts.push(`${n("presenter")} presenter${n("presenter") > 1 ? "s" : ""}`);
  if (n("remote")) parts.push(`${n("remote")} remote`);
  return parts.join(" · ");
}
