// The guided flow's state and API: the user's description, topics and key terms; the builder draft they turn into;
// suggestions from the decision questions (viewer/suggest_routes.py); pilot runs, review marks, voice swaps,
// exports (viewer/builder.py, viewer/export_routes.py). The draft lives in localStorage so a reload or a closed tab
// keeps the user's place; the server holds everything once a run exists.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";
import { api, type Draft } from "./api";

const enc = encodeURIComponent;

export interface KeyTerm {
  term: string;
  say: string | null;
}

/** One change a suggestion made, as dotted builder keys (suggest/effects.py Change). */
export interface Change {
  path: string;
  before: unknown;
  after: unknown;
  overlay: boolean;
}

export interface OfferOption {
  id: string;
  label: string;
  prob: number;
}

export interface Offer {
  group: string;
  question: string;
  options: OfferOption[];
  scope: string;
}

export interface Answer {
  group: string;
  option: string;
  label: string;
  prob: number;
  scope: string;
  source: string;
}

export interface Suggestion {
  applied: Answer[];
  offers: Offer[];
  confirm: Offer[];
  changes: Change[];
  notes: { id: string; text: string }[];
  highlight_exact_numbers: boolean;
  overrides: Record<string, unknown>;
  draft: Draft;
  model: string;
}

export interface ChatMessage {
  from: "user" | "system";
  text: string;
  changes?: Change[];
  offers?: Offer[];
  confirm?: Offer[];
  scope?: "sample" | "dataset";
  undone?: boolean;
  action?: "remake";
}

export interface Guided {
  who: string;
  topics: string[];
  terms: KeyTerm[];
  hours: number;
  runId: string;
  draft: Draft | null;
  chat: ChatMessage[];
  pilot: string | null;
  run: string | null;
}

const KEY = "synthetics.guided.v1";

export function emptyGuided(runId: string): Guided {
  return { who: "", topics: [""], terms: [], hours: 25, runId, draft: null, chat: [], pilot: null, run: null };
}

function load(): Guided | null {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Guided) : null;
  } catch {
    return null;
  }
}

/** The flow's state, saved on every change. `reset` starts a new dataset. */
export function useGuided(runId: string) {
  const [g, setG] = useState<Guided>(() => load() ?? emptyGuided(runId));
  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(g));
    } catch {
      /* private mode: the flow still works, it just does not survive a reload */
    }
  }, [g]);
  const update = useCallback((patch: Partial<Guided> | ((g: Guided) => Partial<Guided>)) => {
    setG((cur) => ({ ...cur, ...(typeof patch === "function" ? patch(cur) : patch) }));
  }, []);
  const reset = useCallback((id: string) => setG(emptyGuided(id)), []);
  return { g, update, reset };
}

/** Meetings for a number of audio hours, from the preview's mean meeting length (25 min until one exists). */
export function meetingsFor(hours: number, meanMinutes: number | null): number {
  const m = meanMinutes && meanMinutes > 0 ? meanMinutes : 25;
  return Math.max(1, Math.round((hours * 60) / m));
}

/** The builder overrides for the user's topics and key terms (blank entries dropped). */
export function contentOverrides(topics: string[], terms: KeyTerm[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const ts = topics.map((t) => t.trim()).filter(Boolean);
  if (ts.length) out["dynamics.topics"] = ts;
  const ks = terms.filter((t) => t.term.trim()).map((t) => ({ term: t.term.trim(), say: t.say?.trim() || null }));
  if (ks.length) out["key_terms"] = ks;
  return out;
}

// ---------------------------------------------------------------- suggestions and topics

export function useSuggest() {
  return useMutation({
    mutationFn: ({ request, draft }: { request: string; draft: Draft }) => api.post<Suggestion>("/suggest", { request, draft }),
  });
}

export function useApplyChoice() {
  return useMutation({
    mutationFn: ({ draft, choices }: { draft: Draft; choices: { group: string; option: string; scope: string }[] }) =>
      api.post<{ changes: Change[]; overrides: Record<string, unknown>; draft: Draft }>("/suggest/apply", { draft, choices }),
  });
}

export async function suggestTopics(exclude: string[], seed: number, n = 5): Promise<{ topic: string; theme: string }[]> {
  const q = new URLSearchParams({ n: String(n), seed: String(seed) });
  for (const e of exclude) if (e.trim()) q.append("exclude", e);
  return (await api.get<{ topics: { topic: string; theme: string }[] }>(`/topics?${q}`)).topics;
}

/** The draft with `changes` taken back: each path returns to its value before the change. */
export function undoChanges(draft: Draft, changes: Change[]): Draft {
  const overrides = { ...draft.overrides };
  for (const c of changes) {
    if (c.before === null || c.before === undefined) delete overrides[c.path];
    else overrides[c.path] = c.before;
  }
  return { ...draft, overrides };
}

/** Plain words for a change: the last two path parts and the values. */
export function changeLabel(c: Change): string {
  const words: Record<string, string> = {
    "acoustics.hybrid": "Calling in", "room_classes.ratio": "Room mix", "acoustics.snr_db": "Noise (SNR)",
    "turn_taking.interruption_rate": "Interruptions", "turn_taking.overlap_target": "Overlap", "acoustics.rt60": "Echo (RT60)",
    "seating.presenters.presentation": "Presentations", "turn_taking.backchannels_per_min": "Backchannels",
    "dynamics.duration_min": "Length", "room_classes.classes": "People", "personas.code_switching": "Language mixing",
    "speaker_pool.ethnicity_mix": "Race mix", "speaker_pool.split_mix": "Split mix", "turn_taking.zipf": "Who talks",
    "dynamics.turns_per_min": "Pace", "seating.wall.by_choice": "Wall seats", "seating.chair_at_head": "Chair at the head",
  };
  const key = Object.keys(words).find((k) => c.path.startsWith(k));
  const cls = /\.(small|large)\b/.exec(c.path)?.[1];
  const name = (key ? words[key] : c.path) + (cls ? ` · ${cls}` : "");
  return `${name}: ${fmt(c.before)} → ${fmt(c.after)}`;
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "default";
  if (typeof v === "number") return v <= 1 && v >= 0 && !Number.isInteger(v) ? `${Math.round(v * 100)}%` : String(v);
  if (typeof v === "object") {
    const w = v as Record<string, unknown>;
    if (w.weights) return fmt(w.weights);
    return Object.entries(w).map(([k, x]) => `${k} ${fmt(x)}`).join(", ");
  }
  return String(v);
}

// ---------------------------------------------------------------- runs: pilots, review, voices, exports

export interface ReviewMark {
  decision: "keep" | "exclude";
  note: string;
  at: string;
}

export const useReview = (run: string) =>
  useQuery({ queryKey: ["review", run], queryFn: () => api.get<Record<string, ReviewMark>>(`/runs/${enc(run)}/review`), enabled: !!run });

export function useMarkMeeting(run: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (b: { meeting_id: string; decision: "keep" | "exclude" | null; note?: string }) =>
      api.post<Record<string, ReviewMark>>(`/runs/${enc(run)}/review`, b),
    onSuccess: (marks) => qc.setQueryData(["review", run], marks),
  });
}

export function useVoiceSwap(run: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (b: { meeting: string; speaker_id: string }) =>
      api.post<{ swap: { to_speaker: string } }>(`/runs/${enc(run)}/meetings/${enc(b.meeting)}/voice-swap`, { speaker_id: b.speaker_id }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["meetings", run] }),
  });
}

export async function deletePilot(run: string): Promise<void> {
  const r = await fetch(`/api/runs/${enc(run)}`, { method: "DELETE" });
  if (!r.ok && r.status !== 404) throw new Error(`could not delete ${run}: ${r.statusText}`);
}

export function audioUrl(run: string, meeting: string, name: string): string {
  return `/api/runs/${enc(run)}/meetings/${enc(meeting)}/audio/${name}`;
}

export interface ExportOptions {
  clips: boolean;
  full: boolean;
  clean: boolean;
  room: boolean;
  noise: boolean;
}

export interface Estimate {
  meetings: number;
  root: string;
  bytes: { mixture: number; full: number; clips_clean: number; clips_room: number; noise: number; labels: number };
}

export interface ExportJob {
  id: string;
  state: "running" | "finished" | "failed";
  done: number;
  total: number;
  out: string;
  error: string | null;
}

/** Bytes of a pack with these options, from the estimate's parts. */
export function packBytes(e: Estimate, o: ExportOptions): number {
  const b = e.bytes;
  const sounds = (o.clean ? 1 : 0) + (o.room ? 1 : 0);
  let n = b.mixture + b.labels;
  if (o.full) n += b.full * sounds;
  if (o.clips) n += (o.clean ? b.clips_clean : 0) + (o.room ? b.clips_room : 0);
  if (o.noise) n += b.noise;
  return n;
}

export const useEstimate = (run: string) =>
  useQuery({ queryKey: ["estimate", run], queryFn: () => api.get<Estimate>(`/runs/${enc(run)}/exports/estimate`), enabled: !!run });

/** Export jobs of a run, newest first; polled while one is running. */
export const useExports = (run: string) =>
  useQuery({
    queryKey: ["exports", run],
    queryFn: () => api.get<ExportJob[]>(`/runs/${enc(run)}/exports`),
    enabled: !!run,
    refetchInterval: (q) => (q.state.data?.some((j) => j.state === "running") ? 1500 : false),
  });

export function useStartExport(run: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (options: ExportOptions) => api.post<ExportJob>(`/runs/${enc(run)}/exports`, { options }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["exports", run] }),
  });
}

export function downloadUrl(run: string, meeting: string, o: ExportOptions): string {
  const q = new URLSearchParams(Object.entries(o).map(([k, v]) => [k, String(v)]));
  return `/api/runs/${enc(run)}/meetings/${enc(meeting)}/download?${q}`;
}
