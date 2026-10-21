// Fetch wrapper, React Query hooks and the shapes of the files the viewer reads. The backend's Pydantic models
// (backend/synthetics/artifacts/*.py) and viewer/api.py are the source of truth for every field name.

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { RoomClass } from "./colors";

/** One problem with a field: `loc` is the dotted config path (or "yaml", "run_id", "confirm"). */
export interface FieldError {
  loc: string;
  msg: string;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public path: string,
    public errors: FieldError[] = [],
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit, as: "json" | "text" = "json"): Promise<T> {
  const r = await fetch(`/api${path}`, init);
  if (!r.ok) {
    let msg = r.statusText;
    let errors: FieldError[] = [];
    try {
      const j = await r.json();
      const d = j.detail ?? j;
      if (typeof d === "string") msg = d;
      else if (d && typeof d.message === "string") {
        msg = d.message;
        errors = Array.isArray(d.errors) ? d.errors : [];
      } else msg = JSON.stringify(d);
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, msg, `/api${path}`, errors);
  }
  return (as === "text" ? await r.text() : await r.json()) as T;
}

async function get<T>(path: string, as: "json" | "text" = "json"): Promise<T> {
  return request<T>(path, undefined, as);
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
}

export const api = { get, post };

// ---------------------------------------------------------------- shapes

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
export type Wall = "N" | "E" | "S" | "W";
export type PositionType = "table" | "wall" | "presenter" | "remote";
export type Split = "train" | "dev" | "test";

export interface RunInfo {
  id: string;
  meetings: number;
  done: number;
  seed: number;
  n_meetings: number;
  config_hash: string;
  revision: number;
  updated: number;
}

export interface StageStatus {
  name: string;
  state: "pending" | "running" | "pass" | "fail" | "retry" | "skipped" | "dropped";
  attempts: number;
  started_at: string | null;
  ended_at: string | null;
  detail: string;
  metrics: Record<string, number | string>;
}

export interface Screen {
  wall: Wall;
  center: number;
  width: number;
  bottom: number;
  top: number;
}

export interface Room {
  length: number;
  width: number;
  height: number;
  front_wall: Wall;
  walls: Record<Wall, "glass" | "plasterboard">;
  floor: "carpet" | "vinyl";
  ceiling: "acoustic_tile" | "plasterboard";
  rt60_target: number;
  screen: Screen | null;
}

export interface SeatRef {
  kind: "table" | "wall";
  index: number;
}

export interface Speakerphone {
  position: Vec3;
  remote_speakers: string[];
}

export interface DeviceChain {
  agc: boolean;
  band_limit_hz: number | null;
  codec_kbps: number | null;
  clipping_level: number | null;
  aec_artifacts: boolean;
}

/** The parts of meeting.json the grid needs (viewer/api.py plan_of). */
export interface Plan {
  room_class: RoomClass;
  n_speakers: number;
  split: Split;
  seed: number;
  fit_resamples: number;
  room: Room;
  table: { shape: "rect" | "round" | "oval"; center: Vec2; length: number; width: number; capacity: number; slots: { xy: Vec2; occupied_by: string | null }[] };
  wall_chairs: { xy: Vec2; wall: Wall; occupied_by: string | null }[];
  people: { speaker_id: string; position_type: PositionType; wall_reason: "overflow" | "by_choice" | null; mouth: Vec3 | null; is_chair: boolean; turn_share: number }[];
  mic: { type: "laptop" | "centre"; position: Vec3; channels: number };
  speakerphone: Speakerphone | null;
  hybrid: boolean;
  presentation: boolean;
  acoustics: { rt60: number; snr_db: number };
  device_chain: DeviceChain;
}

export interface TurnTaking {
  duration: number;
  speech_time: number;
  overlap_share: number;
  backchannels_per_min: number;
  interruption_share: number;
  long_pauses: number;
  voiced_time: number;
  overlap_time: number;
  overlap_target: number | null;
  speaker_changes: number;
  interruptions: number;
  backchannels: number;
}

export type Scalar = number | string | boolean | null;

export interface ManifestEntry {
  run_id: string;
  meeting_id: string;
  seed: number;
  split: Split;
  room_class: RoomClass;
  n_speakers: number;
  duration: number;
  sampled: Record<string, Scalar>;
  stages_enabled: Record<string, boolean>;
  position_counts: Record<PositionType, number>;
  speakers: { speaker_id: string; nsc_speaker: string; split: Split; position_type: PositionType; wall_reason: string | null; turn_share: number; asr_limited: number }[];
  judge: Record<string, Scalar>;
  voice: Record<string, Scalar>;
  turn_taking: TurnTaking;
  files: Record<string, string>;
  acoustic: Record<string, Scalar>;
  labels: Record<string, Scalar>;
}

export type MeetingState = "done" | "failed" | "running" | "pending";

export interface MeetingSummary {
  meeting_id: string;
  state: MeetingState;
  stages: Record<string, StageStatus>;
  plan: Plan | null;
  manifest: ManifestEntry | null;
}

export interface Jobs {
  run: string;
  meetings: number;
  states: Partial<Record<MeetingState, number>>;
  stages: Record<string, Record<string, number>>;
  running_stages: string[];
  active: boolean;
  last_update: string | null;
  process: ProcessState;
}

// ---------------------------------------------------------------- run builder (viewer/builder.py)

export type ProcessState = "idle" | "running" | "finished" | "failed" | "stopped" | "lost";

export interface RunProcess {
  state: ProcessState;
  pid?: number;
  live?: boolean;
  started_at?: string;
  ended_at?: string | null;
  exit_code?: number | null;
  error?: string | null;
}

export interface ProcessInfo {
  run: string;
  process: RunProcess;
  log: string;
  live: boolean;
  gpu_services: Record<string, string[]>;
}

/** What the form sends: the run fields, dotted overrides, and a YAML overlay (viewer/builder.py Draft). */
export interface Draft {
  run_id: string;
  n_meetings: number;
  seed: number;
  overrides: Record<string, unknown>;
  yaml: string;
}

export interface PreviewMeeting {
  meeting_id: string;
  room_class: RoomClass;
  n_speakers: number;
  split: Split;
  rt60: number;
  snr_db: number;
  hybrid: boolean;
  presentation: boolean;
  duration_min: number;
  table_shape: string;
  mic_type: string;
  fit_resamples: number;
  positions: Record<PositionType, number>;
}

export interface SamplerCheck {
  name: string;
  target: number | string;
  observed: number | string;
  tolerance: number | null;
  n: number;
  ok: boolean;
  note: string;
}

export interface Preview {
  config_hash: string;
  revision: number;
  n_meetings: number;
  n_sampled: number;
  overrides: string[];
  meetings: PreviewMeeting[];
  plans: (Plan & { meeting_id: string })[];
  checks: SamplerCheck[];
  failures: { meeting_id: string; error: string }[];
  live: boolean;
  gpu_services: Record<string, string[]>;
  stages_enabled: Record<string, boolean>;
  audio_hours: number;
  estimate_hours: number | null;
}

export interface Defaults {
  config: Record<string, unknown>;
  preview_max: number;
  reserved: string[];
}

export interface Persona {
  display_name: string;
  role: string;
  seniority: string;
  verbosity: "low" | "medium" | "high";
  gender: "F" | "M";
  age_band: string;
  ethnicity: string;
  l1: string;
  accent: string;
  code_switching: "low" | "medium" | "high";
  switch_languages: string[];
}

export interface Person {
  speaker_id: string;
  position_type: PositionType;
  wall_reason: "overflow" | "by_choice" | null;
  seat: SeatRef | null;
  mouth: Vec3 | null;
  facing_deg: number | null;
  presenter: { spot: Vec2; drift_radius: number; facing_screen_p: number; return_seat: SeatRef | null } | null;
  is_chair: boolean;
  turn_share: number;
  nsc_speaker: { speaker: string; split: Split; gender: string; age: string; ethnicity: string; first_language: string; languages: string[] };
  persona: Persona;
}

export interface TableSlot {
  index: number;
  label: string;
  side: string;
  edge_s: number;
  edge_xy: Vec2;
  seat_xy: Vec2;
  occupied_by: string | null;
}

/** meeting.json (artifacts/meeting.py MeetingSpec). */
export interface MeetingSpec {
  meeting_id: string;
  seed: number;
  split: Split;
  room_class: RoomClass;
  n_speakers: number;
  fit_resamples: number;
  room: Room;
  table: { shape: "rect" | "round" | "oval"; center: Vec2; length: number; width: number; height: number; seat_pitch: number; capacity: number; slots: TableSlot[] };
  wall_chairs: { index: number; wall: Wall; offset: number; xy: Vec2; occupied_by: string | null }[];
  people: Person[];
  mic: { type: "laptop" | "centre"; position: Vec3; channels: number; array_geometry: Vec3[]; near_speaker: string | null };
  speakerphone: Speakerphone | null;
  acoustics: {
    rt60: number;
    snr_db: number;
    noise_sources: { kind: string; position: Vec3 | null }[];
    events: Record<string, number>;
    events_per_10min: number;
    orientation_max_deg: number;
    orientation_resample_p: number;
  };
  device_chain: DeviceChain;
  dynamics: {
    meeting_type: string;
    topic: string;
    duration_min: number;
    n_turns_target: number;
    zipf_exponent: number;
    turn_shares: Record<string, number>;
    chair_id: string;
    presentation: { presenters: string[]; start_turn: number; end_turn: number; returns: Record<string, SeatRef | null> } | null;
    hybrid: boolean;
    remote_speakers: string[];
  };
}

export type Interaction = "normal" | "interruption" | "backchannel" | "overlap_start" | "side" | "presentation";

export interface Script {
  meeting_id: string;
  brief: string;
  turns: { turn_id: string; speaker_id: string; interaction: Interaction; text: string; text_norm: string }[];
  generator: { service: string; template_version: string; round: number; thinking: boolean; temperature: number };
}

export interface ScriptValidation {
  decision: string;
  round: number;
  primary: { service: string; rubric_version: string; scores: { overall: number; persona_consistency: Record<string, number> } } | null;
  tiebreak: { service: string; scores: { overall: number; persona_consistency: Record<string, number> } } | null;
}

export interface VoiceValidation {
  utterances: { turn_id: string; speaker_id: string; sv: Record<string, number>; asr_limited: boolean; passed: boolean; skipped: string[] }[];
  speakers: { speaker_id: string; consistency: Record<string, number>; consistency_utts: number; passed: boolean }[];
  kept_failed: string[];
  decision: string;
}

export interface Timeline {
  segments: { turn_id: string; speaker_id: string; kind: Interaction; start: number; end: number; truncated_at: number | null }[];
  stats: TurnTaking;
  dropped_turns: string[];
}

export interface Synthesis {
  service: string;
  voices: Record<string, { speaker_id: string; nsc_speaker: string; clip_id: string }>;
}

export interface AcousticRender {
  rt60_target: number;
  rt60_measured: number;
  snr_db: number;
  channels: number;
  sample_rate: number;
  noise: { kind: string; source: string; start: number | null }[];
  device: Record<string, Scalar>;
}

/** Only the parts of config.yaml the viewer reads; the rest is passed through untyped. */
export interface Dist {
  dist: string;
  lo?: number;
  hi?: number;
  p?: number;
  value?: number;
  weights?: Record<string, number>;
}
export interface RunConfig {
  run_id: string;
  seed: number;
  n_meetings: number;
  sample_rate: number;
  room_classes: { ratio: Dist; classes: Record<RoomClass, { speakers: Dist }> };
  seating: { wall: { by_choice: Dist }; presenters: { presentation: Record<RoomClass, Dist> } };
  acoustics: { rt60: Record<RoomClass, Dist>; snr_db: Record<RoomClass, Dist>; hybrid: Record<RoomClass, Dist>; mic: { laptop_distance: Dist } };
  turn_taking: { overlap_target: Record<RoomClass, Dist>; backchannels_per_min: Dist; interruption_rate: Dist; zipf: { exponent: number } };
  [k: string]: unknown;
}

// ---------------------------------------------------------------- hooks

const enc = encodeURIComponent;

export const useRuns = () => useQuery({ queryKey: ["runs"], queryFn: () => get<RunInfo[]>("/runs") });

export const useConfig = (run: string) =>
  useQuery({ queryKey: ["config", run], queryFn: () => get<{ config_hash: string; config: RunConfig }>(`/runs/${enc(run)}/config`), enabled: !!run });

export const useMeetings = (run: string) =>
  useQuery({ queryKey: ["meetings", run], queryFn: () => get<MeetingSummary[]>(`/runs/${enc(run)}/meetings`), enabled: !!run });

export const useJobs = (run: string) =>
  useQuery({ queryKey: ["jobs", run], queryFn: () => get<Jobs>(`/runs/${enc(run)}/jobs`), enabled: !!run, refetchInterval: 5000 });

export function artifactUrl(run: string, meeting: string, name: string): string {
  return `/api/runs/${enc(run)}/meetings/${enc(meeting)}/${name}`;
}

/** A meeting file; `null` when it does not exist yet (the stage that writes it has not passed). */
export function useArtifact<T>(run: string, meeting: string, name: string, enabled = true) {
  return useQuery({
    queryKey: ["artifact", run, meeting, name],
    enabled: enabled && !!run && !!meeting,
    queryFn: async () => {
      try {
        return await get<T>(`/runs/${enc(run)}/meetings/${enc(meeting)}/${name}`, name.endsWith(".txt") ? "text" : "json");
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
    staleTime: 60_000,
  });
}

export function exportUrl(run: string, params: Record<string, string | number | boolean | undefined>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "" && v !== false) q.set(k, String(v));
  return `/api/runs/${enc(run)}/export${q.size ? `?${q}` : ""}`;
}

export const useViewer = () =>
  useQuery({ queryKey: ["viewer"], queryFn: () => get<{ read_only: boolean; runs_root: string }>("/viewer"), staleTime: Infinity });

export const useDefaults = (enabled = true) =>
  useQuery({ queryKey: ["defaults"], queryFn: () => get<Defaults>("/defaults"), staleTime: Infinity, enabled });

/** Stage 1 in memory for a draft. `draft` should be debounced by the caller; `null` waits. */
export const usePreview = (draft: Draft | null) =>
  useQuery({
    queryKey: ["preview", draft],
    queryFn: () => post<Preview>("/preview", draft),
    enabled: !!draft,
    placeholderData: keepPreviousData,
    retry: false,
    staleTime: Infinity,
  });

export const useProcess = (run: string) =>
  useQuery({
    queryKey: ["process", run],
    queryFn: () => get<ProcessInfo>(`/runs/${enc(run)}/process`),
    enabled: !!run,
    refetchInterval: (q) => (q.state.data?.process.state === "running" ? 3000 : 15000),
  });

export function useCreateRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (draft: Draft) => post<{ id: string; config_hash: string; live: boolean }>("/runs", draft),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runs"] }),
  });
}

/** Start or stop a run's background process; every view of the run refreshes afterwards. */
export function useRunAction(run: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ action, confirm }: { action: "start" | "stop"; confirm?: boolean }) =>
      post<ProcessInfo>(`/runs/${enc(run)}/${action}`, action === "start" ? { confirm: !!confirm } : undefined),
    onSuccess: (info) => {
      qc.setQueryData(["process", run], info);
      qc.invalidateQueries({ queryKey: ["jobs", run] });
      qc.invalidateQueries({ queryKey: ["meetings", run] });
    },
  });
}
