// One row per person of a meeting, with everything the detail screen shows about them. All derived values (3D
// distance, facing relative to the table centre, mean SV scores, counts) are computed here, from the artifacts.

import type { ManifestEntry, MeetingSpec, Person, Script, Synthesis, VoiceValidation } from "./api";
import { dist3, facingVsCentre } from "./geometry";

export interface PersonRow {
  id: string;
  person: Person;
  /** "N5", "E1" (second chair on the E wall), "front", "speakerphone" */
  place: string;
  /** "Table · N5", "Wall · E1", "Presenter · front", "Remote · speakerphone" */
  position: string;
  /** mouth to mic in 3D; remote speakers are heard through the speakerphone, so null */
  micDistance: number | null;
  /** facing relative to the direction of the table centre (degrees), null for remote speakers */
  facing: number | null;
  /** NSC reference speaker after any stage 4 swap */
  nsc: string;
  swapped: boolean;
  /** mean cosine to the reference per SV model, over the utterances that were checked */
  sv: Record<string, number | null>;
  asrLimited: number | null;
  utterances: number | null;
  /** realised share of speech time (manifest), when stage 8 passed */
  realisedShare: number | null;
}

export const SV_MODELS: { key: string; label: string }[] = [
  { key: "sv_wavlm", label: "WavLM" },
  { key: "sv_redimnet", label: "ReDimNet2" },
];

export function wallChairLabel(m: MeetingSpec, index: number): string {
  const chair = m.wall_chairs.find((c) => c.index === index);
  if (!chair) return `chair ${index}`;
  const same = m.wall_chairs.filter((c) => c.wall === chair.wall).sort((a, b) => a.index - b.index);
  return `${chair.wall}${same.findIndex((c) => c.index === index) + 1}`;
}

export function placeOf(m: MeetingSpec, p: Person): string {
  if (p.position_type === "remote") return "speakerphone";
  if (p.position_type === "presenter") return "front";
  if (!p.seat) return "—";
  if (p.seat.kind === "table") return m.table.slots.find((s) => s.index === p.seat!.index)?.label ?? `seat ${p.seat.index}`;
  return wallChairLabel(m, p.seat.index);
}

const KIND_WORD = { table: "Table", wall: "Wall", presenter: "Presenter", remote: "Remote" } as const;

export function peopleRows(
  m: MeetingSpec,
  opts: { manifest?: ManifestEntry | null; synthesis?: Synthesis | null; voice?: VoiceValidation | null; script?: Script | null } = {},
): PersonRow[] {
  const nscAfter: Record<string, string> = {};
  for (const s of opts.manifest?.speakers ?? []) nscAfter[s.speaker_id] = s.nsc_speaker;
  for (const [k, v] of Object.entries(opts.synthesis?.voices ?? {})) nscAfter[k] ??= v.nsc_speaker;
  const realised: Record<string, number> = {};
  for (const s of opts.manifest?.speakers ?? []) realised[s.speaker_id] = s.turn_share;

  const sv: Record<string, Record<string, number[]>> = {};
  const limited: Record<string, number> = {};
  for (const u of opts.voice?.utterances ?? []) {
    limited[u.speaker_id] = (limited[u.speaker_id] ?? 0) + (u.asr_limited ? 1 : 0);
    if (u.skipped.includes("sv")) continue;
    for (const [k, v] of Object.entries(u.sv)) ((sv[u.speaker_id] ??= {})[k] ??= []).push(v);
  }
  const utts: Record<string, number> = {};
  for (const t of opts.script?.turns ?? []) utts[t.speaker_id] = (utts[t.speaker_id] ?? 0) + 1;
  const mean = (xs?: number[]) => (xs && xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

  return m.people.map((p) => {
    const place = placeOf(m, p);
    const nsc = nscAfter[p.speaker_id] ?? p.nsc_speaker.speaker;
    return {
      id: p.speaker_id,
      person: p,
      place,
      position: `${KIND_WORD[p.position_type]} · ${place}`,
      micDistance: p.mouth ? dist3(p.mouth, m.mic.position) : null,
      facing: p.mouth && p.facing_deg !== null ? facingVsCentre(p.mouth, p.facing_deg, m.table.center) : null,
      nsc,
      swapped: nsc !== p.nsc_speaker.speaker,
      sv: Object.fromEntries(SV_MODELS.map(({ key }) => [key, mean(sv[p.speaker_id]?.[key])])),
      asrLimited: opts.voice ? (limited[p.speaker_id] ?? 0) : null,
      utterances: opts.script ? (utts[p.speaker_id] ?? 0) : null,
      realisedShare: realised[p.speaker_id] ?? null,
    };
  });
}

/** The speaker the detail screen opens on: the second-highest turn share, so it is not always the chair. */
export function defaultSelection(m: MeetingSpec): string {
  const byShare = [...m.people].sort((a, b) => b.turn_share - a.turn_share || a.speaker_id.localeCompare(b.speaker_id));
  return (byShare[1] ?? byShare[0])?.speaker_id ?? "";
}
