// The run builder's form model. Each curated field is one leaf of config.yaml, addressed by its dotted path; the
// form sends only the leaves that differ from the defaults, as `init-run --set` overrides. A distribution is one
// leaf (the backend replaces it whole). Everything else goes in the YAML overlay.

export type FieldKind = "range" | "prob" | "number" | "weights" | "toggle";

/** The hover text of a setting: what it is, and why it matters for the training data. */
export interface Info {
  what: string;
  why: string;
}

export interface Field {
  path: string;
  label: string;
  kind: FieldKind;
  /** shown after the inputs; for `range` and `number` */
  unit?: string;
  /** a `range` or `number` shown in % (stored as a 0-1 share) */
  percent?: boolean;
  step?: number;
  min?: number;
  max?: number;
  /** integer bounds (IntUniform) */
  int?: boolean;
  /** for `weights`: the keys, in order */
  keys?: string[];
  hint?: string;
  info: Info;
}

export interface Section {
  id: string;
  title: string;
  note: string;
  fields: Field[];
}

type Rest = Omit<Field, "path" | "label" | "info">;
const perClass = (path: (c: string) => string, label: string, rest: Rest, info: (c: "small" | "large") => Info): Field[] =>
  (["small", "large"] as const).map((c) => ({ path: path(c), label: `${label} · ${c}`, ...rest, info: info(c) }));
const both = (info: Info) => () => info;

export const SECTIONS: Section[] = [
  {
    id: "rooms",
    title: "Rooms and speakers",
    note: "Stage 1. The class boundary is fixed: small rooms have 2 to 6 speakers, large rooms 7 to 20.",
    fields: [
      {
        path: "room_classes.ratio", label: "Room class mix", kind: "weights", keys: ["small", "large"], hint: "Share of meetings in each class",
        info: {
          what: "The share of meetings drawn as small rooms (2 to 6 speakers, laptop mic) or large rooms (7 to 20 speakers, a mic in the middle of the table).",
          why: "Large meetings are the rarest real data and the hardest for diarisation, so the default leans 70% large. Lower it if the model will mostly meet small calls.",
        },
      },
      {
        path: "room_classes.classes.small.speakers", label: "Speakers · small", kind: "range", int: true, min: 2, max: 6,
        info: {
          what: "How many people take part in a small-room meeting, drawn evenly from the range. Presenters and remote participants count.",
          why: "The number of speakers is the main thing that makes diarisation hard. The range must stay inside 2 to 6, because the class boundary is fixed at 7.",
        },
      },
      {
        path: "room_classes.classes.large.speakers", label: "Speakers · large", kind: "range", int: true, min: 7, max: 20,
        info: {
          what: "How many people take part in a large-room meeting, drawn evenly from the range. Presenters and remote participants count.",
          why: "Meetings with more than 15 speakers are where current models do worst and real data is scarcest. The range must stay inside 7 to 20.",
        },
      },
      ...perClass((c) => `dynamics.duration_min.${c}`, "Meeting length", { kind: "range", unit: "min", min: 1, max: 180, step: 1 }, both({
        what: "The target length of each meeting in minutes, drawn evenly from the range. Stage 2 plans the number of turns from it.",
        why: "Longer meetings give more speech per speaker and more of the rare events, such as quiet speakers finally talking. On real backends each minute of audio takes roughly 75 seconds to make (m9-e2e).",
      })),
    ],
  },
  {
    id: "turns",
    title: "Turn-taking",
    note: "Stages 2 and 6. The turn plan meets these by construction; stage 6 scales overlaps to the target.",
    fields: [
      ...perClass((c) => `turn_taking.overlap_target.${c}`, "Overlap target", { kind: "range", percent: true, min: 0, max: 60, step: 1, unit: "% of speech" }, both({
        what: "The share of speech time where two or more people talk at once. Each meeting draws its own target from the range, and stage 6 stretches overlaps until the meeting reaches it.",
        why: "Overlapped speech is where diarisation and ASR fail most. Higher targets make harder training data, but past a point the meetings stop sounding natural.",
      })),
      {
        path: "turn_taking.interruption_rate", label: "Interruptions", kind: "range", percent: true, min: 0, max: 50, step: 1, unit: "% of turn changes",
        info: {
          what: "The share of turn changes where the next speaker cuts in and the current speaker stops early, 0.3 to 1.5 s after the interruption starts.",
          why: "Interruptions give short, cut-off turns and abrupt speaker changes, which models trained on tidy turn-taking handle badly.",
        },
      },
      {
        path: "turn_taking.backchannels_per_min", label: "Backchannels", kind: "range", min: 0, max: 20, step: 0.5, unit: "per min",
        info: {
          what: "Short responses such as \"mm\", \"ok\" or \"yah\" that a listener says during someone else's turn, per minute of meeting.",
          why: "They are brief overlapped speech from a second person. Models often miss them or give them to the main speaker, so the training data needs plenty of them.",
        },
      },
      {
        path: "turn_taking.zipf.exponent", label: "Turn share skew (Zipf exponent)", kind: "number", min: 0, max: 3, step: 0.1, hint: "0 is equal shares; higher gives a longer tail of quiet people",
        info: {
          what: "How unevenly the turns are shared. The speaker ranked n gets a share proportional to 1 / n to this power, and the chair takes the top share.",
          why: "At 0 everyone talks equally. Higher values give a few dominant voices and a long tail of people who rarely speak, which is typical of large meetings and hard for diarisation, since a rare speaker has little audio to learn from.",
        },
      },
      {
        path: "turn_taking.overlap_share", label: "Transitions that overlap", kind: "number", percent: true, min: 0, max: 100, step: 1, unit: "%",
        info: {
          what: "The share of speaker changes where the next person starts 0.3 to 2.0 s before the previous one finishes. The other changes have a short gap of silence.",
          why: "This is how overlap arises naturally at turn changes. Set it too low and stage 6 has to stretch overlaps a lot to reach the overlap target, which sounds less natural.",
        },
      },
      {
        path: "dynamics.turns_per_min", label: "Turns", kind: "range", min: 1, max: 30, step: 0.5, unit: "per min",
        info: {
          what: "How many turns the meeting has per minute, not counting backchannels, drawn from the range for each meeting.",
          why: "More turns per minute means shorter turns and more speaker changes, which is harder for diarisation. It also sets how much script stage 2 has to write.",
        },
      },
    ],
  },
  {
    id: "seating",
    title: "Seating and participants",
    note: "Stage 1. Presenters stand at the front; remote participants are heard through a speakerphone.",
    fields: [
      ...perClass((c) => `seating.presenters.presentation.${c}`, "Presentation segment", { kind: "prob" }, both({
        what: "The chance that a meeting has a presentation: one or two people stand at the front by the screen and talk for a stretch, then may go back to their seat.",
        why: "Standing presenters are further from the mic, move around and sometimes face the screen, so their speech is quieter and more reverberant. A presentation often adds a projector fan to the noise too.",
      })),
      ...perClass((c) => `acoustics.hybrid.${c}`, "Hybrid meeting", { kind: "prob" }, both({
        what: "The chance that some participants join remotely and are heard through a speakerphone on the table.",
        why: "Remote speech comes through a codec and a loudspeaker, with echo cancellation artifacts, so it sounds very different from people in the room. Models need to hear both kinds.",
      })),
      {
        path: "seating.wall.by_choice", label: "Wall seats by choice · large", kind: "prob", hint: "When the table still has room",
        info: {
          what: "In large rooms where the table still has free seats, the chance that one to three people sit on chairs along the wall anyway. People also go to the wall when the table is full.",
          why: "Wall seats are further from the table mic, so those speakers are quieter and more reverberant, the kind of voice models tend to miss.",
        },
      },
    ],
  },
  {
    id: "acoustics",
    title: "Acoustics and device",
    note: "Sampled in stage 1, rendered in stage 7. In revision 2 runs, changing these re-runs only stages 1, 7 and 8.",
    fields: [
      ...perClass((c) => `acoustics.rt60.${c}`, "RT60", { kind: "range", unit: "s", min: 0.1, max: 2, step: 0.05 }, both({
        what: "Reverberation time: how long a sound takes to die away by 60 dB in the room. Each meeting draws a target, and stage 7 tunes the room's absorption until it measures within 15% of it.",
        why: "Reverberation smears speech, most of all for people far from the mic. Large rooms are usually more reverberant, which is why their default range is higher.",
      })),
      ...perClass((c) => `acoustics.snr_db.${c}`, "SNR", { kind: "range", unit: "dB", min: -5, max: 50, step: 1 }, both({
        what: "How much louder speech is than the steady background noise (air conditioning, projector fan) at the mic, in dB. Keyboard noise and events such as doors, chairs and cups are added on top, relative to the speech.",
        why: "Lower SNR means noisier recordings. Data that is too clean leaves a model unready for real rooms; data that is too noisy stops looking like a meeting.",
      })),
      {
        path: "acoustics.mic.centre_array", label: "Centre mic is an array", kind: "prob",
        info: {
          what: "Large rooms record from a mic in the middle of the table. This is the chance that it is a 4-channel array (3.5 cm radius) instead of a single mic. Arrays also write mixture_mc.wav.",
          why: "Multichannel audio lets the fine-tuning team use spatial cues to separate speakers. Every meeting still gets a single-channel mixture.",
        },
      },
      {
        path: "acoustics.device_chain.agc", label: "AGC", kind: "prob",
        info: {
          what: "The chance that the recording device applies automatic gain control, which turns quiet passages up and loud ones down over a few seconds.",
          why: "Laptops and conference phones usually do this. Without AGC the levels are left as the room made them, so far talkers stay quiet; with it, the model learns to cope with levels that pump up and down.",
        },
      },
      {
        path: "acoustics.device_chain.band_limit", label: "Band limit", kind: "prob",
        info: {
          what: "The chance that the recording loses everything above 5 to 7.5 kHz, as cheap mics and some call systems do.",
          why: "High frequencies help tell voices and some consonants apart. Some band-limited audio in training makes models robust to poor devices.",
        },
      },
      {
        path: "acoustics.device_chain.codec", label: "Opus codec", kind: "prob",
        info: {
          what: "The chance that the recording is compressed with the Opus codec at 16 to 32 kbps, as a call or meeting app would.",
          why: "Many real meeting recordings come from conferencing apps, and codec compression leaves artifacts that clean recordings never have.",
        },
      },
      {
        path: "acoustics.device_chain.clipping", label: "Clipping", kind: "prob",
        info: {
          what: "The chance that the signal is cut off at 60 to 90% of full scale, as an overdriven input would.",
          why: "Clipping distorts loud speech. It is rare in good recordings, so the default is low, but a little of it makes models more robust.",
        },
      },
    ],
  },
  {
    id: "splits",
    title: "Splits",
    note: "Each meeting draws its split, then clones voices only from that split's NSC speakers.",
    fields: [
      {
        path: "speaker_pool.split_mix", label: "Split mix", kind: "weights", keys: ["train", "dev", "test"], hint: "Share of meetings per split",
        info: {
          what: "The share of meetings in train, dev and test. Each split has its own fixed NSC speakers (50, 25 and 100), and a meeting only clones voices from its own split.",
          why: "This keeps the splits speaker-disjoint, so test results measure voices the model has never heard. Raise train for training data, or test when building an evaluation set.",
        },
      },
    ],
  },
  {
    id: "stages",
    title: "Stages",
    note: "Ablations. A switched-off stage is recorded in the manifest's stages_enabled, and later stages go on without it.",
    fields: [
      {
        path: "stages.3.enabled", label: "Stage 3 · script validation (LLM judge)", kind: "toggle",
        info: {
          what: "Rule checks and an LLM judge score each script, and a failed script goes back to stage 2 with the judge's feedback, for up to 3 rounds.",
          why: "Switched off, every script is kept and the judge phases are skipped, which saves GPU time. Use it to measure whether judging improves the data.",
        },
      },
      {
        path: "stages.5.enabled", label: "Stage 5 · voice validation", kind: "toggle",
        info: {
          what: "Two speaker-verification models, an ASR round trip and a quality score check every synthesised utterance, with re-synthesis and voice swaps when they fail.",
          why: "Switched off, the run is faster but keeps voices that drift from their speaker or mispronounce words, so labels can disagree with the audio.",
        },
      },
      {
        path: "stages.7.enabled", label: "Stage 7 · acoustic simulation", kind: "toggle",
        info: {
          what: "The room simulation: reverberation, talkers turning, presenters moving, noise, and the device chain.",
          why: "Switched off, no mixture.wav is written; each meeting keeps its clean speaker tracks and labels. Useful for comparing clean and simulated audio.",
        },
      },
    ],
  },
];

export const FIELDS: Field[] = SECTIONS.flatMap((s) => s.fields);

/** Hover text for the Run panel and the YAML overlay, which are not curated config fields. */
export const RUN_INFO: Record<"base" | "runId" | "meetings" | "seed" | "backends" | "yaml", Info> = {
  base: {
    what: "Fill the form from the defaults or from an existing run's saved config. Settings the form does not cover are copied into the YAML overlay.",
    why: "Copying a run is the quickest way to make a variant of it, such as the same meetings with one acoustic setting changed.",
  },
  runId: {
    what: "The folder name under runs/. It cannot be reused or changed later.",
    why: "The viewer, the CLI (synthetics run -r <id>) and every manifest line refer to the run by this id.",
  },
  meetings: {
    what: "How many meetings the run makes. Each one is a separate unit of work with its own id, m-000001 and up.",
    why: "On real backends the 20-meeting m9-e2e run took about 9 hours, so try a new config on a few meetings first.",
  },
  seed: {
    what: "The run's random seed. Every draw for a meeting comes from this seed and the meeting's id.",
    why: "The same seed and settings give the same meetings, so a run can be reproduced. Change it for different meetings with the same settings.",
  },
  backends: {
    what: "Real runs use the platform's models (script LLM, judges, TTS, ASR, speaker verification, aligner) on the GPU. Fake runs use stand-ins that need no GPU.",
    why: "A fake run finishes in minutes and shows that a config works end to end, but its scripts and audio are placeholders, not training data.",
  },
  yaml: {
    what: "Any other config setting, written as in a --config file. Nested keys and dotted keys (turn_taking.gap: ...) both work.",
    why: "The form covers the settings people change most. Everything else the pipeline samples, such as gaps, noise levels or render settings, can still be set here, and it wins over the form.",
  },
};
/** Not a section field, but part of the form: real or fake backends. */
export const MODE_PATH = "backends.mode";
/** Set in their own form fields, or chosen for the new run (`revision`), never copied from another run. */
const NOT_COPIED = new Set(["run_id", "seed", "n_meetings", "revision", "schema_version"]);

export type Values = Record<string, unknown>;

// ---------------------------------------------------------------- paths and leaves

export function getPath(obj: unknown, path: string): unknown {
  let node: unknown = obj;
  for (const part of path.split(".")) {
    if (node === null || typeof node !== "object") return undefined;
    node = (node as Record<string, unknown>)[part];
  }
  return node;
}

const isObj = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === "object" && !Array.isArray(v);

/** Every leaf of a config: scalars, lists, and distributions (a mapping with a `dist` tag) as one leaf each. */
export function leaves(obj: unknown, prefix = ""): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (!isObj(obj) || "dist" in obj) {
    if (prefix) out[prefix] = obj;
    return out;
  }
  const keys = Object.keys(obj);
  if (!keys.length && prefix) out[prefix] = obj;
  for (const k of keys) Object.assign(out, leaves(obj[k], prefix ? `${prefix}.${k}` : k));
  return out;
}

export function same(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) < 1e-12;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => same(x, b[i]));
  if (isObj(a) && isObj(b)) {
    const ka = Object.keys(a);
    return ka.length === Object.keys(b).length && ka.every((k) => k in b && same(a[k], b[k]));
  }
  return false;
}

// ---------------------------------------------------------------- the form

/** The form's values read from a config: every curated field plus the backend mode. */
export function valuesFrom(config: unknown): Values {
  const out: Values = {};
  for (const f of FIELDS) out[f.path] = getPath(config, f.path);
  out[MODE_PATH] = getPath(config, MODE_PATH);
  return out;
}

/** The fields that differ from the defaults, as dotted overrides. */
export function overridesOf(values: Values, defaults: unknown): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [path, v] of Object.entries(values)) if (v !== undefined && !same(v, getPath(defaults, path))) out[path] = v;
  return out;
}

/** A frozen run config as a starting point: the curated fields fill the form, and every other leaf that differs
 *  from the defaults becomes a line of the YAML overlay (`dotted.path: <JSON>`, which is valid YAML). Leaves the
 *  defaults no longer have are left out and listed in `dropped`. */
export function fromRun(config: unknown, defaults: unknown): { values: Values; yaml: string; dropped: string[] } {
  const values = valuesFrom(config);
  const curated = new Set(Object.keys(values));
  const def = leaves(defaults);
  const lines: string[] = [];
  const dropped: string[] = [];
  for (const [path, v] of Object.entries(leaves(config))) {
    // a curated field inside a distribution (turn_taking.zipf.exponent) covers that distribution's leaf
    if (NOT_COPIED.has(path.split(".")[0]) || curated.has(path) || [...curated].some((c) => c.startsWith(`${path}.`))) continue;
    if (!(path in def)) {
      dropped.push(path);
      continue;
    }
    if (!same(v, def[path])) lines.push(`${path}: ${JSON.stringify(v)}`);
  }
  for (const f of FIELDS) if (values[f.path] === undefined) values[f.path] = getPath(defaults, f.path);
  if (values[MODE_PATH] === undefined) values[MODE_PATH] = getPath(defaults, MODE_PATH);
  return { values, yaml: lines.join("\n"), dropped };
}

/** The field a backend error belongs to: the longest field path that is a prefix of its location. */
export function fieldOf(loc: string): string | null {
  let best: string | null = null;
  for (const f of [...FIELDS.map((x) => x.path), MODE_PATH]) {
    if ((loc === f || loc.startsWith(`${f}.`)) && (!best || f.length > best.length)) best = f;
  }
  return best;
}

// ---------------------------------------------------------------- run ids

export const RUN_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

/** `d-20261021`, then `d-20261021-2` and so on past the ids already taken. */
export function suggestRunId(taken: string[], today = new Date()): string {
  const base = `d-${today.getFullYear()}${String(today.getMonth() + 1).padStart(2, "0")}${String(today.getDate()).padStart(2, "0")}`;
  if (!taken.includes(base)) return base;
  for (let i = 2; ; i++) if (!taken.includes(`${base}-${i}`)) return `${base}-${i}`;
}
