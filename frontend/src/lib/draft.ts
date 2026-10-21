// The run builder's form model. Each curated field is one leaf of config.yaml, addressed by its dotted path; the
// form sends only the leaves that differ from the defaults, as `init-run --set` overrides. A distribution is one
// leaf (the backend replaces it whole). Everything else goes in the YAML overlay.

export type FieldKind = "range" | "prob" | "number" | "weights" | "toggle";

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
}

export interface Section {
  id: string;
  title: string;
  note: string;
  fields: Field[];
}

const perClass = (path: (c: string) => string, label: string, rest: Omit<Field, "path" | "label">): Field[] =>
  (["small", "large"] as const).map((c) => ({ path: path(c), label: `${label} · ${c}`, ...rest }));

export const SECTIONS: Section[] = [
  {
    id: "rooms",
    title: "Rooms and speakers",
    note: "Stage 1. The class boundary is fixed: small rooms have 2 to 6 speakers, large rooms 7 to 20.",
    fields: [
      { path: "room_classes.ratio", label: "Room class mix", kind: "weights", keys: ["small", "large"], hint: "Share of meetings in each class" },
      { path: "room_classes.classes.small.speakers", label: "Speakers · small", kind: "range", int: true, min: 2, max: 6 },
      { path: "room_classes.classes.large.speakers", label: "Speakers · large", kind: "range", int: true, min: 7, max: 20 },
      ...perClass((c) => `dynamics.duration_min.${c}`, "Meeting length", { kind: "range", unit: "min", min: 1, max: 180, step: 1 }),
    ],
  },
  {
    id: "turns",
    title: "Turn-taking",
    note: "Stages 2 and 6. The turn plan meets these by construction; stage 6 scales overlaps to the target.",
    fields: [
      ...perClass((c) => `turn_taking.overlap_target.${c}`, "Overlap target", { kind: "range", percent: true, min: 0, max: 60, step: 1, unit: "% of speech" }),
      { path: "turn_taking.interruption_rate", label: "Interruptions", kind: "range", percent: true, min: 0, max: 50, step: 1, unit: "% of turn changes" },
      { path: "turn_taking.backchannels_per_min", label: "Backchannels", kind: "range", min: 0, max: 20, step: 0.5, unit: "per min" },
      { path: "turn_taking.zipf.exponent", label: "Turn share skew (Zipf exponent)", kind: "number", min: 0, max: 3, step: 0.1, hint: "0 is equal shares; higher gives a longer tail of quiet people" },
      { path: "turn_taking.overlap_share", label: "Transitions that overlap", kind: "number", percent: true, min: 0, max: 100, step: 1, unit: "%" },
      { path: "dynamics.turns_per_min", label: "Turns", kind: "range", min: 1, max: 30, step: 0.5, unit: "per min" },
    ],
  },
  {
    id: "seating",
    title: "Seating and participants",
    note: "Stage 1. Presenters stand at the front; remote participants are heard through a speakerphone.",
    fields: [
      ...perClass((c) => `seating.presenters.presentation.${c}`, "Presentation segment", { kind: "prob" }),
      ...perClass((c) => `acoustics.hybrid.${c}`, "Hybrid meeting", { kind: "prob" }),
      { path: "seating.wall.by_choice", label: "Wall seats by choice · large", kind: "prob", hint: "When the table still has room" },
    ],
  },
  {
    id: "acoustics",
    title: "Acoustics and device",
    note: "Sampled in stage 1, rendered in stage 7. In revision 2 runs, changing these re-runs only stages 1, 7 and 8.",
    fields: [
      ...perClass((c) => `acoustics.rt60.${c}`, "RT60", { kind: "range", unit: "s", min: 0.1, max: 2, step: 0.05 }),
      ...perClass((c) => `acoustics.snr_db.${c}`, "SNR", { kind: "range", unit: "dB", min: -5, max: 50, step: 1 }),
      { path: "acoustics.mic.centre_array", label: "Centre mic is an array", kind: "prob" },
      { path: "acoustics.device_chain.agc", label: "AGC", kind: "prob" },
      { path: "acoustics.device_chain.band_limit", label: "Band limit", kind: "prob" },
      { path: "acoustics.device_chain.codec", label: "Opus codec", kind: "prob" },
      { path: "acoustics.device_chain.clipping", label: "Clipping", kind: "prob" },
    ],
  },
  {
    id: "splits",
    title: "Splits",
    note: "Each meeting draws its split, then clones voices only from that split's NSC speakers.",
    fields: [{ path: "speaker_pool.split_mix", label: "Split mix", kind: "weights", keys: ["train", "dev", "test"], hint: "Share of meetings per split" }],
  },
  {
    id: "stages",
    title: "Stages",
    note: "Ablations. A switched-off stage is recorded in the manifest's stages_enabled, and later stages go on without it.",
    fields: [
      { path: "stages.3.enabled", label: "Stage 3 · script validation (LLM judge)", kind: "toggle" },
      { path: "stages.5.enabled", label: "Stage 5 · voice validation", kind: "toggle" },
      { path: "stages.7.enabled", label: "Stage 7 · acoustic simulation", kind: "toggle" },
    ],
  },
];

export const FIELDS: Field[] = SECTIONS.flatMap((s) => s.fields);
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
