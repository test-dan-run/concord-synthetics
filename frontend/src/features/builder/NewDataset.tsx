// The run builder (M14): choose the settings, preview the meetings stage 1 will sample, then create the run. The
// form starts from the defaults or from an existing run's frozen config, and sends only what differs from the
// defaults, as `init-run --set` would. Creating writes the run's frozen config.yaml; the Pipeline screen starts it.

import { type ReactNode, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button, cx, Empty, ErrorNote, PageHeader, Panel, Select, Spinner, Toggle } from "../../components/ui";
import { api, ApiError, type Draft, type FieldError, useCreateRun, useDefaults, usePreview, useRuns, useViewer } from "../../lib/api";
import { fieldOf, fromRun, getPath, MODE_PATH, overridesOf, RUN_ID, SECTIONS, suggestRunId, type Values, valuesFrom } from "../../lib/draft";
import { FieldInput, NumInput } from "./FieldInput";
import { PreviewBody } from "./PreviewPanel";

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

const MODES = [
  { value: "aicore", label: "Real models (GPU)" },
  { value: "fake", label: "Fake (no GPU)" },
];

function Row({ label, hint, htmlFor, children, error }: { label: string; hint?: string; htmlFor?: string; children: ReactNode; error?: string }) {
  return (
    <div className="grid gap-x-4 gap-y-1.5 py-2.5 border-b border-track last:border-0 sm:grid-cols-[minmax(0,240px)_1fr] items-center">
      <label htmlFor={htmlFor} className="flex flex-col gap-0.5 text-[13px]">
        {label}
        {hint && <span className="text-[11px] text-muted">{hint}</span>}
      </label>
      <div className="flex flex-col gap-1 min-w-0">
        {children}
        {error && (
          <span role="alert" className="text-[11px] text-bad">
            {error}
          </span>
        )}
      </div>
    </div>
  );
}

export function NewDataset() {
  const viewer = useViewer();
  const defaults = useDefaults(viewer.data ? !viewer.data.read_only : false);
  const runs = useRuns();
  const nav = useNavigate();
  const create = useCreateRun();

  const [runId, setRunId] = useState("");
  const [n, setN] = useState(20);
  const [seed, setSeed] = useState(1);
  const [values, setValues] = useState<Values | null>(null);
  const [yaml, setYaml] = useState("");
  const [base, setBase] = useState("defaults");
  const [baseNote, setBaseNote] = useState<string | null>(null);
  const [baseError, setBaseError] = useState<unknown>(null);

  const taken = useMemo(() => (runs.data ?? []).map((r) => r.id), [runs.data]);
  const def = defaults.data?.config;
  useEffect(() => {
    if (def && values === null) setValues(valuesFrom(def));
  }, [def, values]);
  useEffect(() => {
    if (runs.data && !runId) setRunId(suggestRunId(taken));
  }, [runs.data, runId, taken]);

  async function startFrom(id: string) {
    setBase(id);
    setBaseError(null);
    if (!def) return;
    if (id === "defaults") {
      setValues(valuesFrom(def));
      setYaml("");
      setBaseNote(null);
      return;
    }
    try {
      const { config } = await api.get<{ config: Record<string, unknown> }>(`/runs/${encodeURIComponent(id)}/config`);
      const r = fromRun(config, def);
      setValues(r.values);
      setYaml(r.yaml);
      setN(Number(config.n_meetings) || n);
      setSeed(Number(config.seed) || 0);
      const lines = r.yaml ? r.yaml.split("\n").length : 0;
      setBaseNote(
        `Copied from ${id}: the form, the seed and the meeting count, and ${lines} other setting${lines === 1 ? "" : "s"} in the YAML overlay. ` +
          `The new run uses the current revision (${String(def.revision)}).` +
          (r.dropped.length ? ` Left out ${r.dropped.length} setting${r.dropped.length === 1 ? "" : "s"} the config no longer has.` : ""),
      );
    } catch (e) {
      setBaseError(e);
    }
  }

  const overrides = useMemo(() => (values && def ? overridesOf(values, def) : {}), [values, def]);
  const previewDraft = useMemo(() => (values ? { run_id: "", n_meetings: n, seed, overrides, yaml } : null), [values, n, seed, overrides, yaml]);
  const debounced = useDebounced(previewDraft, 450);
  const preview = usePreview(debounced);

  const problems: FieldError[] = preview.error instanceof ApiError ? preview.error.errors : [];
  const createProblems: FieldError[] = create.error instanceof ApiError ? create.error.errors : [];
  const byField: Record<string, string> = {};
  const other: FieldError[] = [];
  for (const e of [...problems, ...createProblems]) {
    const f = e.loc === "run_id" || e.loc === "yaml" ? e.loc : fieldOf(e.loc);
    if (f) byField[f] = byField[f] ? `${byField[f]}; ${e.msg}` : e.loc === f ? e.msg : `${e.loc.slice(f.length + 1)}: ${e.msg}`;
    else other.push(e);
  }
  const idProblem = !RUN_ID.test(runId) ? "Letters, digits, dot, dash and underscore; start with a letter or digit." : taken.includes(runId) ? "There is already a run with this id." : undefined;

  if (viewer.data?.read_only)
    return (
      <>
        <PageHeader title="New dataset" />
        <Empty>This viewer was started with --read-only, so it cannot create runs. Use `just init` from a terminal.</Empty>
      </>
    );
  if (defaults.error) return <ErrorNote error={defaults.error} />;
  if (!values || !def) return <Spinner label="Loading the default config" />;

  const set = (path: string, v: unknown) => setValues((old) => ({ ...(old ?? {}), [path]: v }));
  const mode = String(values[MODE_PATH] ?? "aicore");
  const modes = MODES.some((m) => m.value === mode) ? MODES : [...MODES, { value: mode, label: mode }];
  const changed = Object.keys(overrides).length + (yaml.trim() ? yaml.trim().split("\n").length : 0);
  const draft: Draft = { run_id: runId, n_meetings: n, seed, overrides, yaml };
  const ready = !idProblem && !preview.error && !!preview.data && !preview.isFetching && debounced === previewDraft;

  async function onCreate() {
    try {
      const made = await create.mutateAsync(draft);
      nav(`/runs/${encodeURIComponent(made.id)}/pipeline`);
    } catch {
      /* shown from create.error */
    }
  }

  return (
    <>
      <PageHeader
        title="New dataset"
        subtitle="Choose the settings, check the meetings stage 1 will sample, then create the run. Its config is frozen once it is created."
      />
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_440px] items-start">
        <div className="flex flex-col gap-4 min-w-0">
          <Panel title="Run" id="sec-run" subtitle="Each meeting is a function of the seed, its id and the config, so the same settings give the same meetings.">
            <div className="flex flex-col">
              <Row label="Start from" htmlFor="base">
                <Select id="base" value={base} onChange={(e) => startFrom(e.target.value)} className="max-w-[260px]">
                  <option value="defaults">The defaults</option>
                  {taken.map((id) => (
                    <option key={id} value={id}>
                      {`Run ${id}`}
                    </option>
                  ))}
                </Select>
                {baseNote && <span className="text-[11px] text-muted">{baseNote}</span>}
                <ErrorNote error={baseError} />
              </Row>
              <Row label="Run id" hint="The folder under runs/" htmlFor="run-id" error={idProblem ?? byField.run_id}>
                <input
                  id="run-id"
                  value={runId}
                  onChange={(e) => setRunId(e.target.value.trim())}
                  spellCheck={false}
                  className="bg-canvas text-fg border border-line rounded-md min-h-[34px] px-2 text-xs num max-w-[260px]"
                />
              </Row>
              <Row label="Meetings" htmlFor="n">
                <NumInput id="n" value={n} onValue={(v) => setN(Math.max(1, Math.round(v)))} min={1} max={10000} step={1} />
              </Row>
              <Row label="Seed" htmlFor="seed">
                <NumInput id="seed" value={seed} onValue={(v) => setSeed(Math.max(0, Math.round(v)))} min={0} step={1} />
              </Row>
              <Row label="Backends" hint={mode === "fake" ? "No GPU: placeholder scripts and audio, to try a config end to end" : "The platform's models; phases A to D lease the GPU"} error={byField[MODE_PATH]}>
                <span className="flex">
                  <Toggle label="Backends" options={modes} value={mode} onChange={(v) => set(MODE_PATH, v)} />
                </span>
              </Row>
            </div>
          </Panel>

          {SECTIONS.map((s) => (
            <Panel key={s.id} id={`sec-${s.id}`} title={s.title} subtitle={s.note}>
              <div className="flex flex-col">
                {s.fields.map((f) => (
                  <FieldInput key={f.path} field={f} value={values[f.path]} defaultValue={getPath(def, f.path)} onChange={(v) => set(f.path, v)} error={byField[f.path]} />
                ))}
              </div>
            </Panel>
          ))}

          <Panel
            id="sec-yaml"
            title="Advanced: YAML overlay"
            subtitle="Any other setting, as in a --config file. Nested keys or dotted keys both work, and the overlay wins over the form."
          >
            <textarea
              aria-label="YAML overlay"
              value={yaml}
              onChange={(e) => setYaml(e.target.value)}
              spellCheck={false}
              rows={Math.min(16, Math.max(5, yaml.split("\n").length + 1))}
              placeholder={"turn_taking.gap: {dist: lognormal, median: 0.5, sigma: 0.5, cap: 2.5}\nsimulation:\n  ism_max_order: 4"}
              className="bg-canvas text-fg border border-line rounded-md px-3 py-2 text-xs num w-full resize-y"
            />
            {byField.yaml && (
              <span role="alert" className="text-[11px] text-bad whitespace-pre-wrap">
                {byField.yaml}
              </span>
            )}
          </Panel>
        </div>

        <div className="xl:sticky xl:top-6 flex flex-col gap-4 min-w-0">
          <Panel
            id="sec-preview"
            title="Preview"
            subtitle={`Stage 1 for these settings · ${changed ? `${changed} change${changed === 1 ? "" : "s"} from the defaults` : "the defaults"}`}
          >
            {preview.error ? (
              <div className="flex flex-col gap-2">
                <ErrorNote error={preview.error} />
                {other.map((e) => (
                  <span key={e.loc + e.msg} className="text-xs text-muted">
                    <span className="num text-fg">{e.loc}</span> {e.msg}
                  </span>
                ))}
              </div>
            ) : preview.data ? (
              <PreviewBody preview={preview.data} fetching={preview.isFetching} />
            ) : (
              <Spinner label="Sampling" />
            )}
          </Panel>
          <Panel id="sec-create" title="Create the run">
            {preview.data && !preview.error && (
              <p className={cx("m-0 text-[13px]", preview.data.live ? "text-fg" : "text-muted")}>
                {preview.data.live
                  ? `Real backends. Running it leases GPU services on the platform, phase by phase${preview.data.estimate_hours ? `, for roughly ${preview.data.estimate_hours} h (scaled from m9-e2e, which took 9 h for 7.3 h of audio)` : ""}. Creating it does not start it.`
                  : "Fake backends. No GPU and no models; the run takes seconds to minutes."}
              </p>
            )}
            <ErrorNote error={create.error} />
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="primary" disabled={!ready || create.isPending} onClick={onCreate}>
                {create.isPending ? "Creating" : `Create ${runId || "run"}`}
              </Button>
              <span className="text-xs text-muted">Writes runs/{runId}/config.yaml, speaker_splits.json and the meeting folders.</span>
            </div>
          </Panel>
        </div>
      </div>
    </>
  );
}
