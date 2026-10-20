// Screen 2: one meeting's sampled setup to scale (UI doc, "Meeting layout detail"). Everything on it is keyed to
// one selected speaker: a seat, a table row or the persona select changes the selection everywhere at once.

import { useMemo, useState, type ReactNode } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { Chip, cx, Empty, ErrorNote, Panel, PageHeader, Spinner, StatusDot, Tag } from "../../components/ui";
import {
  type AcousticRender,
  artifactUrl,
  type MeetingSpec,
  type MeetingSummary,
  type Script,
  type ScriptValidation,
  type StageStatus,
  type Synthesis,
  type Timeline,
  useArtifact,
  useConfig,
  useMeetings,
  type VoiceValidation,
} from "../../lib/api";
import { CLASS_COLOR } from "../../lib/colors";
import { hours } from "../../lib/format";
import { positionsLine, readFilters, visible, writeFilters } from "../../lib/meetings";
import { defaultSelection, peopleRows } from "../../lib/people";
import { Elevation } from "./Elevation";
import { FloorPlan, type Layers } from "./FloorPlan";
import { PersonaPanel } from "./PersonaPanel";
import { FactPanels, SpeakerCard } from "./SidePanels";
import { SpeakersTable } from "./SpeakersTable";

const STATE_WORD: Record<StageStatus["state"], string> = {
  pass: "done", fail: "failed", dropped: "dropped", running: "running", retry: "retrying", pending: "pending", skipped: "skipped",
};
const STATE_DOT: Record<StageStatus["state"], string> = {
  pass: "done", fail: "failed", dropped: "failed", running: "running", retry: "running", pending: "queued", skipped: "queued",
};

function StageTiles({ stages }: { stages: Record<string, StageStatus> }) {
  const list = Object.entries(stages).sort(([a], [b]) => Number(a) - Number(b));
  const secs = list.reduce((s, [, v]) => (v.started_at && v.ended_at ? s + (Date.parse(v.ended_at) - Date.parse(v.started_at)) / 1000 : s), 0);
  const done = list.filter(([, v]) => v.state === "pass" || v.state === "skipped").length;
  return (
    <Panel id="pipeline" title="Pipeline status" subtitle={`${done === list.length ? "All eight stages done" : `${done} of ${list.length} stages done`} · ${hours(secs / 3600)} in the last attempts`}>
      <ol className="m-0 p-0 list-none grid gap-2 grid-cols-[repeat(auto-fill,minmax(150px,1fr))]">
        {list.map(([n, s]) => (
          <li key={n} className={cx("border rounded-md px-3 py-2 flex flex-col gap-1 min-w-0", s.state === "fail" || s.state === "dropped" ? "border-bad/50" : "border-track")}>
            <span className="flex items-center gap-2 text-[13px]">
              <StatusDot state={STATE_DOT[s.state]} />
              <span className="num text-muted">{n}</span> {s.name}
            </span>
            <span className="text-xs text-muted">
              {STATE_WORD[s.state]}
              {s.attempts > 1 ? ` · ${s.attempts} attempts` : ""}
            </span>
            {s.detail && (
              <span className="text-[11px] text-muted leading-snug break-words" title={s.detail}>
                {s.detail}
              </span>
            )}
          </li>
        ))}
      </ol>
    </Panel>
  );
}

function LayerToggles({ layers, set }: { layers: Layers; set: (l: Layers) => void }) {
  const opts: [keyof Layers, string][] = [["facing", "Facing"], ["paths", "All mic paths"], ["grid", "Grid"]];
  return (
    <span role="group" aria-label="Plan layers" className="inline-flex border border-line rounded-md overflow-hidden">
      {opts.map(([k, label]) => (
        <button
          key={k}
          type="button"
          aria-pressed={layers[k]}
          onClick={() => set({ ...layers, [k]: !layers[k] })}
          className={cx("min-h-8 px-3 text-xs cursor-pointer border-0", layers[k] ? "bg-active text-fg" : "bg-transparent text-muted hover:text-fg")}
        >
          {label}
        </button>
      ))}
    </span>
  );
}

function Legend() {
  const item = (mark: ReactNode, text: string) => (
    <span className="inline-flex items-center gap-1.5">
      {mark}
      {text}
    </span>
  );
  const svg = (children: ReactNode, w = 14) => (
    <svg width={w} height={14} aria-hidden>
      {children}
    </svg>
  );
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-muted">
      {item(svg(<circle cx={7} cy={7} r={5} fill="none" stroke="currentColor" strokeWidth={1.5} />), "Seat · size = turn share")}
      {item(svg(<circle cx={7} cy={7} r={5} fill="none" stroke="currentColor" strokeWidth={3} />), "Chair")}
      {item(svg(<rect x={2} y={2} width={10} height={10} rx={2} fill="none" stroke="currentColor" strokeWidth={1.5} />), "Presenter, standing")}
      {item(svg(<circle cx={7} cy={7} r={5} fill="none" stroke="currentColor" strokeDasharray="2 2" />), "Free table seat")}
      {item(svg(<rect x={1} y={4} width={12} height={6} rx={2} className="fill-track" />), "Wall chairs")}
      {item(svg(<rect x={5} y={1} width={4} height={12} className="fill-focus" />), "Screen")}
      {item(svg(<line x1={2} y1={7} x2={12} y2={7} stroke="currentColor" strokeWidth={2} />), "Facing")}
      {item(svg(<line x1={1} y1={7} x2={21} y2={7} className="stroke-focus" strokeDasharray="4 3" strokeWidth={1.5} />, 22), "Mouth-to-mic path, 3D distance")}
      {item(svg(<line x1={1} y1={7} x2={21} y2={7} stroke="currentColor" strokeDasharray="5 3" strokeWidth={2} />, 22), "Glass")}
      {item(svg(<line x1={1} y1={7} x2={21} y2={7} className="stroke-subtle" strokeWidth={2} />, 22), "Plasterboard")}
    </div>
  );
}

function Header({ run, m, summary, list, search }: { run: string; m: MeetingSpec | null; summary: MeetingSummary | undefined; list: MeetingSummary[]; search: string }) {
  const id = summary?.meeting_id ?? m?.meeting_id ?? "";
  const i = list.findIndex((x) => x.meeting_id === id);
  const prev = i > 0 ? list[i - 1] : undefined;
  const next = i >= 0 && i < list.length - 1 ? list[i + 1] : undefined;
  const base = `/runs/${encodeURIComponent(run)}/meetings`;
  const q = search ? `?${search}` : "";
  const nav = (to: MeetingSummary | undefined, label: string) =>
    to ? (
      <Link to={`${base}/${to.meeting_id}${q}`} className="inline-flex items-center min-h-10 px-4 rounded-md text-[13px] no-underline !text-fg border border-line hover:bg-track" aria-label={`${label}: ${to.meeting_id}`}>
        {label}
      </Link>
    ) : (
      <span aria-disabled className="inline-flex items-center min-h-10 px-4 rounded-md text-[13px] text-subtle border border-line opacity-40 cursor-not-allowed" title={`No ${label.toLowerCase()} meeting in this filter`}>
        {label}
      </span>
    );
  const plan = summary?.plan;
  return (
    <PageHeader
      crumbs={
        <>
          <Link to={`${base}${q}`}>Meetings</Link> · <span className="num">{id}</span>
          {i >= 0 && <span className="num"> · {i + 1} of {list.length} in this filter</span>}
        </>
      }
      title={<span className="num">{id}</span>}
      subtitle={
        plan && (
          <span className="flex flex-wrap items-center gap-1.5 mt-1">
            <Chip>
              <span aria-hidden className="size-2 rounded-sm" style={{ background: CLASS_COLOR[plan.room_class] }} />
              {plan.room_class === "small" ? "Small room" : "Large room"}
            </Chip>
            <Chip>{`${plan.n_speakers} speakers · ${positionsLine(summary!)}`}</Chip>
            {plan.hybrid && <Chip tone="info">Hybrid</Chip>}
            {plan.presentation && <Chip tone="info">Presentation</Chip>}
            <Tag>{plan.split}</Tag>
            <Tag>{`seed ${plan.seed}`}</Tag>
          </span>
        )
      }
      actions={
        <>
          {id && (
            <a href={artifactUrl(run, id, "meeting.json")} target="_blank" rel="noreferrer" className="inline-flex items-center min-h-10 px-4 rounded-md text-[13px] no-underline !text-fg border border-line hover:bg-track">
              View meeting.json
            </a>
          )}
          {nav(prev, "Previous")}
          {nav(next, "Next")}
        </>
      }
    />
  );
}

function Body({ run, m, summary }: { run: string; m: MeetingSpec; summary: MeetingSummary | undefined }) {
  const [selected, setSelected] = useState(() => defaultSelection(m));
  const [layers, setLayers] = useState<Layers>({ facing: true, paths: false, grid: true });
  const id = m.meeting_id;
  const stages = summary?.stages ?? {};
  const passed = (n: number) => stages[String(n)]?.state === "pass";
  const script = useArtifact<Script>(run, id, "script.json", passed(2)).data;
  const validation = useArtifact<ScriptValidation>(run, id, "script_validation.json", passed(3)).data;
  const synthesis = useArtifact<Synthesis>(run, id, "synthesis.json", passed(4)).data;
  const voice = useArtifact<VoiceValidation>(run, id, "voice_validation.json", passed(5)).data;
  const timeline = useArtifact<Timeline>(run, id, "timeline.json", passed(6)).data;
  const render = useArtifact<AcousticRender>(run, id, "render.json", passed(7)).data;
  const config = useConfig(run).data?.config;
  const rows = useMemo(() => peopleRows(m, { manifest: summary?.manifest, synthesis, voice, script }), [m, summary, synthesis, voice, script]);
  const row = rows.find((r) => r.id === selected);

  return (
    <>
      <div className="flex flex-col min-[1000px]:flex-row gap-5 items-start">
        <div className="flex flex-col gap-5 min-w-0 w-full min-[1000px]:flex-1">
          <Panel
            id="floor-plan"
            title="Floor plan"
            subtitle="Top-down · 1 grid square = 1 m · front wall on the left · click a seat or presenter to inspect them"
            actions={<LayerToggles layers={layers} set={setLayers} />}
          >
            <div className="overflow-auto scroll-thin border border-track rounded-md">
              <FloorPlan m={m} rows={rows} selected={selected} onSelect={setSelected} layers={layers} />
            </div>
            <Legend />
          </Panel>
          <Panel id="elevation" title="Side elevation" subtitle="Looking north · heights to scale · N and S seats share x positions, so pairs overlap">
            <div className="overflow-auto scroll-thin border border-track rounded-md">
              <Elevation m={m} rows={rows} selected={selected} />
            </div>
          </Panel>
        </div>
        <div className="flex flex-col gap-5 w-full min-[1000px]:w-[340px] min-[1000px]:shrink-0">
          <SpeakerCard row={row} />
          <FactPanels m={m} config={config} render={render} timeline={timeline} />
        </div>
      </div>
      <PersonaPanel run={run} m={m} selected={selected} onSelect={setSelected} script={script} validation={validation} timeline={timeline} />
      <Panel id="speakers" title="Speakers and positions" subtitle={`Turn shares follow Zipf, exponent ${m.dynamics.zipf_exponent} · NSC references from the ${m.split} split only`}>
        <SpeakersTable m={m} rows={rows} selected={selected} onSelect={setSelected} />
      </Panel>
    </>
  );
}

export function MeetingDetail() {
  const { run = "", meeting = "" } = useParams();
  const [search] = useSearchParams();
  const meetings = useMeetings(run);
  const spec = useArtifact<MeetingSpec>(run, meeting, "meeting.json");
  const qs = writeFilters(readFilters(search)).toString();
  const list = useMemo(() => visible(meetings.data ?? [], readFilters(new URLSearchParams(qs))), [meetings.data, qs]);
  const summary = meetings.data?.find((x) => x.meeting_id === meeting);

  return (
    <>
      <Header run={run} m={spec.data ?? null} summary={summary} list={list} search={qs} />
      {meetings.error && <ErrorNote error={meetings.error} />}
      {spec.error && <ErrorNote error={spec.error} />}
      {summary && <StageTiles stages={summary.stages} />}
      {spec.isLoading || meetings.isLoading ? (
        <Spinner label="Loading meeting" />
      ) : spec.data ? (
        <Body key={meeting} run={run} m={spec.data} summary={summary} />
      ) : (
        !spec.error && <Empty>{`${meeting} has no meeting.json yet: stage 1 has not run for it.`}</Empty>
      )}
    </>
  );
}
