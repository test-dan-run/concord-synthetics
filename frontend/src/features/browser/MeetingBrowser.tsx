// Screen 1: a run's meetings as a grid of cards with to-scale mini plans, filtered from one panel (UI doc,
// "Meeting browser"). Filters live in the URL, apply immediately, and set the order Previous and Next follow on
// the detail screen. The one action is exporting the manifest slice for the current filter.

import { useMemo, useState, type ReactNode } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { MetricTile } from "../../components/MetricTile";
import { Button, Chip, cx, Empty, ErrorNote, PageHeader, Panel, Select, Spinner, StatusDot } from "../../components/ui";
import { exportUrl, type MeetingSummary, useConfig, useMeetings, useRuns } from "../../lib/api";
import { CLASS_COLOR, type RoomClass } from "../../lib/colors";
import { num, pct } from "../../lib/format";
import { DEFAULT_FILTERS, type Filters, kinds, matches, positionsLine, readFilters, rt60, snr, sortMeetings, writeFilters } from "../../lib/meetings";
import { MiniPlan } from "./MiniPlan";

const PAGE = 12;

function Swatch({ cls }: { cls: RoomClass }) {
  return <span aria-hidden className="inline-block size-2 rounded-sm shrink-0" style={{ background: CLASS_COLOR[cls] }} />;
}

/** A segmented control whose options carry a count (and, for room classes, their swatch). */
function CountToggle<T extends string>({ label, value, onChange, options }: { label: string; value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode; count?: number }[] }) {
  return (
    <span role="group" aria-label={label} className="inline-flex flex-wrap border border-line rounded-md overflow-hidden">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === value}
          onClick={() => onChange(o.value)}
          className={cx("min-h-8 px-3 text-xs cursor-pointer border-0 inline-flex items-center gap-1.5", o.value === value ? "bg-active text-fg" : "bg-transparent text-muted hover:text-fg")}
        >
          {o.label}
          {o.count !== undefined && <span className="num text-[11px] text-muted">{o.count}</span>}
        </button>
      ))}
    </span>
  );
}

function Check({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="inline-flex items-center gap-2 text-[13px] cursor-pointer min-h-8">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="size-4 accent-primary cursor-pointer" />
      {children}
    </label>
  );
}

function Legend() {
  const i = (mark: ReactNode, text: string) => (
    <span className="inline-flex items-center gap-1.5">
      <svg width={14} height={14} aria-hidden>
        {mark}
      </svg>
      {text}
    </span>
  );
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-muted border-t border-track pt-3">
      {i(<circle cx={7} cy={7} r={4} fill="none" stroke="currentColor" strokeWidth={1.5} />, "Table seat")}
      {i(<circle cx={7} cy={7} r={4} fill="currentColor" />, "Wall seat")}
      {i(<rect x={3} y={3} width={8} height={8} fill="none" stroke="currentColor" strokeWidth={1.5} />, "Presenter, standing")}
      {i(<circle cx={7} cy={7} r={4.5} className="fill-fg" />, "Chair")}
      {i(<path d="M7 2L12 7L7 12L2 7Z" className="fill-fg" />, "Mic")}
      {i(<rect x={5.5} y={1} width={3} height={12} className="fill-focus" />, "Screen, front wall")}
      <span>Seat colour is the room class · grid squares are 1 m</span>
    </div>
  );
}

const STATE_WORD = { done: "done", running: "running", failed: "failed", pending: "pending" } as const;
const STATE_DOT = { done: "done", running: "running", failed: "failed", pending: "queued" } as const;

function Card({ m, to }: { m: MeetingSummary; to: string }) {
  const p = m.plan!;
  const remote = p.people.filter((x) => x.position_type === "remote").length;
  const r = rt60(m);
  const s = snr(m);
  return (
    <li className="list-none min-w-0">
      <Link
        to={to}
        className="flex flex-col h-full bg-panel border border-track rounded-lg overflow-hidden no-underline !text-fg hover:border-line"
        aria-label={`${m.meeting_id}, ${p.room_class} room, ${p.n_speakers} speakers, ${STATE_WORD[m.state]}`}
      >
        <div className="relative bg-canvas/40 border-b border-track px-2 pt-2 pb-6 flex justify-center">
          <MiniPlan plan={p} />
          <span className="absolute left-2.5 bottom-1.5 text-[11px] text-muted num">
            {`${p.room.length.toFixed(1)} × ${p.room.width.toFixed(1)} × ${p.room.height.toFixed(1)} m`}
          </span>
          {remote > 0 && (
            <span className="absolute right-2.5 top-2.5">
              <Chip tone="info">+ {remote} remote</Chip>
            </span>
          )}
        </div>
        <div className="flex flex-col gap-3 p-4 flex-1">
          <div className="flex items-center gap-2">
            <span className="num font-semibold text-[13px]">{m.meeting_id}</span>
            <span className="ml-auto">
              <Chip>
                <Swatch cls={p.room_class} />
                {p.room_class === "small" ? "Small room" : "Large room"}
              </Chip>
            </span>
          </div>
          <dl className="m-0 grid grid-cols-3 gap-2 text-xs">
            {[["Speakers", num(p.n_speakers)], ["RT60", r === null ? "—" : `${r.toFixed(2)} s`], ["SNR", s === null ? "—" : `${s.toFixed(0)} dB`]].map(([k, v]) => (
              <div key={k} className="flex flex-col gap-0.5">
                <dt className="text-muted">{k}</dt>
                <dd className="m-0 num text-[13px] text-fg">{v}</dd>
              </div>
            ))}
          </dl>
          <span className="text-xs text-muted">{positionsLine(m)}</span>
          <div className="mt-auto flex flex-wrap items-center gap-1.5">
            <span className="rounded bg-track px-1.5 py-px text-[11px] text-muted num">{p.mic.type === "laptop" ? "laptop mic" : `${p.mic.type === "chair" ? "chair" : "centre"} ${p.mic.channels > 1 ? "array" : "mic"}`}</span>
            <span className="rounded bg-track px-1.5 py-px text-[11px] text-muted num">{p.table.shape} table</span>
            <span className="rounded bg-track px-1.5 py-px text-[11px] text-muted num">{p.split}</span>
            <span className="ml-auto inline-flex items-center gap-1.5 text-xs text-muted">
              <StatusDot state={STATE_DOT[m.state]} />
              {STATE_WORD[m.state]}
            </span>
          </div>
        </div>
      </Link>
    </li>
  );
}

export function MeetingBrowser() {
  const { run = "" } = useParams();
  const [search, setSearch] = useSearchParams();
  const f = readFilters(search);
  const set = (patch: Partial<Filters>) => setSearch(writeFilters({ ...f, ...patch }), { replace: true });
  const [shown, setShown] = useState(PAGE);
  const meetings = useMeetings(run);
  const config = useConfig(run).data;
  const info = useRuns().data?.find((r) => r.id === run);
  const all = useMemo(() => meetings.data ?? [], [meetings.data]);
  const sampled = all.filter((m) => m.plan);
  const qs = writeFilters(f).toString();
  const list = useMemo(() => {
    const g = readFilters(new URLSearchParams(qs));
    return sortMeetings(all.filter((m) => matches(m, g)), g.sort);
  }, [all, qs]);

  // tiles, over every sampled meeting of the run
  const ratio = config?.config.room_classes.ratio.weights ?? {};
  const byClass = (c: RoomClass) => sampled.filter((m) => m.plan!.room_class === c);
  const large = byClass("large");
  const withWall = large.filter((m) => kinds(m).has("wall"));
  const overflow = withWall.filter((m) => m.plan!.people.some((p) => p.wall_reason === "overflow")).length;
  const withPres = sampled.filter((m) => kinds(m).has("presenter"));
  const failed = all.filter((m) => m.state === "failed").length;
  const running = all.filter((m) => m.state === "running").length;
  const done = all.filter((m) => m.state === "done").length;
  const countIn = (patch: Partial<Filters>) => all.filter((m) => matches(m, { ...f, ...patch })).length;
  const exportable = list.filter((m) => m.manifest).length;

  return (
    <>
      <PageHeader
        crumbs="Room viewer · Meetings"
        title="Meeting configurations"
        subtitle={
          <span className="num">
            {num(all.length)} meetings · seed {info?.seed ?? "—"} · config {config?.config_hash ?? info?.config_hash ?? "—"}
          </span>
        }
        actions={
          <a
            href={exportable ? exportUrl(run, { room_class: f.room === "all" ? undefined : f.room, split: f.split === "all" ? undefined : f.split, min_speakers: f.minSpeakers, remote: f.remote, presenters: f.presenters, wall: f.wall }) : undefined}
            download
            aria-disabled={!exportable}
            title={exportable ? `Download the manifest lines of the ${exportable} finished meetings in this filter` : "No finished meeting in this filter"}
            className={cx("inline-flex items-center min-h-10 px-4 rounded-md text-[13px] no-underline bg-primary !text-canvas font-semibold hover:brightness-110", !exportable && "opacity-40 cursor-not-allowed")}
          >
            Export manifest slice
          </a>
        }
      />

      {meetings.error ? (
        <ErrorNote error={new Error(`Could not load run ${run} (tried /api/runs/${run}/meetings): ${(meetings.error as Error).message}`)} />
      ) : (
        <>
          <div className="flex flex-wrap gap-3">
            <MetricTile label="Meetings" value={num(all.length)} sub={`${num(done)} done`} />
            {(["small", "large"] as RoomClass[]).map((c) => (
              <MetricTile
                key={c}
                color={CLASS_COLOR[c]}
                label={c === "small" ? "Small rooms" : "Large rooms"}
                value={num(byClass(c).length)}
                sub={`${pct(sampled.length ? byClass(c).length / sampled.length : null)} · target ${pct(ratio[c] ?? null, 0)}`}
              />
            ))}
            <MetricTile label="With presenters" value={pct(sampled.length ? withPres.length / sampled.length : null)} sub={`${num(withPres.length)} meetings · 1–2 standing`} />
            <MetricTile label="Wall seats" value={pct(large.length ? withWall.length / large.length : null)} sub={`of large rooms · ${num(overflow)} overflow`} />
            <MetricTile label="Hybrid" value={pct(sampled.length ? sampled.filter((m) => m.plan!.hybrid).length / sampled.length : null)} sub={`${num(sampled.filter((m) => m.plan!.hybrid).length)} meetings`} />
            <MetricTile label="Needs attention" value={num(failed + running)} sub={`${num(failed)} failed · ${num(running)} running`} />
          </div>

          <Panel id="filters" title="Filters" subtitle="Filters apply as you change them">
            <div className="flex flex-wrap items-end gap-x-6 gap-y-4">
              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-muted">Room class</span>
                <CountToggle
                  label="Room class"
                  value={f.room}
                  onChange={(room) => set({ room })}
                  options={[
                    { value: "all", label: "All", count: countIn({ room: "all" }) },
                    { value: "small", label: <><Swatch cls="small" />Small · laptop</>, count: countIn({ room: "small" }) },
                    { value: "large", label: <><Swatch cls="large" />Large · table mic</>, count: countIn({ room: "large" }) },
                  ]}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-muted">Split</span>
                <CountToggle
                  label="Split"
                  value={f.split}
                  onChange={(split) => set({ split })}
                  options={(["all", "train", "dev", "test"] as const).map((v) => ({ value: v, label: v === "all" ? "All" : v }))}
                />
              </div>
              <label className="flex flex-col gap-1.5 text-xs text-muted min-w-[200px]">
                <span className="flex justify-between">
                  Min speakers <span className="num text-fg">{f.minSpeakers}</span>
                </span>
                <input type="range" min={2} max={20} step={1} value={f.minSpeakers} onChange={(e) => set({ minSpeakers: Number(e.target.value) })} className="accent-primary min-h-8" />
              </label>
            </div>
            <fieldset className="m-0 p-0 border-0 flex flex-wrap items-center gap-x-5 gap-y-1">
              <legend className="text-xs text-muted mb-1.5">Only meetings with</legend>
              <Check checked={f.remote} onChange={(remote) => set({ remote })}>Remote participants</Check>
              <Check checked={f.presenters} onChange={(presenters) => set({ presenters })}>Presenters</Check>
              <Check checked={f.wall} onChange={(wall) => set({ wall })}>Wall seats</Check>
            </fieldset>
            <Legend />
          </Panel>

          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs text-muted num" aria-live="polite">
              {num(Math.min(shown, list.length))} of {num(list.length)} shown
            </span>
            {meetings.isLoading && <Spinner label="Loading meetings" />}
            <span className="ml-auto">
              <Select label="Sort" value={f.sort} onChange={(e) => set({ sort: e.target.value as Filters["sort"] })}>
                <option value="id">Meeting ID</option>
                <option value="speakers">Speakers</option>
                <option value="rt60">RT60</option>
                <option value="snr">SNR</option>
              </Select>
            </span>
          </div>

          {!meetings.isLoading && !list.length ? (
            <Empty>
              <div className="flex flex-col items-center gap-3">
                No meetings match these filters
                <Button size="sm" onClick={() => setSearch(writeFilters({ ...DEFAULT_FILTERS, sort: f.sort }), { replace: true })}>
                  Clear filters
                </Button>
              </div>
            </Empty>
          ) : (
            <ul className="m-0 p-0 grid gap-4 grid-cols-[repeat(auto-fill,minmax(300px,1fr))]">
              {list.slice(0, shown).map((m) => (
                <Card key={m.meeting_id} m={m} to={`/runs/${encodeURIComponent(run)}/meetings/${m.meeting_id}${qs ? `?${qs}` : ""}`} />
              ))}
            </ul>
          )}
          {shown < list.length && (
            <div className="flex justify-center">
              <Button onClick={() => setShown((n) => n + PAGE)}>Load more</Button>
            </div>
          )}
          {all.length > sampled.length && (
            <span className="text-xs text-muted">{num(all.length - sampled.length)} meetings have not been sampled yet and are not shown.</span>
          )}
        </>
      )}
    </>
  );
}
