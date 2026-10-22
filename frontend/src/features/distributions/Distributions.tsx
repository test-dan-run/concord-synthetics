// Screen 3: did the run produce what config.yaml asked for? (UI doc, "Sampler distributions"). Every chart shows
// the run's distribution with the configured range shaded behind it. Per-class charts are small multiples on the
// same axis range; all-meeting charts use `primary` as their single series.

import type { ReactNode } from "react";
import { useParams } from "react-router-dom";
import { cx, ErrorNote, PageHeader, Spinner } from "../../components/ui";
import { type Dist, type MeetingSummary, type Plan, type RunConfig, useConfig, useMeetings } from "../../lib/api";
import { C, CLASS_COLOR, type RoomClass } from "../../lib/colors";
import { deltaPts, num, pct } from "../../lib/format";
import { dist3 } from "../../lib/geometry";
import { type Bin, countBins, histogram } from "../../lib/stats";
import { Histogram } from "./Histogram";

const CLASSES: RoomClass[] = ["small", "large"];
const CLASS_NAME = { small: "small rooms", large: "large rooms" };

function Tile({ title, children, note }: { title: string; children: ReactNode; note?: ReactNode }) {
  return (
    <section aria-label={title} className="bg-panel border border-track rounded-lg px-4 py-3 flex flex-col gap-2 min-w-[180px] flex-1">
      <span className="text-xs text-muted">{title}</span>
      {children}
      {note && <span className="text-[11px] text-subtle leading-snug">{note}</span>}
    </section>
  );
}

/** A per-class share with its gap from target in points; past `regression` points the gap is a regression. */
function Gap({ actual, target, regression }: { actual: number | null; target: number | undefined; regression?: number }) {
  if (actual === null || target === undefined) return null;
  const d = actual - target;
  const bad = regression !== undefined && Math.abs(d) * 100 > regression;
  return (
    <span className={cx("num text-xs", bad ? "text-bad" : "text-muted")} title={bad ? `More than ${regression} points from target` : undefined}>
      {Math.abs(d) * 100 >= 0.05 ? (d > 0 ? "↑" : "↓") : ""}
      {deltaPts(d)}
      {bad && <span className="sr-only"> (regression)</span>}
    </span>
  );
}

function StackBar({ parts }: { parts: { share: number; color: string }[] }) {
  return (
    <span className="flex h-1.5 gap-0.5 rounded overflow-hidden bg-track w-full">
      {parts.map((p, i) => (
        <span key={i} className="block h-full" style={{ width: `${p.share * 100}%`, background: p.color }} />
      ))}
    </span>
  );
}

const range = (d: Dist | undefined, fmt: (v: number) => string, unit = "", prefix = "") =>
  d?.lo !== undefined && d?.hi !== undefined ? { lo: d.lo, hi: d.hi, label: `${prefix}${fmt(d.lo)}–${fmt(d.hi)}${unit}` } : null;
const ticksEvery = (lo: number, hi: number, step: number) => Array.from({ length: Math.floor((hi - lo) / step + 1e-9) + 1 }, (_, i) => +(lo + i * step).toFixed(6));
const binText = (unit: string, digits: number) => (b: Bin) => `${b.lo.toFixed(digits)}–${b.hi.toFixed(digits)}${unit}`;

function distances(plans: Plan[], kind: "table" | "wall" | "presenter") {
  return plans.flatMap((p) => p.people.filter((x) => x.position_type === kind && x.mouth).map((x) => dist3(x.mouth!, p.mic.position)));
}

function Group({ title, note, children }: { title: string; note: string; children: ReactNode }) {
  return (
    <section aria-label={title} className="flex flex-col gap-3">
      <h2 className="m-0 text-[11px] text-muted uppercase tracking-[0.06em] font-medium">
        {title} <span className="normal-case tracking-normal text-subtle ml-2">{note}</span>
      </h2>
      <div className="grid gap-4 lg:grid-cols-2">{children}</div>
    </section>
  );
}

function Body({ cfg, meetings }: { cfg: RunConfig; meetings: MeetingSummary[] }) {
  const plans = meetings.map((m) => m.plan).filter((p): p is Plan => !!p);
  const of = (c: RoomClass) => plans.filter((p) => p.room_class === c);
  const tt = meetings.map((m) => ({ cls: m.plan?.room_class, t: m.manifest?.turn_taking })).filter((x) => x.t && x.cls);
  const ratio = cfg.room_classes.ratio.weights ?? {};
  const share = (xs: Plan[], f: (p: Plan) => boolean) => (xs.length ? xs.filter(f).length / xs.length : null);
  const has = (p: Plan, k: string) => p.people.some((x) => x.position_type === k);
  const presTarget = cfg.seating.presenters.presentation;
  const hybTarget = cfg.acoustics.hybrid;
  const large = of("large");
  const wallMeetings = large.filter((p) => has(p, "wall"));
  const overflow = wallMeetings.filter((p) => p.people.some((x) => x.wall_reason === "overflow")).length;
  const byChoice = wallMeetings.length - overflow;
  const maxWall = Math.max(0, ...large.map((p) => p.people.filter((x) => x.position_type === "wall").length));
  const resampled = plans.filter((p) => p.fit_resamples > 0);
  const twoPres = plans.filter((p) => p.people.filter((x) => x.position_type === "presenter").length === 2).length;
  const splits = { train: 0, dev: 0, test: 0 } as Record<string, number>;
  const shapes: Record<string, number> = {};
  for (const p of plans) {
    splits[p.split] = (splits[p.split] ?? 0) + 1;
    shapes[p.table.shape] = (shapes[p.table.shape] ?? 0) + 1;
  }

  const m1 = (v: number) => v.toFixed(1);
  const m2 = (v: number) => v.toFixed(2);
  const p0 = (v: number) => `${Math.round(v * 100)}%`;
  const spk = cfg.room_classes.classes;

  return (
    <>
      <div className="flex flex-wrap gap-3">
        <Tile title="Room class mix">
          <span className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-1.5 text-xs">
            <span className="text-muted">Actual</span>
            <span className="num">{`${pct(share(plans, (p) => p.room_class === "small"))} / ${pct(share(plans, (p) => p.room_class === "large"))}`}</span>
            <span />
            <StackBar parts={CLASSES.map((c) => ({ share: share(plans, (p) => p.room_class === c) ?? 0, color: CLASS_COLOR[c] }))} />
            <span className="text-muted">Target</span>
            <span className="num">{`${pct(ratio.small ?? null)} / ${pct(ratio.large ?? null)}`}</span>
            <span />
            <StackBar parts={CLASSES.map((c) => ({ share: ratio[c] ?? 0, color: CLASS_COLOR[c] }))} />
          </span>
        </Tile>
        <Tile title="Presentation segment" note={`Target ${p0(presTarget.small.p ?? 0)} small, ${p0(presTarget.large.p ?? 0)} large · ${twoPres} of ${plans.filter((p) => has(p, "presenter")).length} have two presenters`}>
          {CLASSES.map((c) => (
            <span key={c} className="flex items-baseline gap-2 text-xs">
              <span className="text-muted w-12 capitalize">{c}</span>
              <span className="num text-[18px]">{pct(share(of(c), (p) => has(p, "presenter")))}</span>
              <Gap actual={share(of(c), (p) => has(p, "presenter"))} target={presTarget[c].p} />
            </span>
          ))}
        </Tile>
        <Tile title="Hybrid meetings" note={`Gap from target in points · target ${p0(hybTarget.small.p ?? 0)} small, ${p0(hybTarget.large.p ?? 0)} large · over 2 points is shown as a regression`}>
          {CLASSES.map((c) => (
            <span key={c} className="flex items-baseline gap-2 text-xs">
              <span className="text-muted w-12 capitalize">{c}</span>
              <span className="num text-[18px]">{pct(share(of(c), (p) => p.hybrid))}</span>
              <Gap actual={share(of(c), (p) => p.hybrid)} target={hybTarget[c].p} regression={2} />
            </span>
          ))}
        </Tile>
        <Tile title="Wall seating · large rooms" note={`${overflow} overflow · ${byChoice} by choice (p = ${cfg.seating.wall.by_choice.p} target) · most at the wall in one meeting: ${maxWall}`}>
          <span className="num text-[22px] font-medium">{pct(share(large, (p) => has(p, "wall")))}</span>
        </Tile>
        <Tile title="Needed a room resample" note={`Max ${Math.max(0, ...plans.map((p) => p.fit_resamples))} resamples: table, wall chairs and presenter space did not fit on the first draw`}>
          <span className="num text-[22px] font-medium">{pct(share(plans, (p) => p.fit_resamples > 0))}</span>
          <span className="sr-only">{resampled.length} meetings</span>
        </Tile>
        <Tile title="Splits" note={`Tables: ${Object.entries(shapes).map(([k, v]) => `${v} ${k}`).join(" · ")}`}>
          <span className="grid grid-cols-3 gap-2 text-xs">
            {(["train", "dev", "test"] as const).map((s) => (
              <span key={s} className="flex flex-col">
                <span className="text-muted">{s}</span>
                <span className="num text-[18px]">{num(splits[s])}</span>
              </span>
            ))}
          </span>
        </Tile>
      </div>

      <Histogram
        id="h-speakers"
        title="Speakers per meeting"
        subtitle={`All positions, presenters and remote included · small ${spk.small.speakers.lo}–${spk.small.speakers.hi}, large ${spk.large.speakers.lo}–${spk.large.speakers.hi}`}
        bins={countBins(plans.map((p) => p.n_speakers), 2, 20)}
        color={(b) => (b.lo <= (spk.small.speakers.hi ?? 6) ? CLASS_COLOR.small : CLASS_COLOR.large)}
        ticks={ticksEvery(2, 20, 2)}
        tick={(v) => String(v)}
        binLabel={(b) => `${b.lo} speakers`}
        unit="meetings"
        note="Bars are coloured by room class: the class boundary is fixed at 7 speakers."
      />

      <Group title="Rooms and acoustics" note="Stages 1 and 7">
        {CLASSES.map((c) => (
          <Histogram
            key={`rt60${c}`}
            id={`h-rt60-${c}`}
            swatch={CLASS_COLOR[c]}
            title={`RT60 · ${CLASS_NAME[c]}`}
            subtitle={`${of(c).length} meetings · seconds, sampled target`}
            bins={histogram(of(c).map((p) => p.room.rt60_target), 0.2, 1.0, 0.05)}
            color={CLASS_COLOR[c]}
            range={range(cfg.acoustics.rt60[c], m1, " s")}
            ticks={ticksEvery(0.2, 1.0, 0.2)}
            tick={m2}
            binLabel={binText(" s", 2)}
            unit="meetings"
            note={c === "small" ? "Stage 7 calibrates absorption until the measured RT60 is within 15% of this target." : "Large rooms use the hybrid image-source and ray-tracing mode in stage 7."}
          />
        ))}
        {CLASSES.map((c) => (
          <Histogram
            key={`snr${c}`}
            id={`h-snr-${c}`}
            swatch={CLASS_COLOR[c]}
            title={`SNR at mic · ${CLASS_NAME[c]}`}
            subtitle={`${of(c).length} meetings · dB, sampled`}
            bins={histogram(of(c).map((p) => p.acoustics.snr_db), 0, 32, 2)}
            color={CLASS_COLOR[c]}
            range={range(cfg.acoustics.snr_db[c], String, " dB")}
            ticks={ticksEvery(0, 32, 8)}
            tick={(v) => String(v)}
            binLabel={binText(" dB", 0)}
            unit="meetings"
            note="Speech over the noise beds (HVAC, projector fan); keyboard and events sit relative to speech."
          />
        ))}
        {CLASSES.map((c) => {
          const d = distances(of(c), "table");
          return (
            <Histogram
              key={`dist${c}`}
              id={`h-dist-table-${c}`}
              swatch={CLASS_COLOR[c]}
              title={`Mouth-to-mic distance · ${c === "small" ? "laptop" : "table mic"}, table seats`}
              subtitle={`${d.length} seats · metres, 3D`}
              bins={histogram(d, 0, 5, 0.25)}
              color={CLASS_COLOR[c]}
              range={c === "small" ? range(cfg.acoustics.mic.laptop_distance, m1, " m", "nearest seat ") : null}
              ticks={ticksEvery(0, 5, 1)}
              tick={m1}
              binLabel={binText(" m", 2)}
              unit="seats"
              note={c === "small" ? "The laptop sits 0.3–0.5 m (horizontally) in front of one seat: near-far imbalance by design." : `The mic sits at the centre, or in front of the chair (p = ${cfg.acoustics.mic.chair_mic?.p ?? 0}), when the far end of the table is farthest. Wall seats and presenters are charted under Positions.`}
            />
          );
        })}
      </Group>

      <Group title="Positions" note="Stage 1 · wall seats and presenters">
        <Histogram
          id="h-wall-count"
          swatch={CLASS_COLOR.large}
          title="Wall-seated speakers per meeting · large rooms"
          subtitle={`${wallMeetings.length} meetings with wall seats`}
          bins={countBins(wallMeetings.map((p) => p.people.filter((x) => x.position_type === "wall").length), 1, 8)}
          color={CLASS_COLOR.large}
          ticks={ticksEvery(1, 8, 1)}
          tick={(v) => String(v)}
          binLabel={(b) => `${b.lo} at the wall`}
          unit="meetings"
          note={`Overflow when the table is full; otherwise 1–3 by choice with p = ${cfg.seating.wall.by_choice.p}.`}
        />
        <Histogram
          id="h-dist-wall"
          swatch={CLASS_COLOR.large}
          title="Mouth-to-mic distance · wall seats"
          subtitle={`${distances(large, "wall").length} seats · table mic · metres, 3D`}
          bins={histogram(distances(large, "wall"), 0, 6, 0.25)}
          color={CLASS_COLOR.large}
          ticks={ticksEvery(0, 6, 1)}
          tick={m1}
          binLabel={binText(" m", 2)}
          unit="seats"
          note="Further and more reverberant than table seats; stage 7 keeps the lower pickup level."
        />
        {CLASSES.map((c) => (
          <Histogram
            key={`pres${c}`}
            id={`h-dist-pres-${c}`}
            swatch={CLASS_COLOR[c]}
            title={`Mouth-to-mic distance · presenters, ${CLASS_NAME[c]}`}
            subtitle={`${distances(of(c), "presenter").length} presenters · ${c === "small" ? "laptop" : "table"} mic · metres, 3D`}
            bins={histogram(distances(of(c), "presenter"), 0, 6, 0.25)}
            color={CLASS_COLOR[c]}
            ticks={ticksEvery(0, 6, 1)}
            tick={m1}
            binLabel={binText(" m", 2)}
            unit="presenters"
            note="Standing 0.5–1.5 m from the front wall, mouth at 1.5–1.75 m, drifting up to 1 m between turns."
          />
        ))}
      </Group>

      <Group title="Turn-taking" note="Stage 6 assembly · finished meetings">
        {CLASSES.map((c) => (
          <Histogram
            key={`ov${c}`}
            id={`h-overlap-${c}`}
            swatch={CLASS_COLOR[c]}
            title={`Overlap share · ${CLASS_NAME[c]}`}
            subtitle="% of speech time with two or more speakers"
            bins={histogram(tt.filter((x) => x.cls === c).map((x) => x.t!.overlap_share * 100), 0, 36, 2)}
            color={CLASS_COLOR[c]}
            range={(() => {
              const d = cfg.turn_taking.overlap_target[c];
              return d?.lo !== undefined ? { lo: d.lo * 100, hi: d.hi! * 100, label: `${p0(d.lo)}–${p0(d.hi!)}` } : null;
            })()}
            ticks={ticksEvery(0, 36, 8)}
            tick={(v) => String(v)}
            binLabel={binText("%", 0)}
            unit="meetings"
            note="Measured on the assembled timeline in stage 6."
          />
        ))}
        <Histogram
          id="h-backchannels"
          swatch={C.primary}
          title="Backchannels per minute · all meetings"
          subtitle={`${tt.length} meetings`}
          bins={histogram(tt.map((x) => x.t!.backchannels_per_min), 0, 5, 0.25)}
          color={C.primary}
          range={range(cfg.turn_taking.backchannels_per_min, String, " per min")}
          ticks={ticksEvery(0, 5, 1)}
          tick={m2}
          binLabel={binText(" per min", 2)}
          unit="meetings"
          note="From non-speakers, placed inside the current turn; thinned to the planned rate when a meeting comes out shorter."
        />
        <Histogram
          id="h-interruptions"
          swatch={C.primary}
          title="Interruptions · all meetings"
          subtitle="% of turn changes"
          bins={histogram(tt.map((x) => x.t!.interruption_share * 100), 0, 16, 1)}
          color={C.primary}
          range={(() => {
            const d = cfg.turn_taking.interruption_rate;
            return d.lo !== undefined ? { lo: d.lo * 100, hi: d.hi! * 100, label: `${p0(d.lo)}–${p0(d.hi!)}` } : null;
          })()}
          ticks={ticksEvery(0, 16, 4)}
          tick={(v) => String(v)}
          binLabel={binText("%", 0)}
          unit="meetings"
          note="About a tenth of planned interruptions cannot be placed, so a meeting drawn near 5% can land just under it."
        />
      </Group>
    </>
  );
}

export function Distributions() {
  const { run = "" } = useParams();
  const cfg = useConfig(run);
  const meetings = useMeetings(run);
  const n = meetings.data?.filter((m) => m.plan).length ?? 0;
  return (
    <>
      <PageHeader
        crumbs="Room viewer · Distributions"
        title="Sampler distributions"
        subtitle={`What the run produced across ${num(n)} sampled meetings, against the ranges in config.yaml`}
        actions={
          <span className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted" aria-label="Legend">
            <span className="inline-flex items-center gap-1.5"><span aria-hidden className="size-2 rounded-sm bg-cat-1" />Small room · laptop mic</span>
            <span className="inline-flex items-center gap-1.5"><span aria-hidden className="size-2 rounded-sm bg-cat-2" />Large room · table mic</span>
            <span className="inline-flex items-center gap-1.5"><span aria-hidden className="size-2 rounded-sm bg-primary" />All meetings</span>
            <span className="inline-flex items-center gap-1.5"><span aria-hidden className="w-3 h-2 border border-dashed border-muted bg-line/40" />Configured range</span>
          </span>
        }
      />
      {cfg.error && <ErrorNote error={cfg.error} />}
      {meetings.error && <ErrorNote error={meetings.error} />}
      {cfg.isLoading || meetings.isLoading ? <Spinner label="Loading run" /> : cfg.data && meetings.data && <Body cfg={cfg.data.config} meetings={meetings.data} />}
    </>
  );
}
