// The run builder's preview: stage 1 run in memory on the server for the draft (viewer/builder.py preview), so
// these are the meetings the run will sample. Every share and bin is computed here, in the browser.

import type { ReactNode } from "react";
import { Chip, Fact, Spinner } from "../../components/ui";
import type { Preview } from "../../lib/api";
import { C, CLASS_COLOR } from "../../lib/colors";
import { num, pct } from "../../lib/format";
import { countBins, histogram, share } from "../../lib/stats";
import { MiniPlan } from "../browser/MiniPlan";
import { Histogram } from "../distributions/Histogram";

const ticksEvery = (lo: number, hi: number, step: number) => Array.from({ length: Math.floor((hi - lo) / step + 1e-9) + 1 }, (_, i) => +(lo + i * step).toFixed(6));

function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 min-w-0">
      <span className="text-[11px] text-muted">{label}</span>
      <span className="num text-[15px]">{children}</span>
    </div>
  );
}

export function PreviewBody({ preview, fetching }: { preview: Preview; fetching: boolean }) {
  const ms = preview.meetings;
  const n = ms.length;
  const cls = (c: string) => ms.filter((m) => m.room_class === c).length;
  const split = (s: string) => ms.filter((m) => m.split === s).length;
  const failed = preview.checks.filter((c) => !c.ok);
  const capped = preview.n_sampled < preview.n_meetings;
  const maxDur = Math.max(30, ...ms.map((m) => m.duration_min));
  const durStep = maxDur > 90 ? 10 : 5;
  const durHi = Math.ceil(maxDur / durStep) * durStep;
  return (
    <div className="flex flex-col gap-4" aria-busy={fetching}>
      <div className="flex items-center gap-2 text-xs text-muted">
        {capped ? `The first ${num(preview.n_sampled)} of ${num(preview.n_meetings)} meetings` : `All ${num(n)} meetings`} · revision {preview.revision} ·{" "}
        <span className="num">{preview.config_hash}</span>
        {fetching && <span className="ml-auto"><Spinner label="Updating" /></span>}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Stat label="Small / large">{`${cls("small")} / ${cls("large")}`}</Stat>
        <Stat label="Train / dev / test">{`${split("train")} / ${split("dev")} / ${split("test")}`}</Stat>
        <Stat label="Audio">{`${preview.audio_hours.toFixed(1)} h`}</Stat>
        <Stat label="Hybrid">{pct(share(ms.filter((m) => m.hybrid).length, n), 0)}</Stat>
        <Stat label="Presentation">{pct(share(ms.filter((m) => m.presentation).length, n), 0)}</Stat>
        <Stat label="Wall seats">{pct(share(ms.filter((m) => m.positions.wall > 0).length, n), 0)}</Stat>
      </div>

      {preview.failures.length > 0 && (
        <div role="alert" className="border border-bad/50 bg-bad/10 rounded-md px-3 py-2 text-[13px] flex flex-col gap-1">
          <span>{`${preview.failures.length} meeting${preview.failures.length === 1 ? "" : "s"} cannot be sampled with these settings; stage 1 will fail on ${preview.failures.length === 1 ? "it" : "them"}.`}</span>
          {preview.failures.slice(0, 3).map((f) => (
            <span key={f.meeting_id} className="text-xs text-muted">
              <span className="num">{f.meeting_id}</span> {f.error}
            </span>
          ))}
        </div>
      )}

      {preview.plans.length > 0 && (
        <div className="grid grid-cols-2 gap-2" aria-label="Floor plans of the first meetings">
          {preview.plans.map((p) => (
            <figure key={p.meeting_id} className="m-0 bg-canvas/40 border border-track rounded-md p-1.5 flex flex-col gap-1">
              <MiniPlan plan={p} width={180} height={104} />
              <figcaption className="text-[11px] text-muted flex justify-between">
                <span className="num">{p.meeting_id}</span>
                <span>{`${p.n_speakers} speakers · ${p.room_class}`}</span>
              </figcaption>
            </figure>
          ))}
        </div>
      )}

      <Histogram
        id="pv-speakers"
        title="Speakers per meeting"
        bins={countBins(ms.map((m) => m.n_speakers), 2, 20)}
        color={(b) => (b.lo <= 6 ? CLASS_COLOR.small : CLASS_COLOR.large)}
        ticks={ticksEvery(2, 20, 6)}
        tick={String}
        binLabel={(b) => `${b.lo} speakers`}
        unit="meetings"
      />
      <Histogram
        id="pv-duration"
        title="Meeting length"
        subtitle="Minutes"
        bins={histogram(ms.map((m) => m.duration_min), 0, durHi, durStep)}
        color={C.primary}
        ticks={ticksEvery(0, durHi, durHi / 3)}
        tick={(v) => String(Math.round(v))}
        binLabel={(b) => `${b.lo}–${b.hi} min`}
        unit="meetings"
      />
      <Histogram
        id="pv-rt60"
        title="RT60"
        subtitle="Seconds, sampled target"
        bins={histogram(ms.map((m) => m.rt60), 0.2, 1.2, 0.05)}
        color={C.primary}
        ticks={ticksEvery(0.2, 1.2, 0.2)}
        tick={(v) => v.toFixed(1)}
        binLabel={(b) => `${b.lo.toFixed(2)}–${b.hi.toFixed(2)} s`}
        unit="meetings"
      />
      <Histogram
        id="pv-snr"
        title="SNR at the mic"
        subtitle="dB, sampled"
        bins={histogram(ms.map((m) => m.snr_db), 0, 40, 2)}
        color={C.primary}
        ticks={ticksEvery(0, 40, 10)}
        tick={String}
        binLabel={(b) => `${b.lo}–${b.hi} dB`}
        unit="meetings"
      />

      <details className="text-[13px]">
        <summary className="cursor-pointer min-h-8 flex items-center gap-2">
          Sampler checks
          <Chip tone={failed.length ? "warn" : "good"}>{`${preview.checks.length - failed.length} of ${preview.checks.length} within tolerance`}</Chip>
        </summary>
        <p className="text-xs text-muted my-2">
          The same checks as <span className="num">synthetics sampler-report</span>. With few meetings, rates wander from their targets and the
          tolerances widen to allow for it.
        </p>
        <div className="flex flex-col">
          {[...failed, ...preview.checks.filter((c) => c.ok)].map((c) => (
            <Fact
              key={c.name}
              k={
                <span className={c.ok ? undefined : "text-warn"}>
                  {c.ok ? "" : "Off target · "}
                  {c.name}
                </span>
              }
              v={`${c.observed} (target ${c.target})`}
            />
          ))}
        </div>
      </details>
    </div>
  );
}
