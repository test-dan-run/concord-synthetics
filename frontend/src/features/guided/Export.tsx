// Step 5: what goes in the pack, each speaker's own audio by cut (clips or full length) and sound (clean or as heard
// in the room), the noise alone, and where the pack is written. Sizes come from the timelines before anything is
// written. Large packs are written on the server; one meeting at a time can be downloaded from the review screen.

import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Steps } from "../../components/guided";
import { Bar, Button, cx, ErrorNote, Panel, Spinner } from "../../components/ui";
import { useMeetings } from "../../lib/api";
import { bytes, hours, num } from "../../lib/format";
import { type ExportOptions, packBytes, useEstimate, useExports, useReview, useStartExport } from "../../lib/guided";

function Choice({ checked, onChange, title, children }: { checked: boolean; onChange: (v: boolean) => void; title: string; children: string }) {
  return (
    <label className={cx("flex gap-2.5 items-start px-3 py-2.5 border rounded-md cursor-pointer", checked ? "border-focus bg-active/60" : "border-line")}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="mt-1 accent-[var(--color-primary)]" />
      <span className="flex flex-col gap-0.5 min-w-0">
        <span className="text-[13px] font-medium">{title}</span>
        <span className="text-xs text-muted">{children}</span>
      </span>
    </label>
  );
}

export function Export() {
  const { run = "" } = useParams();
  const est = useEstimate(run);
  const meetings = useMeetings(run);
  const review = useReview(run);
  const [o, setO] = useState<ExportOptions>({ clips: true, full: false, clean: true, room: true, noise: true });
  const start = useStartExport(run);
  const jobs = useExports(run);
  const set = (k: keyof ExportOptions) => (v: boolean) => setO({ ...o, [k]: v });
  const marks = review.data ?? {};
  const out = Object.entries(marks).filter(([, m]) => m.decision === "exclude");
  const done = (meetings.data ?? []).filter((m) => m.state === "done");
  const audio = done.filter((m) => !out.some(([id]) => id === m.meeting_id)).reduce((s, m) => s + (m.manifest?.duration ?? 0), 0) / 3600;
  const splits = { train: 0, dev: 0, test: 0 };
  for (const m of done) if (m.manifest && marks[m.meeting_id]?.decision !== "exclude") splits[m.manifest.split] += 1;
  const e = est.data;
  const b = e?.bytes;
  const anyAudio = (o.clips || o.full) && (o.clean || o.room);
  const latest = jobs.data?.[0];

  return (
    <div className="flex flex-col gap-5">
      <Steps current={4} links={{ 3: `/runs/${encodeURIComponent(run)}/review` }} />
      <div className="w-full max-w-[820px] mx-auto flex flex-col gap-5">
        <div>
          <h1 className="m-0 text-[22px] font-semibold tracking-tight">Your dataset</h1>
          <p className="m-0 mt-1 text-[13px] text-muted">Every meeting comes with its audio and exact labels, ready for the fine-tuning team.</p>
        </div>
        <div className="flex flex-wrap gap-3">
          {[
            ["Meetings kept", num(e?.meetings ?? 0)],
            ["Left out", num(out.length)],
            ["Audio", hours(audio)],
            ["Train · dev · test", `${splits.train} · ${splits.dev} · ${splits.test}`],
          ].map(([k, v]) => (
            <div key={k} className="flex-[1_1_140px] min-w-[140px] bg-panel border border-track rounded-lg px-4 py-3">
              <div className="text-xs text-muted">{k}</div>
              <div className="num text-[22px] font-medium">{v}</div>
            </div>
          ))}
        </div>

        <Panel title="What's in the pack, for each meeting" actions={<span className="text-xs text-muted">All audio: <span className="num text-fg">16 kHz</span> · <span className="num text-fg">16-bit</span> · mono WAV</span>}>
          <ul className="list-none m-0 p-0 grid gap-x-5 gap-y-2 [grid-template-columns:repeat(auto-fit,minmax(240px,1fr))] text-[13px]">
            <li className="flex flex-col"><span className="font-medium">The room recording</span><span className="text-xs text-muted">as heard by the meeting mic</span></li>
            <li className="flex flex-col"><span className="font-medium">Who spoke when</span><span className="text-xs text-muted">speaker turns, including overlaps</span></li>
            <li className="flex flex-col"><span className="font-medium">Transcript with word timings</span><span className="text-xs text-muted">readable and normalised versions</span></li>
            <li className="flex flex-col"><span className="font-medium">Meeting details</span><span className="text-xs text-muted">room, seats, mic and speaker profiles</span></li>
          </ul>
        </Panel>

        <Panel title="Each speaker's own audio, and the noise" subtitle="Useful for speaker separation and for checking labels by ear. Pick any mix of the options.">
          <div className="grid gap-x-5 gap-y-3.5 [grid-template-columns:repeat(auto-fit,minmax(260px,1fr))]">
            <fieldset className="border-0 m-0 p-0 flex flex-col gap-2">
              <legend className="pb-1.5 text-[11px] font-medium tracking-wider uppercase text-muted">Cut as</legend>
              <Choice checked={o.clips} onChange={set("clips")} title="Clips, one per line">
                Plus a CSV with speaker, start, end and text. A line cut short by an interruption stops where it was cut.
              </Choice>
              <Choice checked={o.full} onChange={set("full")} title="Full length">
                One file per speaker, as long as the room recording and lined up with it. Silent when they aren't talking.
              </Choice>
            </fieldset>
            <fieldset className="border-0 m-0 p-0 flex flex-col gap-2">
              <legend className="pb-1.5 text-[11px] font-medium tracking-wider uppercase text-muted">Sound</legend>
              <Choice checked={o.clean} onChange={set("clean")} title="Clean">
                The voice on its own, before the room and mic.
              </Choice>
              <Choice checked={o.room} onChange={set("room")} title="As heard in the room">
                The voice after the room echo, the speaker's distance and the mic, at the recording's loudness. No noise.
              </Choice>
            </fieldset>
          </div>
          <Choice checked={o.noise} onChange={set("noise")} title="Background noise on its own · one file per meeting">
            Air-con hum, typing, doors and chairs, as heard at the mic. With the in-room speaker files it adds back up to the recording.
          </Choice>
          {b && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 pt-2.5 border-t border-track text-xs text-muted">
              <span>Clips: clean <span className="num text-fg">{bytes(b.clips_clean)}</span> · in the room <span className="num text-fg">{bytes(b.clips_room)}</span></span>
              <span>Full length: <span className="num text-fg">{bytes(b.full)}</span> each</span>
              <span>Noise: <span className="num text-fg">{bytes(b.noise)}</span></span>
              <span className="ml-auto">Whole pack about <span className="num text-fg">{bytes(packBytes(e!, o))}</span></span>
            </div>
          )}
          {(o.clips || o.full) && !(o.clean || o.room) && <p className="m-0 text-xs text-warn">Pick clean, in the room, or both, or untick the cuts.</p>}
          {!anyAudio && !(o.clips || o.full) && <p className="m-0 text-xs text-muted">No speaker audio: the pack has the room recordings and labels only.</p>}
        </Panel>

        {out.length > 0 && (
          <Panel title="Left out of the pack" actions={<Link to={`/runs/${encodeURIComponent(run)}/review`} className="text-xs">Change in review</Link>}>
            <ul className="list-none m-0 p-0">
              {out.map(([id, m]) => (
                <li key={id} className="grid grid-cols-[64px_minmax(0,1fr)] gap-2.5 py-2 text-[13px]">
                  <span className="num text-xs text-muted">{id}</span>
                  <span className="text-muted">{m.note || "no note"}</span>
                </li>
              ))}
            </ul>
          </Panel>
        )}

        <Panel title="Where it goes">
          <p className="m-0 text-[13px]">The pack is written to a folder on the server:</p>
          <code className="num text-xs text-muted bg-canvas border border-line rounded-md px-2.5 py-2 overflow-x-auto whitespace-nowrap">{e ? `${e.root}/${run}-<date>/` : "…"}</code>
          <p className="m-0 text-xs text-muted">Packs are too large for a browser download. To take one meeting, use Download on the review screen.</p>
        </Panel>

        {latest && (
          <Panel title={latest.state === "finished" ? "Exported" : latest.state === "failed" ? "Export failed" : "Exporting"}>
            {latest.state === "running" && <Bar value={latest.done} max={latest.total} />}
            <p className="m-0 text-[13px]">
              <span className="num">{latest.done}</span> of <span className="num">{latest.total}</span> meetings · <code className="num text-xs">{latest.out}</code>
            </p>
            {latest.error && <p className="m-0 text-xs text-bad">{latest.error}</p>}
          </Panel>
        )}
        {est.isLoading && <Spinner label="Working out sizes" />}
        {(est.error || start.error) && <ErrorNote error={est.error ?? start.error} />}
        <div className="flex flex-wrap justify-between items-center gap-3">
          <Link to={`/runs/${encodeURIComponent(run)}/review`} className="inline-flex items-center min-h-10 px-4 rounded-md border border-line !text-fg text-[13px] no-underline">
            Back to review
          </Link>
          <Button variant="primary" onClick={() => start.mutate(o)} disabled={start.isPending || !e?.meetings || ((o.clips || o.full) && !(o.clean || o.room)) || latest?.state === "running"}>
            Export {num(e?.meetings ?? 0)} meetings
          </Button>
        </div>
      </div>
    </div>
  );
}
