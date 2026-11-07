// Step 4: the run makes its meetings in waves; each finished meeting can be listened to and kept or left out at
// once. Review marks go to review.json beside the run and never change a meeting. Speaker lanes come from the
// timeline, drawn in one neutral tone and named by speaker id, never colour-coded.

import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Player, Steps } from "../../components/guided";
import { Bar, Button, cx, ErrorNote, Panel, Spinner } from "../../components/ui";
import { type MeetingSummary, type Script, type Timeline, useArtifact, useConfig, useMeetings, useProcess, useRunAction } from "../../lib/api";
import { clock, num } from "../../lib/format";
import { downloadUrl, useMarkMeeting, useReview } from "../../lib/guided";

const STAGE_WORDS: Record<string, string> = { "1": "Planning", "2": "Writing scripts", "3": "Checking scripts", "4": "Recording voices", "5": "Checking voices", "6": "Putting meetings together", "7": "Adding the room", "8": "Writing labels" };

function waveState(ms: MeetingSummary[]): { word: string; tone: string } {
  if (ms.every((m) => m.state === "done")) return { word: "Ready", tone: "bg-good" };
  const running = ms.flatMap((m) => Object.entries(m.stages).filter(([, s]) => s.state === "running").map(([k]) => k));
  if (running.length) return { word: STAGE_WORDS[running[0]] ?? "In progress", tone: "bg-primary animate-pulse" };
  if (ms.some((m) => m.state !== "pending" || Object.values(m.stages).some((s) => s.state === "pass"))) return { word: "In progress", tone: "bg-primary" };
  return { word: "Waiting", tone: "bg-pending" };
}

function Lanes({ tl, t }: { tl: Timeline; t: number }) {
  const dur = tl.stats.duration || 1;
  const talk = new Map<string, number>();
  for (const s of tl.segments) talk.set(s.speaker_id, (talk.get(s.speaker_id) ?? 0) + s.end - s.start);
  const top = [...talk.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k]) => k);
  const overlaps = useMemo(() => {
    const ev = tl.segments.flatMap((s) => [[s.start, 1], [s.end, -1]] as [number, number][]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const out: [number, number][] = [];
    let n = 0, from = 0;
    for (const [x, d] of ev) {
      if (n >= 2 && n + d < 2) out.push([from, x]);
      if (n < 2 && n + d >= 2) from = x;
      n += d;
    }
    return out;
  }, [tl]);
  const pos = (x: number) => `${(x / dur) * 100}%`;
  return (
    <div role="img" aria-label={`Who speaks when: ${top.join(", ")}, and where people talk at once`} className="flex flex-col gap-1 relative">
      {[...top, "All"].map((sid) => (
        <div key={sid} className="grid grid-cols-[36px_minmax(0,1fr)] gap-2 items-center">
          <span className="num text-[11px] text-muted">{sid}</span>
          <div className="relative h-2.5 bg-canvas rounded-sm overflow-hidden">
            {sid === "All"
              ? overlaps.map(([a, b]) => <div key={a} className="absolute inset-y-0 hatch border border-[color:var(--color-overlap)]/70" style={{ left: pos(a), width: pos(Math.max(b - a, dur / 400)) }} />)
              : tl.segments.filter((s) => s.speaker_id === sid).map((s) => <div key={s.turn_id} className="absolute inset-y-0 bg-muted rounded-sm" style={{ left: pos(s.start), width: pos(Math.max(s.end - s.start, dur / 800)) }} />)}
            <div className="absolute -inset-y-1 w-0.5 bg-focus" style={{ left: pos(Math.max(t, 0)) }} />
          </div>
        </div>
      ))}
      <p className="m-0 ml-11 text-[11px] text-muted">
        Hatched: two people talking at once · <span className="num">{top.length}</span> of <span className="num">{talk.size}</span> speakers shown
      </p>
    </div>
  );
}

function Review({ run, m, mark, onMark }: { run: string; m: MeetingSummary; mark?: string; onMark: (d: "keep" | "exclude" | null) => void }) {
  const tl = useArtifact<Timeline>(run, m.meeting_id, "timeline.json");
  const script = useArtifact<Script>(run, m.meeting_id, "script.json");
  const [t, setT] = useState(0);
  const lines = useMemo(() => {
    const text = new Map((script.data?.turns ?? []).map((x) => [x.turn_id, x.text]));
    return (tl.data?.segments ?? []).map((s) => ({ ...s, text: text.get(s.turn_id) ?? "" }));
  }, [tl.data, script.data]);
  const i = Math.max(0, lines.findIndex((s) => s.end > t));
  const near = lines.slice(Math.max(0, i - 1), i + 3);
  const p = m.plan;
  return (
    <Panel
      title={
        <>
          {p ? (p.room_class === "small" ? "Small room" : "Large room") : "Meeting"} <span className="num font-normal text-muted">{m.meeting_id}</span>
        </>
      }
      subtitle={p ? `${p.n_speakers} people${p.hybrid ? ` · ${p.people.filter((x) => x.position_type === "remote").length} calling in` : ""}${p.presentation ? " · presentation" : ""}` : undefined}
      actions={
        <a href={downloadUrl(run, m.meeting_id, { clips: true, full: false, clean: true, room: true, noise: false })} className="inline-flex items-center gap-1.5 min-h-8 px-2.5 rounded-md border border-line text-xs no-underline !text-fg hover:bg-track" download>
          Download this meeting
        </a>
      }
      className="flex-[1.4_1_480px]"
    >
      <Player src={`/api/runs/${encodeURIComponent(run)}/meetings/${encodeURIComponent(m.meeting_id)}/audio/mixture.wav`} onTime={setT} label="this meeting" />
      {tl.data ? <Lanes tl={tl.data} t={t} /> : <Spinner label="Loading the timeline" />}
      <ol className="list-none m-0 p-0 border-t border-track">
        {near.map((s) => (
          <li key={s.turn_id} className={cx("grid grid-cols-[52px_36px_minmax(0,1fr)] gap-2 py-2 text-[13px] px-1 rounded", t >= s.start && t < s.end ? "bg-active/60" : "text-muted")}>
            <span className="num text-xs">{clock(s.start, false)}</span>
            <span className="num text-xs">{s.speaker_id}</span>
            <span>{s.text}</span>
          </li>
        ))}
      </ol>
      <div className="flex flex-wrap justify-between items-center gap-2.5 pt-1 border-t border-track">
        <Link to={`/runs/${encodeURIComponent(run)}/meetings/${encodeURIComponent(m.meeting_id)}`} className="text-xs">
          Room and speaker details
        </Link>
        <div role="group" aria-label="Keep this meeting?" className="flex gap-2">
          <Button aria-pressed={mark === "exclude"} onClick={() => onMark(mark === "exclude" ? null : "exclude")}>
            {mark === "exclude" ? "Left out · undo" : "Leave out"}
          </Button>
          <Button variant="primary" aria-pressed={mark === "keep"} onClick={() => onMark("keep")}>
            {mark === "keep" ? "Kept" : "Keep · next meeting"}
          </Button>
        </div>
      </div>
    </Panel>
  );
}

export function GenerateReview() {
  const { run = "" } = useParams();
  const cfg = useConfig(run);
  const meetings = useMeetings(run);
  const proc = useProcess(run);
  const action = useRunAction(run);
  const review = useReview(run);
  const markM = useMarkMeeting(run);
  const [sel, setSel] = useState<string | null>(null);
  const [all, setAll] = useState(false);
  const list = meetings.data ?? [];
  const wave = (cfg.data?.config.phases as { wave_size?: number | null } | undefined)?.wave_size || list.length || 1;
  const waves = useMemo(() => {
    const out: MeetingSummary[][] = [];
    for (let k = 0; k < list.length; k += wave) out.push(list.slice(k, k + wave));
    return out;
  }, [list, wave]);
  const ready = list.filter((m) => m.state === "done");
  const marks = review.data ?? {};
  const selected = ready.find((m) => m.meeting_id === sel) ?? ready.find((m) => !marks[m.meeting_id]) ?? ready[0];
  const running = proc.data?.process.state === "running";
  const onMark = (d: "keep" | "exclude" | null) => {
    if (!selected) return;
    markM.mutate({ meeting_id: selected.meeting_id, decision: d });
    if (d === "keep") {
      const next = ready.find((m) => m.meeting_id !== selected.meeting_id && !marks[m.meeting_id]);
      if (next) setSel(next.meeting_id);
    }
  };
  const reviewed = Object.values(marks).length;
  const left = Object.values(marks).filter((x) => x.decision === "exclude").length;

  return (
    <div className="flex flex-col gap-5">
      <Steps current={3} links={{ 4: `/runs/${encodeURIComponent(run)}/export` }} />
      <div className="flex flex-wrap justify-between items-end gap-3">
        <div>
          <h1 className="m-0 text-[22px] font-semibold tracking-tight">Making your meetings</h1>
          <p className="m-0 mt-1 text-[13px] text-muted">Listen to each meeting as soon as it's ready. You can close this page; the work carries on.</p>
        </div>
        {running ? (
          <Button onClick={() => action.mutate({ action: "stop" })} disabled={action.isPending}>
            Pause
          </Button>
        ) : (
          ready.length < list.length && (
            <Button variant="primary" onClick={() => action.mutate({ action: "start", confirm: !!proc.data?.live })} disabled={action.isPending}>
              Resume
            </Button>
          )
        )}
      </div>
      {action.error && <ErrorNote error={action.error} />}
      {meetings.isLoading ? (
        <Spinner label="Loading the run" />
      ) : (
        <Panel title={<><span className="num">{num(ready.length)}</span> of <span className="num">{num(list.length)}</span> ready</>}>
          <Bar value={ready.length} max={list.length || 1} color="var(--color-good)" />
          <ol className="list-none m-0 p-0 grid gap-2.5 [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))] text-[13px]">
            {waves.map((w, k) => {
              const st = waveState(w);
              return (
                <li key={k} className="flex items-center gap-2 px-3 py-2.5 border border-track rounded-md">
                  <span aria-hidden className={cx("size-2 rounded-full shrink-0", st.tone)} />
                  <span>
                    Batch {k + 1} · <span className="num">{w.length}</span> meetings
                  </span>
                  <span className="ml-auto text-muted">{st.word}</span>
                </li>
              );
            })}
          </ol>
          {proc.data?.process.state === "failed" && <p className="m-0 text-xs text-bad">The run stopped with an error: {proc.data.process.error}</p>}
        </Panel>
      )}
      <div className="flex flex-wrap gap-5 items-start">
        <Panel
          title="Ready to review"
          subtitle={
            <>
              <span className="num">{reviewed}</span> reviewed · <span className="num">{left}</span> left out
            </>
          }
          className="flex-[1_1_420px]"
        >
          {ready.length === 0 && <p className="m-0 text-[13px] text-muted">The first meetings appear here when their batch is done.</p>}
          <ul className="list-none m-0 p-0">
            {(all ? ready : ready.slice(0, 12)).map((m) => {
              const mk = marks[m.meeting_id]?.decision;
              return (
                <li key={m.meeting_id}>
                  <button
                    type="button"
                    aria-current={m.meeting_id === selected?.meeting_id}
                    onClick={() => setSel(m.meeting_id)}
                    className={cx("w-full grid grid-cols-[64px_minmax(0,1fr)_auto] gap-2.5 items-center px-2 py-2.5 bg-transparent border-0 text-fg text-left text-[13px] cursor-pointer rounded hover:bg-track/60", m.meeting_id === selected?.meeting_id && "bg-active/60")}
                  >
                    <span className="num text-xs text-muted">{m.meeting_id}</span>
                    <span>
                      {m.plan?.room_class === "small" ? "Small room" : "Large room"} · <span className="num">{m.plan?.n_speakers}</span> people · <span className="num">{Math.round((m.manifest?.duration ?? 0) / 60)} min</span>
                    </span>
                    <span className={cx("text-xs", mk === "keep" ? "text-good" : "text-muted")}>{mk === "keep" ? "Kept" : mk === "exclude" ? "Left out" : "New"}</span>
                  </button>
                </li>
              );
            })}
          </ul>
          {ready.length > 12 && (
            <Button size="sm" variant="ghost" onClick={() => setAll((v) => !v)}>
              {all ? "Show fewer" : `Show all ${ready.length}`}
            </Button>
          )}
        </Panel>
        {selected && <Review key={selected.meeting_id} run={run} m={selected} mark={marks[selected.meeting_id]?.decision} onMark={onMark} />}
      </div>
      {ready.length > 0 && (
        <div className="flex justify-end">
          <Link to={`/runs/${encodeURIComponent(run)}/export`} className="inline-flex items-center min-h-10 px-5 rounded-md bg-primary !text-canvas font-semibold text-[13px] no-underline">
            Go to export
          </Link>
        </div>
      )}
    </div>
  );
}
