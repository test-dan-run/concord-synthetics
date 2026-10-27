// The run's background process: start or resume it, stop it, and follow its stages and log. Starting a run
// on real backends asks first, listing the platform services each GPU phase leases. Stopping interrupts it as
// Ctrl-C would; Start again continues from the status files, and finished stages are cache hits.

import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Bar, Button, Chip, Empty, ErrorNote, Fact, PageHeader, Panel, Spinner, StatusDot } from "../../components/ui";
import { type ProcessState, useConfig, useJobs, useProcess, useRunAction, useViewer } from "../../lib/api";
import { ago, num } from "../../lib/format";

const STAGES: [string, string][] = [
  ["1", "Meeting sampler"],
  ["2", "Script generation"],
  ["3", "Script validation"],
  ["4", "Voice synthesis"],
  ["5", "Voice validation"],
  ["6", "Conversation assembly"],
  ["7", "Acoustic simulation"],
  ["8", "Labels and packaging"],
];
const PHASE_WORDS: Record<string, string> = {
  A: "Script generation",
  B: "Script judge",
  B2: "Tie-break judge",
  C: "Voice synthesis and checks",
  D: "Forced alignment",
};
const STATE_WORDS: Record<ProcessState, string> = {
  idle: "Not started from the viewer",
  running: "Running",
  finished: "Finished",
  failed: "Failed",
  stopped: "Stopped",
  lost: "Ended without a record",
};
const STATE_TONE = { idle: "default", running: "info", finished: "good", failed: "bad", stopped: "warn", lost: "warn" } as const;
const STATE_DOT = { idle: "queued", running: "running", finished: "done", failed: "failed", stopped: "blocked", lost: "blocked" } as const;

function Confirm({ open, services, meetings, onCancel, onConfirm }: { open: boolean; services: Record<string, string[]>; meetings: number; onCancel: () => void; onConfirm: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal?.();
    if (!open && d.open) d.close?.();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onCancel}
      aria-labelledby="confirm-title"
      className="m-auto bg-panel text-fg border border-line rounded-lg p-5 max-w-[520px] w-[calc(100%-32px)] backdrop:bg-canvas/70"
    >
      <div className="flex flex-col gap-3">
        <h2 id="confirm-title" className="m-0 text-base font-semibold">
          Start a real run?
        </h2>
        <p className="m-0 text-[13px] text-muted">
          {`This runs ${num(meetings)} meeting${meetings === 1 ? "" : "s"} on the platform's models and takes hours. Each GPU phase starts its services with stack.py up, which may stop other GPU services to make room, and stops them afterwards. Other projects share the machine.`}
        </p>
        <div className="flex flex-col">
          {Object.entries(services).map(([ph, svcs]) => (
            <Fact key={ph} k={`Phase ${ph} · ${PHASE_WORDS[ph] ?? ""}`} v={<span className="text-xs">{svcs.join(", ") || "none"}</span>} />
          ))}
        </div>
        <div className="flex justify-end gap-2 pt-1">
          <Button onClick={onCancel}>Cancel</Button>
          <Button variant="primary" onClick={onConfirm}>
            Start the run
          </Button>
        </div>
      </div>
    </dialog>
  );
}

function Log({ text, running }: { text: string; running: boolean }) {
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [text]);
  return (
    <Panel id="log" title="Log" subtitle={running ? "The last 80 lines of process.log, every 3 seconds" : "The last 80 lines of process.log"}>
      {text ? (
        <pre ref={ref} tabIndex={0} aria-label="Run log" className="m-0 max-h-[420px] overflow-auto scroll-thin bg-canvas border border-track rounded-md p-3 text-[11px] leading-relaxed num whitespace-pre">
          {text}
        </pre>
      ) : (
        <Empty>No log yet. The viewer writes one when it starts the run.</Empty>
      )}
    </Panel>
  );
}

export function Pipeline() {
  const { run = "" } = useParams();
  const viewer = useViewer();
  const proc = useProcess(run);
  const jobs = useJobs(run);
  const cfg = useConfig(run);
  const action = useRunAction(run);
  const [confirming, setConfirming] = useState(false);

  if (proc.isLoading) return <Spinner label="Loading the run" />;
  if (proc.error) return <ErrorNote error={proc.error} />;
  const info = proc.data!;
  const p = info.process;
  const state = p.state;
  const running = state === "running";
  const meetings = jobs.data?.meetings ?? cfg.data?.config.n_meetings ?? 0;
  const begun = Object.values(jobs.data?.stages ?? {}).some((s) => Object.keys(s).some((k) => k !== "pending"));
  const readOnly = !!viewer.data?.read_only;
  const mode = (cfg.data?.config as { backends?: { mode?: string } } | undefined)?.backends?.mode ?? (info.live ? "aicore" : "fake");

  const start = () => (info.live ? setConfirming(true) : action.mutate({ action: "start" }));

  return (
    <>
      <PageHeader
        crumbs={<span className="num">{run}</span>}
        title="Pipeline"
        subtitle={`${num(meetings)} meetings · ${info.live ? "real backends" : `${mode} backends, no GPU`}`}
        actions={
          !readOnly && (
            <>
              {running ? (
                <Button variant="danger" disabled={action.isPending} onClick={() => action.mutate({ action: "stop" })}>
                  Stop
                </Button>
              ) : (
                <Button variant="primary" disabled={action.isPending} onClick={start}>
                  {begun ? "Resume" : "Start run"}
                </Button>
              )}
            </>
          )
        }
      />
      <ErrorNote error={action.error} />
      <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)] items-start">
        <Panel id="process" title="Process">
          <span className="flex items-center gap-2">
            <Chip tone={STATE_TONE[state]}>
              <StatusDot state={STATE_DOT[state]} />
              {STATE_WORDS[state]}
            </Chip>
          </span>
          <div className="flex flex-col">
            {p.started_at && <Fact k="Started" v={ago(p.started_at)} />}
            {p.ended_at && <Fact k="Ended" v={ago(p.ended_at)} />}
            {p.exit_code !== undefined && p.exit_code !== null && <Fact k="Exit code" v={p.exit_code} />}
            {p.pid && running && <Fact k="Process" v={p.pid} />}
            {jobs.data?.last_update && <Fact k="Last stage update" v={ago(jobs.data.last_update)} />}
          </div>
          {p.error && <ErrorNote error={p.error} />}
          <p className="m-0 text-xs text-muted">
            {state === "idle" && "Start runs every stage, phase by phase. Stopping is safe at any point; Resume continues where it stopped."}
            {state === "stopped" && "Resume continues from the status files; finished stages are cache hits."}
            {state === "lost" && "The process ended without writing how (killed, or the machine restarted). Resume continues from the status files."}
            {state === "failed" && "See the log. Resume retries from where it failed once the cause is fixed."}
            {state === "finished" && "manifest.jsonl lists the meetings that passed stage 8. Resume re-runs only what is not a cache hit."}
            {running && "The run's own CLI is `synthetics status -r " + run + "`; the counts here refresh every few seconds."}
          </p>
        </Panel>

        <Panel id="stages" title="Stages" subtitle="Meetings per stage, from their status.json files">
          <div className="overflow-x-auto">
            <table className="w-full text-[13px] border-collapse min-w-[520px]">
              <caption className="sr-only">Meetings in each state, per stage</caption>
              <thead>
                <tr className="text-xs text-muted text-left">
                  <th scope="col" className="font-normal py-1.5">Stage</th>
                  <th scope="col" className="font-normal py-1.5 w-[30%]">Passed</th>
                  <th scope="col" className="font-normal py-1.5 text-right">Pass</th>
                  <th scope="col" className="font-normal py-1.5 text-right">Running</th>
                  <th scope="col" className="font-normal py-1.5 text-right">Failed</th>
                  <th scope="col" className="font-normal py-1.5 text-right">Skipped</th>
                  <th scope="col" className="font-normal py-1.5 text-right">Pending</th>
                </tr>
              </thead>
              <tbody>
                {STAGES.map(([k, name]) => {
                  const c = jobs.data?.stages[k] ?? {};
                  const pass = c.pass ?? 0;
                  const run_ = (c.running ?? 0) + (c.retry ?? 0);
                  const fail = (c.fail ?? 0) + (c.dropped ?? 0);
                  return (
                    <tr key={k} className="border-t border-track">
                      <td className="py-2">
                        <span className="num text-muted mr-2">{k}</span>
                        {name}
                      </td>
                      <td className="py-2 pr-3">
                        <Bar value={pass} max={meetings || 1} />
                      </td>
                      <td className="py-2 text-right num">{num(pass)}</td>
                      <td className="py-2 text-right num">{run_ ? num(run_) : "—"}</td>
                      <td className={fail ? "py-2 text-right num text-bad" : "py-2 text-right num"}>{fail ? num(fail) : "—"}</td>
                      <td className="py-2 text-right num">{c.skipped ? num(c.skipped) : "—"}</td>
                      <td className="py-2 text-right num">{c.pending ? num(c.pending) : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
      <Log text={info.log} running={running} />
      <Confirm
        open={confirming}
        services={info.gpu_services}
        meetings={meetings}
        onCancel={() => setConfirming(false)}
        onConfirm={() => {
          setConfirming(false);
          action.mutate({ action: "start", confirm: true });
        }}
      />
    </>
  );
}

