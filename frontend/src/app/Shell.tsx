import { useEffect, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation, useNavigate, useParams } from "react-router-dom";
import { useJobs, useRuns } from "../lib/api";
import { ago, num } from "../lib/format";
import { SyntheticsIcon } from "../components/icons";
import { Bar, cx } from "../components/ui";

function Item({ to, children }: { to: string; children: ReactNode }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        cx("flex items-center gap-2.5 px-2.5 min-h-10 rounded-md no-underline text-[13px]", isActive ? "bg-active !text-fg" : "!text-muted hover:!text-fg hover:bg-track")
      }
    >
      {children}
    </NavLink>
  );
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] text-muted uppercase tracking-[0.06em] px-2.5 pb-1.5">{title}</span>
      {children}
    </div>
  );
}

const STAGE_WORDS: Record<string, string> = {
  "1": "Sampling", "2": "Script generation", "3": "Script judging", "4": "Voice synthesis",
  "5": "Voice checks", "6": "Assembly", "7": "Acoustic simulation", "8": "Labels",
};

/** The run's progress from its status.json files: what is running, and how many meetings are done. */
function JobsCard({ run }: { run: string }) {
  const { data } = useJobs(run);
  if (!data) return null;
  const done = data.states.done ?? 0;
  const stage = data.running_stages[0];
  return (
    <section aria-label="Run progress" className="mt-auto flex flex-col gap-2 p-3 border border-track rounded-md text-xs">
      <span className="flex items-center gap-2">
        <span aria-hidden className={cx("size-2 rounded-full", data.active ? "bg-primary animate-pulse" : "bg-subtle")} />
        {data.active ? `Running · stage ${stage}` : "No stage running"}
      </span>
      <Bar value={done} max={data.meetings || 1} />
      <span className="text-[11px] text-muted">
        {data.active ? `${STAGE_WORDS[stage] ?? "Stage"} · ` : ""}
        <span className="num">{num(done)} of {num(data.meetings)}</span> meetings done
      </span>
      {!data.active && data.last_update && <span className="text-[11px] text-subtle">Last update {ago(data.last_update)}</span>}
    </section>
  );
}

function RunSelect({ run }: { run: string }) {
  const { data } = useRuns();
  const nav = useNavigate();
  const loc = useLocation();
  const section = loc.pathname.includes("/distributions") ? "distributions" : "meetings";
  return (
    <label className="flex flex-col gap-1.5 px-2.5 text-xs text-muted">
      Active run
      <select
        value={run}
        onChange={(e) => nav(`/runs/${encodeURIComponent(e.target.value)}/${section}`)}
        className="bg-panel text-fg border border-line rounded-md min-h-[34px] px-2 text-xs num"
      >
        {(data ?? []).map((r) => (
          <option key={r.id} value={r.id}>
            {r.id}
          </option>
        ))}
        {!data?.some((r) => r.id === run) && <option value={run}>{run}</option>}
      </select>
    </label>
  );
}

export function Shell() {
  const { run = "" } = useParams();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => setOpen(false), [loc.pathname]);
  const base = `/runs/${encodeURIComponent(run)}`;
  return (
    <div className="min-h-screen bg-canvas text-fg flex flex-col lg:flex-row">
      <div className="bg-panel border-b lg:border-b-0 lg:border-r border-track lg:w-[216px] lg:shrink-0">
        <nav aria-label="Primary" className="px-3 py-4 lg:py-5 flex flex-col gap-6 lg:sticky lg:top-0 lg:h-screen lg:overflow-y-auto scroll-thin">
          <div className="flex items-center gap-2.5 px-2">
            <span className="text-primary">
              <SyntheticsIcon size={22} />
            </span>
            <div className="flex flex-col">
              <span className="font-semibold text-[15px] tracking-tight">Synthetics</span>
              <span className="text-[11px] text-muted">Room config viewer</span>
            </div>
            <button
              type="button"
              aria-expanded={open}
              aria-controls="nav-groups"
              onClick={() => setOpen((o) => !o)}
              className="lg:hidden ml-auto min-h-10 px-3 rounded-md border border-line bg-transparent text-fg text-[13px] cursor-pointer"
            >
              {open ? "Close" : "Menu"}
            </button>
          </div>
          <div id="nav-groups" className={cx("flex-col gap-6 flex-1", open ? "flex" : "hidden lg:flex")}>
            <Group title="Room viewer">
              <Item to={`${base}/meetings`}>Meetings</Item>
              <Item to={`${base}/distributions`}>Distributions</Item>
            </Group>
            <Group title="Run">
              <RunSelect run={run} />
            </Group>
            {run && <JobsCard run={run} />}
          </div>
        </nav>
      </div>
      <main className="flex-1 min-w-0 px-4 py-6 sm:px-8 sm:pt-7 sm:pb-10 flex flex-col gap-5">
        <Outlet />
      </main>
    </div>
  );
}
