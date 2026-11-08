// Step 2: the plan in plain words, three example meetings, and a chat to change it. Every change re-runs stage 1
// in memory (the builder's preview), so the summary and the examples follow at once with no GPU. "Make sample
// meetings" creates the pilot run (the draft's first three meetings) and starts it.

import { useEffect, useMemo, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { ChatPanel, replyText, Steps } from "../../components/guided";
import { Button, Chip, ErrorNote, Panel, Spinner } from "../../components/ui";
import { api, ApiError, type Draft, type Preview, usePreview, useRuns } from "../../lib/api";
import { hours, num, pct } from "../../lib/format";
import { suggestRunId } from "../../lib/draft";
import { type ChatMessage, deletePilot, meetingsFor, undoChanges, useApplyChoice, useGuided, useSuggest } from "../../lib/guided";
import { MiniPlan } from "../browser/MiniPlan";

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

function range(xs: number[]): string {
  if (!xs.length) return "–";
  const lo = Math.min(...xs), hi = Math.max(...xs);
  return lo === hi ? String(lo) : `${lo}–${hi}`;
}

/** The preview as a few plain sentences. */
function summary(p: Preview): string[] {
  const ms = p.meetings;
  const small = ms.filter((m) => m.room_class === "small");
  const large = ms.filter((m) => m.room_class === "large");
  const share = (xs: typeof ms, f: (m: (typeof ms)[number]) => boolean) => (xs.length ? xs.filter(f).length / xs.length : 0);
  const out = [];
  if (large.length) out.push(`${pct(large.length / ms.length, 0)} in large rooms, ${range(large.map((m) => m.n_speakers))} people around a table mic`);
  if (small.length) out.push(`${pct(small.length / ms.length, 0)} in small rooms, ${range(small.map((m) => m.n_speakers))} people around a laptop`);
  out.push(`${pct(share(ms, (m) => m.hybrid), 0)} have people calling in; ${pct(share(ms, (m) => m.presentation), 0)} start with a presentation`);
  out.push(`Meetings last ${range(ms.map((m) => Math.round(m.duration_min)))} minutes`);
  return out;
}

export function Plan() {
  const runs = useRuns();
  const { g, update } = useGuided(suggestRunId(runs.data?.map((r) => r.id) ?? []));
  const nav = useNavigate();
  const suggest = useSuggest();
  const apply = useApplyChoice();
  const draft = g.draft;
  const debounced = useDebounced(draft, 400);
  const preview = usePreview(debounced);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<unknown>(null);

  // the meeting count follows the hours asked for, at the plan's mean meeting length
  const meanMin = useMemo(() => {
    const ms = preview.data?.meetings ?? [];
    return ms.length ? ms.reduce((s, m) => s + m.duration_min, 0) / ms.length : null;
  }, [preview.data]);
  useEffect(() => {
    if (!draft || !meanMin) return;
    const n = meetingsFor(g.hours, meanMin);
    if (Math.abs(n - draft.n_meetings) / draft.n_meetings > 0.05) update({ draft: { ...draft, n_meetings: n } });
  }, [meanMin, g.hours, draft, update]);

  if (!draft) return <Navigate to="/create" replace />;

  const push = (m: ChatMessage[], d?: Draft) => update((cur) => ({ chat: [...cur.chat, ...m], ...(d ? { draft: d } : {}) }));
  const onSend = (text: string) => {
    push([{ from: "user", text }]);
    suggest.mutate(
      { request: text, draft },
      {
        onSuccess: (s) =>
          push(
            [{ from: "system", text: replyText(s.changes.length, s.offers.length, s.confirm.length, s.notes.map((n) => n.id), s.model === "fake"), changes: s.changes, offers: s.offers, confirm: s.confirm }],
            s.draft,
          ),
        onError: (e) => push([{ from: "system", text: `That did not work: ${e instanceof Error ? e.message : String(e)}` }]),
      },
    );
  };
  const onUndo = (i: number) =>
    update((cur) => ({
      draft: cur.draft ? undoChanges(cur.draft, cur.chat[i].changes ?? []) : cur.draft,
      chat: cur.chat.map((m, k) => (k === i ? { ...m, undone: true } : m)),
    }));
  const onPick = (offer: { group: string; scope: string }, option: string) =>
    apply.mutate(
      { draft, choices: [{ group: offer.group, option, scope: offer.scope }] },
      { onSuccess: (r) => push([{ from: "system", text: "Done.", changes: r.changes }], r.draft) },
    );

  const makeSamples = async () => {
    setCreateError(null);
    const p = preview.data;
    if (p?.live && !window.confirm(`Making sample meetings uses the shared GPU (${Object.values(p.gpu_services).flat().join(", ")}). Start now?`)) return;
    setCreating(true);
    try {
      if (g.pilot) await deletePilot(g.pilot);
      const created = await api.post<{ id: string; live: boolean }>("/runs", { ...draft, pilot: true });
      await api.post(`/runs/${encodeURIComponent(created.id)}/start`, { confirm: created.live });
      update({ pilot: created.id });
      nav(`/create/pilot/${encodeURIComponent(created.id)}`);
    } catch (e) {
      setCreateError(e);
    } finally {
      setCreating(false);
    }
  };

  const p = preview.data;
  const problems = preview.error instanceof ApiError ? preview.error.errors : [];
  const topics = (draft.overrides["dynamics.topics"] as string[] | undefined) ?? [];
  const terms = (draft.overrides["key_terms"] as { term: string }[] | undefined) ?? [];
  const examples = (p?.plans ?? []).slice(0, 3);

  return (
    <div className="flex flex-col gap-5">
      <Steps current={1} links={{ 0: "/create", ...(g.pilot ? { 2: `/create/pilot/${g.pilot}` } : {}) }} />
      <div className="flex flex-wrap gap-5 items-start">
        <div className="flex-[999_1_560px] min-w-0 flex flex-col gap-5">
          <div>
            <h1 className="m-0 text-[22px] font-semibold tracking-tight">Here's the plan</h1>
            <p className="m-0 mt-1 text-[13px] text-muted">Read it through. Ask for changes on the right, or go back and rewrite your description.</p>
          </div>

          <Panel
            title={
              <>
                <span className="num">{num(draft.n_meetings)}</span> meetings · about <span className="num">{p ? hours(p.audio_hours * (draft.n_meetings / Math.max(p.n_sampled, 1))) : "…"}</span> of audio
              </>
            }
            actions={preview.isFetching ? <Spinner label="Updating" /> : null}
          >
            {p ? (
              <ul className="m-0 pl-5 flex flex-col gap-1.5 text-sm">
                {summary(p).map((s) => (
                  <li key={s}>{s}</li>
                ))}
                <li>{topics.length ? `Topics: ${topics.slice(0, 4).join("; ")}${topics.length > 4 ? ` and ${topics.length - 4} more` : ""}` : "Topics from the built-in list"}{terms.length ? ` · ${terms.length} key terms` : ""}</li>
              </ul>
            ) : (
              !preview.error && <Spinner label="Working out the plan" />
            )}
            {p?.estimate_hours != null && (
              <p className="m-0 text-xs text-muted">
                Making them all takes about <span className="num">{hours(p.estimate_hours)}</span> on the GPU.
              </p>
            )}
          </Panel>

          {(problems.length > 0 || (p?.failures.length ?? 0) > 0) && (
            <div role="note" className="flex flex-col gap-1 px-3.5 py-3 border border-warn/50 bg-warn/10 rounded-md text-[13px]">
              <strong className="font-semibold text-warn">Couldn't do exactly</strong>
              {problems.map((e) => (
                <span key={e.loc + e.msg}>{e.msg}</span>
              ))}
              {p && p.failures.length > 0 && <span>{p.failures.length} meetings could not be laid out with these settings; they will be left out.</span>}
            </div>
          )}
          {preview.error && !problems.length && <ErrorNote error={preview.error} />}

          <section aria-labelledby="examples" className="flex flex-col gap-3">
            <h2 id="examples" className="m-0 text-sm font-semibold">
              Three example meetings
            </h2>
            <div className="grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(230px,1fr))]">
              {examples.map((plan) => {
                const row = p?.meetings.find((m) => m.meeting_id === plan.meeting_id);
                return (
                  <article key={plan.meeting_id} className="bg-panel border border-track rounded-lg overflow-hidden">
                    <div className="bg-canvas flex justify-center">
                      <MiniPlan plan={plan} width={260} height={150} />
                    </div>
                    <div className="p-3 flex flex-col gap-1.5">
                      <div className="flex justify-between gap-2 items-baseline">
                        <h3 className="m-0 text-[13px] font-semibold">{row?.room_class === "small" ? "Small room" : "Large room"}</h3>
                        <Chip>{row?.mic_type === "laptop" ? "laptop mic" : "table mic"}</Chip>
                      </div>
                      {row && (
                        <p className="m-0 text-xs text-muted">
                          <span className="num">{row.n_speakers}</span> people · <span className="num">{row.positions.remote}</span> calling in · <span className="num">{Math.round(row.duration_min)} min</span>
                          {row.presentation ? " · presentation" : ""}
                        </p>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          </section>

          {createError != null && <ErrorNote error={createError} />}
          <div className="flex flex-wrap justify-between items-center gap-3">
            <Button onClick={() => nav("/create")}>Back</Button>
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-xs text-muted">Makes 3 sample meetings first{p?.live ? " · uses the shared GPU" : ""}</span>
              <Button variant="primary" onClick={() => void makeSamples()} disabled={creating || !p || !!preview.error}>
                {creating ? "Starting…" : "Make sample meetings"}
              </Button>
            </div>
          </div>
        </div>

        <ChatPanel hint="Changes update the plan straight away." messages={g.chat} busy={suggest.isPending || apply.isPending} onSend={onSend} onUndo={onUndo} onPick={onPick} />
      </div>
    </div>
  );
}
