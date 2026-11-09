// Step 3: listen to the pilot's three meetings before making them all. Each sample shows who is in the room, their
// voices, and the conversation line by line with a play button per line. The chat changes either this sample
// only (another voice for one speaker, which re-synthesises just that speaker) or every meeting (a settings
// change, after which the samples are remade). "Looks good" creates the full run from the same draft and starts it.

import { useEffect, useMemo, useState } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";
import { ChatPanel, PlayClip, Player, replyText, Steps } from "../../components/guided";
import { Button, Chip, cx, ErrorNote, Panel, Spinner } from "../../components/ui";
import { api, type MeetingSpec, type Script, type Timeline, useArtifact, useMeetings, useProcess, useRuns } from "../../lib/api";
import { clock, num } from "../../lib/format";
import { suggestRunId } from "../../lib/draft";
import { audioUrl, type ChatMessage, deletePilot, useGuided, useSuggest, useVoiceSwap } from "../../lib/guided";

const VOICE_WORDS = /\b(voice|sounds?|robotic|accent|tone|pitch|unnatural|fake)\b/i;
const SPEAKER = /\bS(\d{2,3})\b/i;
const STAGE_WORDS: Record<string, string> = { "1": "Planning", "2": "Writing the script", "3": "Checking the script", "4": "Recording voices", "5": "Checking voices", "6": "Putting it together", "7": "Adding the room", "8": "Writing labels" };

function personLine(p: MeetingSpec["people"][number]): string {
  const where = { table: "At the table", wall: "By the wall", presenter: "Presenting", remote: "Calling in" }[p.position_type];
  return p.is_chair ? "Chair" : where;
}

function Sample({ run, meeting }: { run: string; meeting: string }) {
  const m = useArtifact<MeetingSpec>(run, meeting, "meeting.json");
  const script = useArtifact<Script>(run, meeting, "script.json");
  const tl = useArtifact<Timeline>(run, meeting, "timeline.json");
  const [more, setMore] = useState(false);
  const [t, setT] = useState(-1);
  const times = useMemo(() => new Map((tl.data?.segments ?? []).map((s) => [s.turn_id, s])), [tl.data]);
  if (m.isLoading) return <Spinner label="Loading the sample" />;
  if (!m.data) return <p className="text-[13px] text-muted m-0">This sample is still being planned.</p>;
  const people = [...m.data.people].sort((a, b) => b.turn_share - a.turn_share);
  const firstTurn = (sid: string) => script.data?.turns.find((x) => x.speaker_id === sid && x.interaction === "normal")?.turn_id;
  const turns = script.data?.turns ?? [];
  const shown = more ? turns : turns.slice(0, 12);
  const hasAudio = !!tl.data;
  return (
    <div className="flex flex-col gap-5">
      <section aria-labelledby="people" className="flex flex-col gap-2.5">
        <div className="flex flex-wrap justify-between items-baseline gap-2">
          <h2 id="people" className="m-0 text-sm font-semibold">
            Who's in the room
          </h2>
          <span className="text-xs text-muted">
            {m.data.dynamics.meeting_type} · {m.data.dynamics.topic}
          </span>
        </div>
        <ul className="list-none m-0 p-0 grid gap-2.5 [grid-template-columns:repeat(auto-fill,minmax(176px,1fr))]">
          {people.slice(0, more ? people.length : 6).map((p) => {
            const turn = firstTurn(p.speaker_id);
            return (
              <li key={p.speaker_id} className="bg-panel border border-track rounded-lg p-3 flex flex-col gap-1.5">
                <div className="flex justify-between items-baseline">
                  <span className="num text-xs text-muted">{p.speaker_id}</span>
                  <Chip>{personLine(p)}</Chip>
                </div>
                <span className="text-[13px] font-semibold capitalize">{p.persona.role}</span>
                <span className="text-xs text-muted">
                  {p.persona.seniority} · talks {p.persona.verbosity === "high" ? "a lot" : p.persona.verbosity === "low" ? "little" : "some"} · {p.persona.code_switching} mixing
                </span>
                {turn && hasAudio && <PlayClip src={audioUrl(run, meeting, `utterances/${turn}.wav`)} label={`${p.speaker_id}'s voice`} text="Voice" />}
              </li>
            );
          })}
        </ul>
      </section>
      <Panel title="The conversation" actions={hasAudio ? <span className="text-xs text-muted num">{num(turns.length)} lines</span> : null}>
        {hasAudio && <Player src={audioUrl(run, meeting, "mixture.wav")} onTime={setT} label="the room recording" />}
        {!script.data && <p className="m-0 text-[13px] text-muted">The script is being written.</p>}
        <ol className="list-none m-0 p-0">
          {shown.map((x) => {
            const seg = times.get(x.turn_id);
            const now = seg && t >= seg.start && t < seg.end;
            return (
              <li key={x.turn_id} className={cx("grid grid-cols-[60px_44px_minmax(0,1fr)_auto] gap-2.5 items-start py-2 px-1 rounded", now && "bg-active/60")}>
                <span className="num text-xs text-muted pt-0.5">{seg ? clock(seg.start, false) : "–"}</span>
                <span className="num text-xs pt-0.5">{x.speaker_id}</span>
                <span className={cx("text-sm", x.interaction === "backchannel" && "text-muted")}>
                  {x.text}
                  {x.interaction !== "normal" && x.interaction !== "presentation" && <span className="ml-2 text-[11px] font-medium px-1.5 rounded bg-track text-muted">{x.interaction.replace("_", " ")}</span>}
                </span>
                {hasAudio && seg ? <PlayClip src={audioUrl(run, meeting, `utterances/${x.turn_id}.wav`)} label={`line ${x.turn_id}`} /> : <span />}
              </li>
            );
          })}
        </ol>
        {turns.length > 12 && (
          <Button size="sm" variant="ghost" onClick={() => setMore((v) => !v)}>
            {more ? "Show fewer" : `Show all ${turns.length} lines`}
          </Button>
        )}
      </Panel>
    </div>
  );
}

export function Pilot() {
  const { run = "" } = useParams();
  const runs = useRuns();
  const { g, update } = useGuided(suggestRunId(runs.data?.map((r) => r.id) ?? []));
  const nav = useNavigate();
  const meetings = useMeetings(run);
  const proc = useProcess(run);
  const suggest = useSuggest();
  const swap = useVoiceSwap(run);
  const [tab, setTab] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const running = proc.data?.process.state === "running";
  const list = meetings.data ?? [];
  const current = list[tab];
  useEffect(() => {
    if (running) {
      const id = setInterval(() => void meetings.refetch(), 4000);
      return () => clearInterval(id);
    }
  }, [running, meetings]);
  if (!g.draft) return <Navigate to="/create" replace />;
  const draft = g.draft;

  const push = (m: ChatMessage[]) => update((cur) => ({ chat: [...cur.chat, ...m] }));
  const restart = () => api.post(`/runs/${encodeURIComponent(run)}/start`, { confirm: !!proc.data?.live }).then(() => proc.refetch());

  const onSend = (text: string) => {
    push([{ from: "user", text }]);
    const spk = SPEAKER.exec(text);
    if (spk && VOICE_WORDS.test(text) && current) {
      const sid = `S${spk[1].padStart(2, "0")}`;
      swap.mutate(
        { meeting: current.meeting_id, speaker_id: sid },
        {
          onSuccess: async () => {
            push([{ from: "system", text: `I gave ${sid} a different voice with the same background. It is being recorded now.`, scope: "sample" }]);
            await restart();
          },
          onError: (e) => push([{ from: "system", text: e instanceof Error ? e.message : String(e) }]),
        },
      );
      return;
    }
    suggest.mutate(
      { request: text, draft },
      {
        onSuccess: (s) => {
          const reply = replyText(s.changes.length, s.offers.length, s.confirm.length, s.notes.map((n) => n.id), s.model === "fake");
          update((cur) => ({ draft: s.draft, chat: [...cur.chat, { from: "system", text: s.changes.length ? `${reply} That changes how every meeting is made, so the samples need remaking.` : reply, changes: s.changes, offers: s.offers, confirm: s.confirm, scope: s.changes.length ? "dataset" : undefined, action: s.changes.length ? "remake" : undefined }] }));
        },
      },
    );
  };

  const remake = async () => {
    setBusy(true);
    setError(null);
    try {
      if (running) await api.post(`/runs/${encodeURIComponent(run)}/stop`);
      await deletePilot(run);
      const created = await api.post<{ id: string; live: boolean }>("/runs", { ...g.draft, pilot: true });
      await api.post(`/runs/${encodeURIComponent(created.id)}/start`, { confirm: created.live });
      update({ pilot: created.id });
      nav(`/create/pilot/${encodeURIComponent(created.id)}`, { replace: true });
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const approve = async () => {
    const live = !!proc.data?.live;
    if (live && !window.confirm(`Making all ${draft.n_meetings} meetings reserves the shared GPU for many hours. Start now?`)) return;
    setBusy(true);
    setError(null);
    try {
      let id = draft.run_id;
      const taken = new Set(runs.data?.map((r) => r.id));
      for (let k = 2; taken.has(id); k++) id = `${draft.run_id}-${k}`;
      const created = await api.post<{ id: string; live: boolean }>("/runs", { ...draft, run_id: id, pilot: false });
      await api.post(`/runs/${encodeURIComponent(created.id)}/start`, { confirm: created.live });
      update({ run: created.id });
      nav(`/runs/${encodeURIComponent(created.id)}/review`);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const stage = current ? Object.entries(current.stages).find(([, s]) => s.state === "running" || s.state === "retry")?.[0] : undefined;
  return (
    <div className="flex flex-col gap-5">
      <Steps current={2} links={{ 0: "/create", 1: "/create/plan" }} />
      <div className="flex flex-wrap gap-5 items-start">
        <div className="flex-[999_1_560px] min-w-0 flex flex-col gap-5">
          <div>
            <h1 className="m-0 text-[22px] font-semibold tracking-tight">Listen to three sample meetings</h1>
            <p className="m-0 mt-1 text-[13px] text-muted">
              Check the voices and the conversation before we make all <span className="num">{draft.n_meetings}</span>.
            </p>
          </div>
          {meetings.isLoading && <Spinner label="Loading samples" />}
          {list.length > 0 && (
            <div role="tablist" aria-label="Sample meetings" className="flex flex-wrap gap-1 border-b border-track">
              {list.map((m, i) => (
                <button
                  key={m.meeting_id}
                  type="button"
                  role="tab"
                  aria-selected={i === tab}
                  onClick={() => setTab(i)}
                  className={cx("min-h-10 px-3.5 bg-transparent border-0 cursor-pointer text-[13px] -mb-px", i === tab ? "border-b-2 border-primary text-fg font-medium" : "text-muted")}
                  style={i === tab ? { borderBottom: "2px solid var(--color-primary)" } : undefined}
                >
                  Sample {i + 1}
                  <span className="ml-2 text-[11px] text-muted">{m.state === "done" ? "ready" : m.state}</span>
                </button>
              ))}
            </div>
          )}
          {current && current.state !== "done" && (
            <p role="status" className="m-0 flex items-center gap-2 text-[13px] text-muted">
              <span aria-hidden className={cx("size-2 rounded-full", running ? "bg-primary animate-pulse" : "bg-subtle")} />
              {running ? `${STAGE_WORDS[stage ?? ""] ?? "Waiting its turn"}…` : current.state === "failed" ? "This sample failed; remake the samples or ask for a change." : "Not started yet."}
            </p>
          )}
          {current && <Sample key={current.meeting_id} run={run} meeting={current.meeting_id} />}
          {error != null && <ErrorNote error={error} />}
          <div className="flex flex-wrap justify-between items-center gap-3">
            <Button onClick={() => nav("/create/plan")}>Back to plan</Button>
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={() => void remake()} disabled={busy}>
                Remake samples
              </Button>
              <Button variant="primary" onClick={() => void approve()} disabled={busy || running || !list.some((m) => m.state === "done")}>
                Looks good, make all {draft.n_meetings}
              </Button>
            </div>
          </div>
        </div>
        <ChatPanel
          hint="Each reply says whether it changes this sample or every meeting."
          messages={g.chat}
          busy={suggest.isPending || swap.isPending || busy}
          onSend={onSend}
          placeholder="e.g. S03 sounds robotic"
          extra={(m) =>
            m.action === "remake" ? (
              <div className="flex gap-2">
                <Button size="sm" variant="primary" onClick={() => void remake()} disabled={busy}>
                  Remake samples
                </Button>
              </div>
            ) : null
          }
        />
      </div>
    </div>
  );
}
