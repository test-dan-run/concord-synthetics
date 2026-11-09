// Step 1: the user says what meetings they need, in their own words, plus the topics (one per meeting theme), key
// terms and how much audio. "Make a plan" sends the description to the decision questions and opens the plan;
// nothing is created yet.

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Steps } from "../../components/guided";
import { Button, ErrorNote, Panel, Toggle } from "../../components/ui";
import { useRuns } from "../../lib/api";
import { suggestRunId } from "../../lib/draft";
import { contentOverrides, type KeyTerm, meetingsFor, suggestTopics, useGuided, useSuggest } from "../../lib/guided";

const CHIPS = ["Busy steering committee", "Quiet 1:1s", "Heated debate", "Someone presents slides", "Noisy room", "People dialling in"];
const HOURS = [5, 25, 50];

export function Describe() {
  const runs = useRuns();
  const { g, update, reset } = useGuided(suggestRunId(runs.data?.map((r) => r.id) ?? []));
  const nav = useNavigate();
  const suggest = useSuggest();
  const [term, setTerm] = useState<KeyTerm>({ term: "", say: "" });
  const [topicSeed, setTopicSeed] = useState(1);
  const [custom, setCustom] = useState(!HOURS.includes(g.hours));

  const topics = g.topics.length ? g.topics : [""];
  const setTopic = (i: number, v: string) => update({ topics: topics.map((t, k) => (k === i ? v : t)) });
  const filled = topics.filter((t) => t.trim()).length;
  const n = meetingsFor(g.hours, null);
  const perTopic = filled ? Math.round(n / filled) : 0;

  const addTerm = () => {
    if (!term.term.trim()) return;
    update({ terms: [...g.terms, { term: term.term.trim(), say: term.say?.trim() || null }] });
    setTerm({ term: "", say: "" });
  };
  const suggestSome = async () => {
    const got = await suggestTopics(topics, topicSeed, 3);
    setTopicSeed((s) => s + 1);
    update({ topics: [...topics.filter((t) => t.trim()), ...got.map((x) => x.topic)] });
  };
  const pasteList = () => {
    const text = window.prompt("Paste one topic per line");
    if (text) update({ topics: [...topics.filter((t) => t.trim()), ...text.split("\n").map((t) => t.trim()).filter(Boolean)] });
  };

  const makePlan = () => {
    const content = contentOverrides(topics, g.terms);
    const base = { run_id: g.runId, n_meetings: n, seed: 1, overrides: content, yaml: "" };
    if (!g.who.trim()) {
      update({ draft: base, chat: [] });
      nav("/create/plan");
      return;
    }
    suggest.mutate(
      { request: g.who, draft: base },
      {
        onSuccess: (s) => {
          update({
            draft: { ...base, overrides: { ...s.draft.overrides, ...content } },
            chat: [
              { from: "user", text: g.who },
              { from: "system", text: "Here is what I took from your description.", changes: s.changes, offers: s.offers, confirm: s.confirm },
            ],
          });
          nav("/create/plan");
        },
      },
    );
  };

  return (
    <div className="flex flex-col gap-5">
      <Steps current={0} links={g.draft ? { 1: "/create/plan" } : {}} />
      <div className="w-full max-w-[760px] mx-auto flex flex-col gap-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="m-0 text-[22px] font-semibold tracking-tight">What meetings do you need?</h1>
            <p className="m-0 mt-1 text-[13px] text-muted">Describe them the way you would to a colleague. You'll see a plan before anything is made.</p>
          </div>
          {(g.who || g.draft) && (
            <Button size="sm" variant="ghost" onClick={() => reset(suggestRunId(runs.data?.map((r) => r.id) ?? []))}>
              Start over
            </Button>
          )}
        </div>

        <Panel>
          <label htmlFor="who" className="text-sm font-semibold">
            Who's meeting, and where?
          </label>
          <textarea
            id="who"
            rows={5}
            value={g.who}
            onChange={(e) => update({ who: e.target.value })}
            placeholder="Mostly weekly team meetings in big boardrooms, 10 to 15 people. Some small design reviews around a laptop. A few people dial in. The chair is senior; juniors rarely speak. Lots of Singlish."
            className="w-full box-border resize-y bg-canvas text-fg border border-line rounded-md px-3 py-2.5 text-sm leading-relaxed"
          />
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-muted mr-0.5">Try adding</span>
            {CHIPS.map((c) => (
              <Button key={c} size="sm" onClick={() => update({ who: `${g.who.trim()}${g.who.trim() ? ". " : ""}${c}.` })}>
                + {c}
              </Button>
            ))}
          </div>
        </Panel>

        <Panel>
          <div className="flex flex-col gap-0.5">
            <h2 className="m-0 text-sm font-semibold">What do they talk about?</h2>
            <p className="m-0 text-[13px] text-muted">Add one topic per meeting theme. Each meeting is built around one of them, so more topics give you more varied meetings.</p>
          </div>
          <ol className="list-none m-0 p-0 flex flex-col gap-2">
            {topics.map((t, i) => (
              <li key={i} className="flex items-center gap-2">
                <span className="num text-xs text-muted w-6 text-right shrink-0">{i + 1}</span>
                <label htmlFor={`topic-${i}`} className="sr-only">
                  Topic {i + 1}
                </label>
                <input
                  id={`topic-${i}`}
                  value={t}
                  onChange={(e) => setTopic(i, e.target.value)}
                  placeholder={i === 0 ? "Rolling out a new HR system before the financial year end" : "Another topic"}
                  className="flex-1 min-w-0 min-h-10 bg-canvas text-fg border border-line rounded-md px-3 text-sm"
                />
                {t.trim() && <span className="text-xs text-muted shrink-0 min-w-[92px]">about <span className="num">{perTopic}</span> meetings</span>}
                <Button size="sm" aria-label={`Remove topic ${i + 1}`} onClick={() => update({ topics: topics.filter((_, k) => k !== i) })}>
                  ×
                </Button>
              </li>
            ))}
          </ol>
          <div className="flex flex-wrap gap-2 pl-8">
            <Button size="sm" className="border-dashed" onClick={() => update({ topics: [...topics, ""] })}>
              + Add another topic
            </Button>
            <Button size="sm" onClick={pasteList}>
              Paste a list
            </Button>
            <Button size="sm" onClick={() => void suggestSome()}>
              Suggest topics for me
            </Button>
          </div>
          {!filled && <p className="m-0 text-xs text-muted pl-8">No topics: meetings use the built-in list of Singapore workplace topics.</p>}

          <div className="flex flex-col gap-2 pt-1">
            <span className="text-[13px] font-medium">
              Key terms to use <span className="text-muted font-normal">· optional</span>
            </span>
            {g.terms.length > 0 && (
              <ul className="flex flex-wrap gap-1.5 list-none m-0 p-0">
                {g.terms.map((k, i) => (
                  <li key={`${k.term}-${i}`} className="flex items-center gap-1.5 pl-2 pr-1 py-1 border border-track bg-track rounded text-xs">
                    <span className="font-medium">{k.term}</span>
                    {k.say && <span className="text-muted">{k.say}</span>}
                    <button type="button" aria-label={`Remove ${k.term}`} onClick={() => update({ terms: g.terms.filter((_, j) => j !== i) })} className="bg-transparent border-0 text-muted cursor-pointer px-0.5 text-sm leading-none">
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <form
              className="flex flex-wrap gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                addTerm();
              }}
            >
              <label htmlFor="term" className="sr-only">
                New key term
              </label>
              <input id="term" value={term.term} onChange={(e) => setTerm({ ...term, term: e.target.value })} placeholder="Add a term, e.g. SingPass" className="flex-[2_1_200px] min-w-0 min-h-10 bg-canvas text-fg border border-line rounded-md px-3 text-[13px]" />
              <label htmlFor="say" className="sr-only">
                How is it said?
              </label>
              <input id="say" value={term.say ?? ""} onChange={(e) => setTerm({ ...term, say: e.target.value })} placeholder="How is it said? e.g. letters" className="flex-[1_1_170px] min-w-0 min-h-10 bg-canvas text-fg border border-line rounded-md px-3 text-[13px]" />
              <Button type="submit" disabled={!term.term.trim()}>
                Add
              </Button>
            </form>
            <p className="m-0 text-xs text-muted">Leave out real people's names. Speakers get made-up names. Write "letters" if a term is spelt out, like PDPC.</p>
          </div>
        </Panel>

        <Panel title="How much audio do you need?">
          <div className="flex flex-wrap items-center gap-3">
            <Toggle
              label="Amount of audio"
              options={[...HOURS.map((h) => ({ value: String(h), label: `${h} h` })), { value: "other", label: "Other" }]}
              value={custom ? "other" : String(g.hours)}
              onChange={(v) => {
                setCustom(v === "other");
                if (v !== "other") update({ hours: Number(v) });
              }}
            />
            {custom && (
              <label className="flex items-center gap-2 text-[13px] text-muted">
                Hours
                <input type="number" min={1} max={2000} value={g.hours} onChange={(e) => update({ hours: Math.max(1, Number(e.target.value) || 1) })} className="num w-24 min-h-10 bg-canvas text-fg border border-line rounded-md px-3" />
              </label>
            )}
          </div>
          <p className="m-0 text-[13px] text-muted">
            About <span className="num text-fg">{n}</span> meetings. The plan shows how long they take to make.
          </p>
        </Panel>

        {suggest.error && <ErrorNote error={suggest.error} />}
        <div className="flex flex-wrap justify-end items-center gap-3">
          <span className="text-xs text-muted">Nothing runs yet. This only builds a plan.</span>
          <Button variant="primary" onClick={makePlan} disabled={suggest.isPending}>
            {suggest.isPending ? "Reading your description…" : "Make a plan"}
          </Button>
        </div>
      </div>
    </div>
  );
}
