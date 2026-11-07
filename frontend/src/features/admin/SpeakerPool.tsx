// Admin: the voice pool meetings clone from. An admin imports a curated voice folder as a draft, reviews each new
// speaker (clips, the main clip to clone, leave out), and publishes a new pool version. Splits are assigned by the
// server once and for good; admins never pick them. Only `synthetics viewer --admin` serves these routes.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { PlayClip } from "../../components/guided";
import { Button, Chip, cx, ErrorNote, PageHeader, Panel, Spinner } from "../../components/ui";
import { api } from "../../lib/api";
import { num } from "../../lib/format";

const SPLITS = ["train", "dev", "test"] as const;
const GROUPS = ["CHINESE", "MALAY", "INDIAN", "OTHERS"];

interface PoolInfo {
  version: string;
  speakers: number;
  counts: Record<string, number>;
  coverage: Record<string, Record<string, number>>;
  runs_using: number;
  versions: { version: string; path: string }[];
  draft: { source: string; version: string } | null;
}

interface DraftSpeaker {
  speaker: string;
  gender: string;
  age: string;
  ethnicity: string;
  first_language: string;
  source: string;
  consent: string;
  split: string | null;
  by_partner: boolean;
  status: "ready" | "excluded" | "left_out";
  reason: string | null;
  usable_clips: number;
  best_mos: number | null;
  main_clip: string | null;
  clips: { clip_id: string; text: string; mos: number; duration: number }[];
}

interface DraftView {
  source: string;
  version: string;
  speakers: DraftSpeaker[];
  counts: Record<string, number>;
  added: Record<string, number>;
  coverage: Record<string, Record<string, { base: number; added: number }>>;
  rules: { mos_floor: number; min_clips: number };
  ready: number;
  not_ready: number;
}

async function patch(path: string, body: unknown): Promise<DraftView> {
  const r = await fetch(`/api${path}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? r.statusText);
  return r.json();
}

function Coverage({ pool, draft }: { pool: PoolInfo; draft?: DraftView }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-[13px]">
        <caption className="sr-only">Speakers by ethnicity and split</caption>
        <thead>
          <tr>
            {["Split", ...GROUPS.map((g) => g[0] + g.slice(1).toLowerCase()), "Total"].map((h) => (
              <th key={h} scope="col" className="text-left text-[11px] font-medium uppercase tracking-wider text-muted px-3 py-2 border-b border-line">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {SPLITS.map((s) => (
            <tr key={s}>
              <td className="px-3 py-2 border-b border-track capitalize">{s}</td>
              {GROUPS.map((g) => {
                const added = draft?.coverage[s]?.[g]?.added ?? 0;
                return (
                  <td key={g} className="px-3 py-2 border-b border-track num">
                    {pool.coverage[s]?.[g] ?? 0}
                    {added > 0 && <span className="text-good"> +{added}</span>}
                  </td>
                );
              })}
              <td className="px-3 py-2 border-b border-track num">
                {pool.counts[s] ?? 0}
                {(draft?.added[s] ?? 0) > 0 && <span className="text-good"> +{draft?.added[s]}</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SpeakerReview({ s, onPatch }: { s: DraftSpeaker; onPatch: (b: { main_clip?: string; excluded?: boolean }) => void }) {
  return (
    <Panel title={<>{s.speaker} <span className="font-normal text-muted">· choose the voice to clone</span></>} subtitle="Meetings clone from the main clip. The others are backups if a line fails its voice check." className="flex-[1.4_1_480px]">
      <ul className="list-none m-0 p-0">
        {s.clips.map((c) => (
          <li key={c.clip_id} className={cx("grid grid-cols-[auto_84px_minmax(0,1fr)_44px_52px] gap-2.5 items-center px-2 py-2 rounded", c.clip_id === s.main_clip && "bg-active/60")}>
            <PlayClip src={`/api/admin/voices/${s.speaker}/${c.clip_id}`} label={`clip ${c.clip_id}`} />
            {c.clip_id === s.main_clip ? <Chip tone="info">Main</Chip> : <Button size="sm" variant="ghost" onClick={() => onPatch({ main_clip: c.clip_id })}>Make main</Button>}
            <span className="text-[13px] truncate">“{c.text}”</span>
            <span className="num text-xs">{c.mos.toFixed(1)}</span>
            <span className="num text-xs text-muted">{c.duration.toFixed(1)} s</span>
          </li>
        ))}
      </ul>
      <div className="flex justify-end gap-2 pt-1 border-t border-track">
        <Button onClick={() => onPatch({ excluded: s.status !== "excluded" })}>{s.status === "excluded" ? "Put back" : "Leave out"}</Button>
      </div>
    </Panel>
  );
}

export function SpeakerPool() {
  const qc = useQueryClient();
  const pool = useQuery({ queryKey: ["admin-pool"], queryFn: () => api.get<PoolInfo>("/admin/pool") });
  const draft = useQuery({
    queryKey: ["admin-draft"],
    queryFn: async () => {
      try {
        return await api.get<DraftView>("/admin/pool/draft");
      } catch {
        return null;
      }
    },
  });
  const [source, setSource] = useState("/home/user/code/datasets/nsc-part3-voices");
  const [sel, setSel] = useState<string | null>(null);
  const setDraft = (d: DraftView | null) => qc.setQueryData(["admin-draft"], d);
  const create = useMutation({ mutationFn: () => api.post<DraftView>("/admin/pool/drafts", { source }), onSuccess: setDraft });
  const edit = useMutation({ mutationFn: ({ sid, b }: { sid: string; b: unknown }) => patch(`/admin/pool/draft/speakers/${sid}`, b), onSuccess: setDraft });
  const pub = useMutation({
    mutationFn: () => api.post<{ version: string; path: string; next: string }>("/admin/pool/draft/publish"),
    onSuccess: () => {
      setDraft(null);
      void qc.invalidateQueries({ queryKey: ["admin-pool"] });
    },
  });
  const d = draft.data ?? undefined;
  const selected = d?.speakers.find((s) => s.speaker === sel) ?? d?.speakers.find((s) => s.status === "ready");

  if (pool.isLoading) return <Spinner label="Loading the speaker pool" />;
  if (pool.error || !pool.data) return <ErrorNote error={pool.error ?? "The admin routes are off: start the viewer with --admin."} />;
  const p = pool.data;
  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        crumbs="Admin"
        title="Speaker pool"
        subtitle={<>The real voices meetings clone from. Pool <span className="num text-fg">{p.version}</span> · in use by <span className="num text-fg">{p.runs_using}</span> runs</>}
      />
      <div className="flex flex-wrap gap-3">
        {[
          ["Speakers", num(p.speakers)],
          ["Train · dev · test", SPLITS.map((s) => p.counts[s] ?? 0).join(" · ")],
          ["Draft additions", num(d ? d.ready : 0)],
        ].map(([k, v]) => (
          <div key={k} className="flex-[1_1_140px] min-w-[140px] bg-panel border border-track rounded-lg px-4 py-3">
            <div className="text-xs text-muted">{k}</div>
            <div className="num text-[22px] font-medium">{v}</div>
          </div>
        ))}
      </div>
      <Panel title="Who's in each split" subtitle={d ? "Current pool, then + draft" : "Splits are assigned once per speaker, automatically, so they never overlap across runs."}>
        <Coverage pool={p} draft={d} />
      </Panel>

      {!d ? (
        <Panel title="Add speakers" subtitle="Import a curated voice folder (speakers.csv and manifest.jsonl, as nsc-part3-voices is laid out).">
          <form
            className="flex flex-wrap gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              create.mutate();
            }}
          >
            <label htmlFor="src" className="sr-only">
              Voice folder
            </label>
            <input id="src" value={source} onChange={(e) => setSource(e.target.value)} className="num flex-1 min-w-0 min-h-10 bg-canvas text-fg border border-line rounded-md px-3 text-xs" />
            <Button type="submit" variant="primary" disabled={create.isPending}>
              {create.isPending ? "Reading…" : "Import"}
            </Button>
          </form>
          {create.error && <ErrorNote error={create.error} />}
        </Panel>
      ) : (
        <>
          <Panel
            title={<>Draft: pool <span className="num">{d.version}</span></>}
            subtitle={<><span className="num">{d.speakers.length}</span> new speakers from <span className="num">{d.source}</span></>}
            actions={<Button size="sm" variant="ghost" onClick={() => void fetch("/api/admin/pool/draft", { method: "DELETE" }).then(() => setDraft(null))}>Discard draft</Button>}
          >
            <div className="overflow-x-auto max-h-[420px] overflow-y-auto scroll-thin">
              <table className="w-full min-w-[860px] border-collapse text-[13px]">
                <caption className="sr-only">New speakers in the draft</caption>
                <thead className="sticky top-0 bg-panel">
                  <tr>
                    {["Speaker", "Source", "Gender · age", "Ethnicity", "First language", "Usable clips", "Best clip", "Split", "Status"].map((h) => (
                      <th key={h} scope="col" className="text-left text-[11px] font-medium uppercase tracking-wider text-muted px-3 py-2 border-b border-line whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {d.speakers.map((s) => (
                    <tr key={s.speaker} onClick={() => s.status !== "left_out" && setSel(s.speaker)} className={cx("cursor-pointer hover:bg-track/60", s.speaker === selected?.speaker && "bg-active/60")}>
                      <td className="px-3 py-2 border-b border-track num">{s.speaker}</td>
                      <td className="px-3 py-2 border-b border-track">{s.source}</td>
                      <td className="px-3 py-2 border-b border-track">{s.gender} · <span className="num">{s.age}</span></td>
                      <td className="px-3 py-2 border-b border-track capitalize">{s.ethnicity.toLowerCase()}</td>
                      <td className="px-3 py-2 border-b border-track capitalize">{s.first_language.toLowerCase()}</td>
                      <td className="px-3 py-2 border-b border-track num">{s.usable_clips}</td>
                      <td className="px-3 py-2 border-b border-track num">{s.best_mos?.toFixed(1) ?? "–"}</td>
                      <td className="px-3 py-2 border-b border-track capitalize">{s.split ?? "–"}{s.by_partner && <span className="text-muted text-xs"> · with partner</span>}</td>
                      <td className="px-3 py-2 border-b border-track">
                        <span className={cx("inline-flex items-center gap-1.5", s.status === "left_out" && "text-warn")}>
                          <span aria-hidden className={cx("size-2 rounded-full", s.status === "ready" ? "bg-good" : s.status === "excluded" ? "bg-pending" : "bg-warn")} />
                          {s.status === "ready" ? "Ready" : s.status === "excluded" ? "Left out" : s.reason}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="m-0 text-xs text-muted">
              Best clip is the audio quality score from 1 to 5 after cleanup; clips below <span className="num">{d.rules.mos_floor}</span> aren't used, and a speaker needs at least <span className="num">{d.rules.min_clips}</span>. Clips where a name was bleeped out are never used. People who recorded together always share a split.
            </p>
          </Panel>
          <div className="flex flex-wrap gap-5 items-start">
            {selected && <SpeakerReview s={selected} onPatch={(b) => edit.mutate({ sid: selected.speaker, b })} />}
            <Panel title={<>Publish pool <span className="num">{d.version}</span></>} className="flex-[1_1_340px]">
              <ul className="m-0 pl-5 flex flex-col gap-1.5 text-[13px]">
                <li>Runs already made keep the pool they started with. Only new runs use this one.</li>
                <li>The <span className="num">{p.speakers}</span> current speakers keep their split forever.</li>
                <li>Each new speaker gets one split for good, assigned automatically.</li>
              </ul>
              {d.not_ready > 0 && (
                <div role="note" className="px-3 py-2.5 border border-warn/50 bg-warn/10 rounded-md text-xs">
                  <span className="font-semibold text-warn">{d.not_ready} speakers not included</span> · they are left out, or their clips don't meet the bar.
                </div>
              )}
              {(edit.error || pub.error) && <ErrorNote error={edit.error ?? pub.error} />}
              {pub.data && <p className="m-0 text-xs text-good">Published {pub.data.path}. {pub.data.next}.</p>}
              <Button variant="primary" onClick={() => pub.mutate()} disabled={pub.isPending || d.ready === 0}>
                Publish with {d.ready} speakers
              </Button>
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}
