// Pieces the guided flow shares: the step bar, an audio player, the "ask for changes" chat panel and its change
// chips. Speakers are never colour-coded: a speaker is named by id, and selection is the only highlight.

import { type FormEvent, type ReactNode, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { ChatMessage, Offer } from "../lib/guided";
import { changeLabel } from "../lib/guided";
import { clock } from "../lib/format";
import { Button, Chip, cx } from "./ui";

export const STEPS = ["Describe", "Plan", "Listen to samples", "Generate and review", "Export"] as const;

export function Steps({ current, links = {} }: { current: number; links?: Partial<Record<number, string>> }) {
  return (
    <nav aria-label="Steps">
      <ol className="flex flex-wrap gap-1 list-none m-0 p-0 text-[13px]">
        {STEPS.map((s, i) => {
          const done = i < current;
          const body = (
            <>
              {done ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--color-good)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                  <path d="M5 12l5 5L20 7" />
                </svg>
              ) : (
                <span className={cx("num text-xs", i === current && "text-primary")}>{i + 1}</span>
              )}
              {s}
            </>
          );
          const cls = cx("flex items-center gap-2 px-2.5 py-1.5 rounded-md", i === current ? "bg-active text-fg font-medium" : "text-muted");
          const to = links[i];
          return (
            <li key={s} aria-current={i === current ? "step" : undefined}>
              {to && i !== current ? (
                <Link to={to} className={cx(cls, "no-underline !text-muted hover:!text-fg")}>
                  {body}
                </Link>
              ) : (
                <span className={cls}>{body}</span>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function PlayIcon({ playing }: { playing: boolean }) {
  return playing ? (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M6 4h4v16H6zM14 4h4v16h-4z" />
    </svg>
  ) : (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M7 4l13 8-13 8z" />
    </svg>
  );
}

/** One shared <audio> for the page, so starting a clip stops the one playing. */
let current: HTMLAudioElement | null = null;

/** A play button for a short clip (a line, a voice). */
export function PlayClip({ src, label, text }: { src: string; label: string; text?: string }) {
  const [playing, setPlaying] = useState(false);
  const ref = useRef<HTMLAudioElement | null>(null);
  useEffect(() => () => {
    ref.current?.pause();
  }, []);
  const toggle = () => {
    if (!ref.current) {
      ref.current = new Audio(src);
      ref.current.onended = ref.current.onpause = () => setPlaying(false);
      ref.current.onerror = () => setPlaying(false);
    }
    if (playing) {
      ref.current.pause();
      return;
    }
    if (current && current !== ref.current) current.pause();
    current = ref.current;
    ref.current.currentTime = 0;
    void ref.current.play().then(() => setPlaying(true), () => setPlaying(false));
  };
  return (
    <Button size="sm" onClick={toggle} aria-label={`${playing ? "Pause" : "Play"} ${label}`} aria-pressed={playing}>
      <PlayIcon playing={playing} />
      {text}
    </Button>
  );
}

/** A long recording with a seek bar; `onTime` reports the position for highlighting the line being spoken. */
export function Player({ src, onTime, label }: { src: string; onTime?: (t: number) => void; label: string }) {
  const ref = useRef<HTMLAudioElement>(null);
  const [t, setT] = useState(0);
  const [d, setD] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    setT(0);
    setPlaying(false);
  }, [src]);
  const toggle = () => {
    const a = ref.current;
    if (!a) return;
    if (a.paused) {
      if (current && current !== a) current.pause();
      current = a;
      void a.play();
    } else a.pause();
  };
  return (
    <div className="flex items-center gap-3">
      <audio
        ref={ref}
        src={src}
        preload="metadata"
        onLoadedMetadata={(e) => setD(e.currentTarget.duration || 0)}
        onTimeUpdate={(e) => {
          setT(e.currentTarget.currentTime);
          onTime?.(e.currentTarget.currentTime);
        }}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
      />
      <Button variant="primary" onClick={toggle} aria-label={`${playing ? "Pause" : "Play"} ${label}`} className="!px-3 shrink-0">
        <PlayIcon playing={playing} />
      </Button>
      <label className="flex-1 min-w-0">
        <span className="sr-only">Position in {label}</span>
        <input
          type="range"
          min={0}
          max={d || 0}
          step={0.1}
          value={t}
          onChange={(e) => {
            if (ref.current) ref.current.currentTime = Number(e.target.value);
          }}
          className="w-full accent-[var(--color-focus)]"
        />
      </label>
      <span className="num text-xs text-muted shrink-0">
        {clock(t, false)} / {clock(d, false)}
      </span>
    </div>
  );
}

export function ChangeChips({ m, onUndo }: { m: ChatMessage; onUndo?: () => void }) {
  if (!m.changes?.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {m.changes.map((c) => (
        <Chip key={c.path} tone={m.undone ? "default" : "info"}>
          <span className={cx(m.undone && "line-through")}>{changeLabel(c)}</span>
        </Chip>
      ))}
      {onUndo && !m.undone && (
        <Button size="sm" variant="ghost" onClick={onUndo}>
          Undo
        </Button>
      )}
    </div>
  );
}

function OfferButtons({ offer, onPick }: { offer: Offer; onPick: (o: Offer, option: string) => void }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs text-muted">{offer.question}</span>
      <div className="flex flex-wrap gap-1.5">
        {offer.options.map((o) => (
          <Button key={o.id} size="sm" onClick={() => onPick(offer, o.id)}>
            {o.label}
          </Button>
        ))}
      </div>
    </div>
  );
}

/** The side panel where the user asks for changes in their own words. */
export function ChatPanel({
  title = "Ask for changes",
  hint,
  messages,
  busy,
  onSend,
  onUndo,
  onPick,
  extra,
  placeholder = "Describe a change…",
}: {
  title?: string;
  hint: ReactNode;
  messages: ChatMessage[];
  busy: boolean;
  onSend: (text: string) => void;
  onUndo?: (index: number) => void;
  onPick?: (o: Offer, option: string, index: number) => void;
  extra?: (m: ChatMessage, index: number) => ReactNode;
  placeholder?: string;
}) {
  const [text, setText] = useState("");
  const end = useRef<HTMLLIElement>(null);
  useEffect(() => {
    void end.current?.scrollIntoView?.({ block: "nearest" }); // a Promise in some browsers: never an effect cleanup
  }, [messages.length]);
  const send = (e: FormEvent) => {
    e.preventDefault();
    if (!text.trim() || busy) return;
    onSend(text.trim());
    setText("");
  };
  return (
    <aside aria-labelledby="chat-title" className="flex-[1_1_340px] min-w-0 bg-panel border border-track rounded-lg flex flex-col max-h-[80vh] lg:sticky lg:top-6">
      <div className="px-4 py-3.5 border-b border-track">
        <h2 id="chat-title" className="m-0 text-sm font-semibold">
          {title}
        </h2>
        <p className="m-0 mt-0.5 text-xs text-muted">{hint}</p>
      </div>
      <ol className="list-none m-0 p-4 flex flex-col gap-3 text-[13px] overflow-y-auto scroll-thin flex-1" aria-live="polite">
        {messages.length === 0 && <li className="text-muted text-xs">Try “make half of them hybrid” or “noisier rooms”.</li>}
        {messages.map((m, i) =>
          m.from === "user" ? (
            <li key={i} className="self-end max-w-[85%] bg-active rounded-md px-2.5 py-2">
              {m.text}
            </li>
          ) : (
            <li key={i} className="max-w-[92%] flex flex-col gap-2">
              <span>{m.text}</span>
              {m.scope && <Chip tone={m.scope === "dataset" ? "warn" : "default"}>{m.scope === "dataset" ? "Every meeting" : "This sample only"}</Chip>}
              <ChangeChips m={m} onUndo={onUndo ? () => onUndo(i) : undefined} />
              {[...(m.offers ?? []), ...(m.confirm ?? [])].map((o) => (
                <OfferButtons key={o.group} offer={o} onPick={(off, opt) => onPick?.(off, opt, i)} />
              ))}
              {extra?.(m, i)}
            </li>
          ),
        )}
        {busy && (
          <li role="status" className="text-xs text-muted">
            Thinking…
          </li>
        )}
        <li ref={end} aria-hidden />
      </ol>
      <form onSubmit={send} className="mt-auto px-4 pt-3 pb-4 border-t border-track flex gap-2">
        <label htmlFor="chat-msg" className="sr-only">
          Describe a change
        </label>
        <input
          id="chat-msg"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={placeholder}
          className="flex-1 min-w-0 min-h-10 bg-canvas text-fg border border-line rounded-md px-3 text-[13px]"
        />
        <Button type="submit" disabled={busy || !text.trim()}>
          Send
        </Button>
      </form>
    </aside>
  );
}

/** The reply to a suggestion, in plain words, from what changed and what is left to choose. */
export function replyText(applied: number, offers: number, confirm: number, notes: string[], fake: boolean): string {
  const parts: string[] = [];
  if (applied) parts.push(`Done: ${applied === 1 ? "one setting changed" : `${applied} settings changed`}.`);
  if (offers) parts.push(offers === 1 ? "One thing could go two ways; pick one." : `${offers} things could go two ways; pick for each.`);
  if (confirm) parts.push("Some changes need your OK before they apply.");
  if (notes.includes("out_of_scope")) parts.push("Part of that is not a setting (topics, emotions, a speaker's accent): use the topics and key terms instead.");
  if (notes.includes("exact_numbers")) parts.push("For an exact number, set it under Advanced settings.");
  if (!parts.length) parts.push("I could not find a setting to change in that. Try naming what should change: rooms, people, noise, interruptions, hybrid.");
  if (fake) parts.push("(Rough suggestions: the decision model is not running, so this used word matching.)");
  return parts.join(" ");
}
