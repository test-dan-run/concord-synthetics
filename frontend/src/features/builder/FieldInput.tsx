// One curated config field of the run builder: a range (lo to hi), a probability, a number, category weights or
// a switch. Values are the config's own (shares stay 0-1 and are shown in %); typing never loses a half-typed
// number, because each input keeps its own text until it parses.

import { useEffect, useId, useState, type InputHTMLAttributes } from "react";
import { cx } from "../../components/ui";
import type { Dist } from "../../lib/api";
import type { Field } from "../../lib/draft";

const round = (v: number) => String(+v.toFixed(6));

export function NumInput({
  value,
  onValue,
  scale = 1,
  className,
  ...p
}: Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange"> & { value: number | undefined; onValue: (v: number) => void; scale?: number }) {
  const shown = value === undefined || Number.isNaN(value) ? "" : round(value * scale);
  const [text, setText] = useState(shown);
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setText(shown);
  }, [shown, focused]);
  return (
    <input
      type="number"
      inputMode="decimal"
      {...p}
      value={text}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      onChange={(e) => {
        setText(e.target.value);
        const n = Number(e.target.value);
        if (e.target.value.trim() !== "" && Number.isFinite(n)) onValue(n / scale);
      }}
      className={cx("bg-canvas text-fg border border-line rounded-md min-h-[34px] px-2 text-xs num w-[84px]", className)}
    />
  );
}

function Unit({ children }: { children?: string }) {
  return children ? <span className="text-xs text-muted">{children}</span> : null;
}

export function FieldInput({ field, value, onChange, defaultValue, error }: { field: Field; value: unknown; onChange: (v: unknown) => void; defaultValue: unknown; error?: string }) {
  const id = useId();
  const scale = field.percent ? 100 : 1;
  const changed = JSON.stringify(value) !== JSON.stringify(defaultValue);
  const step = field.int ? 1 : (field.step ?? (field.percent ? 1 : 0.01));
  const bounds = { min: field.min, max: field.max, step };
  const d = (value ?? {}) as Dist;

  let control;
  if (field.kind === "range") {
    const lo = d.dist === "fixed" ? d.value : d.lo;
    const hi = d.dist === "fixed" ? d.value : d.hi;
    const set = (k: "lo" | "hi", v: number) => {
      const base = d.dist === "fixed" ? { dist: field.int ? "int_uniform" : "uniform", lo, hi } : d;
      onChange({ ...base, [k]: field.int ? Math.round(v) : v });
    };
    control = (
      <span className="flex flex-wrap items-center gap-2">
        <NumInput id={id} aria-label={`${field.label}, lowest`} value={lo} onValue={(v) => set("lo", v)} scale={scale} {...bounds} />
        <span className="text-muted text-xs">to</span>
        <NumInput aria-label={`${field.label}, highest`} value={hi} onValue={(v) => set("hi", v)} scale={scale} {...bounds} />
        <Unit>{field.unit}</Unit>
      </span>
    );
  } else if (field.kind === "prob") {
    control = (
      <span className="flex items-center gap-2">
        <input
          type="range"
          aria-hidden
          tabIndex={-1}
          min={0}
          max={100}
          step={1}
          value={(d.p ?? 0) * 100}
          onChange={(e) => onChange({ dist: "bernoulli", p: Number(e.target.value) / 100 })}
          className="accent-primary w-28 min-h-8"
        />
        <NumInput id={id} value={d.p} onValue={(v) => onChange({ dist: "bernoulli", p: Math.min(1, Math.max(0, v)) })} scale={100} min={0} max={100} step={1} className="w-[68px]" />
        <Unit>%</Unit>
      </span>
    );
  } else if (field.kind === "number") {
    control = (
      <span className="flex items-center gap-2">
        <NumInput id={id} value={value as number} onValue={onChange} scale={scale} {...bounds} />
        <Unit>{field.unit}</Unit>
      </span>
    );
  } else if (field.kind === "weights") {
    const w = d.weights ?? {};
    const total = Object.values(w).reduce((a, b) => a + b, 0) || 1;
    control = (
      <span className="flex flex-wrap items-center gap-x-4 gap-y-2">
        {(field.keys ?? Object.keys(w)).map((k, i) => (
          <label key={k} className="flex items-center gap-2 text-xs text-muted">
            {k}
            <NumInput
              id={i === 0 ? id : undefined}
              value={w[k] ?? 0}
              onValue={(v) => onChange({ dist: "categorical", weights: { ...w, [k]: Math.max(0, v) } })}
              scale={100}
              min={0}
              step={1}
              className="w-[68px]"
            />
            <span className="num text-subtle" aria-label={`${k} share`}>{`${Math.round(((w[k] ?? 0) / total) * 100)}%`}</span>
          </label>
        ))}
      </span>
    );
  } else {
    control = (
      <input id={id} type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} className="size-4 accent-primary cursor-pointer" />
    );
  }

  return (
    <div className={cx("grid gap-x-4 gap-y-1.5 py-2.5 border-b border-track last:border-0 sm:grid-cols-[minmax(0,240px)_1fr] items-center", field.kind === "toggle" && "sm:grid-cols-[minmax(0,1fr)_auto]")}>
      <label htmlFor={id} className="flex flex-col gap-0.5 text-[13px] min-w-0">
        <span className="flex items-center gap-2">
          {field.label}
          {changed && <span className="text-[11px] text-primary" title="Differs from the default">changed</span>}
        </span>
        {field.hint && <span className="text-[11px] text-muted">{field.hint}</span>}
      </label>
      <div className="flex flex-col gap-1 min-w-0">
        {control}
        {error && (
          <span role="alert" className="text-[11px] text-bad">
            {error}
          </span>
        )}
      </div>
    </div>
  );
}
