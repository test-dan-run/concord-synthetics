// A histogram with the configured range shaded behind it (UI doc, "Sampler distributions"). Marks follow the
// dataviz skill: bars at most 24 px wide with a 4 px rounded data end, square at the baseline, a 2 px gap between
// bars, a hairline solid grid, a per-bar tooltip on hover and focus, and a table view of every bin.

import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { cx, Panel } from "../../components/ui";
import { C } from "../../lib/colors";
import { num } from "../../lib/format";
import type { Bin } from "../../lib/stats";

const H = 190;

/** The wrapper's width, so the SVG is drawn at its real size and text never scales with the chart. */
function useWidth(initial = 520) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(initial);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}
const M = { l: 34, r: 12, t: 12, b: 26 };

export interface HistogramProps {
  id: string;
  title: ReactNode;
  subtitle?: ReactNode;
  note?: ReactNode;
  bins: Bin[];
  /** one colour for every bar, or per bar */
  color: string | ((b: Bin) => string);
  range?: { lo: number; hi: number; label: string } | null;
  /** x tick values (in data units) */
  ticks: number[];
  tick: (v: number) => string;
  /** bin label in the tooltip and the table */
  binLabel: (b: Bin) => string;
  unit: string;
  swatch?: string;
}

function Bars({ bins, color, range, ticks, tick, binLabel, unit, label }: Omit<HistogramProps, "id" | "title" | "subtitle" | "note" | "swatch"> & { label: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const [ref, W] = useWidth();
  const lo = bins[0]?.lo ?? 0;
  const hi = bins[bins.length - 1]?.hi ?? 1;
  const span = hi - lo || 1;
  const x = (v: number) => M.l + ((v - lo) / span) * (W - M.l - M.r);
  const top = Math.max(1, ...bins.map((b) => b.count));
  const yMax = Math.max(2, Math.ceil(top / 2) * 2); // even, so the half-way tick is a whole count
  const y = (v: number) => H - M.b - (v / yMax) * (H - M.t - M.b);
  const slot = (W - M.l - M.r) / Math.max(1, bins.length);
  const bw = Math.max(2, Math.min(24, slot - 2));
  const fill = (b: Bin) => (typeof color === "string" ? color : color(b));
  const hb = hover === null ? null : bins[hover];
  return (
    <div className="relative" ref={ref}>
      <svg width={W} height={H} className="block" role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
        {/* recessive hairline grid at 0, half and the top */}
        {[0, yMax / 2, yMax].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke={C.track} strokeWidth={1} />
            <text x={M.l - 6} y={y(v) + 3} textAnchor="end" fontSize={10} fill={C.muted} className="num">
              {v}
            </text>
          </g>
        ))}
        {range && (
          <g>
            <rect x={x(Math.max(lo, range.lo))} y={M.t} width={Math.max(0, x(Math.min(hi, range.hi)) - x(Math.max(lo, range.lo)))} height={H - M.t - M.b} fill={C.line} opacity={0.35} />
            {[range.lo, range.hi].map((v) =>
              v >= lo && v <= hi ? <line key={v} x1={x(v)} x2={x(v)} y1={M.t} y2={H - M.b} stroke={C.muted} strokeDasharray="3 3" strokeWidth={1} /> : null,
            )}
          </g>
        )}
        {bins.map((b, i) => {
          const cx0 = M.l + slot * i + slot / 2;
          const h = H - M.b - y(b.count);
          const r = Math.min(4, bw / 2, h);
          const left = cx0 - bw / 2;
          const yt = y(b.count);
          const path = b.count
            ? `M${left} ${H - M.b}V${yt + r}Q${left} ${yt} ${left + r} ${yt}H${left + bw - r}Q${left + bw} ${yt} ${left + bw} ${yt + r}V${H - M.b}Z`
            : "";
          return (
            <g key={i}>
              {path && <path d={path} fill={fill(b)} opacity={hover === null || hover === i ? 1 : 0.35} />}
              {/* the hit target is the whole slot, bigger than the mark */}
              <rect
                x={M.l + slot * i}
                y={M.t}
                width={slot}
                height={H - M.t - M.b}
                fill="transparent"
                onMouseEnter={() => setHover(i)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
                tabIndex={-1}
              />
            </g>
          );
        })}
        <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} stroke={C.line} strokeWidth={1} />
        {range && (
          <g className="pointer-events-none">
            <rect x={x(Math.max(lo, range.lo)) + 2} y={M.t + 2} width={range.label.length * 6.2 + 8} height={14} rx={2} fill={C.canvas} opacity={0.85} />
            <text x={x(Math.max(lo, range.lo)) + 6} y={M.t + 12} fontSize={10} fill={C.muted} className="num">
              {range.label}
            </text>
          </g>
        )}
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 15} textAnchor="middle" fontSize={10} fill={C.muted} className="num">
            {tick(t)}
          </text>
        ))}
      </svg>
      {hb && (
        <div
          role="status"
          className="pointer-events-none absolute bg-canvas border border-line rounded-md px-2 py-1 text-xs whitespace-nowrap"
          style={{ left: M.l + slot * hover! + slot / 2, top: 0, transform: "translateX(-50%)" }}
        >
          <span className="text-muted">{binLabel(hb)}</span> · <span className="num">{num(hb.count)}</span> {unit}
        </div>
      )}
    </div>
  );
}

function BinTable({ bins, binLabel, unit, caption }: { bins: Bin[]; binLabel: (b: Bin) => string; unit: string; caption: string }) {
  return (
    <div className="border border-track rounded-md overflow-auto scroll-thin max-h-[240px]">
      <table className="w-full text-[13px] border-collapse">
        <caption className="sr-only">{caption}</caption>
        <thead className="sticky top-0 bg-panel">
          <tr className="border-b border-line text-[11px] text-muted uppercase tracking-wider">
            <th scope="col" className="px-3 py-1.5 text-left font-medium">Bin</th>
            <th scope="col" className="px-3 py-1.5 text-right font-medium">{unit}</th>
          </tr>
        </thead>
        <tbody>
          {bins.map((b, i) => (
            <tr key={i} className="border-b border-track last:border-0">
              <td className="px-3 py-1 num">{binLabel(b)}</td>
              <td className="px-3 py-1 num text-right">{num(b.count)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Histogram(p: HistogramProps) {
  const [table, setTable] = useState(false);
  const total = p.bins.reduce((s, b) => s + b.count, 0);
  const label = typeof p.title === "string" ? p.title : p.id;
  return (
    <Panel
      id={p.id}
      title={
        <span className="inline-flex items-center gap-2">
          {p.swatch && <span aria-hidden className="size-2 rounded-sm" style={{ background: p.swatch }} />}
          {p.title}
        </span>
      }
      subtitle={p.subtitle}
      actions={
        <button
          type="button"
          aria-pressed={table}
          onClick={() => setTable((t) => !t)}
          className={cx("min-h-8 px-2.5 text-xs rounded-md border border-line cursor-pointer", table ? "bg-active text-fg" : "bg-transparent text-muted hover:text-fg")}
        >
          Table
        </button>
      }
    >
      {total === 0 ? (
        <span className="text-[13px] text-muted">No data in this run yet.</span>
      ) : table ? (
        <BinTable bins={p.bins} binLabel={p.binLabel} unit={p.unit} caption={label} />
      ) : (
        <Bars {...p} label={`${label}: histogram of ${total} ${p.unit}`} />
      )}
      {p.note && <span className="text-xs text-muted">{p.note}</span>}
    </Panel>
  );
}
