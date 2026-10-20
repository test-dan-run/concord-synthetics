// Histogram binning for the distributions screen. Every derived number is computed here, in the browser.

export interface Bin {
  lo: number;
  hi: number;
  count: number;
}

/** Fixed bins of width `step` from `lo` to `hi`; values outside go into the end bins. */
export function histogram(values: number[], lo: number, hi: number, step: number): Bin[] {
  const n = Math.max(1, Math.round((hi - lo) / step));
  const bins: Bin[] = Array.from({ length: n }, (_, i) => ({ lo: lo + i * step, hi: lo + (i + 1) * step, count: 0 }));
  for (const v of values) {
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    const i = Math.min(n - 1, Math.max(0, Math.floor((v - lo) / step + 1e-9)));
    bins[i].count += 1;
  }
  return bins;
}

/** Integer bins: one per value from lo to hi inclusive. */
export function countBins(values: number[], lo: number, hi: number): Bin[] {
  return histogram(values.map((v) => v + 0.5), lo, hi + 1, 1).map((b) => ({ ...b, hi: b.lo }));
}

export const share = (n: number, d: number) => (d ? n / d : null);
