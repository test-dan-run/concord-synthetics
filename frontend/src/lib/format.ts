// Number and time formatting. Every value rendered with these goes inside a `.num` element.

export const DASH = "—";

export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return `${(v * 100).toFixed(digits)}%`;
}

/** Signed percentage-point delta, e.g. "+1.2" / "−0.4". */
export function deltaPts(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  const x = v * 100;
  if (Math.abs(x) < 0.05) return "±0.0";
  return `${x > 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;
}

export function num(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function secs(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined) return DASH;
  return `${v.toFixed(digits)} s`;
}

/** mm:ss.mmm */
export function clock(t: number | null | undefined, ms = true): string {
  if (t === null || t === undefined || Number.isNaN(t)) return DASH;
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  const ss = ms ? s.toFixed(3).padStart(6, "0") : Math.floor(s).toString().padStart(2, "0");
  return `${m.toString().padStart(2, "0")}:${ss}`;
}

export function hours(h: number | null | undefined): string {
  if (h === null || h === undefined) return DASH;
  if (h < 1) return `${(h * 60).toFixed(h * 60 < 10 ? 1 : 0)} min`;
  return `${h.toFixed(1)} h`;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function ago(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const t = new Date(/(Z|[+-]\d\d:?\d\d)$/.test(iso) ? iso : `${iso}Z`).getTime();
  if (Number.isNaN(t)) return DASH;
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

/** Metres: "1.60 m". */
/** A mic in a few words: "laptop", or where a large room's table mic sits and whether it is an array. */
export function micName(mic: { type: string; channels: number }): string {
  if (mic.type === "laptop") return "laptop";
  const place = mic.type === "chair" ? "at the chair" : "centre";
  return mic.channels > 1 ? `${place} · ${mic.channels}-ch array` : `${place} · single`;
}

export function metres(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return `${v.toFixed(digits)} m`;
}

/** Signed degrees: "+50°", "−10°", "±0°". */
export function signedDeg(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  const r = Math.round(v);
  return `${r > 0 ? "+" : r < 0 ? "−" : "±"}${Math.abs(r)}°`;
}

/** hh:mm:ss.mmm for meeting times. */
export function hms(t: number | null | undefined): string {
  if (t === null || t === undefined || Number.isNaN(t)) return DASH;
  const h = Math.floor(t / 3600);
  return `${h.toString().padStart(2, "0")}:${clock(t - h * 3600)}`;
}

export function db(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return `${v.toFixed(digits)} dB`;
}
