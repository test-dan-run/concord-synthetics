// AeroAudio colours used from code (SVG). CSS uses the Tailwind theme tokens instead.

export const C = {
  canvas: "#0A0F1D",
  panel: "#111827",
  track: "#1E293B",
  active: "#283548",
  line: "#334155",
  fg: "#F8FAFC",
  muted: "#94A3B8",
  subtle: "#64748B",
  focus: "#38BDF8",
  primary: "#06B6D4",
  secondary: "#6366F1",
  good: "#10B981",
  warn: "#F59E0B",
  bad: "#F43F5E",
  overlap: "#DC143C",
};

/** Categorical palette (DESIGN_SYSTEM §2), assigned in this order and never cycled. */
export const CAT = ["#3987E5", "#D95926", "#199E70", "#C98500", "#D55181", "#9085E9"];

export type RoomClass = "small" | "large";
/** Room classes own the first two category slots everywhere in the viewer. */
export const CLASS_COLOR: Record<RoomClass, string> = { small: CAT[0], large: CAT[1] };
export const CLASS_TOKEN: Record<RoomClass, string> = { small: "bg-cat-1", large: "bg-cat-2" };
