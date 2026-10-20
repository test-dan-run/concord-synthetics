// SVG pieces shared by the floor plan, the side elevation and the browser's mini plans (UI doc, "Plan drawing
// conventions"). Coordinates are already projected to pixels.

import { C } from "../lib/colors";

/** Glass walls are dashed `muted`, plasterboard solid `subtle`: pattern, not only tone, tells them apart. */
export function wallStyle(material: "glass" | "plasterboard", width = 2.5) {
  return material === "glass"
    ? { stroke: C.muted, strokeWidth: width, strokeDasharray: "7 5" }
    : { stroke: C.subtle, strokeWidth: width };
}

/** The table's true shape: rect, round (circle) or oval (a stadium: straight long sides, round ends). */
export function TableOutline({ shape, cx, cy, w, h }: { shape: "rect" | "round" | "oval"; cx: number; cy: number; w: number; h: number }) {
  const common = { fill: C.active, stroke: C.line, strokeWidth: 1 };
  if (shape === "round") return <ellipse cx={cx} cy={cy} rx={w / 2} ry={h / 2} {...common} />;
  const r = shape === "oval" ? Math.min(w, h) / 2 : 3;
  return <rect x={cx - w / 2} y={cy - h / 2} width={w} height={h} rx={r} ry={r} {...common} />;
}

export function MicMark({ x, y, size = 7, color = C.fg }: { x: number; y: number; size?: number; color?: string }) {
  return <path d={`M${x} ${y - size}L${x + size} ${y}L${x} ${y + size}L${x - size} ${y}Z`} fill={color} stroke={C.canvas} strokeWidth={1} />;
}

/** A label on a dark plate, so it stays readable over grid lines and paths. */
export function PlateLabel({ x, y, children, color = C.muted, anchor = "middle", size = 11 }: { x: number; y: number; children: string; color?: string; anchor?: "start" | "middle" | "end"; size?: number }) {
  const w = children.length * size * 0.62 + 8;
  const left = anchor === "start" ? x - 4 : anchor === "end" ? x - w + 4 : x - w / 2;
  return (
    <g>
      <rect x={left} y={y - size + 1} width={w} height={size + 5} rx={2} fill={C.canvas} opacity={0.85} />
      <text x={x} y={y + 1} textAnchor={anchor} fontSize={size} fill={color} className="num">
        {children}
      </text>
    </g>
  );
}
