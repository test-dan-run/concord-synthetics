// A browser card's to-scale floor plan (UI doc, "Meeting browser"): room, table, seats in the room-class colour
// (table seats hollow, wall seats solid), presenters as hollow squares, the chair filled white, the mic as a
// diamond and the screen on the front wall. Decorative: the card's text carries the same facts.

import { MicMark, TableOutline } from "../../components/plan";
import type { Plan } from "../../lib/api";
import { C, CLASS_COLOR } from "../../lib/colors";
import { projector } from "../../lib/geometry";

export const MINI_W = 330;
export const MINI_H = 184;

export function MiniPlan({ plan, width = MINI_W, height = MINI_H }: { plan: Plan; width?: number; height?: number }) {
  const { length: L, width: W } = plan.room;
  const pad = 10;
  const S = Math.min((width - 2 * pad) / L, (height - 2 * pad) / W);
  const ox = (width - L * S) / 2 - pad;
  const oy = (height - W * S) / 2 - pad;
  const pr = projector(W, S, pad);
  const X = (x: number) => ox + pr.x(x);
  const Y = (y: number) => oy + pr.y(y);
  const color = CLASS_COLOR[plan.room_class];
  const t = plan.table;
  return (
    <svg width={width} height={height} aria-hidden className="block max-w-full h-auto" viewBox={`0 0 ${width} ${height}`}>
      <rect x={X(0)} y={Y(W)} width={L * S} height={W * S} fill={C.canvas} stroke={C.subtle} strokeWidth={1.5} />
      <g stroke={C.track} strokeWidth={1}>
        {Array.from({ length: Math.ceil(L) - 1 }, (_, i) => (
          <line key={`x${i}`} x1={X(i + 1)} y1={Y(W) + 1} x2={X(i + 1)} y2={Y(0) - 1} />
        ))}
        {Array.from({ length: Math.ceil(W) - 1 }, (_, i) => (
          <line key={`y${i}`} x1={X(0) + 1} y1={Y(i + 1)} x2={X(L) - 1} y2={Y(i + 1)} />
        ))}
      </g>
      {plan.presentation && plan.room.screen && (
        <rect x={X(0) - 1.5} y={Y(plan.room.screen.center + plan.room.screen.width / 2)} width={3} height={plan.room.screen.width * S} fill={C.focus} />
      )}
      <TableOutline shape={t.shape} cx={X(t.center[0])} cy={Y(t.center[1])} w={t.length * S} h={t.width * S} />
      {plan.people
        .filter((p) => p.mouth)
        .map((p) => {
          const cx = X(p.mouth![0]);
          const cy = Y(p.mouth![1]);
          if (p.position_type === "presenter") return <rect key={p.speaker_id} x={cx - 4} y={cy - 4} width={8} height={8} fill="none" stroke={color} strokeWidth={1.5} />;
          if (p.is_chair) return <circle key={p.speaker_id} cx={cx} cy={cy} r={4.5} fill={C.fg} />;
          if (p.position_type === "wall") return <circle key={p.speaker_id} cx={cx} cy={cy} r={4} fill={color} />;
          return <circle key={p.speaker_id} cx={cx} cy={cy} r={3.8} fill="none" stroke={color} strokeWidth={1.5} />;
        })}
      <MicMark x={X(plan.mic.position[0])} y={Y(plan.mic.position[1])} size={4.5} />
    </svg>
  );
}
