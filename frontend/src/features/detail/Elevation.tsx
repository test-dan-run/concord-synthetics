// Side elevation at the plan's scale, looking north (UI doc, "Side elevation"): floor, ceiling, table, mic and every
// mouth height. Presenters are drawn standing, as a line from the floor to the mouth.

import { MicMark, PlateLabel } from "../../components/plan";
import type { MeetingSpec } from "../../lib/api";
import { C } from "../../lib/colors";
import { metres } from "../../lib/format";
import type { PersonRow } from "../../lib/people";
import { PX_PER_M } from "./FloorPlan";

const PAD = 40;

export function Elevation({ m, rows, selected }: { m: MeetingSpec; rows: PersonRow[]; selected: string }) {
  const S = PX_PER_M;
  const { length: L, height: H } = m.room;
  const X = (x: number) => PAD + 20 + x * S;
  const Z = (z: number) => PAD + (H - z) * S;
  const width = L * S + 2 * PAD + 40;
  const height = H * S + 2 * PAD;
  const t = m.table;
  const x0 = X(t.center[0] - t.length / 2);
  const x1 = X(t.center[0] + t.length / 2);
  const mic = m.mic.position;
  const sel = rows.find((r) => r.id === selected);
  const seated = rows.filter((r) => r.person.mouth);
  return (
    <svg width={width} height={height} role="img" aria-label={`Side elevation of ${m.meeting_id}: ceiling ${H} m, table ${t.height} m, mic ${mic[2].toFixed(2)} m`} className="block">
      <rect x={X(0)} y={Z(H)} width={L * S} height={H * S} fill={C.canvas} />
      <line x1={X(0)} y1={Z(0)} x2={X(L)} y2={Z(0)} stroke={C.subtle} strokeWidth={2} />
      <line x1={X(0)} y1={Z(H)} x2={X(L)} y2={Z(H)} stroke={C.subtle} strokeDasharray="6 5" />
      <line x1={X(0)} y1={Z(0)} x2={X(0)} y2={Z(H)} stroke={C.line} />
      <line x1={X(L)} y1={Z(0)} x2={X(L)} y2={Z(H)} stroke={C.line} />
      <text x={X(L / 2)} y={Z(H) - 8} textAnchor="middle" fontSize={11} fill={C.muted} className="num">{`Ceiling · ${metres(H)}`}</text>
      <text x={X(L / 2)} y={Z(0) + 18} textAnchor="middle" fontSize={11} fill={C.muted} className="num">Floor · 0 m</text>
      <text x={X(0) - 8} y={Z(0) + 18} textAnchor="middle" fontSize={10} fill={C.muted}>front</text>
      {m.room.screen && (
        <g>
          <rect x={X(0) - 2} y={Z(m.room.screen.top)} width={5} height={(m.room.screen.top - m.room.screen.bottom) * S} fill={C.focus} />
          <text x={X(0) + 8} y={Z(m.room.screen.top) - 6} fontSize={11} fill={C.muted}>Screen</text>
        </g>
      )}

      {/* table top and legs */}
      <rect x={x0} y={Z(t.height)} width={x1 - x0} height={4} fill={C.active} stroke={C.line} />
      <line x1={x0 + 6} y1={Z(t.height)} x2={x0 + 6} y2={Z(0)} stroke={C.line} strokeWidth={2} />
      <line x1={x1 - 6} y1={Z(t.height)} x2={x1 - 6} y2={Z(0)} stroke={C.line} strokeWidth={2} />
      <text x={x1 + 6} y={Z(t.height) + 12} fontSize={11} fill={C.muted} className="num">{`Table · ${metres(t.height)}`}</text>

      {/* mouths; presenters stand */}
      {seated.map((r) => {
        const [x, , z] = r.person.mouth!;
        const on = r.id === selected;
        const color = on ? C.focus : C.panel;
        if (r.person.position_type === "presenter")
          return (
            <g key={r.id}>
              <line x1={X(x)} y1={Z(0)} x2={X(x)} y2={Z(z)} stroke={on ? C.focus : C.muted} strokeWidth={1.5} />
              <rect x={X(x) - 6} y={Z(z) - 6} width={12} height={12} rx={2} fill={color} stroke={C.fg} />
            </g>
          );
        return <circle key={r.id} cx={X(x)} cy={Z(z)} r={6} fill={color} stroke={on ? C.focus : C.fg} strokeWidth={1.2} />;
      })}

      {sel?.person.mouth && (
        <g>
          <line x1={X(sel.person.mouth[0])} y1={Z(sel.person.mouth[2])} x2={X(mic[0])} y2={Z(mic[2])} stroke={C.focus} strokeWidth={1.5} strokeDasharray="5 4" />
          <PlateLabel x={X(sel.person.mouth[0])} y={Z(sel.person.mouth[2]) - 14} color={C.focus}>
            {`${sel.id} mouth · ${metres(sel.person.mouth[2])}`}
          </PlateLabel>
        </g>
      )}
      <MicMark x={X(mic[0])} y={Z(mic[2])} size={6} />
      <PlateLabel x={X(mic[0])} y={Z(mic[2]) + 22} color={C.fg}>{`MIC ${metres(mic[2])}`}</PlateLabel>
    </svg>
  );
}
