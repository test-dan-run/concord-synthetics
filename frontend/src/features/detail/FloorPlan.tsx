// Top-down floor plan at 90 px per metre (UI doc, "Floor plan"). The front wall is always on the left; the plan
// keeps its scale and scrolls inside its panel. Seats are buttons that select a speaker.

import type { KeyboardEvent } from "react";
import { MicMark, PlateLabel, TableOutline, wallStyle } from "../../components/plan";
import type { MeetingSpec, Wall } from "../../lib/api";
import { C } from "../../lib/colors";
import { metres, micName } from "../../lib/format";
import { projector, seatRadius, speakerNumber } from "../../lib/geometry";
import type { PersonRow } from "../../lib/people";

export const PX_PER_M = 90;
const PAD = 60;

export interface Layers {
  facing: boolean;
  paths: boolean;
  grid: boolean;
}

const WALL_NAME: Record<Wall, string> = { N: "N", E: "E", S: "S", W: "W" };
const FLOOR = { carpet: "carpet", vinyl: "vinyl" };
const CEILING = { acoustic_tile: "acoustic tile", plasterboard: "plasterboard" };

export function FloorPlan({ m, rows, selected, onSelect, layers }: { m: MeetingSpec; rows: PersonRow[]; selected: string; onSelect: (id: string) => void; layers: Layers }) {
  const S = PX_PER_M;
  const { length: L, width: W } = m.room;
  const pr = projector(W, S, PAD);
  const X = pr.x;
  const Y = pr.y;
  const width = L * S + 2 * PAD;
  const height = W * S + 2 * PAD;
  const byId = Object.fromEntries(rows.map((r) => [r.id, r]));
  const sel = byId[selected];
  const mic = m.mic.position;
  const mx = X(mic[0]);
  const my = Y(mic[1]);

  // wall segments in drawing coordinates
  const walls: Record<Wall, [number, number, number, number]> = {
    N: [X(0), Y(W), X(L), Y(W)],
    S: [X(0), Y(0), X(L), Y(0)],
    W: [X(0), Y(0), X(0), Y(W)],
    E: [X(L), Y(0), X(L), Y(W)],
  };
  const free = m.table.slots.filter((s) => !s.occupied_by);
  const returns: Record<string, string> = {};
  for (const p of m.people)
    if (p.presenter?.return_seat?.kind === "table") returns[`t${p.presenter.return_seat.index}`] = p.speaker_id;
    else if (p.presenter?.return_seat?.kind === "wall") returns[`w${p.presenter.return_seat.index}`] = p.speaker_id;

  // wall chair strips: one per wall that has chairs
  const strips = (["N", "S", "E"] as Wall[]).flatMap((w) => {
    const chairs = m.wall_chairs.filter((c) => c.wall === w);
    if (!chairs.length) return [];
    const along = chairs.map((c) => (w === "E" ? c.xy[1] : c.xy[0]));
    const lo = Math.min(...along) - 0.3;
    const hi = Math.max(...along) + 0.3;
    const across = w === "E" ? chairs[0].xy[0] : chairs[0].xy[1];
    return [{ w, lo, hi, across }];
  });

  const key = (id: string) => (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect(id);
    }
  };
  const seated = rows.filter((r) => r.person.mouth);

  return (
    <svg width={width} height={height} role="group" aria-label={`Floor plan of ${m.meeting_id}, ${L} by ${W} metres`} className="block">
      <rect x={X(0)} y={Y(W)} width={L * S} height={W * S} fill={C.canvas} />
      {layers.grid && (
        <g stroke={C.track} strokeWidth={1}>
          {Array.from({ length: Math.ceil(L) - 1 }, (_, i) => (
            <line key={`x${i}`} x1={X(i + 1)} y1={Y(W)} x2={X(i + 1)} y2={Y(0)} />
          ))}
          {Array.from({ length: Math.ceil(W) - 1 }, (_, i) => (
            <line key={`y${i}`} x1={X(0)} y1={Y(i + 1)} x2={X(L)} y2={Y(i + 1)} />
          ))}
        </g>
      )}

      {/* walls, by material, labelled with their side */}
      {(Object.keys(walls) as Wall[]).map((w) => {
        const [x1, y1, x2, y2] = walls[w];
        return <line key={w} x1={x1} y1={y1} x2={x2} y2={y2} strokeLinecap="square" {...wallStyle(m.room.walls[w])} />;
      })}
      <text x={X(0)} y={Y(W) - 26} fontSize={11} fill={C.muted}>{`N wall · ${m.room.walls.N}`}</text>
      <text x={X(0)} y={Y(0) + 40} fontSize={11} fill={C.muted}>{`S wall · ${m.room.walls.S}`}</text>
      <text transform={`translate(${X(0) - 34} ${Y(W / 2)}) rotate(-90)`} textAnchor="middle" fontSize={11} fill={C.muted}>
        {`${WALL_NAME[m.room.front_wall]} (front) wall · ${m.room.walls.W}`}
      </text>
      <text transform={`translate(${X(L) + 30} ${Y(W / 2)}) rotate(90)`} textAnchor="middle" fontSize={11} fill={C.muted}>
        {`E (back) wall · ${m.room.walls.E}`}
      </text>
      {/* dimensions and room facts around the outline */}
      <text x={X(L / 2)} y={Y(W) - 26} textAnchor="middle" fontSize={11} fill={C.fg} className="num">{metres(L)}</text>
      <text transform={`translate(${X(0) - 18} ${Y(W / 2)}) rotate(-90)`} textAnchor="middle" fontSize={11} fill={C.fg} className="num">{metres(W)}</text>
      <text x={X(L)} y={Y(W) - 26} textAnchor="end" fontSize={11} fill={C.muted} className="num">{`RT60 target ${m.room.rt60_target.toFixed(2)} s`}</text>
      <text x={X(L)} y={Y(0) + 40} textAnchor="end" fontSize={11} fill={C.muted}>
        {`Floor ${FLOOR[m.room.floor]} · ceiling ${CEILING[m.room.ceiling]} · height ${metres(m.room.height)}`}
      </text>

      {/* screen on the front wall */}
      {m.room.screen && (
        <g>
          <rect x={X(0) - 2} y={Y(m.room.screen.center + m.room.screen.width / 2)} width={5} height={m.room.screen.width * S} fill={C.focus} />
          <text x={X(0) + 10} y={Y(m.room.screen.center + m.room.screen.width / 2) - 6} fontSize={11} fill={C.muted}>Screen</text>
        </g>
      )}

      {/* wall chair strips and empty wall chairs */}
      {strips.map(({ w, lo, hi, across }) =>
        w === "E" ? (
          <rect key={w} x={X(across) - 0.2 * S} y={Y(hi)} width={0.4 * S} height={(hi - lo) * S} rx={4} fill={C.track} />
        ) : (
          <rect key={w} x={X(lo)} y={Y(across) - 0.2 * S} width={(hi - lo) * S} height={0.4 * S} rx={4} fill={C.track} />
        ),
      )}
      {m.wall_chairs.filter((c) => !c.occupied_by).map((c) => (
        <circle key={`wc${c.index}`} cx={X(c.xy[0])} cy={Y(c.xy[1])} r={7} fill="none" stroke={C.subtle} strokeDasharray="2 3" />
      ))}

      {/* table, free seats, return seats */}
      <TableOutline shape={m.table.shape} cx={X(m.table.center[0])} cy={Y(m.table.center[1])} w={m.table.length * S} h={m.table.width * S} />
      {free.map((s) => (
        <g key={`free${s.index}`}>
          <circle cx={X(s.seat_xy[0])} cy={Y(s.seat_xy[1])} r={11} fill="none" stroke={C.subtle} strokeDasharray="3 3" />
          {returns[`t${s.index}`] && <PlateLabel x={X(s.seat_xy[0])} y={Y(s.seat_xy[1]) - 18}>{`${returns[`t${s.index}`]} returns here`}</PlateLabel>}
        </g>
      ))}

      {/* presenter drift areas */}
      {m.people.filter((p) => p.presenter).map((p) => (
        <g key={`drift${p.speaker_id}`}>
          <circle cx={X(p.presenter!.spot[0])} cy={Y(p.presenter!.spot[1])} r={p.presenter!.drift_radius * S} fill="none" stroke={C.subtle} strokeDasharray="4 4" />
          <text x={X(p.presenter!.spot[0])} y={Y(p.presenter!.spot[1]) + p.presenter!.drift_radius * S + 14} textAnchor="middle" fontSize={10} fill={C.muted}>
            {`Presenter drift · within ${metres(p.presenter!.drift_radius, 1)}`}
          </text>
        </g>
      ))}

      {/* mic paths: all (layer), then the selected one on top */}
      {layers.paths &&
        seated.map((r) => (
          <line key={`path${r.id}`} x1={X(r.person.mouth![0])} y1={Y(r.person.mouth![1])} x2={mx} y2={my} stroke={C.line} strokeWidth={1} />
        ))}

      {/* speakerphone */}
      {m.speakerphone && (
        <g>
          <rect
            x={X(m.speakerphone.position[0]) - 9}
            y={Y(m.speakerphone.position[1]) - 6}
            width={18}
            height={12}
            rx={3}
            fill={C.track}
            stroke={sel?.person.position_type === "remote" ? C.focus : C.muted}
            strokeWidth={sel?.person.position_type === "remote" ? 2 : 1}
          />
          {/* on the side away from the mic, so the two labels never meet */}
          <PlateLabel
            x={X(m.speakerphone.position[0]) + (m.speakerphone.position[0] < mic[0] ? -12 : 12)}
            y={Y(m.speakerphone.position[1]) - 14}
            anchor={m.speakerphone.position[0] < mic[0] ? "end" : "start"}
          >
            {`Speakerphone · ${m.speakerphone.remote_speakers.join(", ")}`}
          </PlateLabel>
        </g>
      )}

      <MicMark x={mx} y={my} />
      <PlateLabel x={mx} y={my + 24} color={C.fg}>
        {`MIC · ${micName(m.mic)} · z ${metres(m.mic.position[2])}`}
      </PlateLabel>

      {/* selected speaker's mouth-to-mic path, always shown */}
      {sel?.person.mouth && (
        <g>
          <line x1={X(sel.person.mouth[0])} y1={Y(sel.person.mouth[1])} x2={mx} y2={my} stroke={C.focus} strokeWidth={1.5} strokeDasharray="5 4" />
          {(() => {
            // beside the path, not on it, so a short path keeps its seat and the mic visible
            const sx = X(sel.person.mouth[0]);
            const sy = Y(sel.person.mouth[1]);
            const len = Math.hypot(mx - sx, my - sy) || 1;
            const off = 16;
            return (
              <PlateLabel x={(sx + mx) / 2 - ((my - sy) / len) * off} y={(sy + my) / 2 + ((mx - sx) / len) * off} color={C.focus}>
                {metres(sel.micDistance)}
              </PlateLabel>
            );
          })()}
        </g>
      )}

      {/* seats: buttons, size by turn share, chair heavier, selected filled */}
      {seated.map((r) => {
        const p = r.person;
        const cx = X(p.mouth![0]);
        const cy = Y(p.mouth![1]);
        const rad = seatRadius(p.turn_share);
        const on = r.id === selected;
        const [dx, dy] = pr.dir(p.facing_deg ?? 0);
        const label = `Select ${r.id}, ${p.persona.role}, ${r.position}`;
        return (
          <g
            key={r.id}
            role="button"
            tabIndex={0}
            aria-pressed={on}
            aria-label={label}
            onClick={() => onSelect(r.id)}
            onKeyDown={key(r.id)}
            className="cursor-pointer outline-none [&:focus-visible>.ring]:stroke-focus"
          >
            <title>{`${r.id} · ${p.persona.role} · ${r.position}`}</title>
            {layers.facing && p.facing_deg !== null && (
              <line x1={cx + dx * rad} y1={cy + dy * rad} x2={cx + dx * (rad + 11)} y2={cy + dy * (rad + 11)} stroke={C.subtle} strokeWidth={2} strokeLinecap="round" />
            )}
            {p.position_type === "presenter" ? (
              <rect x={cx - rad} y={cy - rad} width={rad * 2} height={rad * 2} rx={5} fill={on ? C.focus : C.panel} stroke={C.fg} strokeWidth={p.is_chair ? 3 : 1.5} />
            ) : (
              <circle cx={cx} cy={cy} r={rad} fill={on ? C.focus : C.panel} stroke={C.fg} strokeWidth={p.is_chair ? 3 : 1.5} />
            )}
            <circle className="ring" cx={cx} cy={cy} r={rad + 4} fill="none" stroke="transparent" strokeWidth={2} />
            <text x={cx} y={cy + 4} textAnchor="middle" fontSize={11} fontWeight={600} fill={on ? C.canvas : C.fg} className="num pointer-events-none">
              {speakerNumber(r.id)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
