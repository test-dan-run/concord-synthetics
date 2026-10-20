// The detail screen's right column: the selected-speaker card and seven fact panels from meeting.json (UI doc,
// "Selected speaker and config").

import type { ReactNode } from "react";
import { Fact, Panel } from "../../components/ui";
import type { AcousticRender, MeetingSpec, RunConfig, Timeline, Wall } from "../../lib/api";
import { DASH, db, hms, metres, num, pct, signedDeg } from "../../lib/format";
import { SV_MODELS, placeOf, type PersonRow, wallChairLabel } from "../../lib/people";

const WALL_SIDE: Record<Wall, string> = { N: "N (side)", S: "S (side)", E: "E (back)", W: "W (front)" };
const join = (xs: string[]) => (xs.length ? xs.join(" · ") : DASH);

export function SpeakerCard({ row }: { row: PersonRow | undefined }) {
  if (!row) return null;
  const p = row.person;
  const facing =
    p.position_type === "remote" ? "speakerphone" : row.facing === null ? DASH : `${signedDeg(row.facing)} vs table centre`;
  return (
    <Panel
      id="speaker-card"
      title={
        <span className="flex items-center gap-2.5">
          <span className="inline-flex items-center justify-center size-7 rounded-full bg-focus text-canvas text-[11px] font-semibold num">{row.id.slice(1)}</span>
          {row.id} · {p.persona.role}
        </span>
      }
      subtitle={`${p.persona.display_name} · ${row.position}${p.is_chair ? " · chairs the meeting" : ""}`}
    >
      <div>
        <Fact k="Position" v={row.position} />
        <Fact k="L1 · code-switching" v={`${p.persona.l1} · ${p.persona.code_switching}`} />
        <Fact k="Turn share" v={`${pct(p.turn_share)}${row.utterances !== null ? ` · ${num(row.utterances)} utts` : ""}`} />
        <Fact k="Mouth height" v={p.mouth ? metres(p.mouth[2]) : "remote"} />
        <Fact k="Mic distance, 3D" v={row.micDistance === null ? "via speakerphone" : metres(row.micDistance)} />
        <Fact k="Facing" v={facing} />
        {p.presenter && <Fact k="Faces the screen" v={`on ${pct(p.presenter.facing_screen_p, 0)} of turns`} />}
        <Fact k="NSC reference" v={`P3 · spk ${row.nsc}${row.swapped ? " (swapped in stage 4)" : ""}`} />
        <Fact k={`SV · ${SV_MODELS.map((s) => s.label).join(" / ")}`} v={SV_MODELS.map((s) => (row.sv[s.key] === null ? DASH : row.sv[s.key]!.toFixed(2))).join(" / ")} />
        <Fact k="ASR-limited utterances" v={row.asrLimited === null ? DASH : row.asrLimited === 0 ? "0 (fallback off)" : num(row.asrLimited)} />
      </div>
    </Panel>
  );
}

function FactPanel({ title, id, children }: { title: string; id: string; children: ReactNode }) {
  return (
    <Panel title={title} id={id}>
      <div>{children}</div>
    </Panel>
  );
}

export function FactPanels({ m, config, render, timeline }: { m: MeetingSpec; config?: RunConfig; render?: AcousticRender | null; timeline?: Timeline | null }) {
  const r = m.room;
  const range = config?.room_classes.classes[m.room_class]?.speakers;
  const t = m.table;
  const used = t.slots.filter((s) => s.occupied_by).length;
  const sides: Record<string, number> = {};
  for (const s of t.slots) sides[s.side] = (sides[s.side] ?? 0) + 1;
  const layout = Object.entries(sides).map(([k, v]) => (k === "arc" ? `${v} around` : `${k} ${v}`)).join(" · ");

  const wall = m.people.filter((p) => p.position_type === "wall");
  const wallChairs = m.wall_chairs.filter((c) => c.occupied_by);
  const wallSides = [...new Set(wallChairs.map((c) => WALL_SIDE[c.wall]))];
  const presenters = m.people.filter((p) => p.position_type === "presenter");
  const pres = m.dynamics.presentation;
  const presSegs = timeline?.segments.filter((s) => s.kind === "presentation") ?? [];
  const presTimes = presSegs.length ? `${hms(Math.min(...presSegs.map((s) => s.start)))}–${hms(Math.max(...presSegs.map((s) => s.end)))}, then Q&A` : null;
  const back = (sid: string) => {
    const ref = pres?.returns[sid];
    if (!ref) return "stays standing";
    return ref.kind === "table" ? `returns to table seat ${t.slots.find((s) => s.index === ref.index)?.label ?? ref.index}` : `returns to wall seat ${wallChairLabel(m, ref.index)}`;
  };

  const mic = m.mic;
  const byMat: Record<string, Wall[]> = {};
  for (const [w, mat] of Object.entries(r.walls) as [Wall, string][]) (byMat[mat] ??= []).push(w);
  const ac = m.acoustics;
  const bg = ac.noise_sources.filter((n) => n.kind !== "keyboard").map((n) => (n.kind === "hvac" ? "HVAC" : n.kind.replace("_", " ")));
  const events = Object.entries(ac.events).filter(([, n]) => n > 0).map(([k, n]) => `${k} ${n}`);
  const dc = m.device_chain;
  const near = mic.near_speaker ? m.people.find((p) => p.speaker_id === mic.near_speaker) : undefined;

  return (
    <>
      <FactPanel title="Room" id="facts-room">
        <Fact k="Class" v={`${m.room_class}${range?.lo !== undefined ? ` · ${range.lo}–${range.hi}` : ""}`} />
        <Fact k="L × W × H" v={`${r.length.toFixed(2)} × ${r.width.toFixed(2)} × ${r.height.toFixed(2)} m`} />
        <Fact k="Volume" v={`${num(r.length * r.width * r.height, 1)} m³`} />
        <Fact k="Front wall" v={`${r.front_wall}${r.screen ? ` · screen ${metres(r.screen.width, 1)}` : " · no screen"}`} />
        <Fact k="RT60 target" v={`${r.rt60_target.toFixed(2)} s`} />
        <Fact k="RT60 measured" v={render ? `${render.rt60_measured.toFixed(2)} s` : "not rendered yet"} />
        <Fact k="Room resamples" v={num(m.fit_resamples)} />
      </FactPanel>
      <FactPanel title="Table seats" id="facts-table">
        <Fact k="Table" v={`${t.shape} ${t.length.toFixed(2)} × ${t.width.toFixed(2)} m`} />
        <Fact k="Seats used" v={`${used} of ${t.capacity}`} />
        <Fact k="Edge per person" v={metres(t.seat_pitch)} />
        <Fact k="Layout" v={layout} />
        <Fact k="Table height" v={metres(t.height)} />
      </FactPanel>
      <FactPanel title="Off-table positions" id="facts-off">
        <Fact k="Wall seats" v={wall.length ? `${wall.length} · ${wallSides.join(", ")} wall, ${metres(m.wall_chairs.find((c) => c.occupied_by)?.offset)} out` : "none"} />
        {wall.length > 0 && (
          <Fact k="Why at the wall" v={wall[0].wall_reason === "by_choice" ? "by choice · table not full" : "overflow · table full"} />
        )}
        <Fact k="Presenters" v={presenters.length ? presenters.map((p) => `${p.speaker_id} · ${metres(p.presenter!.spot[0])} from front wall`).join("; ") : "none"} />
        {presenters.map((p) => (
          <Fact key={p.speaker_id} k={`${p.speaker_id} mouth · drift`} v={`${metres(p.mouth?.[2])} · within ${metres(p.presenter!.drift_radius, 1)}`} />
        ))}
        {pres && <Fact k="Presentation" v={presTimes ?? `turns ${pres.start_turn}–${pres.end_turn}`} />}
        {presenters.map((p) => (
          <Fact key={`b${p.speaker_id}`} k={`After presenting, ${p.speaker_id}`} v={back(p.speaker_id)} />
        ))}
        <Fact k="Remote" v={m.dynamics.remote_speakers.length ? `${m.dynamics.remote_speakers.join(", ")} via speakerphone` : "none"} />
      </FactPanel>
      <FactPanel title="Microphone" id="facts-mic">
        <Fact k="Type" v={mic.type === "laptop" ? "laptop" : mic.channels > 1 ? `centre array, ${mic.channels}-ch` : "centre, single"} />
        <Fact k="Position" v={mic.position.map((v) => v.toFixed(2)).join(", ")} />
        {mic.type === "centre" ? (
          <Fact k="Offset from centre" v={`${(mic.position[0] - t.center[0]).toFixed(2)}, ${(mic.position[1] - t.center[1]).toFixed(2)} m`} />
        ) : (
          <Fact k="In front of" v={near ? `${near.speaker_id} · ${placeOf(m, near)}` : DASH} />
        )}
        <Fact k="Above the table" v={metres(mic.position[2] - t.height)} />
        <Fact k="Output" v={`${mic.channels > 1 ? `${mic.channels}-ch + mono` : "mono"} · ${num((render?.sample_rate ?? config?.sample_rate ?? 16000) / 1000)} kHz`} />
      </FactPanel>
      <FactPanel title="Surfaces" id="facts-surfaces">
        {Object.entries(byMat).map(([mat, ws]) => (
          <Fact key={mat} k={`${ws.join(" and ")} wall${ws.length > 1 ? "s" : ""}`} v={mat} />
        ))}
        <Fact k="Floor" v={r.floor} />
        <Fact k="Ceiling" v={r.ceiling.replace("_", " ")} />
      </FactPanel>
      <FactPanel title="Noise" id="facts-noise">
        <Fact k="SNR at mic, sampled" v={db(ac.snr_db, 1)} />
        <Fact k="SNR at mic, realised" v={render ? db(render.snr_db, 1) : "not rendered yet"} />
        <Fact k="Background" v={join(bg)} />
        <Fact k="Events" v={join(events)} />
        <Fact k="Keyboard" v={ac.noise_sources.some((n) => n.kind === "keyboard") ? "on" : "off"} />
      </FactPanel>
      <FactPanel title="Device chain" id="facts-device">
        <Fact k="AGC" v={dc.agc ? "on" : "off"} />
        <Fact k="Band limit" v={dc.band_limit_hz ? `${(dc.band_limit_hz / 1000).toFixed(1)} kHz` : "off"} />
        <Fact k="Codec" v={dc.codec_kbps ? `Opus ${dc.codec_kbps} kbps` : "off"} />
        <Fact k="Clipping" v={dc.clipping_level ? `at ${pct(dc.clipping_level, 0)} of full scale` : "off"} />
        <Fact k="AEC artifacts" v={dc.aec_artifacts ? "on · hybrid" : "off"} />
      </FactPanel>
    </>
  );
}
