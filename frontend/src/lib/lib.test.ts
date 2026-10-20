import { describe, expect, it } from "vitest";
import large from "../test/meeting-large.json";
import small from "../test/meeting-small.json";
import type { MeetingSpec, MeetingSummary, Plan } from "./api";
import { ago, signedDeg } from "./format";
import { dist3, facingVsCentre, projector, wrapDeg } from "./geometry";
import { DEFAULT_FILTERS, matches, readFilters, sortMeetings, visible, writeFilters } from "./meetings";
import { defaultSelection, peopleRows, placeOf } from "./people";
import { countBins, histogram } from "./stats";

const L = large as unknown as MeetingSpec;
const S = small as unknown as MeetingSpec;

function summary(m: MeetingSpec, extra: Partial<Plan> = {}): MeetingSummary {
  const plan = {
    room_class: m.room_class, n_speakers: m.n_speakers, split: m.split, seed: m.seed, fit_resamples: m.fit_resamples, room: m.room,
    table: { ...m.table, slots: m.table.slots.map((s) => ({ xy: s.seat_xy, occupied_by: s.occupied_by })) },
    wall_chairs: m.wall_chairs, people: m.people, mic: m.mic, speakerphone: m.speakerphone, hybrid: m.dynamics.hybrid,
    presentation: !!m.dynamics.presentation, acoustics: m.acoustics, device_chain: m.device_chain, ...extra,
  } as unknown as Plan;
  return { meeting_id: m.meeting_id, state: "done", stages: {}, plan, manifest: null };
}

describe("geometry", () => {
  it("wraps angles and measures in 3D", () => {
    expect(wrapDeg(190)).toBe(-170);
    expect(wrapDeg(-180)).toBe(180);
    expect(dist3([0, 0, 0], [3, 4, 12])).toBe(13);
  });
  it("facing the table centre reads 0, turning left is positive", () => {
    expect(facingVsCentre([0, 0, 1], 45, [1, 1])).toBeCloseTo(0);
    expect(facingVsCentre([0, 0, 1], 90, [1, 1])).toBeCloseTo(45);
  });
  it("puts the front wall left and N at the top", () => {
    const p = projector(5, 90, 10);
    expect(p.x(0)).toBe(10);
    expect(p.y(5)).toBe(10); // the N wall (y = W) is the top edge
    expect(p.y(0)).toBe(460);
    expect(p.dir(90)[1]).toBeCloseTo(-1); // +y in the room is up on screen
  });
});

describe("people", () => {
  it("labels every position and measures every local talker", () => {
    const rows = peopleRows(L);
    expect(rows).toHaveLength(L.n_speakers);
    const wall = rows.filter((r) => r.person.position_type === "wall");
    expect(wall.length).toBe(4);
    expect(wall.every((r) => /^Wall · [NSE]\d$/.test(r.position))).toBe(true);
    expect(rows.filter((r) => r.person.position_type === "remote").every((r) => r.micDistance === null && r.position === "Remote · speakerphone")).toBe(true);
    expect(rows.filter((r) => r.person.mouth).every((r) => r.micDistance! > 0 && r.micDistance! < 15)).toBe(true);
    const tableRow = rows.find((r) => r.person.position_type === "table")!;
    expect(placeOf(L, tableRow.person)).toBe(L.table.slots.find((s) => s.occupied_by === tableRow.id)!.label);
  });
  it("opens on the second-highest turn share", () => {
    const sorted = [...S.people].sort((a, b) => b.turn_share - a.turn_share);
    expect(defaultSelection(S)).toBe(sorted[1].speaker_id);
  });
  it("prefers the voice after a stage 4 swap", () => {
    const id = S.people[0].speaker_id;
    const rows = peopleRows(S, { synthesis: { service: "x", voices: { [id]: { speaker_id: id, nsc_speaker: "99999999", clip_id: "c" } } } });
    expect(rows[0].nsc).toBe("99999999");
    expect(rows[0].swapped).toBe(true);
  });
});

describe("browser filters", () => {
  it("round-trips through the URL and leaves defaults out", () => {
    expect(writeFilters(DEFAULT_FILTERS).toString()).toBe("");
    const f = { ...DEFAULT_FILTERS, room: "large" as const, minSpeakers: 9, wall: true, sort: "rt60" as const };
    expect(readFilters(writeFilters(f))).toEqual(f);
    expect(readFilters(new URLSearchParams("room=huge&min=99")).room).toBe("all");
  });
  it("matches by class, split, size and position types", () => {
    const l = summary(L);
    const s = summary(S);
    expect(matches(l, { ...DEFAULT_FILTERS, wall: true })).toBe(true);
    expect(matches(s, { ...DEFAULT_FILTERS, wall: true })).toBe(false);
    expect(matches(s, { ...DEFAULT_FILTERS, presenters: true })).toBe(true);
    expect(matches(l, { ...DEFAULT_FILTERS, room: "small" })).toBe(false);
    expect(matches(l, { ...DEFAULT_FILTERS, minSpeakers: L.n_speakers + 1 })).toBe(false);
    expect(visible([s, l], { ...DEFAULT_FILTERS, sort: "speakers" }).map((m) => m.meeting_id)).toEqual([L.meeting_id, S.meeting_id]);
  });
  it("sorts by id when keys tie", () => {
    const a = { ...summary(S), meeting_id: "m-000002" };
    const b = { ...summary(S), meeting_id: "m-000001" };
    expect(sortMeetings([a, b], "snr").map((m) => m.meeting_id)).toEqual(["m-000001", "m-000002"]);
  });
});

describe("stats and format", () => {
  it("bins values, clamping the ends", () => {
    const bins = histogram([0.21, 0.24, 0.26, 5], 0.2, 0.4, 0.05);
    expect(bins.map((b) => b.count)).toEqual([2, 1, 0, 1]);
    expect(countBins([2, 2, 7, 20], 2, 20).filter((b) => b.count).map((b) => [b.lo, b.count])).toEqual([[2, 2], [7, 1], [20, 1]]);
  });
  it("reads offsets in timestamps and signs degrees with a true minus", () => {
    expect(ago("2000-01-01T00:00:00+00:00")).toMatch(/d ago$/);
    expect(ago("not a time")).toBe("—");
    expect(signedDeg(-10.4)).toBe("−10°");
    expect(signedDeg(0.2)).toBe("±0°");
  });
});
