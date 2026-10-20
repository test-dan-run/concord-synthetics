import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import large from "../../test/meeting-large.json";
import type { MeetingSpec } from "../../lib/api";
import { peopleRows } from "../../lib/people";
import { excerpt } from "./PersonaPanel";
import { FloorPlan } from "./FloorPlan";

const M = large as unknown as MeetingSpec;

describe("floor plan", () => {
  it("draws every local talker as a button and selects by click or keyboard", () => {
    const onSelect = vi.fn();
    const rows = peopleRows(M);
    render(<FloorPlan m={M} rows={rows} selected="S02" onSelect={onSelect} layers={{ facing: true, paths: true, grid: true }} />);
    const seats = screen.getAllByRole("button");
    expect(seats).toHaveLength(M.people.filter((p) => p.mouth).length);
    const s02 = seats.find((b) => b.getAttribute("aria-label")!.startsWith("Select S02,"))!;
    expect(s02.getAttribute("aria-pressed")).toBe("true");
    const other = seats.find((b) => b.getAttribute("aria-pressed") === "false")!;
    fireEvent.click(other);
    fireEvent.keyDown(other, { key: "Enter" });
    expect(onSelect).toHaveBeenCalledTimes(2);
  });
  it("labels the walls by side and material", () => {
    const { container } = render(<FloorPlan m={M} rows={peopleRows(M)} selected="S01" onSelect={() => {}} layers={{ facing: false, paths: false, grid: false }} />);
    expect(container.textContent).toContain(`N wall · ${M.room.walls.N}`);
    expect(container.textContent).toContain("(front) wall");
  });
});

describe("persona excerpt", () => {
  it("is five turns around the persona's longest main turn", () => {
    const turns = Array.from({ length: 12 }, (_, i) => ({
      turn_id: `t${i}`, speaker_id: i % 3 === 0 ? "S02" : "S01", interaction: "normal" as const, text: i === 6 ? "a much longer turn than the others" : "short", text_norm: "",
    }));
    const script = { meeting_id: "m", brief: "", turns, generator: { service: "", template_version: "", round: 1, thinking: false, temperature: 0 } };
    expect(excerpt(script, "S02")).toEqual([4, 5, 6, 7, 8]);
    expect(excerpt(script, "S09")).toEqual([]);
  });
});
