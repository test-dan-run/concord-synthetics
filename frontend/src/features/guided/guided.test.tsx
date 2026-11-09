import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatPanel, replyText } from "../../components/guided";
import type { Draft } from "../../lib/api";
import { changeLabel, contentOverrides, type Estimate, meetingsFor, packBytes, undoChanges } from "../../lib/guided";
import { Describe } from "./Describe";

afterEach(cleanup);

describe("guided helpers", () => {
  it("turns hours into meetings at the mean length", () => {
    expect(meetingsFor(25, 25)).toBe(60);
    expect(meetingsFor(25, null)).toBe(60);
    expect(meetingsFor(1, 120)).toBe(1);
  });

  it("drops blank topics and terms and keeps pronunciations", () => {
    expect(contentOverrides(["  HR rollout ", ""], [{ term: "PDPC", say: "letters" }, { term: " ", say: null }])).toEqual({
      "dynamics.topics": ["HR rollout"],
      key_terms: [{ term: "PDPC", say: "letters" }],
    });
    expect(contentOverrides([""], [])).toEqual({});
  });

  it("undoes a change back to its value before, or to the default", () => {
    const d: Draft = { run_id: "r", n_meetings: 4, seed: 1, yaml: "", overrides: { "acoustics.hybrid.large.p": 0.5, "x.y": 2 } };
    const out = undoChanges(d, [
      { path: "acoustics.hybrid.large.p", before: 0.3, after: 0.5, overlay: false },
      { path: "x.y", before: null, after: 2, overlay: false },
    ]);
    expect(out.overrides).toEqual({ "acoustics.hybrid.large.p": 0.3 });
  });

  it("names changes in plain words", () => {
    expect(changeLabel({ path: "acoustics.hybrid.large.p", before: 0.3, after: 0.5, overlay: false })).toBe("Calling in · large: 30% → 50%");
  });

  it("adds up the parts of a pack", () => {
    const e: Estimate = { meetings: 2, root: "/x", bytes: { mixture: 100, full: 1000, clips_clean: 80, clips_room: 90, noise: 100, labels: 5 } };
    expect(packBytes(e, { clips: true, full: false, clean: true, room: true, noise: true })).toBe(100 + 5 + 80 + 90 + 100);
    expect(packBytes(e, { clips: false, full: true, clean: true, room: true, noise: false })).toBe(105 + 2000);
  });

  it("says when the model is the fake", () => {
    expect(replyText(1, 0, 0, [], true)).toMatch(/word matching/);
    expect(replyText(0, 0, 0, ["out_of_scope"], false)).toMatch(/topics and key terms/);
  });
});

describe("ChatPanel", () => {
  it("sends a message, shows change chips and undoes them", () => {
    const onSend = vi.fn();
    const onUndo = vi.fn();
    render(
      <ChatPanel
        hint="hint"
        busy={false}
        onSend={onSend}
        onUndo={onUndo}
        messages={[
          { from: "user", text: "more hybrid" },
          { from: "system", text: "Done.", changes: [{ path: "acoustics.hybrid.small.p", before: 0.1, after: 0.5, overlay: false }] },
        ]}
      />,
    );
    expect(screen.getByText("Calling in · small: 10% → 50%")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(onUndo).toHaveBeenCalledWith(1);
    fireEvent.change(screen.getByLabelText("Describe a change"), { target: { value: "noisier" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).toHaveBeenCalledWith("noisier");
  });
});

describe("Describe", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.stubGlobal("fetch", vi.fn(async () => new Response("[]", { headers: { "Content-Type": "application/json" } })));
  });
  afterEach(() => vi.unstubAllGlobals());

  const show = () =>
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter>
          <Describe />
        </MemoryRouter>
      </QueryClientProvider>,
    );

  it("adds and removes topics, and adds a key term with how it is said", () => {
    show();
    fireEvent.change(screen.getByLabelText("Topic 1"), { target: { value: "HR rollout" } });
    fireEvent.click(screen.getByRole("button", { name: "+ Add another topic" }));
    fireEvent.change(screen.getByLabelText("Topic 2"), { target: { value: "Budget" } });
    expect(screen.getAllByText(/meetings$/).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Remove topic 1" }));
    expect((screen.getByLabelText("Topic 1") as HTMLInputElement).value).toBe("Budget");
    fireEvent.change(screen.getByLabelText("New key term"), { target: { value: "PDPC" } });
    fireEvent.change(screen.getByLabelText("How is it said?"), { target: { value: "letters" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    expect(screen.getByRole("button", { name: "Remove PDPC" })).toBeTruthy();
    expect(JSON.parse(localStorage.getItem("synthetics.guided.v1") ?? "{}").terms).toEqual([{ term: "PDPC", say: "letters" }]);
  });
});
