import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FIELDS, RUN_INFO } from "../../lib/draft";
import { FieldInput } from "./FieldInput";

describe("setting explanations", () => {
  it("every form setting says what it is and why it matters", () => {
    for (const info of [...FIELDS.map((f) => f.info), ...Object.values(RUN_INFO)]) {
      expect(info.what.length).toBeGreaterThan(30);
      expect(info.why.length).toBeGreaterThan(30);
    }
  });

  it("shows on hover, focus and click, hides on Escape, and stays out of the input's label", () => {
    const field = FIELDS.find((f) => f.path === "acoustics.rt60.large")!;
    const value = { dist: "uniform", lo: 0.4, hi: 0.9 };
    render(<FieldInput field={field} value={value} defaultValue={value} onChange={() => {}} />);
    const button = screen.getByRole("button", { name: "About RT60 · large" });
    const tip = screen.getByRole("tooltip", { hidden: true });
    expect(button.getAttribute("aria-describedby")).toBe(tip.id);
    expect(tip.textContent).toContain("Reverberation time");
    expect(tip.textContent).toContain("Why it matters");
    expect(tip.className).toContain("hidden");

    fireEvent.mouseEnter(button.parentElement!);
    expect(tip.className).not.toContain("hidden");
    fireEvent.mouseLeave(button.parentElement!);
    expect(tip.className).toContain("hidden");

    fireEvent.focus(button);
    expect(button.getAttribute("aria-expanded")).toBe("true");
    fireEvent.keyDown(button, { key: "Escape" });
    expect(tip.className).toContain("hidden");

    fireEvent.click(button);
    expect(tip.className).not.toContain("hidden");
    fireEvent.click(button);
    expect(tip.className).toContain("hidden");

    // the range's first input keeps its own name; the info button is not inside its <label>
    expect(screen.getByLabelText("RT60 · large").tagName).toBe("INPUT");
  });
});
