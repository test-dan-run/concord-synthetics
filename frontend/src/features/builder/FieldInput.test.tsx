import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FIELDS, overridesOf } from "../../lib/draft";
import { FieldInput } from "./FieldInput";

const race = FIELDS.find((f) => f.path === "speaker_pool.ethnicity_mix")!;

afterEach(cleanup);

describe("race mix", () => {
  it("starts as in the pool, and switching that off sets weights that start from the pool's shares", () => {
    const onChange = vi.fn();
    render(<FieldInput field={race} value={null} defaultValue={null} onChange={onChange} />);
    const pool = screen.getByRole("checkbox", { name: /As in the pool/ });
    expect((pool as HTMLInputElement).checked).toBe(true);
    expect(screen.queryByLabelText("Malay share")).toBeNull();

    fireEvent.click(pool);
    expect(onChange).toHaveBeenCalledWith({ dist: "categorical", weights: race.unset!.start });
  });

  it("shows each race by name with its share, and switching back to the pool sends null", () => {
    const onChange = vi.fn();
    const value = { dist: "categorical", weights: { CHINESE: 1, MALAY: 2, INDIAN: 1, OTHERS: 0 } };
    render(<FieldInput field={race} value={value} defaultValue={null} onChange={onChange} />);
    expect(screen.getByLabelText("Malay share").textContent).toBe("50%");
    expect(screen.getByLabelText("Others share").textContent).toBe("0%");

    fireEvent.click(screen.getByRole("checkbox", { name: /As in the pool/ }));
    expect(onChange).toHaveBeenCalledWith(null);
  });

  it("is sent only when it differs from the default", () => {
    const defaults = { speaker_pool: { ethnicity_mix: null } };
    expect(overridesOf({ [race.path]: null }, defaults)).toEqual({});
    const mix = { dist: "categorical", weights: { MALAY: 1 } };
    expect(overridesOf({ [race.path]: mix }, defaults)).toEqual({ [race.path]: mix });
  });
});
