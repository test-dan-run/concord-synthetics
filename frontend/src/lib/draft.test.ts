import { describe, expect, it } from "vitest";
import { fieldOf, FIELDS, fromRun, getPath, leaves, MODE_PATH, overridesOf, same, suggestRunId, valuesFrom } from "./draft";

const DEFAULTS = {
  revision: 2,
  room_classes: { ratio: { dist: "categorical", weights: { small: 0.3, large: 0.7 } }, classes: { small: { speakers: { dist: "int_uniform", lo: 2, hi: 6 } } } },
  turn_taking: { zipf: { dist: "zipf", exponent: 1.0 }, gap: { dist: "lognormal", median: 0.5, sigma: 0.5, cap: 2.5 }, overlap_share: 0.5 },
  acoustics: { hybrid: { small: { dist: "bernoulli", p: 0.1 }, large: { dist: "bernoulli", p: 0.3 } } },
  stages: { "3": { enabled: true, overrides: {} } },
  simulation: { ism_max_order: 3, ray_tracing: ["large"] },
  backends: { mode: "aicore" },
};

describe("draft", () => {
  it("every curated field is a unique dotted path", () => {
    const paths = FIELDS.map((f) => f.path);
    expect(new Set(paths).size).toBe(paths.length);
    expect(getPath(DEFAULTS, "stages.3.enabled")).toBe(true);
    expect(getPath(DEFAULTS, "nothing.here")).toBeUndefined();
  });

  it("treats a distribution as one leaf", () => {
    const l = leaves(DEFAULTS);
    expect(l["turn_taking.gap"]).toEqual(DEFAULTS.turn_taking.gap);
    expect(l["simulation.ray_tracing"]).toEqual(["large"]);
    expect(l["stages.3.overrides"]).toEqual({});
    expect("turn_taking.gap.median" in l).toBe(false);
  });

  it("sends only the fields that differ from the defaults", () => {
    const v = valuesFrom(DEFAULTS);
    expect(overridesOf(v, DEFAULTS)).toEqual({});
    v["acoustics.hybrid.large"] = { dist: "bernoulli", p: 0.9 };
    v["stages.3.enabled"] = false;
    v[MODE_PATH] = "fake";
    expect(overridesOf(v, DEFAULTS)).toEqual({ "acoustics.hybrid.large": { dist: "bernoulli", p: 0.9 }, "stages.3.enabled": false, "backends.mode": "fake" });
    expect(same({ a: 0.1 + 0.2 }, { a: 0.3 })).toBe(true);
  });

  it("copies a run: curated fields to the form, the other differences to the YAML overlay", () => {
    const run = {
      ...DEFAULTS,
      revision: 1,
      run_id: "old",
      seed: 7,
      n_meetings: 20,
      turn_taking: { ...DEFAULTS.turn_taking, zipf: { dist: "zipf", exponent: 1.4 }, gap: { dist: "lognormal", median: 0.3, sigma: 0.6, cap: 2.0 } },
      simulation: { ism_max_order: 4, ray_tracing: ["large"], retired_field: 1 },
      backends: { mode: "fake" },
    };
    const r = fromRun(run, DEFAULTS);
    expect(r.values["turn_taking.zipf.exponent"]).toBe(1.4);
    expect(r.values[MODE_PATH]).toBe("fake");
    // revision, run fields and the curated zipf stay out of the overlay; a field the defaults lack is dropped
    expect(r.yaml.split("\n").sort()).toEqual(['simulation.ism_max_order: 4', 'turn_taking.gap: {"dist":"lognormal","median":0.3,"sigma":0.6,"cap":2}']);
    expect(r.dropped).toEqual(["simulation.retired_field"]);
  });

  it("maps a backend error to its field", () => {
    expect(fieldOf("room_classes.classes.small.speakers.hi")).toBe("room_classes.classes.small.speakers");
    expect(fieldOf("acoustics.hybrid.large")).toBe("acoustics.hybrid.large");
    expect(fieldOf("simulation.ism_max_order")).toBeNull();
  });

  it("suggests a free run id", () => {
    const d = new Date(2026, 9, 21);
    expect(suggestRunId([], d)).toBe("d-20261021");
    expect(suggestRunId(["d-20261021", "d-20261021-2"], d)).toBe("d-20261021-3");
  });
});
