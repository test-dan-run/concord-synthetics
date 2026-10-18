"""Assign the selected pool to fixed splits: partner pairs together, Others all in test,
every other (ethnicity, gender) cell and age band proportional to split size. Local search, seeded."""
import csv, collections, random, sys
picked = list(csv.DictReader(open(sys.argv[1])))
SPLITS = {"train": 50, "dev": 25, "test": 100}
AGES = ["1X", "2X", "3X", "4X", "5X", "6X"]
ids = {r["speaker"]: r for r in picked}
units, seen = [], set()
for r in sorted(picked, key=lambda r: r["speaker"]):
    if r["speaker"] in seen: continue
    u = tuple([r["speaker"]] + ([r["partner"]] if r["partner"] in ids else []))
    seen.update(u); units.append(u)
eg = lambda sid: ids[sid]["ethnicity"] + ids[sid]["gender"]
cells = collections.Counter(eg(s) for s in ids)
ages = collections.Counter(ids[s]["age"] for s in ids)
fixed = {u: "test" for u in units if any(ids[s]["ethnicity"] == "OTHERS" for s in u)}
n_other = sum(len(u) for u in fixed)
def targets(s):
    # Others are pinned to test; the rest of each split is filled proportionally.
    free = SPLITS[s] - (n_other if s == "test" else 0)
    tot = len(ids) - n_other
    t = {c: n * free / tot for c, n in cells.items() if not c.startswith("O")}
    t |= {c: (n if s == "test" else 0) for c, n in cells.items() if c.startswith("O")}
    a = {x: ages[x] * SPLITS[s] / len(ids) for x in AGES}
    return t, a
T = {s: targets(s) for s in SPLITS}
def cost(assign):
    c = 0.0
    for s in SPLITS:
        members = [x for u, sp in assign.items() if sp == s for x in u]
        if len(members) != SPLITS[s]: c += 1000 * abs(len(members) - SPLITS[s])
        ce = collections.Counter(eg(x) for x in members); ca = collections.Counter(ids[x]["age"] for x in members)
        c += sum((ce[k] - v) ** 2 for k, v in T[s][0].items()) / SPLITS[s]
        c += 0.5 * sum((ca[k] - v) ** 2 for k, v in T[s][1].items()) / SPLITS[s]
    return c
rng = random.Random(20261018)
best = None
for restart in range(20):
    order = [u for u in units if u not in fixed]; rng.shuffle(order)
    assign, size = dict(fixed), collections.Counter({"test": n_other})
    for u in order:
        s = next(s for s in ("test", "train", "dev") if size[s] + len(u) <= SPLITS[s]) if rng.random() < 0.5 else \
            rng.choice([s for s in SPLITS if size[s] + len(u) <= SPLITS[s]])
        assign[u] = s; size[s] += len(u)
    cur = cost(assign); free = [u for u in units if u not in fixed]
    for _ in range(20000):
        a, b = rng.sample(free, 2)
        if assign[a] == assign[b] or len(a) != len(b): continue
        assign[a], assign[b] = assign[b], assign[a]
        c = cost(assign)
        if c <= cur: cur = c
        else: assign[a], assign[b] = assign[b], assign[a]
    if best is None or cur < best[0]: best = (cur, dict(assign))
cur, assign = best
for u, s in assign.items():
    for x in u: ids[x]["split"] = s
picked.sort(key=lambda r: (list(SPLITS).index(r["split"]), r["speaker"]))
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(picked[0])); w.writeheader(); w.writerows(picked)
print("cost", round(cur, 3))
for s in SPLITS:
    sub = [r for r in picked if r["split"] == s]
    print(s, len(sub), sorted(collections.Counter(r["ethnicity"][0] + r["gender"] for r in sub).items()),
          sorted(collections.Counter(r["age"] for r in sub).items()),
          sorted(collections.Counter(r["first_language"] for r in sub).items()))
print("pairs split across:", sum(1 for r in picked if r["partner"] in ids and ids[r["partner"]]["split"] != r["split"]))
