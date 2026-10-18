import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1])))
QUOTA = {("CHINESE","F"):43, ("CHINESE","M"):43, ("MALAY","F"):22, ("MALAY","M"):22,
         ("INDIAN","F"):20, ("INDIAN","M"):20, ("OTHERS","F"):3, ("OTHERS","M"):2}
SPLITS = {"train": 50, "dev": 25, "test": 100}
AGE_W = {"1X":0.08, "2X":0.25, "3X":0.25, "4X":0.22, "5X":0.12, "6X":0.08}
AGES = list(AGE_W)
def allocate(n, avail):
    alloc = {a: 0 for a in AGE_W}
    while n > 0:
        open_ = [a for a in AGE_W if alloc[a] < avail.get(a, 0)]
        if not open_: break
        tot = sum(AGE_W[a] for a in open_)
        a = max(open_, key=lambda a: (AGE_W[a]/tot*(sum(alloc.values())+1) - alloc[a], AGE_W[a]))
        alloc[a] += 1; n -= 1
    return alloc
cells = collections.defaultdict(list)
for r in rows: cells[(r["ethnicity"], r["gender"])].append(r)
picked = []
for cell, q in QUOTA.items():
    by_age = collections.defaultdict(list)
    for r in cells[cell]: by_age[r["age"]].append(r)
    for a, k in allocate(q, {a: len(v) for a, v in by_age.items()}).items():
        picked += sorted(by_age[a], key=lambda r: (-float(r["ref_candidates_min"]), r["speaker"]))[:k]
ids = {r["speaker"]: r for r in picked}
# Units: partner pairs that are both picked stay together; everyone else is a single.
units, seen = [], set()
for r in sorted(picked, key=lambda r: (r["ethnicity"], r["gender"], AGES.index(r["age"]), r["speaker"])):
    if r["speaker"] in seen: continue
    u = [r] + ([ids[r["partner"]]] if r["partner"] in ids else [])
    seen.update(x["speaker"] for x in u); units.append(u)
N = len(picked)
share = {s: n / N for s, n in SPLITS.items()}
# per-split targets for each stratum (ethnicity, gender) and each age band
def key_eg(r): return (r["ethnicity"], r["gender"])
tgt_eg = {s: {c: q * share[s] for c, q in QUOTA.items()} for s in SPLITS}
age_tot = collections.Counter(r["age"] for r in picked)
tgt_age = {s: {a: age_tot[a] * share[s] for a in AGES} for s in SPLITS}
cnt = {s: collections.Counter() for s in SPLITS}
cnt_age = {s: collections.Counter() for s in SPLITS}
size = collections.Counter()
assign = {}
rarity = {c: q for c, q in QUOTA.items()}
for u in sorted(units, key=lambda u: (min(rarity[key_eg(r)] for r in u), -len(u))):   # rarest strata first, then pairs
    def cost(s):
        if size[s] + len(u) > SPLITS[s]: return float("inf")
        c = 0.0
        for r in u:
            c += (cnt[s][key_eg(r)] + 1 - tgt_eg[s][key_eg(r)]) + 0.5 * (cnt_age[s][r["age"]] + 1 - tgt_age[s][r["age"]])
        return c - 0.01 * (SPLITS[s] - size[s])
    s = min(SPLITS, key=cost)
    for r in u:
        assign[r["speaker"]] = s; cnt[s][key_eg(r)] += 1; cnt_age[s][r["age"]] += 1
    size[s] += len(u)
for r in picked: r["split"] = assign[r["speaker"]]
picked.sort(key=lambda r: (list(SPLITS).index(r["split"]), r["speaker"]))
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["split"]); w.writeheader(); w.writerows(picked)
print("total", N, dict(size))
for k in ["ethnicity", "gender", "age", "first_language"]:
    print(k, sorted(collections.Counter(r[k] for r in picked).items()))
for s in SPLITS:
    sub = [r for r in picked if r["split"] == s]
    print(s, len(sub), sorted(collections.Counter((r["ethnicity"][0] + r["gender"]) for r in sub).items()),
          sorted(collections.Counter(r["age"] for r in sub).items()),
          sorted(collections.Counter(r["first_language"] for r in sub).items()))
print("pairs", sum(1 for r in picked if r["partner"] in ids)//2,
      "split across:", sum(1 for r in picked if r["partner"] in ids and assign[r["partner"]] != r["split"]))
print("min clean clips", min(int(r["ref_candidates"]) for r in picked), "min clean min", min(float(r["ref_candidates_min"]) for r in picked))
