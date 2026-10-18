import sys, os, re, json, zipfile, csv, collections, openpyxl
xlsx, sdm_dir, out = sys.argv[1:4]
wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
rows = list(wb["Separate Room"].iter_rows(values_only=True))
hdr = [str(h).strip() for h in rows[0]]
info = {}
for r in rows[1:]:
    if not any(r): continue
    d = dict(zip(hdr, r)); sid = str(d["Speaker ID"]).strip().zfill(8); info[sid] = d
BAD = re.compile(r"#|<|\*\*|\[unk|~")
res = []
for f in sorted(os.listdir(sdm_dir)):
    sid = f[7:-4]
    z = zipfile.ZipFile(os.path.join(sdm_dir, f))
    names = set(z.namelist())
    man = [json.loads(l) for l in z.read(f"SPEAKER{sid}/manifest.json").decode("utf-8").splitlines() if l.strip()]
    dur = [m["source_end"] - m["source_start"] for m in man]
    wav_ok = sum(1 for m in man if f"WAVE/{m['pair_id'].replace('_','')}.wav" in names)
    cand = [m for m, d in zip(man, dur) if 5 <= d <= 15 and not BAD.search(m["text"]) and len(m["text"].split()) >= 8]
    d = info.get(sid, {})
    res.append(dict(speaker=sid, conf=d.get("Conference ID"), gender=d.get("Gender"), age=d.get("Age"),
        ethnicity=d.get("Ethnicity"), first_language=d.get("First Language"),
        languages=str(d.get("Language Proficiency ") or "").replace(" ", ""), partner=str(d.get("Partner ID") or "").zfill(8),
        n_segments=len(man), speech_min=round(sum(dur)/60, 1), wav_matched=wav_ok,
        ref_candidates=len(cand), ref_candidates_min=round(sum(m["source_end"]-m["source_start"] for m in cand)/60, 1)))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print(len(res), "speakers; missing info:", sum(1 for r in res if r["gender"] is None))
print("wav match rate", sum(r["wav_matched"] for r in res)/sum(r["n_segments"] for r in res))
for k in ["gender", "ethnicity", "age", "first_language"]:
    print(k, collections.Counter(r[k] for r in res).most_common())
import statistics as st
c = [r["ref_candidates"] for r in res]; print("ref candidates: min", min(c), "median", st.median(c), "<5:", sum(x < 5 for x in c))
s = [r["speech_min"] for r in res]; print("speech min: min", min(s), "median", st.median(s))
