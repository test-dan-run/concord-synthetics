import csv, sys, openpyxl
wb = openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=True)
rows = list(wb["Separate Room"].iter_rows(values_only=True))
hdr = [str(h).strip() for h in rows[0]]
info = {str(d["Speaker ID"]).strip().zfill(8): d for d in (dict(zip(hdr, r)) for r in rows[1:] if any(r))}
sel = list(csv.DictReader(open(sys.argv[2])))
cols = ["speaker", "split", "sdm_zip", "conf", "partner", "gender", "age", "ethnicity", "first_language", "languages",
        "education", "occupation", "relationship", "n_segments", "speech_min", "ref_candidates", "ref_candidates_min"]
with open(sys.argv[3], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for r in sel:
        d = info[r["speaker"]]
        w.writerow({**{k: r[k] for k in cols if k in r}, "sdm_zip": f"SDM/SPEAKER{r['speaker']}.zip",
                    "languages": str(d["Language Proficiency"]).replace(" ", ""), "education": d["Education"],
                    "occupation": d["Occupation"], "relationship": d["Relationship"]})
