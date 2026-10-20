#!/usr/bin/env python3
"""Do the two "where" views agree? The evidence map (ConvNeXt cover-up, backend/engine.py) and the crypt
measurements (backend/glands.py) are built independently. If both are meaningful, crypts measured as dilated should
sit where covering the tissue lowers P(SSL) (positive evidence) more than round, regular crypts do.
TRAINING images only (400 not used to set the crypt thresholds). Uses the 16x16 map level the app shows first.
Writes docs/results/evidence_vs_crypts.json."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.draw import polygon as fill_polygon
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research"))
from engine import Engine  # noqa: E402

ann = pd.read_csv(ROOT / "data/mhist/annotations.csv")
tr = ann[ann.Partition == "train"]
used = set(pd.concat([d.sample(min(len(d), 300), random_state=0) for _, d in tr.groupby("Majority Vote Label")])["Image Name"])
held = tr[~tr["Image Name"].isin(used)].sample(400, random_state=2)
e = Engine()
rows = []
for _, r in held.iterrows():
    res = e.analyze(Image.open(ROOT / "data/mhist/images" / r["Image Name"]).convert("RGB").resize((560, 560)))
    if "error" in res:
        continue
    H = res["heat"]; ev = (np.array(H["p_ssa"]).reshape(H["rows"], H["cols"]) - 0.5) * 2
    for g in res["glands"]:
        if g.get("label"):
            # evidence averaged over the whole crypt outline (its centre is often the empty lumen)
            o = np.array(g["outline"])
            rr, cc = fill_polygon(o[:, 1] * H["rows"], o[:, 0] * H["cols"], shape=(H["rows"], H["cols"]))
            cells = np.zeros((H["rows"], H["cols"]), bool); cells[rr, cc] = True
            area = float(ev[cells].mean()) if cells.any() else g["evidence"]
            rows.append({"image": r["Image Name"], "ssl": r["Majority Vote Label"] == "SSA", "tags": g["tags"],
                         "evidence": g["evidence"], "evidence_area": area})
d = pd.DataFrame(rows)
out = {"images": int(d.image.nunique()), "labelled_crypts": len(d)}
for tag in ("dilated", "serrated", "round regular"):
    m = d.tags.apply(lambda t: tag in t)
    out[tag] = {"crypts": int(m.sum()), "mean_evidence_centre": round(float(d[m].evidence.mean()), 3),
                "mean_evidence_area": round(float(d[m].evidence_area.mean()), 3),
                "in_SSL_images": round(float(d[m & d.ssl].evidence_area.mean()), 3),
                "in_HP_images": round(float(d[m & ~d.ssl].evidence_area.mean()), 3),
                "share_pushing_toward_SSL_area": round(float((d[m].evidence_area > 0).mean()), 3)}
dil, rnd = d[d.tags.apply(lambda t: "dilated" in t)].evidence_area, d[d.tags.apply(lambda t: "round regular" in t)].evidence_area
out["dilated_vs_round_p"] = float(mannwhitneyu(dil, rnd).pvalue) if len(dil) and len(rnd) else None
# within each class separately (pooling mixes the classes: HP fields carry HP-pushing evidence everywhere)
for name, cls in (("SSL", d.ssl), ("HP", ~d.ssl)):
    a, b = dil[cls[dil.index]], rnd[cls[rnd.index]]
    out[f"dilated_vs_round_p_in_{name}_images"] = float(mannwhitneyu(a, b).pvalue) if len(a) and len(b) else None
(ROOT / "docs/results/evidence_vs_crypts.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
