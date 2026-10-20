#!/usr/bin/env python3
"""The deployed app on another hospital's slides: UniToPatho (Turin; Hamamatsu scanner, lighter H&E).
Each patch's centre 448 um (one MHIST field) goes through the app's own path (Engine.analyze, default cut-off).
Classes: hyperplastic polyp (should be called HP), tubular adenoma low grade and normal colon (neither HP nor SSL;
reported, not scored). Also: how often the unlike-training-data warning fires, and the crypt labels per class
(serration should be common in HP and rare in normal colon). No model or threshold is chosen from this.
Writes docs/results/unitopatho_panel_check.json."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research"))
from engine import Engine  # noqa: E402

D = ROOT / "data/unitopatho/unitopath-public/800"
meta = pd.concat([pd.read_csv(D / "train.csv"), pd.read_csv(D / "test.csv")]).set_index("image_id")
e = Engine()
e._dense_heat = lambda snip, tiles, sub=4: {"rows": 1, "cols": 1, "p_ssa": [0.5], "tissue": [1.0], "features": {}}

out = {}
for cls in ("HP", "TA.LG", "NORM"):
    files = sorted((D / cls).glob("*.png"))
    rows, t0 = [], time.time()
    for f in files:
        im = Image.open(f).convert("RGB"); c = round(448 / meta.loc[f.name, "mpp"]); w, h = im.size
        r = e.analyze(im.crop(((w - c) // 2, (h - c) // 2, (w + c) // 2, (h + c) // 2)))
        if "error" in r:
            rows.append({"error": r["error"]}); continue
        lab = [g for g in r["glands"] if g["label"]]
        rows.append({"wsi": meta.loc[f.name, "wsi"], "p_ssl": r["panel"]["p_ssl"], "call": r["panel"]["top"],
                     "members": [j["top"] for j in r["judges"]], "ood": bool(r["ood"] and r["ood"]["flag"]),
                     "n_crypts": len(r["glands"]), "n_labelled": len(lab),
                     "dilated": sum("dilated" in g["tags"] for g in lab), "serrated": sum("serrated" in g["tags"] for g in lab),
                     "round": sum("round regular" in g["tags"] for g in lab)})
    d = pd.DataFrame([x for x in rows if "error" not in x])
    lab_tot = max(1, int(d.n_labelled.sum()))
    out[cls] = {"fields": len(files), "slides": int(d.wsi.nunique()), "errors": len(rows) - len(d),
                "called_HP": round(float((d.call == "HP").mean()), 4),
                "median_p_ssl": round(float(d.p_ssl.median()), 3), "p90_p_ssl": round(float(d.p_ssl.quantile(0.9)), 3),
                "members_disagree": round(float(d.members.apply(lambda m: len(set(m)) > 1).mean()), 4),
                "unlike_training_flagged": round(float(d.ood.mean()), 4),
                "crypts_per_field": round(float(d.n_crypts.mean()), 1), "labelled_per_field": round(float(d.n_labelled.mean()), 2),
                "share_dilated": round(d.dilated.sum() / lab_tot, 3), "share_serrated": round(d.serrated.sum() / lab_tot, 3),
                "share_round_regular": round(d["round"].sum() / lab_tot, 3), "seconds": round(time.time() - t0)}
    print(cls, json.dumps(out[cls]), flush=True)
(ROOT / "docs/results/unitopatho_panel_check.json").write_text(json.dumps(out, indent=2))
print("wrote docs/results/unitopatho_panel_check.json")
