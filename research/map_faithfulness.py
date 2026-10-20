#!/usr/bin/env python3
"""Is the evidence map faithful? Standard deletion test on TRAINING images (held out from the crypt thresholds):
blur the 15% of the field the map ranks most SSL-pushing (for SSL images) or most HP-pushing (for HP images) and
measure how much ConvNeXt's logit moves the other way, against blurring a random 15% of tissue. A faithful map moves
the score much more. Uses the 16x16 map level shown first in the app. Writes docs/results/map_faithfulness.json."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research"))
from engine import Engine  # noqa: E402

ann = pd.read_csv(ROOT / "data/mhist/annotations.csv")
tr = ann[ann.Partition == "train"]
used = set(pd.concat([d.sample(min(len(d), 300), random_state=0) for _, d in tr.groupby("Majority Vote Label")])["Image Name"])
held = tr[~tr["Image Name"].isin(used)]
pick = pd.concat([held[held["Majority Vote Label"] == c].sample(60, random_state=3) for c in ("SSA", "HP")])
e = Engine()
net = e.panel["nets"]["convnext_base"]
rng = np.random.default_rng(0)


@torch.no_grad()
def logit(t):
    with torch.autocast("cuda" if str(e.device).startswith("cuda") else "cpu", dtype=torch.bfloat16):
        z = net.m(t.unsqueeze(0).to(e.device))
    return float((net.cls(torch.nn.functional.normalize(z.float(), dim=-1)).squeeze(-1) * 10.0)[0])


rows = []
for _, r in pick.iterrows():
    im = Image.open(ROOT / "data/mhist/images" / r["Image Name"]).convert("RGB").resize((560, 560))
    tiles = e.make_grid(im, min(im.size))
    H = e._evidence_heat(im, tiles, 16)
    ev = np.array(H["p_ssa"]).reshape(16, 16) - 0.5
    tissue = np.array(H["tissue"]).reshape(16, 16) >= 0.25
    inner = np.zeros((16, 16), bool); inner[1:-1, 1:-1] = True       # cells the model sees
    ok = tissue & inner
    if ok.sum() < 20:
        continue
    ssl = r["Majority Vote Label"] == "SSA"
    score = ev if ssl else -ev                                       # evidence FOR the image's own class
    k = max(1, int(round(0.15 * ok.sum())))
    order = np.argsort(np.where(ok, score, -9).ravel())[::-1][:k]
    top = np.zeros(256, bool); top[order] = True; top = top.reshape(16, 16)
    rand = np.zeros(256, bool); rand[rng.choice(np.flatnonzero(ok.ravel()), k, replace=False)] = True; rand = rand.reshape(16, 16)
    x = net.pre(im); blur = net.pre(im.filter(ImageFilter.GaussianBlur(radius=14)))
    def masked(cells):
        t = x.clone()
        for rr, cc in np.argwhere(cells):                            # field cell -> 16 px block of the 224 crop
            y0, x0 = (rr - 1) * 16, (cc - 1) * 16
            t[:, y0:y0 + 16, x0:x0 + 16] = blur[:, y0:y0 + 16, x0:x0 + 16]
        return t
    base = logit(x)
    sign = 1 if ssl else -1                                         # positive = moved away from the image's class
    rows.append({"ssl": bool(ssl), "drop_top": sign * (base - logit(masked(top))), "drop_random": sign * (base - logit(masked(rand)))})
d = pd.DataFrame(rows)
out = {"images": len(d), "cells_removed": "15% of tissue cells the model sees"}
for name, g in (("SSL images", d[d.ssl]), ("HP images", d[~d.ssl]), ("all", d)):
    out[name] = {"n": len(g), "mean_logit_drop_top_evidence": round(float(g.drop_top.mean()), 3),
                 "mean_logit_drop_random": round(float(g.drop_random.mean()), 3),
                 "share_where_top_beats_random": round(float((g.drop_top > g.drop_random).mean()), 3)}
(ROOT / "docs/results/map_faithfulness.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
