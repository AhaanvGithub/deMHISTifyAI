#!/usr/bin/env python3
"""Does the app reproduce the panel's official test result? A deployment check, not a new evaluation: nothing is
chosen from it (the panel, weights and cut-off were fixed on training data and scored once in research/panel.py).

Each MHIST test tile is turned into what the app would receive from a screen snip:
  native   : the tile enlarged 2.5x (bilinear), as a viewer shows it on screen
  scaled150: the same, then through a 150% display-scaling round trip (down to 1/1.5 and back up)
and scored by the app's own path (backend/engine.py Engine.analyze, default cut-off). The evidence map and crypt
finder are skipped here (they don't affect the call). Writes docs/results/app_panel_check.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research"))
from engine import Engine  # noqa: E402
from select_prompts import load_labels  # noqa: E402

lab = load_labels(ROOT / "data/mhist/annotations.csv")
names = np.array(sorted(lab["image"]))
lab = lab.set_index("image").loc[names]
ts = lab[lab["split"] == "test"]
y = (ts["label"] == "SSA").to_numpy().astype(int)
votes = ts["ssa_votes"].to_numpy()

e = Engine()
e._dense_heat = lambda snip, tiles, sub=4: {"rows": 1, "cols": 1, "p_ssa": [0.5], "tissue": [1.0], "features": {}}
e._glands = staticmethod(lambda snip, tiles, heat: [])


def render(img, variant):
    big = img.resize((560, 560), Image.BILINEAR)
    if variant == "scaled150":
        big = big.resize((373, 373), Image.BILINEAR).resize((560, 560), Image.BILINEAR)
    return big


official = json.loads((ROOT / "docs/results/panel.json").read_text())["TEST_chosen_panel"]
out = {"official_test": official}
for variant in ("native", "scaled150"):
    p, calls, flags, t0 = [], [], [], time.time()
    for i, n in enumerate(ts.index):
        r = e.analyze(render(Image.open(ROOT / "data/mhist/images" / n).convert("RGB"), variant))
        if "error" in r:
            p.append(np.nan); calls.append(0); flags.append(False)
            continue
        p.append(r["panel"]["p_ssl"]); calls.append(int(r["panel"]["top"] == "SSA")); flags.append(bool(r["ood"] and r["ood"]["flag"]))
        if (i + 1) % 200 == 0:
            print(f"  {variant} {i + 1}/{len(ts)} ({time.time() - t0:.0f}s)", flush=True)
    p, calls = np.array(p), np.array(calls)
    ok = ~np.isnan(p)
    out[variant] = {"images": int(ok.sum()), "errors": int((~ok).sum()),
                    "bal_acc": round(float(balanced_accuracy_score(y[ok], calls[ok])), 4),
                    "auc": round(float(roc_auc_score(y[ok], p[ok])), 4),
                    "acc_vs_majority": round(float((calls[ok] == y[ok]).mean()), 4),
                    "ssl_caught": round(float(calls[ok][y[ok] == 1].mean()), 4),
                    "hp_caught": round(float(1 - calls[ok][y[ok] == 0].mean()), 4),
                    "spearman_vs_votes": round(float(spearmanr(p[ok], votes[ok])[0]), 4),
                    "unlike_training_flagged": round(float(np.mean(flags)), 4),
                    "seconds": round(time.time() - t0)}
    print(variant, json.dumps(out[variant]), flush=True)
(ROOT / "docs/results/app_panel_check.json").write_text(json.dumps(out, indent=2))
print("wrote docs/results/app_panel_check.json")
