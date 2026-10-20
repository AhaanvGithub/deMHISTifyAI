#!/usr/bin/env python3
"""Do the measured crypt labels (backend/glands.py) agree with the pathologists? TRAINING images only, excluding
the 600 used to set the thresholds (research/gland_thresholds.py). Per image: share of labelled crypts that are
dilated / round regular, against the number of 7 pathologists who called it SSL (Spearman), and AUC for SSL vs HP.
Writes docs/results/gland_check.json.

    python research/gland_check.py [finder]     finder = classical (default) or a name in research/crypt_finders.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import glands as G  # noqa: E402
sys.path.insert(0, str(ROOT / "research"))
import crypt_finders as CF  # noqa: E402
from crypt_gt import DATA  # noqa: E402

FINDER = sys.argv[1] if len(sys.argv) > 1 else "classical"
find = CF.FINDERS[FINDER]

ann = pd.read_csv(DATA / "mhist/annotations.csv")
train = ann[ann.Partition == "train"]
used = set(pd.concat([d.sample(min(len(d), 300), random_state=0) for _, d in train.groupby("Majority Vote Label")])["Image Name"])
held = train[~train["Image Name"].isin(used)]
rows = []
for _, r in held.iterrows():
    gl = [g for g in find(Image.open(DATA / "mhist/images" / r["Image Name"])) if g["label"]]
    n = len(gl)
    rows.append({"image": r["Image Name"], "ssl": r["Majority Vote Label"] == "SSA",
                 "votes": r["Number of Annotators who Selected SSA (Out of 7)"], "n_labelled": n,
                 "dilated": np.mean(["dilated" in g["tags"] for g in gl]) if n else np.nan,
                 "round": np.mean(["round regular" in g["tags"] for g in gl]) if n else np.nan,
                 "serrated": np.mean(["serrated" in g["tags"] for g in gl]) if n else np.nan})
d = pd.DataFrame(rows)
ok = d.dropna()
res = {"finder": FINDER, "images": len(d), "images_with_labelled_crypts": len(ok)}
for k in ("dilated", "round", "serrated"):
    res[k] = {"spearman_vs_ssl_votes": round(float(spearmanr(ok[k], ok.votes)[0]), 3),
              "auc_ssl_vs_hp": round(float(roc_auc_score(ok.ssl, ok[k])), 3),
              "mean_share_hp": round(float(ok[~ok.ssl][k].mean()), 3), "mean_share_ssl": round(float(ok[ok.ssl][k].mean()), 3)}
(ROOT / "docs/results" / ("gland_check.json" if FINDER == "classical" else f"gland_check_{FINDER}.json")).write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
