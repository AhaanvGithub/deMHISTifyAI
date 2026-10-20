#!/usr/bin/env python3
"""Set the gland label thresholds (backend/glands.py) from MHIST TRAINING images only.

Measures every crypt in 600 training images (stratified, seed 0) and sets each threshold from the distribution:
  dilated crypt       : crypt interior diameter at or above the 90th percentile
  serrated lumen      : lumen roughness at or above the 75th percentile (crypts with a measurable lumen)
  round regular crypt : solidity at or above the 60th percentile, eccentricity at or below 0.75, and no wider
                        than the median crypt
Then reports how often each label occurs in SSL vs HP images (a sanity check, not a tuning step).
Writes backend/config/gland_thresholds.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import glands as G  # noqa: E402

ann = pd.read_csv(ROOT / "data/mhist/annotations.csv")
tr = pd.concat([d.sample(min(len(d), 300), random_state=0)
                for _, d in ann[ann.Partition == "train"].groupby("Majority Vote Label")])
rows = []
for _, r in tr.iterrows():
    for g in G.measure(Image.open(ROOT / "data/mhist/images" / r["Image Name"])):
        rows.append({**{k: g[k] for k in ("crypt_um", "lumen_um", "roughness", "solidity", "eccentricity")},
                     "label": r["Majority Vote Label"], "image": r["Image Name"]})
d = pd.DataFrame(rows)
T = {"dilated_um": round(float(d.crypt_um.quantile(0.90)), 1),
     "serrated_roughness": round(float(d.roughness.dropna().quantile(0.75)), 3),
     "round_solidity": round(float(d.solidity.quantile(0.60)), 3),
     "round_eccentricity": 0.75,
     "round_max_um": round(float(d.crypt_um.median()), 1),
     "source": f"{len(d)} crypts in {d.image.nunique()} MHIST training images (research/gland_thresholds.py)"}
(ROOT / "backend/config/gland_thresholds.json").write_text(json.dumps(T, indent=2))
print(json.dumps(T, indent=2))

G._T = T
lab = pd.DataFrame([{"label": r.label, "name": G.label([dict(r._asdict())])[0]["label"]} for r in d.itertuples()])
share = lab.groupby("label")["name"].value_counts(normalize=True).unstack(fill_value=0).round(3)
print("\nshare of crypts per label, by image class (SSA = SSL):\n", share)
print("\ncrypts found per image: HP", round(d[d.label == "HP"].groupby("image").size().mean(), 1),
      "| SSL", round(d[d.label == "SSA"].groupby("image").size().mean(), 1))
