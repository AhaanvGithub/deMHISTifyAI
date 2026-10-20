#!/usr/bin/env python3
"""Set the gland label thresholds (backend/glands.py) from MHIST TRAINING images only.

Measures every crypt in 600 training images (stratified, seed 0) and sets each threshold from the distribution:
  dilated crypt       : crypt interior width (short axis) at or above the 90th percentile
  serrated lumen      : lumen roughness at or above the 75th percentile (crypts with a measurable lumen)
  round regular crypt : solidity at or above the 60th percentile, eccentricity at or below 0.75, and no wider
                        than the median crypt
Then reports how often each label occurs in SSL vs HP images (a sanity check, not a tuning step).
Writes backend/config/gland_thresholds.json (classical finder) or gland_thresholds_<finder>.json.

    python research/gland_thresholds.py [finder]     finder = classical (default) or a name in research/crypt_finders.py
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
sys.path.insert(0, str(ROOT / "research"))
import crypt_finders as CF  # noqa: E402
from crypt_gt import DATA  # noqa: E402

FINDER = sys.argv[1] if len(sys.argv) > 1 else "classical"
measure = G.measure if FINDER == "classical" else CF.MEASURERS[FINDER]

ann = pd.read_csv(DATA / "mhist/annotations.csv")
tr = pd.concat([d.sample(min(len(d), 300), random_state=0)
                for _, d in ann[ann.Partition == "train"].groupby("Majority Vote Label")])
# basis "gland" (finder name ending in _g): the same rules on the crypt's own outline instead of its pale interior
BY_GLAND = FINDER.endswith("_g")
rows = []
for _, r in tr.iterrows():
    for g in measure(Image.open(DATA / "mhist/images" / r["Image Name"])):
        if BY_GLAND:
            if g.get("gland_width_um") is None:
                continue
            g = {**g, "crypt_um": g["gland_um"], "width_um": g["gland_width_um"], "solidity": g["gland_solidity"],
                 "eccentricity": g["gland_eccentricity"]}
        elif g["ring"] < G.MIN_RING:
            continue
        rows.append({**{k: g[k] for k in ("crypt_um", "width_um", "lumen_um", "roughness", "solidity", "eccentricity", "ring")},
                     "label": r["Majority Vote Label"], "image": r["Image Name"]})
d = pd.DataFrame(rows)
T = {**({"basis": "gland"} if BY_GLAND else {}),"dilated_width_um": round(float(d.width_um.quantile(0.90)), 1),
     "serrated_roughness": round(float(d.roughness.dropna().quantile(0.75)), 3),
     "round_solidity": round(float(d.solidity.quantile(0.60)), 3),
     "round_eccentricity": 0.75,
     "round_max_um": round(float(d.crypt_um.median()), 1),
     "source": f"{FINDER} finder: {len(d)} crypts in {d.image.nunique()} MHIST training images (research/gland_thresholds.py)"}
(ROOT / "backend/config" / ("gland_thresholds.json" if FINDER == "classical" else f"gland_thresholds_{FINDER}.json")).write_text(json.dumps(T, indent=2))
print(json.dumps(T, indent=2))

lab = pd.DataFrame([{"label": r.label, "name": G.label([{**dict(r._asdict()), **({"gland_um": r.crypt_um, "gland_width_um": r.width_um, "gland_solidity": r.solidity,
                                    "gland_eccentricity": r.eccentricity} if BY_GLAND else {})}], T)[0]["label"]} for r in d.itertuples()])
share = lab.groupby("label")["name"].value_counts(normalize=True).unstack(fill_value=0).round(3)
print("\nshare of crypts per label, by image class (SSA = SSL):\n", share)
print("\ncrypts found per image: HP", round(d[d.label == "HP"].groupby("image").size().mean(), 1),
      "| SSL", round(d[d.label == "SSA"].groupby("image").size().mean(), 1))
