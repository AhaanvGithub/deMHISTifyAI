#!/usr/bin/env python3
"""Measure MHIST's own stain reference (median dye colours and dye strengths over the training images) and
check that normalising MHIST test images toward it barely changes them. Writes backend/config/stain_mhist.json."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research"))
import stain  # noqa: E402
from select_prompts import load_labels  # noqa: E402

lab = load_labels(ROOT / "data/mhist/annotations.csv")
img = lambda n: np.asarray(Image.open(ROOT / "data/mhist/images" / n).convert("RGB"))  # noqa: E731

mats, maxs = [], []
for n in lab.loc[lab["split"] == "train", "image"]:
    r = stain.estimate(img(n))
    if r is not None:
        mats.append(r[0]); maxs.append(r[1])
he = np.median(np.stack(mats), axis=0)
he = he / np.linalg.norm(he, axis=0)
mx = np.median(np.stack(maxs), axis=0)
stain.REF_PATH.write_text(json.dumps({"stain_matrix": he.round(5).tolist(), "max_conc": mx.round(5).tolist(),
                                      "n_images": len(mats), "source": "MHIST train, median per-image Macenko estimate"}, indent=2))
print(f"MHIST reference from {len(mats)} training images:\n  stain matrix (cols H, E):\n{he.round(4)}\n  max conc {mx.round(4)}")

# self-check: how much do MHIST test images change when normalised toward the MHIST reference vs the generic one?
generic = (np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]), np.array([1.9705, 1.0308]))
test = lab.loc[lab["split"] == "test", "image"].sample(200, random_state=0)
d_m, d_g = [], []
for n in test:
    a = img(n); src = stain.estimate(a)
    if src is None:
        continue
    d_m.append(np.abs(stain.normalize(a, src).astype(int) - a).mean())
    d_g.append(np.abs(stain.normalize(a, src, generic).astype(int) - a).mean())
print(f"mean pixel change on 200 MHIST test images: toward MHIST reference {np.mean(d_m):.1f}, "
      f"toward the generic (old) reference {np.mean(d_g):.1f}  (0-255 scale)")
