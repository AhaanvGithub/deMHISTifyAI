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

# --- references for the alternatives (colour averaging, Vahadane, derived): 400 random training images ---
import time
t0 = time.time()
sample = lab.loc[lab["split"] == "train", "image"].sample(400, random_state=0)
labs, vm, med, p99 = [], [], [], []
for n in sample:
    a = img(n)
    mu, sd = stain.reinhard_stats(a); labs.append((mu, sd))
    he = stain.estimate_vahadane(a)
    if he is None:
        continue
    vm.append(he)
    m_, p_ = stain.conc_stats(a, he); med.append(m_); p99.append(p_)
vmat = np.median(np.stack(vm), axis=0); vmat = vmat / np.linalg.norm(vmat, axis=0)
ref = json.loads(stain.REF_PATH.read_text())
ref.update({"reinhard_mean": np.mean([x[0] for x in labs], 0).round(4).tolist(),
            "reinhard_std": np.mean([x[1] for x in labs], 0).round(4).tolist(),
            "vahadane_matrix": vmat.round(5).tolist(),
            "vahadane_median": np.median(np.stack(med), 0).round(5).tolist(),
            "vahadane_p99": np.median(np.stack(p99), 0).round(5).tolist(),
            "alternatives_n_images": len(vm)})
stain.REF_PATH.write_text(json.dumps(ref, indent=2))
print(f"alternatives reference from {len(vm)} images in {time.time() - t0:.0f}s:\n  Vahadane matrix (H, E):\n{vmat.round(4)}"
      f"\n  Lab mean {ref['reinhard_mean']} std {ref['reinhard_std']}")
