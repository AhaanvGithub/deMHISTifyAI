#!/usr/bin/env python3
"""Measure the reference nucleus scale on MHIST training images (once) and check that the
estimator recovers known zoom factors on held-out test images. Writes backend/config/scale.json."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from scale import SCALE_PATH, char_sigma  # noqa: E402

ann = pd.read_csv(ROOT / "data/mhist/annotations.csv")
img = lambda n: Image.open(ROOT / "data/mhist/images" / n).convert("RGB")  # noqa: E731
ref = float(np.median([char_sigma(img(n)) for n in ann[ann.Partition == "train"]["Image Name"].sample(200, random_state=0)]))
print(f"reference sigma (MHIST train, 200 images): {ref:.3f}px")

test = ann[ann.Partition == "test"]["Image Name"].sample(40, random_state=1)
check = {}
for z in (0.6, 0.75, 1.0, 1.5, 2.0, 3.5, 5.0):
    est = [char_sigma(img(n).resize((int(224 * z),) * 2, Image.BICUBIC)) / ref for n in test]
    check[z] = [round(float(np.median(est)), 2), round(float(np.percentile(est, 25)), 2), round(float(np.percentile(est, 75)), 2)]
    print(f"  true zoom x{z}: estimated x{check[z][0]} (IQR {check[z][1]}-{check[z][2]})")

SCALE_PATH.write_text(json.dumps({"sigma": round(ref, 4), "n_images": 200, "check_true_vs_est": check}, indent=2))
print("wrote", SCALE_PATH)
