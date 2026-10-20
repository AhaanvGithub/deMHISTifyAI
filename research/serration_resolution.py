#!/usr/bin/env python3
"""Can "serrated lumen" be seen at MHIST's ~2 um/px? Test on UniToPatho (Turin; 0.44 um/px): hyperplastic polyps
(HP: serrated, saw-tooth crypt lumens) vs normal colon (NORM: smooth lumens), same lab and scanner.

Each 800 um patch -> its central 448 um field. Crypts are found once, on the field shrunk to MHIST's 2.0 um/px (the
finder given on the command line). For each crypt, the lumen (near-white pixels inside the crypt) is measured at
0.44 (native), 1.0 and 2.0 um/px, always within the same crypt outline: roughness = lumen perimeter / perimeter of
its convex hull (backend/glands.py). If serration is resolvable at a given scale, HP crypts should be rougher than
normal crypts there: AUC (HP vs NORM) per crypt and per field (mean over its crypts). Labels are slide-level
(HP vs normal); not every crypt in an HP patch is serrated, so these AUCs understate per-crypt separability equally
at every scale - the comparison across scales is the point.

    python research/serration_resolution.py [finder]      -> docs/results/serration_resolution.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crypt_gt as GT  # noqa: E402
import crypt_finders as CF  # noqa: E402

SRC = GT.DATA / "unitopatho/unitopath-public/800"
MPP = 0.4415
SCALES = {"0.44 um/px (native)": None, "1.0 um/px": 448, "2.0 um/px (MHIST)": 224}


def roughness(img: Image.Image, outline) -> float:
    from skimage import measure as M
    from skimage.draw import polygon
    a = np.asarray(img, np.float32) / 255
    gray, sat = a.mean(2), a.max(2) - a.min(2)
    n = a.shape[0]
    m = np.zeros((n, n), bool)
    p = np.asarray(outline) * n
    rr, cc = polygon(p[:, 1], p[:, 0], (n, n))
    m[rr, cc] = True
    lab = M.label((gray > 0.84) & (sat < 0.1) & m)
    if not lab.max():
        return np.nan
    lp = max(M.regionprops(lab), key=lambda q: q.area)
    if lp.area * (448 / n) ** 2 < 12 * (448 / 256) ** 2:   # same minimum lumen area (in um^2) as glands.py
        return np.nan
    return float(M.perimeter(lp.image) / max(M.perimeter(lp.image_convex), 1e-6))


def main(finder):
    find = CF.FINDERS[finder]
    rows = []
    files = [(c, f) for c in ("HP", "NORM") for f in sorted((SRC / c).glob("*.png"))]
    side = round(448 / MPP)
    for k, (cls, f) in enumerate(files, 1):
        im = Image.open(f).convert("RGB")
        o = (im.width - side) // 2
        native = im.crop((o, o, o + side, o + side))
        views = {s: (native if px is None else native.resize((px, px), Image.LANCZOS)) for s, px in SCALES.items()}
        for j, g in enumerate(find(views["2.0 um/px (MHIST)"])):
            r = {s: roughness(v, g["outline"]) for s, v in views.items()}
            rows.append({"cls": cls, "patch": f.name, "crypt": j, **r})
        if k % 50 == 0:
            print(f"{k}/{len(files)}", flush=True)
    d = pd.DataFrame(rows)
    res = {"what": __doc__.split("\n\n")[0], "finder": finder, "patches": {c: int(n) for c, n in d.groupby("cls").patch.nunique().items()},
           "scales": {}}
    for s in SCALES:
        x = d.dropna(subset=[s])
        y = x.cls == "HP"
        per = x.groupby(["cls", "patch"])[s].mean().reset_index()
        res["scales"][s] = {"crypts_with_lumen": len(x),
                            "auc_hp_vs_norm_per_crypt": round(float(roc_auc_score(y, x[s])), 3),
                            "auc_hp_vs_norm_per_field": round(float(roc_auc_score(per.cls == "HP", per[s])), 3),
                            "median_roughness_hp": round(float(x[y][s].median()), 3),
                            "median_roughness_norm": round(float(x[~y][s].median()), 3)}
    both = d.dropna(subset=list(SCALES))
    from scipy.stats import spearmanr
    res["same_crypt_rank_agreement_spearman"] = {
        "native vs 1.0": round(float(spearmanr(both[list(SCALES)[0]], both[list(SCALES)[1]])[0]), 3),
        "native vs 2.0": round(float(spearmanr(both[list(SCALES)[0]], both[list(SCALES)[2]])[0]), 3)}
    return res


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "classical"
    res = main(name)
    (Path(__file__).resolve().parent.parent / "docs/results" / f"serration_resolution_{name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
