"""Gland finder: outlines crypts in one field and measures them (no learned model).

A field (one MHIST-sized square, ~448 um) is resized to 256 px (~1.75 um/px) so screen snips and MHIST tiles are
measured at the same scale. In H&E a crypt is a pale interior (goblet-cell mucin + lumen) enclosed by a dark ring
of epithelial nuclei, so:
  - crypt interior = pale pixels merged across the thin cytoplasm between goblet cells, enclosed by tissue and
    ringed by haematoxylin-dark pixels (which rules out tears and glass);
  - gland outline  = the interior plus the nuclear layer (~12 um);
  - lumen          = the brightest, nearly white part of the interior (the luminal border is where serration is).
Each crypt is measured: interior and lumen size (um), lumen roughness (perimeter / convex-hull perimeter: a
saw-toothed, serrated border is rougher), solidity and eccentricity of the interior. Labels use thresholds from
the same measurements on MHIST training images (config/gland_thresholds.json, research/gland_thresholds.py).
They are measurements, not a validated detector.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

S = 256                     # working size of one field
UM_PER_PX = 448 / S         # ~1.75 um per working pixel
NUC_UM = 12                 # nuclear layer added around the interior to outline the gland
MIN_CRYPT_UM = 14           # smaller pale specks are single goblet cells, not crypts
THRESH_PATH = Path(__file__).resolve().parent / "config" / "gland_thresholds.json"


def _arrays(field: Image.Image):
    from skimage.color import rgb2hed
    a = np.asarray(field.convert("RGB").resize((S, S), Image.LANCZOS), dtype=np.float32) / 255.0
    gray, sat = a.mean(2), a.max(2) - a.min(2)
    return a, gray, sat, rgb2hed(a)[..., 0]


def measure(field: Image.Image) -> List[dict]:
    """All crypts in one field with their measurements and outline (coordinates 0..1 of the field)."""
    from skimage import measure as M, morphology, segmentation

    a, gray, sat, hema = _arrays(field)
    pale = (gray > 0.74) & (sat < 0.2)
    tissue = morphology.remove_small_objects(~((gray > 0.86) & (sat < 0.08)), 64)   # not bare glass
    interior = morphology.binary_closing(pale, morphology.disk(2))                   # merge goblet cells
    interior = morphology.remove_small_holes(interior, 40)
    interior = morphology.binary_opening(interior, morphology.disk(1))
    lab = M.label(interior)
    h_tissue = np.percentile(hema[tissue & ~interior], 55) if (tissue & ~interior).any() else np.inf
    min_area = np.pi * (MIN_CRYPT_UM / UM_PER_PX / 2) ** 2
    keep, props = np.zeros_like(lab), []
    for p in M.regionprops(lab):
        r0, c0, r1, c1 = p.bbox
        if r0 == 0 or c0 == 0 or r1 == S or c1 == S:        # cut by the field edge or open to glass
            continue
        if p.area < min_area or p.area > 0.15 * S * S:
            continue
        me = lab == p.label
        # nuclei sit at the base of the epithelium, up to ~20 um from a wide lumen: look in a band 4-18 um out
        ring = morphology.binary_dilation(me, morphology.disk(10)) & ~morphology.binary_dilation(me, morphology.disk(2))
        if tissue[ring].mean() < 0.8 or np.percentile(hema[ring], 75) < h_tissue:   # needs a dark nuclear ring
            continue
        keep[me] = len(props) + 1
        props.append(p)
    if not props:
        return []
    glands = segmentation.expand_labels(keep, distance=NUC_UM / UM_PER_PX) * tissue
    whiteish = (gray > 0.84) & (sat < 0.1)
    out = []
    for i, p in enumerate(props, 1):
        cs = M.find_contours(np.pad(glands == i, 1).astype(float), 0.5)
        if not cs:
            continue
        c = M.approximate_polygon(max(cs, key=len) - 1, tolerance=0.7)
        lum = M.label(whiteish & (keep == i))
        lp = max(M.regionprops(lum), key=lambda q: q.area) if lum.max() else None
        rough = float(M.perimeter(lp.image) / max(M.perimeter(lp.image_convex), 1e-6)) if lp is not None and lp.area >= 12 else None
        out.append({
            "outline": [[round(float(x) / S, 4), round(float(y) / S, 4)] for y, x in c],
            "center": [round(float(p.centroid[1]) / S, 4), round(float(p.centroid[0]) / S, 4)],
            "crypt_um": round(2 * np.sqrt(p.area / np.pi) * UM_PER_PX, 1),
            "lumen_um": round(2 * np.sqrt(lp.area / np.pi) * UM_PER_PX, 1) if lp is not None else 0.0,
            "roughness": round(rough, 3) if rough is not None else None,
            "solidity": round(float(p.solidity), 3),
            "eccentricity": round(float(p.eccentricity), 3),
        })
    return out


_T: Optional[dict] = None


def label(glands: List[dict]) -> List[dict]:
    """Name each crypt from its measurements (thresholds from MHIST training images); None = nothing clear."""
    global _T
    if _T is None:
        _T = json.loads(THRESH_PATH.read_text()) if THRESH_PATH.exists() else {}
    for g in glands:
        g["label"], g["points_to"] = None, None
        if not _T:
            continue
        if g["crypt_um"] >= _T["dilated_um"]:
            g["label"], g["points_to"] = "dilated crypt", "SSA"
        elif g["roughness"] is not None and g["roughness"] >= _T["serrated_roughness"]:
            g["label"], g["points_to"] = "serrated lumen", None   # HP and SSL are both serrated
        elif (g["solidity"] >= _T["round_solidity"] and g["eccentricity"] <= _T["round_eccentricity"]
              and g["crypt_um"] <= _T["round_max_um"]):
            g["label"], g["points_to"] = "round regular crypt", "HP"
    return glands
