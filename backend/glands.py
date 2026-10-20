"""Gland finder: outlines crypts in one field and measures them (no learned model).

A field (one MHIST-sized square, ~448 um) is resized to 256 px (~1.75 um/px) so screen snips and MHIST tiles are
measured at the same scale. In H&E a crypt is a pale interior (goblet-cell mucin + lumen) enclosed by a dark ring
of epithelial nuclei, so:
  - crypt interior = pale pixels merged across the thin cytoplasm between goblet cells, enclosed by tissue and
    ringed by haematoxylin-dark pixels (which rules out tears and glass);
  - gland outline  = the interior grown out to the outer side of its band of nuclei (per direction, <= 40 um);
  - lumen          = the brightest, nearly white part of the interior (the luminal border is where serration is).
Each crypt is measured: interior size and width (um; dilation means wide, not long), lumen size, lumen roughness (perimeter / convex-hull perimeter: a
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
MAX_EPI_UM = 40             # the gland edge is searched up to this far out from the crypt interior
MIN_RING = 0.7             # share of directions that must show a clear band of nuclei
LABEL_RING = 0.85          # ... and a label needs an even clearer ring
MIN_CRYPT_UM = 14           # smaller pale specks are single goblet cells, not crypts
THRESH_PATH = Path(__file__).resolve().parent / "config" / "gland_thresholds.json"


def _arrays(field: Image.Image):
    from skimage.color import rgb2hed
    a = np.asarray(field.convert("RGB").resize((S, S), Image.LANCZOS), dtype=np.float32) / 255.0
    gray, sat = a.mean(2), a.max(2) - a.min(2)
    return a, gray, sat, rgb2hed(a)[..., 0]


def _grow_to_gland_edge(keep: np.ndarray, n: int, hema: np.ndarray, tissue: np.ndarray):
    """Grow each crypt interior out to the gland's edge: the outer side of its band of epithelial nuclei.
    Each tissue pixel belongs to its nearest crypt. Per crypt and per 24 angular sectors, the smoothed
    haematoxylin is followed outward with distance: the nuclear band is its peak within MAX_EPI_UM, and the edge
    is where it falls halfway back to the stroma level beyond. Tall epithelium (nuclei far from a wide lumen)
    and thin epithelium both get their own edge. Also returns, per crypt, the share of directions with a clear
    nuclear band: a real crypt is ringed all the way round, a pale gap in stroma is not."""
    from scipy import ndimage as ndi
    from skimage import morphology

    hs = ndi.gaussian_filter(hema, 1.5)
    strong = np.percentile(hs[tissue], 65) if tissue.any() else np.inf   # a "clear" nuclear band is above this
    ring_share = {}
    dist, (iy, ix) = ndi.distance_transform_edt(keep == 0, return_indices=True)
    owner = keep[iy, ix]
    dmax = int(MAX_EPI_UM / UM_PER_PX)
    out = np.zeros_like(keep)
    yy, xx = np.mgrid[0:S, 0:S]
    for i in range(1, n + 1):
        me = keep == i
        cy, cx = np.argwhere(me).mean(0)
        band = (owner == i) & (dist <= dmax + 6) & tissue
        if not band.any():
            out[me] = i
            continue
        ang = (np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi) * 24
        sec = np.minimum(ang.astype(int), 23)
        grow = me.copy()
        seen = clear = 0
        for k in range(24):
            px = band & (sec == k)
            if px.sum() < 6:
                continue
            seen += 1
            d = np.round(dist[px]).astype(int)
            prof = np.bincount(d, weights=hs[px], minlength=dmax + 8) / np.maximum(np.bincount(d, minlength=dmax + 8), 1)
            lo = max(1, int(3 / UM_PER_PX))
            peak = lo + int(np.argmax(prof[lo : dmax + 1]))
            clear += prof[peak] >= strong
            base = np.median(prof[dmax : dmax + 7]) if prof[dmax : dmax + 7].any() else prof[peak] * 0.5
            half = base + 0.5 * (prof[peak] - base)
            after = np.flatnonzero(prof[peak : dmax + 1] < half)
            edge = peak + (int(after[0]) if after.size else dmax - peak)
            grow |= px & (dist <= edge)
        grow = morphology.opening(morphology.closing(grow, morphology.disk(2)), morphology.disk(1))
        grow = ndi.binary_fill_holes(grow) & ((owner == i) | me)
        out[grow & (out == 0)] = i
        ring_share[i] = clear / max(seen, 1)
    return out, ring_share


def measure(field: Image.Image) -> List[dict]:
    """All crypts in one field with their measurements and outline (coordinates 0..1 of the field)."""
    from skimage import measure as M, morphology, segmentation

    a, gray, sat, hema = _arrays(field)
    tissue = morphology.remove_small_objects(~((gray > 0.86) & (sat < 0.08)), max_size=63)   # not bare glass
    # "pale" is relative to this slide's staining (light and dark labs differ): brighter than halfway from the
    # tissue's typical brightness to white, and no more coloured than typical tissue
    g_med = np.median(gray[tissue]) if tissue.any() else 0.6
    white = max(np.percentile(gray, 99), g_med + 0.1)
    pale = (gray > g_med + 0.5 * (white - g_med)) & (sat < max(0.08, np.median(sat[tissue]) if tissue.any() else 0.2))
    interior = morphology.closing(pale, morphology.disk(2))                   # merge goblet cells
    interior = morphology.remove_small_holes(interior, max_size=39)
    interior = morphology.opening(interior, morphology.disk(1))
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
        ring = morphology.dilation(me, morphology.disk(10)) & ~morphology.dilation(me, morphology.disk(2))
        if tissue[ring].mean() < 0.8 or np.percentile(hema[ring], 75) < h_tissue:   # needs a dark nuclear ring
            continue
        keep[me] = len(props) + 1
        props.append(p)
    if not props:
        return []
    glands, ring_share = _grow_to_gland_edge(keep, len(props), hema, tissue)
    whiteish = (gray > 0.84) & (sat < 0.1)
    out = []
    for i, p in enumerate(props, 1):
        if ring_share.get(i, 0) < MIN_RING:                     # not ringed by epithelium: stroma or a tear
            continue
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
            "width_um": round(float(p.axis_minor_length) * UM_PER_PX, 1),   # dilation = wide, not long
            "lumen_um": round(2 * np.sqrt(lp.area / np.pi) * UM_PER_PX, 1) if lp is not None else 0.0,
            "roughness": round(rough, 3) if rough is not None else None,
            "solidity": round(float(p.solidity), 3),
            "eccentricity": round(float(p.eccentricity), 3),
            "ring": round(float(ring_share.get(i, 0)), 2),
        })
    return out


_T: Optional[dict] = None


def label(glands: List[dict]) -> List[dict]:
    """Name each crypt from its measurements (thresholds from MHIST training images). A crypt can carry several
    tags (e.g. dilated and serrated); `label` is the main one, used for its colour. None = nothing clear."""
    global _T
    if _T is None:
        _T = json.loads(THRESH_PATH.read_text()) if THRESH_PATH.exists() else {}
    for g in glands:
        g["label"], g["points_to"], g["tags"] = None, None, []
        if not _T or g.get("ring", 1) < LABEL_RING:              # only name crypts with a clear, continuous ring
            continue
        if g["width_um"] >= _T["dilated_width_um"]:
            g["tags"].append("dilated")
        if g["roughness"] is not None and g["roughness"] >= _T["serrated_roughness"]:
            g["tags"].append("serrated")                         # HP and SSL are both serrated
        if (not g["tags"] and g["solidity"] >= _T["round_solidity"] and g["eccentricity"] <= _T["round_eccentricity"]
                and g["crypt_um"] <= _T["round_max_um"]):
            g["tags"].append("round regular")
        if g["tags"]:
            main = g["tags"][0]
            g["label"] = {"dilated": "dilated crypt", "serrated": "serrated lumen", "round regular": "round regular crypt"}[main]
            g["points_to"] = {"dilated": "SSA", "serrated": None, "round regular": "HP"}[main]
    return glands
