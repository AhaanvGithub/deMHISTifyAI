"""Gland finder: outlines crypts in one field and measures them.

find() is what the app uses: crypt outlines from the learned gland outliner (backend/gland_net.py; a U-Net trained
on expert GlaS gland outlines at MHIST's scale) when its weights are present, else from the classical finder
below. Both are measured and labelled the same way. On expert outlines (GlaS benign glands, out-of-patient) the
learned outliner finds glands at F1 0.89 (outline IoU 0.88) vs 0.36 (0.71) for the classical one, and a field
gives far more nearly the same crypts whether it comes from the raw tile or a screen snip
(docs/results/crypt_finder.md).

Classical finder (no learned model):

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
from types import SimpleNamespace
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
        g = _describe(glands == i, keep == i, p, whiteish, ring_share.get(i, 0))
        if g:
            out.append(g)
    return out


def _describe(gland: np.ndarray, interior: np.ndarray, p, whiteish: np.ndarray, ring: Optional[float],
              rough_too: bool = True) -> Optional[dict]:
    """Outline and measurements of one crypt: its gland mask, its pale interior (regionprops `p`), its ring share."""
    from skimage import measure as M
    cs = M.find_contours(np.pad(gland, 1).astype(float), 0.5)
    if not cs:
        return None
    c = M.approximate_polygon(max(cs, key=len) - 1, tolerance=0.7)
    lum = M.label(whiteish & interior)
    lp = max(M.regionprops(lum), key=lambda q: q.area) if lum.max() else None
    rough = (float(M.perimeter(lp.image) / max(M.perimeter(lp.image_convex), 1e-6))
             if rough_too and lp is not None and lp.area >= 12 else None)
    return {
        "outline": [[round(float(x) / S, 4), round(float(y) / S, 4)] for y, x in c],
        "center": [round(float(p.centroid[1]) / S, 4), round(float(p.centroid[0]) / S, 4)],
        "crypt_um": round(2 * np.sqrt(p.area / np.pi) * UM_PER_PX, 1),
        "width_um": round(float(p.axis_minor_length) * UM_PER_PX, 1),   # dilation = wide, not long
        "lumen_um": round(2 * np.sqrt(lp.area / np.pi) * UM_PER_PX, 1) if lp is not None else 0.0,
        "roughness": round(rough, 3) if rough is not None else None,
        "solidity": round(float(p.solidity), 3),
        "eccentricity": round(float(p.eccentricity), 3),
        "ring": round(float(ring), 2) if ring is not None else None,
    }


def measure_masks(field: Image.Image, masks: List[np.ndarray], fast: bool = False) -> List[dict]:
    """Same measurements for crypts outlined by another finder (boolean gland masks, any square size). The crypt
    interior is the pale part inside each gland, found exactly as in measure(); a gland with no pale interior
    (a collapsed, slit-like crypt) keeps its outline but gets no measurements and so no label.
    fast: skip what labels from the outline don't use (the nuclear-ring share and lumen roughness), ~2x faster."""
    from skimage import measure as M, morphology
    from skimage.transform import resize

    a, gray, sat, hema = _arrays(field)
    tissue = morphology.remove_small_objects(~((gray > 0.86) & (sat < 0.08)), max_size=63)
    g_med = np.median(gray[tissue]) if tissue.any() else 0.6
    white = max(np.percentile(gray, 99), g_med + 0.1)
    pale = (gray > g_med + 0.5 * (white - g_med)) & (sat < max(0.08, np.median(sat[tissue]) if tissue.any() else 0.2))
    pale = morphology.opening(morphology.remove_small_holes(morphology.closing(pale, morphology.disk(2)), max_size=39),
                              morphology.disk(1))
    whiteish = (gray > 0.84) & (sat < 0.1)
    gl = [resize(m.astype(np.float32), (S, S), order=1) >= 0.5 for m in masks]
    keep, props, idx = np.zeros((S, S), int), [], []
    for k, m in enumerate(gl):
        lab = M.label(pale & m)
        if lab.max():
            p = max(M.regionprops(lab), key=lambda q: q.area)
            keep[(lab == p.label) & (keep == 0)] = len(props) + 1
            props.append(p)
            idx.append(k)
    _, ring_share = _grow_to_gland_edge(keep, len(props), hema, tissue) if props and not fast else (None, {})
    out = []
    for k, m in enumerate(gl):
        if k in idx:
            i = idx.index(k) + 1
            g = _describe(m, keep == i, props[i - 1], whiteish, None if fast else ring_share.get(i, 0), not fast)
        else:
            cy, cx = np.argwhere(m).mean(0) if m.any() else (0, 0)
            g = _describe(m, np.zeros_like(m), SimpleNamespace(centroid=(cy, cx), area=0, axis_minor_length=0,
                                                               solidity=0, eccentricity=0), whiteish, None if fast else 0, not fast)
            if g:
                g.update(crypt_um=None, width_um=None, solidity=None, eccentricity=None)
        if g:
            gp = M.regionprops(m.astype(np.uint8))[0]
            g.update(gland_um=round(2 * np.sqrt(gp.area / np.pi) * UM_PER_PX, 1),
                     gland_width_um=round(float(gp.axis_minor_length) * UM_PER_PX, 1),
                     gland_solidity=round(float(gp.solidity), 3), gland_eccentricity=round(float(gp.eccentricity), 3))
            out.append(g)
    return out


_T: Optional[dict] = None
_LOCK = __import__("threading").Lock()
SHOW_SERRATED = False   # lumen roughness barely tells serrated from smooth crypts, even at 0.44 um/px
                        # (UniToPatho HP vs normal, AUC ~0.57-0.59 at every scale): not shown


def finder() -> str:
    import gland_net
    return "unet" if gland_net.available() else "classical"


def find(field: Image.Image) -> List[dict]:
    """Crypts in one field, outlined by the best available finder, measured and labelled."""
    name = finder()
    if name == "unet":
        import gland_net
        with _LOCK:                                   # one network, shared by the server's threads
            masks = gland_net.outlines(field)
        # labels from the crypt's own outline: on 1,575 held-out MHIST training images "dilated" tracks the
        # pathologists' SSL votes at Spearman 0.45 (0.32 from the pale interior, 0.18 with the classical finder)
        return label(measure_masks(field, masks, fast=True), thresholds("unet_g"))
    return label(measure(field))


def thresholds(finder: str = "classical") -> dict:
    """Label thresholds for a finder's crypts (each set from that finder's crypts on MHIST training images)."""
    path = THRESH_PATH if finder == "classical" else THRESH_PATH.with_name(f"gland_thresholds_{finder}.json")
    return json.loads(path.read_text()) if path.exists() else {}


def label(glands: List[dict], T: Optional[dict] = None) -> List[dict]:
    """Name each crypt from its measurements (thresholds from MHIST training images). A crypt can carry several
    tags (e.g. dilated and serrated); `label` is the main one, used for its colour. None = nothing clear."""
    global _T
    if T is None:
        if _T is None:
            _T = thresholds()
        T = _T
    # "dilated" means wider than the crypts around it: above the MHIST threshold AND >= 1.5x this field's median
    # width (when the field has 3+ clear crypts). On another lab's normal colon this halves false "dilated" calls
    # (14% -> 7% of crypts); on 1,575 held-out MHIST training images agreement is about the same (AUC 0.589 -> 0.582,
    # within noise). A judgement call: the labels explain the call, they don't make it.
    # basis "gland": measure the crypt's own outline (a learned finder's outlines are trustworthy); otherwise the
    # pale interior (the classical finder's outlines are not, so only crypts with a clear nuclear ring are named)
    by_gland = (T or {}).get("basis") == "gland"
    k = (lambda key: "gland_" + {"crypt_um": "um", "width_um": "width_um"}.get(key, key)) if by_gland else (lambda key: key)
    sure = (lambda g: g.get(k("width_um")) is not None) if by_gland else (lambda g: g.get("ring", 1) >= LABEL_RING)
    ringed = [g[k("width_um")] for g in glands if sure(g)]
    field_median = float(np.median(ringed)) if len(ringed) >= 3 else 0.0
    for g in glands:
        g["label"], g["points_to"], g["tags"] = None, None, []
        if not T or not sure(g):                                 # only name crypts measured with confidence
            continue
        if g[k("width_um")] >= T["dilated_width_um"] and g[k("width_um")] >= 1.5 * field_median:
            g["tags"].append("dilated")
        if SHOW_SERRATED and g["roughness"] is not None and g["roughness"] >= T["serrated_roughness"]:
            g["tags"].append("serrated")                         # HP and SSL are both serrated
        if (not g["tags"] and g[k("solidity")] >= T["round_solidity"] and g[k("eccentricity")] <= T["round_eccentricity"]
                and g[k("crypt_um")] <= T["round_max_um"]):
            g["tags"].append("round regular")
        if g["tags"]:
            main = g["tags"][0]
            g["label"] = {"dilated": "dilated crypt", "serrated": "serrated lumen", "round regular": "round regular crypt"}[main]
            g["points_to"] = {"dilated": "SSA", "serrated": None, "round regular": "HP"}[main]
    return glands
