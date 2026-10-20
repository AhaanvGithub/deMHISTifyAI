"""Stain normalisation toward MHIST.

H&E colour differs between labs (dye shade, dye strength, scanner) while the tissue pattern does not: nuclei
take up haematoxylin, cytoplasm and mucin take eosin. Following Macenko et al. (2009), each image is split
into a stain matrix (the colour of each dye) and per-pixel dye concentrations (the pattern). Normalising keeps
the concentration pattern and repaints it with MHIST's dye colours at MHIST-like strength.

The reference is measured on MHIST itself (backend/config/stain_mhist.json, research/stain_reference.py), so
MHIST-like images barely change. Stain parameters should be estimated from all tissue on screen (per slide),
not from the snip alone: a mucin-rich SSL field genuinely differs in colour and must keep that difference.
"""

import json
from pathlib import Path

import numpy as np

REF_PATH = Path(__file__).resolve().parent / "config" / "stain_mhist.json"
IO = 240.0      # background intensity
BETA = 0.15     # optical density below this is background
ALPHA = 1.0     # percentile for robust stain-angle extremes


def _od(rgb):
    return -np.log((rgb.reshape(-1, 3).astype(np.float64) + 1) / IO)


def estimate(rgb, max_pixels=400_000, seed=0):
    """(stain matrix [3, 2] with columns H then E, robust max concentrations [2]) from an RGB image or pixel
    array; None if there is too little tissue."""
    od = _od(np.asarray(rgb))
    od = od[~np.any(od < BETA, axis=1)]
    if len(od) < 500:
        return None
    if len(od) > max_pixels:
        od = od[np.random.default_rng(seed).choice(len(od), max_pixels, replace=False)]
    _, vecs = np.linalg.eigh(np.cov(od.T))
    plane = vecs[:, 1:3]
    t = od @ plane
    phi = np.arctan2(t[:, 1], t[:, 0])
    lo, hi = np.percentile(phi, ALPHA), np.percentile(phi, 100 - ALPHA)
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    he = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T  # haematoxylin first
    he = he * np.sign(he.sum(0))  # optical densities are positive
    he = he / np.linalg.norm(he, axis=0)
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    return he, np.percentile(conc, 99, axis=1)


def load_reference():
    r = json.loads(REF_PATH.read_text())
    return np.array(r["stain_matrix"]), np.array(r["max_conc"])


def normalize(rgb, src, ref=None):
    """Repaint `rgb` from the source stain parameters `src` (from estimate(), ideally on the whole slide) to
    the MHIST reference `ref`."""
    he_src, max_src = src
    he_ref, max_ref = ref if ref is not None else load_reference()
    a = np.asarray(rgb)
    h, w = a.shape[:2]
    conc = np.linalg.lstsq(he_src, _od(a).T, rcond=None)[0]
    conc *= (max_ref / max_src)[:, None]
    out = IO * np.exp(-he_ref @ conc)
    return np.clip(out, 0, 255).T.reshape(h, w, 3).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------
# Alternatives compared against Macenko (research/stain_compare.py). All map toward MHIST's own reference
# (`ref` dict from stain_mhist.json); source statistics should come from the whole slide/screen.
# ---------------------------------------------------------------------------------------------------------

def _tissue_mask(rgb):
    return ~np.any(_od(np.asarray(rgb)) < BETA, axis=1)


def reinhard_stats(rgb):
    """mean and std of tissue pixels in CIE-Lab (colour averaging)"""
    from skimage.color import rgb2lab
    lab = rgb2lab(np.asarray(rgb).reshape(-1, 1, 3)).reshape(-1, 3)[_tissue_mask(rgb)]
    return lab.mean(0), lab.std(0) + 1e-6


def reinhard(rgb, src, ref):
    """Reinhard colour transfer: shift and scale each Lab channel of tissue pixels to MHIST's mean/std."""
    from skimage.color import lab2rgb, rgb2lab
    a = np.asarray(rgb)
    lab = rgb2lab(a)
    m = _tissue_mask(a).reshape(a.shape[:2])
    mu_s, sd_s = src
    lab[m] = (lab[m] - mu_s) / sd_s * np.array(ref["reinhard_std"]) + np.array(ref["reinhard_mean"])
    return (np.clip(lab2rgb(lab), 0, 1) * 255).astype(np.uint8)


def estimate_vahadane(rgb, n_pixels=8_000, sparsity=0.0, seed=0):
    """Vahadane et al. (2016) idea - stain colours from a NON-NEGATIVE factorisation of tissue optical density
    (dyes only add colour), solved with NMF so it is fast enough to run per image (~0.07 s). Vahadane's
    sparsity term is off by default: at optical-density scale even a tiny L1 penalty collapses one dye.
    Returns stain matrix [3, 2] (H, E) or None."""
    from sklearn.decomposition import NMF
    od = _od(np.asarray(rgb))
    od = od[~np.any(od < BETA, axis=1)]
    if len(od) < 500:
        return None
    od = od[np.random.default_rng(seed).choice(len(od), min(n_pixels, len(od)), replace=False)]
    nmf = NMF(n_components=2, init="nndsvda", alpha_W=sparsity, l1_ratio=1.0, max_iter=400, random_state=seed)
    nmf.fit(od)  # od ~ W (amounts, sparse) @ H (dye colours)
    he = nmf.components_.T  # [3, 2]
    he = he / (np.linalg.norm(he, axis=0) + 1e-9)
    if he[0, 0] < he[0, 1]:  # haematoxylin absorbs more red light: put it first
        he = he[:, ::-1]
    return he


def concentrations(rgb, he):
    """non-negative dye amounts [2, n_pixels] and the leftover optical density [3, n_pixels]"""
    od = _od(np.asarray(rgb)).T
    conc = np.clip(np.linalg.lstsq(he, od, rcond=None)[0], 0, None)
    return conc, od - he @ conc


def conc_stats(rgb, he):
    """per-dye median and 99th percentile of the amounts on tissue pixels"""
    conc, _ = concentrations(rgb, he)
    c = conc[:, _tissue_mask(rgb)]
    return np.percentile(c, 50, axis=1), np.percentile(c, 99, axis=1)


def vahadane(rgb, he_src, max_src, ref):
    """Vahadane separation, repaint with MHIST's (Vahadane) dye colours, match each dye's 99th percentile."""
    a = np.asarray(rgb)
    conc, _ = concentrations(a, he_src)
    conc *= (np.array(ref["vahadane_p99"]) / (max_src + 1e-9))[:, None]
    out = IO * np.exp(-np.array(ref["vahadane_matrix"]) @ conc)
    return np.clip(out, 0, 255).T.reshape(a.shape).astype(np.uint8)


def derived(rgb, he_src, src_stats, ref):
    """First-principles variant: non-negative (Vahadane) separation; dye amounts mapped piecewise-linearly so
    0, median and 99th percentile land on MHIST's (fixes pale AND dark slides); the leftover optical density
    (blood, pigment, folds: neither dye) is kept unchanged instead of being forced into H or E."""
    a = np.asarray(rgb)
    conc, resid = concentrations(a, he_src)
    (med_s, p99_s), med_r, p99_r = src_stats, np.array(ref["vahadane_median"]), np.array(ref["vahadane_p99"])
    for k in range(2):
        c = conc[k]
        lo = c <= med_s[k]
        c[lo] = c[lo] * (med_r[k] / (med_s[k] + 1e-9))
        c[~lo] = med_r[k] + (c[~lo] - med_s[k]) * ((p99_r[k] - med_r[k]) / (p99_s[k] - med_s[k] + 1e-9))
    out = IO * np.exp(-(np.array(ref["vahadane_matrix"]) @ conc + resid))
    return np.clip(out, 0, 255).T.reshape(a.shape).astype(np.uint8)
