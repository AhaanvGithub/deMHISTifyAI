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
