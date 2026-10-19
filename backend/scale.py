"""Magnification matching via nucleus scale.

The judges were trained on MHIST tiles, each covering a fixed field of view. A researcher's
snip can be at any zoom, so a fixed 224-screen-pixel window may see far more or far less
tissue than the judges were trained on. Nuclei have a fairly stable physical size, so the
characteristic blob scale of the hematoxylin channel tells us the zoom relative to MHIST:

    tile_px = 224 * sigma(snip) / sigma(MHIST)

sigma = the scale at which a scale-normalised Laplacian-of-Gaussian responds most strongly to
hematoxylin blobs (nuclei). Scale-normalised LoG is zoom-covariant, so unlike blob counting it
doesn't break when nuclei merge after upscaling. Checked on MHIST test images resized by known
factors (research/measure_scale.py). The reference is stored in backend/config/scale.json.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

SCALE_PATH = Path(__file__).resolve().parent / "config" / "scale.json"
SIGMAS = np.geomspace(0.6, 48, 44)  # below ~0.6px the response is dominated by pixel noise


def char_sigma(img: Image.Image, max_side: int = 1600) -> float:
    from scipy import ndimage
    from skimage.color import rgb2hed

    img = img.convert("RGB")
    f = min(1.0, max_side / max(img.size))
    if f < 1.0:
        img = img.resize((int(img.width * f), int(img.height * f)), Image.BILINEAR)
    h = rgb2hed(np.asarray(img))[..., 0]
    h = (h - h.mean()) / (h.std() + 1e-6)
    resp = np.full(len(SIGMAS), -np.inf)
    for i, s in enumerate(SIGMAS):
        if s * 6 > min(h.shape):
            break
        resp[i] = np.percentile(-s * s * ndimage.gaussian_laplace(h, s), 99)
    i = int(np.argmax(resp))
    if 0 < i < len(SIGMAS) - 1 and np.isfinite(resp[i + 1]):  # parabolic refinement in log-sigma
        y0, y1, y2 = resp[i - 1 : i + 2]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-9)
        return float(np.exp(np.log(SIGMAS[i]) + d * np.log(SIGMAS[1] / SIGMAS[0]))) / f
    return float(SIGMAS[i]) / f


def tile_px_for(img: Image.Image, base: int = 224, lo: int = 64, hi: int = 1400):
    """Window size (in snip pixels) matching the MHIST field of view, plus diagnostics."""
    if not SCALE_PATH.exists():
        return base, {"auto": False, "reason": "no reference scale"}
    ref = json.loads(SCALE_PATH.read_text())["sigma"]
    scale = char_sigma(img) / ref
    raw = base * scale
    px = int(np.clip(round(raw), lo, hi))
    return px, {"auto": True, "zoom_vs_training": round(scale, 2), "tile_px": px, "clamped": px != round(raw)}
