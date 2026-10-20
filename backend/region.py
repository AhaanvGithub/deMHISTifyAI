"""Find the tissue inside a screen snip and place the square field the models will score.

Tissue on an H&E slide is pink-to-purple (red and blue above green, clearly coloured); slide glass is near white and
viewer chrome is grey or dark. Inside the researcher's selection, tissue_square() returns the largest square that is
mostly tissue (>= 90%), positioned to cover as much tissue as possible. The models score one square field, as they
were trained, and the researcher can drag or resize the square.
"""

from typing import Tuple

import numpy as np
from PIL import Image

Box = Tuple[int, int, int, int]  # x0, y0, x1, y1 in snip pixels
MIN_TISSUE = 0.9                 # a square must be at least this much tissue


def tissue_mask(img: Image.Image, work: int = 256):
    """Boolean tissue mask at a working size, and the scale from snip pixels to it."""
    from scipy import ndimage

    w, h = img.size
    s = work / max(w, h)
    a = np.asarray(img.convert("RGB").resize((max(int(w * s), 8), max(int(h * s), 8))), dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    gray, sat = a.mean(2), a.max(2) - a.min(2)
    m = (sat > 0.07) & (gray > 0.15) & (gray < 0.93) & (r > g + 0.02) & (b > g - 0.02)   # stained (H&E hues)
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)                                   # lumens and pale mucin are tissue too
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.01 * m.size))   # drop specks
    return m, s


def tissue_square(img: Image.Image) -> Box:
    w, h = img.size
    m, s = tissue_mask(img)
    H, W = m.shape
    ii = np.pad(m.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))   # integral image
    full = min(W, H)
    if m.sum() == 0:
        side = full; y, x = (H - side) // 2, (W - side) // 2
    else:
        best = None
        for side in sorted({max(16, int(full * f)) for f in np.linspace(1.0, 0.3, 15)}, reverse=True):
            tot = ii[side:, side:] - ii[:-side, side:] - ii[side:, :-side] + ii[:-side, :-side]
            y, x = np.unravel_index(int(tot.argmax()), tot.shape)
            frac = tot[y, x] / side ** 2
            if best is None or frac > best[0]:
                best = (frac, side, y, x)
            if frac >= MIN_TISSUE:                                     # largest square that is mostly tissue
                best = (frac, side, y, x)
                break
        _, side, y, x = best
    x0, y0 = int(x / s), int(y / s)
    d = int(min(side / s, w - x0, h - y0))
    return (x0, y0, x0 + d, y0 + d)
