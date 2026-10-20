"""Read a slide viewer's scale bar from a screenshot: microns per screen pixel.

1. Windows' built-in OCR finds labels such as "100 µm", "50 um", "0.5 mm" (OCR often misreads µ).
2. Next to each label, look for a scale bar: a thin, solid horizontal bar (dark, white or coloured) whose length
   is many times its height, centred over or beside the label.
3. microns per pixel = label value / bar length. The closest plausible bar wins.
Returns None when no labelled bar is found; the app then falls back to the tissue-colour square.
"""

import asyncio
import io
import re
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

# OCR misreads "µ" as u, p, j, w, y or d, or drops it; "mm" is millimetres
LABEL = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|[µμujpdwy]?m)\b", re.IGNORECASE)


async def _ocr_async(img: Image.Image):
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

    buf = io.BytesIO(); img.save(buf, format="PNG")
    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream)
    writer.write_bytes(buf.getvalue())
    await writer.store_async()
    writer.detach_stream()
    stream.seek(0)
    bmp = await (await BitmapDecoder.create_async(stream)).get_software_bitmap_async()
    engine = OcrEngine.try_create_from_user_profile_languages()
    res = await engine.recognize_async(bmp)
    out = []
    for line in res.lines:
        words = list(line.words)
        rects = [w.bounding_rect for w in words]
        out.append((line.text, [(w.text, (r.x, r.y, r.width, r.height)) for w, r in zip(words, rects)]))
    return out


def ocr_lines(img: Image.Image):
    return asyncio.run(_ocr_async(img))


def _labels(img: Image.Image, zoom: int = 3):
    """(microns, x, y, w, h) for each standalone scale label OCR finds in a small crop (read at `zoom`x).
    Only short lines that are just a label count, so "0.45 mm" inside a sentence is ignored."""
    big = img.resize((img.width * zoom, img.height * zoom), Image.LANCZOS)
    found = []
    for text, words in ocr_lines(big):
        m = LABEL.fullmatch(text.strip())
        if not m or not words:
            continue
        val = float(m.group(1).replace(",", "."))
        um = val * 1000 if m.group(2).lower() == "mm" else val
        x0 = min(r[0] for _, r in words); y0 = min(r[1] for _, r in words)
        x1 = max(r[0] + r[2] for _, r in words); y1 = max(r[1] + r[3] for _, r in words)
        found.append((um, x0 / zoom, y0 / zoom, (x1 - x0) / zoom, (y1 - y0) / zoom))
    return found


def _bars(a: np.ndarray):
    """Thin, solid horizontal bars: (x0, y0, length, height)."""
    from scipy import ndimage

    gray, sat = a.mean(2), a.max(2) - a.min(2)
    out = []
    for mask in (gray < 0.25, gray > 0.92, sat > 0.45):            # dark, white or strongly coloured bars
        lab, n = ndimage.label(mask)
        for sl in ndimage.find_objects(lab):
            if sl is None:
                continue
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if not (2 <= h <= 16 and w >= 20 and w >= 5 * h):
                continue
            fill = mask[sl].mean()
            if fill < 0.85:                                         # a solid bar, not text or a frame
                continue
            out.append((sl[1].start, sl[0].start, w, h))
    return out


def _text_only(crop: Image.Image) -> Image.Image:
    """Grey crop with coloured (tissue) pixels blanked to the background: Windows OCR gives up on a crop that
    contains stained tissue, while scale labels and bars are nearly always black or white."""
    a = np.asarray(crop, dtype=np.float32) / 255.0
    g = a.mean(2)
    coloured = (a.max(2) - a.min(2)) > 0.12
    bg = np.median(g[~coloured]) if (~coloured).any() else 1.0
    g[coloured] = bg
    # unstained tissue (white lumens, grey mucin) survives the colour test: blank every blob bigger than a letter
    # that isn't a thin bar
    from scipy import ndimage
    lab, _ = ndimage.label(np.abs(g - bg) > 0.2)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h > 30 and not (h <= 16 and w >= 5 * h):
            g[sl][lab[sl] == i] = bg
    return Image.fromarray((g * 255).astype(np.uint8)).convert("RGB")


def read_scale(img: Image.Image) -> Optional[dict]:
    """{"um_per_px", "label_um", "bar_px", "bar": [x0, y0, x1, y1]} in screenshot pixels, or None.
    Bars are found first; OCR then reads only a small, enlarged crop around each bar (fast, and Windows OCR is
    much more reliable on a small text-like crop than on a whole screen)."""
    img = img.convert("RGB")
    a = np.asarray(img, dtype=np.float32) / 255.0
    best = None
    for bx, by, bw, bh in sorted(_bars(a), key=lambda b: -b[2])[:12]:
        pad_x, pad_y = max(140, bw), max(40, 6 * bh)  # labels can sit beside the bar, not just above or below
        X0, Y0 = max(0, bx - pad_x), max(0, by - pad_y)
        X1, Y1 = min(img.width, bx + bw + pad_x), min(img.height, by + bh + pad_y)
        crop = _text_only(img.crop((X0, Y0, X1, Y1)))
        inv = ImageOps.invert(crop)  # viewers often draw white labels on dark overlays; OCR reads dark-on-light best
        labels = next((r for r in (_labels(c, z) for c in (crop, inv) for z in (3, 1)) if r), [])
        for um, lx, ly, lw, lh in labels:
            lx, ly = lx + X0, ly + Y0
            lcx, lcy = lx + lw / 2, ly + lh / 2
            bcx, bcy = bx + bw / 2, by + bh / 2
            beside = abs(bcy - lcy) < 1.5 * lh and (abs(bx + bw - lx) < 3 * lh or abs(lx + lw - bx) < 3 * lh)
            stacked = abs(bcy - lcy) < 4 * lh and bx - lh <= lcx <= bx + bw + lh
            if not (beside or stacked):
                continue
            d = abs(bcy - lcy) + 0.25 * abs(bcx - lcx)
            if best is None or d < best[0]:
                best = (d, um, bx, by, bw, bh)
    if best is None:
        return None
    _, um, bx, by, bw, bh = best
    return {"um_per_px": um / bw, "label_um": um, "bar_px": int(bw), "bar": [int(bx), int(by), int(bx + bw), int(by + bh)]}
