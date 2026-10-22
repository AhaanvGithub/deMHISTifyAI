"""Read QuPath's scale bar from a screenshot: microns per screen pixel.

QuPath draws its scale bar in the viewer's bottom-left corner: a thin bar with a label such as "100 µm" centred
under it, both black at about 50% opacity, so they halve the brightness of whatever lies behind (white panel or
stained tissue). Dividing the corner by a local estimate of the background (a grey closing, which fills in thin
dark lines and strokes) turns the bar and the label into the same clean dark marks on any background:
1. The bar: a run of rows darkened by the same factor in every colour channel, with sharp ends.
2. The label: Windows' built-in OCR reads the background-divided crop under the bar (OCR often misreads µ).
3. microns per pixel = label value / bar length, if the label is centred on the bar.
Only the bottom-left corner is searched. Returns None when no labelled bar is found.
"""


import asyncio
import io
import re
from typing import Optional

import numpy as np
from PIL import Image

# A number, then the unit. OCR garbles "µm" in many ways ("um", "pm", "urn", "prn", "1.1m", or drops it), but reads
# "mm" reliably: anything other than "mm" (up to 4 characters ending in m or n, or nothing) is taken as microns.
# A label only counts under a bar, and the resulting scale must be plausible (see read_scale).
LABEL = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|[^\s\d]{0,3}[mn]|)", re.IGNORECASE)


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


def _round_value(um: float) -> bool:
    """Scale bars use round lengths: 1, 2, 2.5 or 5 times a power of ten (10, 20, 25, 50, 100, 200, 250, 500 um, ...).
    Anything else is treated as an OCR slip; a wrong scale is worse than none (the app then uses the tissue square)."""
    m = um / 10 ** np.floor(np.log10(um))
    return any(abs(m - v) < 1e-6 for v in (1, 2, 2.5, 5, 10))


def _normalise(text: str) -> str:
    """Undo OCR's usual slips in a small label's number: letters read for digits ("1OO", "S0") and a number split
    in two ("1 00")."""
    toks = text.strip().split()
    digitish = lambda t: any(c.isdigit() for c in t) and all(c.isdigit() or c in "OoQDlI|!SsZz.," for c in t)
    while len(toks) > 1 and toks[0].isdigit() and toks[1].isdigit():
        toks[:2] = [toks[0] + toks[1]]
    if toks and digitish(toks[0]):
        toks[0] = toks[0].translate(str.maketrans("OoQDlI|!SsZz", "000011115522"))
    return " ".join(toks)


OCR_MAX = 4000   # Windows OCR refuses images over ~10000 px a side (OcrEngine.MaxImageDimension); stay well under


def _labels(img: Image.Image, zoom: int = 3):
    """(microns, x, y, w, h, has unit, number box (x0, y0, x1, y1)) for each standalone scale label OCR finds in a
    small crop (read at `zoom`x).
    Only short lines that are just a label count, so "0.45 mm" inside a sentence is ignored."""
    zoom = max(1, min(zoom, OCR_MAX // max(img.width, img.height, 1)))
    big = img.resize((img.width * zoom, img.height * zoom), Image.LANCZOS)
    found = []
    try:
        lines = ocr_lines(big)
    except OSError:          # one unreadable crop (e.g. a window-wide toolbar edge) must not end the whole search
        return []
    for text, words in lines:
        m = LABEL.fullmatch(_normalise(text))
        if not m or not words:
            continue
        num = m.group(1).replace(",", ".")
        if len(num) > 1 and num[0] == "0" and "." not in num:     # "05" is "0.5" with the point lost by OCR
            num = "0." + num[1:]
        val = float(num)
        um = val * 1000 if m.group(2).lower() == "mm" else val
        if um <= 0 or not _round_value(um):                       # e.g. "300" is most likely a misread "500"
            continue
        x0 = min(r[0] for _, r in words); y0 = min(r[1] for _, r in words)
        x1 = max(r[0] + r[2] for _, r in words); y1 = max(r[1] + r[3] for _, r in words)
        nums = [r for t, r in words if any(c.isdigit() for c in t)] or [words[0][1]]   # the number's own box
        nbox = (min(r[0] for r in nums) / zoom, min(r[1] for r in nums) / zoom,
                max(r[0] + r[2] for r in nums) / zoom, max(r[1] + r[3] for r in nums) / zoom)
        found.append((um, x0 / zoom, y0 / zoom, (x1 - x0) / zoom, (y1 - y0) / zoom, bool(m.group(2)), nbox))
    return found



CORNER = (0.35, 0.15)   # bottom-left search area: left 35% x bottom 15% of the screen
CENTRED = 0.05          # the label is centred under its bar within 5% of the bar length (+3 px)


def _darkening(a: np.ndarray, size):
    """Per pixel: (least, most) channel ratio of the image to its grey closing over `size` (rows, cols), and the
    closing's brightness. The closing fills in dark marks thinner than the window, so a see-through black overlay
    gives the same ratio (QuPath: ~0.5) in every channel whatever lies behind it, while stained tissue detail
    darkens the channels unevenly."""
    from scipy import ndimage
    cl = np.stack([ndimage.grey_closing(a[..., c], size=size) for c in range(3)], 2)
    r = (a + 1 / 255) / (cl + 1 / 255)
    return r.min(2), r.max(2), cl.max(2)


def _grow(on: np.ndarray, off: np.ndarray, x: int, step: int):
    """Last bar column going from `x` in direction `step`, through bar ("on") and undecided columns (where the bar
    crosses something too dark to measure); the bar ends where at least 2 of the next 4 columns are clearly
    undarkened and none is "on". None if it runs into the crop edge or the end is unclear (more than 2 undecided
    columns between the last "on" one and the end)."""
    last, n = x, len(on)
    while 0 <= x + step < n:
        x += step
        if on[x]:
            last = x
        elif off[x]:
            ahead = [j for j in range(x, x + 4 * step, step) if 0 <= j < n]
            if off[ahead].sum() >= 2 and not on[ahead].any():
                return last if abs(x - last) - 1 <= 2 else None
    return None


def _bar_candidates(a: np.ndarray):
    """(x0, y0, length, height, ratio) of see-through dark bars in the corner crop `a`, nearest the corner first."""
    from scipy import ndimage
    rmin, rmax, bg = _darkening(a, (13, 1))                       # bars up to ~12 px thick
    m = (rmax < 0.75) & (rmax - rmin < 0.12) & (bg > 0.1)        # evenly darkened, and not on a black background
    # only unbroken horizontal runs of >= 30 px (gaps of a few px bridged), so tissue touching the bar falls away
    lab, _ = ndimage.label(ndimage.binary_opening(ndimage.binary_closing(m, np.ones((1, 5), bool)), np.ones((1, 30), bool)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        # the bar's rows: the longest stack of rows each about as long as the typical row (one row may run on into
        # tissue as dark as the bar)
        counts = (lab[sl] == i).sum(1)
        good = counts >= 0.5 * np.median(counts[counts >= 30])
        runs = np.split(np.flatnonzero(good), np.flatnonzero(np.diff(np.flatnonzero(good)) > 1) + 1)
        rows = max(runs, key=len)
        if not 1 <= rows.size <= 12:
            continue
        y0, bh = int(rows[0]) + sl[0].start, int(rows.size)
        prof = np.median(rmax[y0:y0 + bh], 0)                    # per column: how much the bar's rows are darkened
        spread = np.median(rmax[y0:y0 + bh] - rmin[y0:y0 + bh], 0)
        cols = np.flatnonzero(m[y0:y0 + bh, sl[1]].mean(0) >= 0.75) + sl[1].start
        if cols.size < 20:
            continue
        rb = float(np.median(prof[cols]))
        if y0 < 1 or y0 + bh >= a.shape[0]:
            continue
        # two tests per column: the bar's rows against the closing (fails where the tissue behind changes over the
        # bar's height), and the bar's own top and bottom edges, a one-pixel step down by the bar's ratio
        top = (a[y0] + 1 / 255) / (a[y0 - 1] + 1 / 255)
        bot = (a[y0 + bh - 1] + 1 / 255) / (a[y0 + bh] + 1 / 255)
        # ratios of dark pixels are coarse (8-bit levels): widen the tolerances where the background is dark
        q = 2 / (255 * np.maximum(np.minimum(a[y0 - 1].mean(1), a[y0 + bh].mean(1)), 0.02))
        edge_on = ((np.abs(np.median(top, 1) - rb) < 0.12 + q) & (np.abs(np.median(bot, 1) - rb) < 0.12 + q)
                   & (np.ptp(top, 1) < 0.15 + 2 * q) & (np.ptp(bot, 1) < 0.15 + 2 * q))
        edge_off = (np.median(top, 1) > rb + 0.3) & (np.median(bot, 1) > rb + 0.3)
        # the edges decide at the ends: tissue darker in the middle of the bar's rows can pass the closing test
        on = ((np.abs(prof - rb) < 0.1 + q) & (spread < 0.12 + 2 * q) & ~edge_off) | edge_on
        off = edge_off                                            # no step at either edge: past the bar's end
        seed = int(cols[len(cols) // 2])
        x0, x1 = _grow(on, off, seed, -1), _grow(on, off, seed, 1)
        if x0 is None or x1 is None or x1 - x0 + 1 < 30 or on[x0:x1 + 1].mean() < 0.8:
            continue
        out.append((x0, y0, x1 - x0 + 1, bh, rb))
    out = list({c[:3]: c for c in out}.values())                  # pieces of one bar grow to the same bar
    out.sort(key=lambda b: b[0] + (a.shape[0] - b[1]))
    return out


TEXT_SPREAD = 0.15   # a see-through black overlay darkens all channels equally, even at a letter's soft edges


def _text_mask(a: np.ndarray, rb: float, k: int):
    """(soft, ink): soft is dark text on white, from pixels darkened evenly in all channels (as the bar is)
    relative to the background estimated by a k x k closing, with unevenly darkened pixels (stained tissue detail)
    blanked; ink keeps only the dark blobs that contain pixels darkened by the bar's own ratio `rb`, which drops
    most tissue clutter around the label."""
    from scipy import ndimage
    rmin, rmax, _ = _darkening(a, (k, k))
    soft = np.clip((rmax - rb + 0.05) / (0.95 - rb), 0, 1)
    soft[rmax - rmin > TEXT_SPREAD] = 1
    core = (np.abs(rmax - rb) < 0.06) & (rmax - rmin < 0.05)
    lab, _ = ndimage.label(soft < 0.75, np.ones((3, 3)))
    ink = np.isin(lab, np.unique(lab[core])) & (lab > 0)
    return soft, ink


def _img(v: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def _label_under(a: np.ndarray, bx: int, by: int, bw: int, bh: int, rb: float):
    """(microns, label centre x) read under a bar, or None. Two readers on the background-divided crop: glyph
    matching against QuPath's font (_glyph_read) and Windows OCR at a few zooms (_ocr_read). The glyph reading is
    taken unless OCR reads anything else; without one, OCR's reading must fit the label's ink."""
    H, W = a.shape[:2]
    cx = bx + bw / 2
    hw = max(bw / 2, 16 * bh) + 8
    k = max(5, 2 * bh + 1)
    X0, X1 = max(0, int(cx - hw)), min(W, int(cx + hw))
    Y0, Y1 = min(H, by + bh + 1), min(H, by + bh + 1 + max(20, 12 * bh))
    if Y1 - Y0 < 8:
        return None
    px, py = max(0, X0 - k), max(0, Y0 - k)                       # the closing needs context around the crop
    soft, ink = _text_mask(a[py:Y1 + k, px:X1 + k], rb, k)
    soft, ink = soft[Y0 - py:Y1 - py, X0 - px:X1 - px], ink[Y0 - py:Y1 - py, X0 - px:X1 - px]
    glyphs = _glyph_read(soft, ink, bh)
    votes = _ocr_read(soft, ink, bh, glyphs[0] if glyphs else None)
    if glyphs:
        um, centre = glyphs
        if set(votes) <= {um}:
            return um, X0 + centre
        return None                                               # OCR reads something else: no guess
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda kv: -len(kv[1]))
    um, boxes = ranked[0]
    if len(ranked) > 1 and 2 * len(ranked[1][1]) >= len(boxes):
        return None                                               # readings disagree: no guess
    for nbox in boxes:
        centre = _check_number(ink, um, nbox, soft, lenient=len(boxes) >= 2)   # OCR agreed with itself
        if centre is not None:
            return um, X0 + centre
    return None


def _ocr_read(soft: np.ndarray, ink: np.ndarray, bh: int, expect: Optional[float] = None) -> dict:
    """{microns: [number boxes]} read by Windows OCR from a few views and zooms of the label crop; stops once
    `expect` is read, or a value is read twice and nothing else is."""
    views = (_img(np.where(ink, soft, 1)), _img(soft), _img(np.where(ink & (soft < 0.6), 0, 1)))
    zooms = sorted({max(2, round(z / bh)) for z in (16, 24, 12)}, key=lambda z: abs(z - 16 / bh))
    votes = {}
    for v in views:
        for z in zooms:
            for um, *_, nbox in _labels(v, z):
                votes.setdefault(um, []).append(nbox)
            top = sorted((len(bs) for bs in votes.values()), reverse=True) + [0]
            if expect in votes or top[0] >= 3 or (top[0] >= 2 and top[1] == 0):
                break
        else:
            continue
        break
    # OCR drops a digit more often than it adds one ("100" -> "10"): a value 10x another one read wins over it
    for um in sorted(votes):
        if um * 10 in votes:
            votes[um * 10] += votes.pop(um)
    return votes


GLYPHS = "0125\u00b5m"   # all QuPath's labels need: 1, 2 or 5 times a power of ten, in um or mm
_GLYPH_CACHE = None


def _glyph_templates() -> dict:
    """{glyph: ink image} in the label font at 60 px, cropped to the ink."""
    global _GLYPH_CACHE
    if _GLYPH_CACHE is None:
        from PIL import ImageDraw, ImageFont
        font = ImageFont.truetype(LABEL_FONT, 60)
        out = {}
        for c in GLYPHS:
            t = Image.new("L", (120, 120), 0)
            ImageDraw.Draw(t).text((20, 90), c, fill=255, font=font, anchor="ls")
            g = np.asarray(t, np.float32) / 255
            ys, xs = np.nonzero(g > 0.3)
            out[c] = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        _GLYPH_CACHE = out
    return _GLYPH_CACHE


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def _norm(g: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).resize((20, 28), Image.BILINEAR),
                      np.float32) / 255


def _digit(g: np.ndarray) -> Optional[str]:
    """Which of 0, 1, 2, 5 the glyph image `g` (ink, cropped) is in QuPath's font, or None if not clearly one:
    its shape must match one template clearly better than the others, and its width/height ratio must fit (a
    "2" that lost its left half to the tissue filter can look like a "1" once stretched to the template size)."""
    tpl = _glyph_templates()
    sc = sorted(((_ncc(_norm(g), _norm(tpl[d])), d) for d in "0125"), reverse=True)
    best = sc[0][1]
    aspect = (g.shape[1] / g.shape[0]) / (tpl[best].shape[1] / tpl[best].shape[0])
    if sc[0][0] < 0.55 or sc[0][0] - sc[1][0] < 0.15 or not 0.75 <= aspect <= 1.33:
        return None
    return best


def _glyph_read(soft: np.ndarray, ink: np.ndarray, bh: int):
    """(microns, label centre x) from the label's glyphs, or None. The number: a run of glyphs as tall as each
    other on one baseline, close together, each matching one of 0, 1, 2, 5 in QuPath's font clearly better than
    the others, reading 1, 2 or 5 followed by zeros. The unit: the ink after a word space, as wide as "µm" or "mm"
    in the font; "µm" if its first glyph hangs below the baseline. Exactly one number must fit."""
    from scipy import ndimage
    try:
        tpl = _glyph_templates()
        unit_w = {u: _ink_size(u)[0] / _ink_size("0")[1] for u in ("\u00b5m", "mm")}   # per digit height
        glyph_w = {g: _ink_size(g)[0] / _ink_size("0")[1] for g in ("\u00b5", "m")}
    except OSError:
        return None                                               # no font
    lab, _ = ndimage.label(ink, np.ones((3, 3)))
    comps = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == i
        if m.sum() >= 6 and 2 * bh <= m.shape[0] <= 7 * bh:
            comps.append((sl[1].start, sl[1].stop, sl[0].start, sl[0].stop, np.where(m, 1 - soft[sl], 0)))
    comps.sort(key=lambda c: c[0])
    found = set()
    for i0 in range(len(comps)):
        c0 = comps[i0]
        base, h = c0[3], c0[3] - c0[2]
        if (c0[1] - c0[0]) > 1.1 * h:
            continue                                              # digits are taller than wide
        same = lambda c: abs(c[3] - base) <= 1 and abs((c[3] - c[2]) - h) <= max(1, 0.12 * h)
        # the run must start here: nothing digit-like just before it
        if any(same(c) and 0 <= c0[0] - c[1] <= 0.45 * h for c in comps[:i0]):
            continue
        number = [c0]
        for c in comps[i0 + 1:]:
            gap = c[0] - number[-1][1]
            if gap > 0.45 * h:
                break
            if same(c) and gap >= -1:
                number.append(c)
        digits = ""
        for c in number:
            d = _digit(c[4])
            if d is None:
                break
            digits += d
        if len(digits) != len(number) or digits[0] == "0" or digits[1:].strip("0"):
            continue                                              # QuPath: 1, 2 or 5 followed by zeros
        # the unit: ink columns after a word space, in the rows from digit top to descender bottom
        x_end = number[-1][1]
        rows = ink[max(0, base - h):min(ink.shape[0], base + int(0.6 * h) + 1)]
        cols = rows.any(0)
        start = next((x for x in range(x_end + 1, min(len(cols), x_end + int(0.9 * h) + 1)) if cols[x]), None)
        if start is None or start - x_end < 0.35 * h:
            continue
        stop = start
        while stop + 1 < len(cols) and cols[stop + 1:stop + 2 + int(0.35 * h)].any():
            stop += 1
        width = (stop - start + 1) / h
        if ink[max(0, base - h):base - int(0.85 * h), start:stop + 1].any():
            continue                                              # "µm" and "mm" have no ascenders: that's a digit
        # which unit: the first glyph's width (the micro sign is narrow, "m" wide) and what lies under the baseline
        # at the unit's start (the micro sign's tail, dark in the soft view where the ink mask lost it); they must
        # not disagree, and one must say it clearly
        word = cols[start:stop + 1]
        first = np.flatnonzero(~word)
        w1 = (first[0] if first.size else len(word)) / h          # up to the first gap inside the unit word
        below = soft[base + 1 + int(0.15 * h):min(soft.shape[0], base + int(0.6 * h) + 1),
                     start:start + max(1, int(0.45 * (stop - start + 1)))]
        tail = (below < 0.5).any(1).sum() >= max(2, 0.15 * h)
        by_width = "\u00b5m" if w1 < glyph_w["\u00b5"] * 1.2 else ("mm" if abs(w1 / glyph_w["m"] - 1) < 0.15 else None)
        if by_width == "mm" and tail or by_width is None and not tail:
            continue
        unit = by_width or "\u00b5m"
        if abs(width / unit_w[unit] - 1) > 0.2:
            continue
        found.add((float(digits) * (1 if unit != "mm" else 1000), (number[0][0] + stop + 1) / 2))
    if len({f[0] for f in found}) != 1:
        return None
    return min(found)


LABEL_FONT = "C:/Windows/Fonts/segoeui.ttf"   # QuPath on Windows draws its labels in Segoe UI


def _ink_size(text: str, size: float = 60):
    """(width, height) of `text`'s ink in the label font."""
    from PIL import ImageDraw, ImageFont
    font = ImageFont.truetype(LABEL_FONT, size)
    t = Image.new("L", (int(font.getlength(text)) + 20, int(2 * size)), 0)
    ImageDraw.Draw(t).text((10, 1.5 * size), text, fill=255, font=font, anchor="ls")
    x0, y0, x1, y1 = t.point(lambda p: 255 if p > 100 else 0).getbbox()
    return x1 - x0, y1 - y0


def _check_number(ink: np.ndarray, um: float, nbox, soft: np.ndarray = None, lenient: bool = False) -> Optional[float]:
    """Centre x of the whole label if the number OCR read fits the label's ink, else None. The number's ink (inside
    OCR's box for it) must be as wide for its digit height as the number rendered in QuPath's font, and closer to
    it than to the same leading digits with one zero more or less (OCR drops or adds a "0": "100" -> "10"). The
    label's centre is then predicted from where the number starts and the whole label's rendered width, so it
    does not depend on OCR boxing the unit too."""
    # OCR may box only part of the number ("50" of "500"): look a digit height either side of its box, and take
    # every digit that runs on from the ones inside it
    reach = int(np.ceil(nbox[3] - nbox[1]))
    x0, y0, x1, y1 = (int(np.floor(nbox[0])) - reach, int(np.floor(nbox[1])) - 2,
                      int(np.ceil(nbox[2])) + reach, int(np.ceil(nbox[3])) + 2)
    box = ink[max(0, y0):y1, max(0, x0):x1]
    # the digits' rows: the longest run of rows with a fair share of ink (a speck above or below has little)
    count = box.sum(1)
    dense = np.flatnonzero(count >= 0.2 * count.max()) if count.max() else np.zeros(0, int)
    if dense.size < 5:
        return None
    runs = np.split(dense, np.flatnonzero(np.diff(dense) > 2) + 1)
    rows = max(runs, key=len)
    h = rows[-1] - rows[0] + 1
    # the digits' columns: clusters of ink columns (split at gaps wider than a digit gap) that reach into OCR's box
    on = np.flatnonzero(box[rows[0]:rows[-1] + 1].any(0))
    if on.size < 2:
        return None
    clusters = np.split(on, np.flatnonzero(np.diff(on) > max(2, 0.3 * h)) + 1)
    inner = (nbox[0] - max(0, x0), nbox[2] - max(0, x0))
    clusters = [c for c in clusters if c[-1] >= inner[0] + 1 and c[0] <= inner[1] - 1]
    if not clusters:
        return None
    cols = np.r_[clusters[0][0], clusters[-1][-1]]
    w = cols[-1] - cols[0] + 1
    if soft is not None:                                          # OCR confuses 1, 2 and 5 at this size: the first
        from scipy import ndimage                                 # glyph must be the leading digit read
        band = box[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        lab, _ = ndimage.label(band, np.ones((3, 3)))
        tall = [(sl[1].start, i) for i, sl in enumerate(ndimage.find_objects(lab), 1)
                if sl is not None and sl[0].stop - sl[0].start >= 0.8 * h]
        if not tall:
            return None
        i = min(tall)[1]
        sl = ndimage.find_objects(lab)[i - 1]
        sb = soft[max(0, y0):y1, max(0, x0):x1][rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        g = np.where(lab[sl] == i, 1 - sb[sl], 0)
        lead = f"{um / 1000 if um >= 1000 else um:g}"[0]
        d = _digit(g)
        if d is None and not lenient or d is not None and d != lead and not (um >= 100 and d == f"{um / 1000:g}"[0]):
            return None                                           # (unreadable is let through only if `lenient`)
    best = None
    for value, unit in [(um, "µm")] + ([(um / 1000, "mm")] if um >= 100 else []):   # "1000 µm" or "1 mm"
        try:
            fits = {f: np.log((w / h) / np.divide(*_ink_size(f"{value * f:g}"))) for f in (0.1, 1, 10) if value * f >= 1}
        except OSError:
            return None                                           # no font to check with
        if 1 in fits and min(fits, key=lambda f: abs(fits[f])) == 1 and abs(fits[1]) <= 0.18:
            if best is None or abs(fits[1]) < best[0]:
                best = (abs(fits[1]), value, unit)
    if best is None:
        return None
    _, value, unit = best
    _, nh = _ink_size(f"{value:g}")
    lw, _ = _ink_size(f"{value:g} {unit}")
    return max(0, x0) + cols[0] + lw * h / nh / 2


def read_scale(img: Image.Image) -> Optional[dict]:
    """{"um_per_px", "label_um", "bar_px", "bar": [x0, y0, x1, y1]} in screenshot pixels, or None.
    Only QuPath's place for the bar is searched: the bottom-left corner of the screen."""
    img = img.convert("RGB")
    W, H = img.size
    cw, ch = round(W * CORNER[0]), max(60, round(H * CORNER[1]))
    y_off = H - ch
    a = np.asarray(img.crop((0, y_off, cw, H)), dtype=np.float32) / 255.0
    for bx, by, bw, bh, rb in _bar_candidates(a)[:4]:
        got = _label_under(a, bx, by, bw, bh, rb)
        if not got:
            continue
        um, lcx = got
        if abs(bx + bw / 2 - lcx) > CENTRED * bw + 3 or not 0.05 <= um / bw <= 5:
            continue
        return {"um_per_px": um / bw, "label_um": um, "bar_px": int(bw),
                "bar": [int(bx), int(by + y_off), int(bx + bw), int(by + bh + y_off)]}
    return None
