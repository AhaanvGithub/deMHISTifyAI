#!/usr/bin/env python3
"""Stress test for the field placement: scale-bar reading (backend/scalebar.py) and the tissue square
(backend/region.py) on generated viewer screens with known truth. Random but seeded; no model involved.
Each screen: a viewer background (dark, light or grey), a toolbar, MHIST or UniToPatho tissue shown at a known
microns-per-pixel, and (usually) a scale bar with a label in a random font, size, colour, unit and position.
Writes docs/results/scale_stress.json."""

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import scalebar as SB  # noqa: E402
from region import tissue_square  # noqa: E402

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
FONTS = ["segoeui.ttf", "arial.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf", "consola.ttf"]
FONTS = [f for f in FONTS if (Path("C:/Windows/Fonts") / f).exists()]
mhist = sorted((ROOT / "data/mhist/images").glob("*.png"))
uto = sorted((ROOT / "data/unitopatho/unitopath-public/800/HP").glob("*.png"))


def screen(i):
    W, H = rng.choice([(1600, 1000), (1920, 1080), (1366, 768)])
    bgc = rng.choice([(30, 30, 34), (245, 245, 245), (128, 128, 130), (255, 255, 255)])
    im = Image.new("RGB", (W, H), bgc)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 40), fill=rng.choice([(60, 60, 64), (230, 230, 232)]))
    d.text((12, 10), "File  Edit  View  Analyze  Help", fill=(200, 200, 200), font=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14))
    # tissue at a known scale: MHIST tiles are 448 um wide; UniToPatho patches 800 um
    if rng.random() < 0.5:
        src, src_um = Image.open(rng.choice(mhist)).convert("RGB"), 448
    else:
        src, src_um = Image.open(rng.choice(uto)).convert("RGB"), 800
    um_per_px = rng.choice([0.25, 0.4, 0.5, 0.8, 1.0, 1.5])
    side = int(src_um / um_per_px)
    side = min(side, H - 120, W - 120)
    um_per_px = src_um / side
    tx, ty = rng.randint(40, W - side - 40), rng.randint(60, H - side - 40)
    im.paste(src.resize((side, side), Image.BILINEAR), (tx, ty))
    truth = None
    info = {}
    if rng.random() < 0.85:
        label_um = rng.choice([20, 50, 100, 200, 250, 500, 1000])
        bar = int(round(label_um / um_per_px))
        if not 25 <= bar <= 600:
            label_um = 100; bar = int(round(100 / um_per_px))
        fs = rng.choice([11, 13, 15, 18, 22])
        font = ImageFont.truetype(f"C:/Windows/Fonts/{rng.choice(FONTS)}", fs)
        unit = "mm" if label_um >= 1000 and rng.random() < 0.7 else rng.choice(["µm", "μm", "um"])
        text = f"{label_um / 1000:g} mm" if unit == "mm" else f"{label_um} {unit}"
        dark = sum(bgc) < 300
        col = rng.choice([(255, 255, 255)] if dark else [(0, 0, 0), (20, 20, 20)]) if rng.random() < 0.85 else (255, 220, 0)
        bh = rng.choice([3, 4, 6, 8])
        # bar in a margin (bottom-left/right of the viewer) or on an overlay box
        bx = rng.choice([30, W - bar - 160])
        by = H - rng.randint(30, 60)
        if rng.random() < 0.3:  # overlay panel behind bar + label
            d.rectangle((bx - 10, by - fs - 16, bx + bar + 150, by + bh + 10), fill=(255, 255, 255) if sum(col) < 300 else (0, 0, 0))  # contrasting overlay
        d.rectangle((bx, by, bx + bar - 1, by + bh - 1), fill=col)
        tw = d.textlength(text, font=font)
        pos = rng.choice(["above", "below", "right"])
        if pos == "above":
            d.text((bx + bar / 2 - tw / 2, by - fs - 6), text, fill=col, font=font)
        elif pos == "below":
            d.text((bx + bar / 2 - tw / 2, by + bh + 3), text, fill=col, font=font)
        else:
            d.text((bx + bar + 10, by - fs / 2), text, fill=col, font=font)
        truth = um_per_px
        info = {'font': fs, 'pos': pos, 'unit': unit, 'colour': col, 'bar_px': bar, 'bar_h': bh, 'bg': bgc, 'text': text}
    if rng.random() < 0.25:  # 125% or 150% display scaling round trip blurs everything slightly
        f = rng.choice([1.25, 1.5]); im = im.resize((int(W / f), int(H / f)), Image.BILINEAR).resize((W, H), Image.BILINEAR)
        info["scaled"] = f
    return im, truth, (tx, ty, side), info


rows = []
for i in range(60):
    im, truth, (tx, ty, side), info = screen(i)
    r = SB.read_scale(im)
    got = r["um_per_px"] if r else None
    if truth is not None:  # diagnostics: was the bar itself found among the candidates?
        cands = SB._bars(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0)
        info["bar_found"] = any(abs(c[2] - int(info["bar_px"])) <= 3 for c in cands)
        info["n_candidates"] = len(cands)
    if truth is None:
        ok = got is None
    else:
        ok = got is not None and abs(got - truth) / truth < 0.03
    # tissue square inside a selection that covers the tissue plus margins
    sel = im.crop((max(0, tx - 80), max(0, ty - 60), min(im.width, tx + side + 80), min(im.height, ty + side + 60)))
    b = tissue_square(sel)
    ox, oy = max(0, tx - 80), max(0, ty - 60)
    inter = max(0, min(b[2] + ox, tx + side) - max(b[0] + ox, tx)) * max(0, min(b[3] + oy, ty + side) - max(b[1] + oy, ty))
    inside = inter / max(1, (b[2] - b[0]) * (b[3] - b[1]))
    rows.append({**{k: str(v) for k, v in info.items()}, "truth": truth, "got": got, "scale_ok": ok, "square_inside_tissue": round(inside, 3),
                 "square_vs_tissue_side": round((b[2] - b[0]) / side, 3)})
with_bar = [r for r in rows if r["truth"] is not None]
no_bar = [r for r in rows if r["truth"] is None]
res = {"screens": len(rows),
       "with_bar": len(with_bar), "scale_read_within_3pct": sum(r["scale_ok"] for r in with_bar),
       "scale_wrong_value": sum(1 for r in with_bar if r["got"] is not None and not r["scale_ok"]),
       "scale_missed": sum(1 for r in with_bar if r["got"] is None),
       "no_bar": len(no_bar), "no_bar_correctly_none": sum(r["scale_ok"] for r in no_bar),
       "square_mostly_on_tissue (>=95% inside)": sum(r["square_inside_tissue"] >= 0.95 for r in rows),
       "square_median_side_vs_tissue": float(np.median([r["square_vs_tissue_side"] for r in rows])),
       "rows": rows}
(ROOT / f"docs/results/scale_stress{'_seed' + sys.argv[1] if len(sys.argv) > 1 else ''}.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
