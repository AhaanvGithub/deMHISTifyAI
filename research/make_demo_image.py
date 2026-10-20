#!/usr/bin/env python3
"""Build data/demo/demo_hp_ssl.png: two squares side by side, each one MHIST field (448 x 448 um), with a 100 um
scale bar, labelled A and B (answers in demo_hp_ssl_answers.txt, not on the image). The app opens it at start-up.

A = a UniToPatho hyperplastic polyp (outside hospital, scanned at 0.44 um/px, so it is sharp): the centre 448 um of
    the patch whose ConvNeXt-Base score is closest to the HP median - a typical case, not a hand-picked easy one.
B = an MHIST test-split SSL with 7/7 pathologist votes (the first such tile by name, chosen without looking at any
    model score). MHIST only exists at 224 px (~2 um/px), so B is upscaled and looks softer than A.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/demo"
FIELD_UM = 448
GAP, MARGIN, TOP = 140, 70, 120

# A: typical UniToPatho HP
D = ROOT / "data/unitopatho/unitopath-public/800"
meta = pd.concat([pd.read_csv(D / "train.csv"), pd.read_csv(D / "test.csv")]).set_index("image_id")
scores = pd.read_csv(ROOT / "data/unitopatho/convnext_base_scores_hp_ta_norm.csv")
hp = scores[scores.cls == "HP"]
hp_row = hp.iloc[(hp.p_ssl - hp.p_ssl.median()).abs().argmin()]
mpp = meta.loc[hp_row.image, "mpp"]
c = round(FIELD_UM / mpp)
im = Image.open(D / "HP" / hp_row.image).convert("RGB")
w, h = im.size
a = im.crop(((w - c) // 2, (h - c) // 2, (w + c) // 2, (h + c) // 2))
S = a.size[0]
px_per_um = S / FIELD_UM

# B: first 7/7 SSL in the MHIST test split, upscaled to the same physical size
ann = pd.read_csv(ROOT / "data/mhist/annotations.csv")
ssl_name = sorted(ann[(ann.Partition == "test") & (ann["Number of Annotators who Selected SSA (Out of 7)"] == 7)]["Image Name"])[0]
b = Image.open(ROOT / "data/mhist/images" / ssl_name).convert("RGB").resize((S, S), Image.LANCZOS)

W, H = MARGIN * 2 + 2 * S + GAP, TOP + S + 150
canvas = Image.new("RGB", (W, H), (250, 250, 250))
d = ImageDraw.Draw(canvas)
font = lambda n: ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", n)
d.text((MARGIN, 30), "Each square is one 0.45 mm field, the size the model was trained on. Snip one square.",
       fill=(60, 60, 60), font=font(30))
bar = round(100 * px_per_um)
for i, (img, label) in enumerate([(a, "A"), (b, "B")]):
    x = MARGIN + i * (S + GAP)
    canvas.paste(img, (x, TOP))
    d.text((x, TOP + S + 20), label, fill=(30, 30, 30), font=font(44))
    y = TOP + S + 50
    d.rectangle((x + S - bar, y, x + S, y + 10), fill=(20, 20, 20))
    d.text((x + S - bar, y + 16), "100 µm", fill=(20, 20, 20), font=font(26))
OUT.mkdir(parents=True, exist_ok=True)
canvas.save(OUT / "demo_hp_ssl.png")
(OUT / "demo_hp_ssl_answers.txt").write_text(
    f"A  HP   UniToPatho {hp_row.image} (wsi {hp_row.wsi}); ConvNeXt-Base P(SSL) {hp_row.p_ssl:.2f} = HP median\n"
    f"B  SSL  MHIST {ssl_name} (test split, 7/7 SSA votes); upscaled from 224 px\n"
    f"Each square = {FIELD_UM} um; {px_per_um:.3f} px/um in the image\n", encoding="utf-8")
print(f"saved {canvas.size}, field {S} px, A={hp_row.image[:40]}..., B={ssl_name}")
