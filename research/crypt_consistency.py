#!/usr/bin/env python3
"""Same field, same crypts? A crypt finder run on a raw MHIST tile vs on a simulated screen snip of that tile.

A snip is what the app gets when the tile is shown in a viewer and captured from the screen: the tile drawn at a
larger size (smooth interpolation, like a slide viewer), then the app's field box, which never lines up exactly with
the tile. Variants: zoom 1.5x / 2.37x / 3.3x with the snip covering exactly the tile, and zoom 2.37x with the box
shifted by 4% of the field (its outer strip is lost; crypts there are not counted). Outlines from the snip are mapped
back to tile coordinates and matched one-to-one to the raw-tile outlines (IoU >= 0.5).

Score = F1 between the two crypt sets (1.0 = identical crypts) and, on matched crypts, how often the label agrees.
Training images only: 100 not among the 600 that set the gland thresholds (seed 3).

    python research/crypt_consistency.py [finder]       finder = classical (default) or a name in crypt_finders.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import crypt_gt as GT  # noqa: E402
import crypt_finders as CF  # noqa: E402

VARIANTS = [("zoom 1.5", 1.5, 0.0), ("zoom 2.37", 2.37, 0.0), ("zoom 3.3", 3.3, 0.0), ("zoom 2.37, box shifted 4%", 2.37, 0.04)]


def images(n=100):
    ann = pd.read_csv(GT.DATA / "mhist/annotations.csv")
    train = ann[ann.Partition == "train"]
    used = set(pd.concat([d.sample(min(len(d), 300), random_state=0)
                          for _, d in train.groupby("Majority Vote Label")])["Image Name"]) | set(GT.fields()["image"])
    return list(train[~train["Image Name"].isin(used)].sample(n, random_state=3)["Image Name"])


def snip(tile: Image.Image, zoom: float, shift: float):
    """Screen snip of `tile` drawn at `zoom`, with the field box moved by `shift` of the field (down and right).
    Returns the snip and a function mapping snip 0..1 coordinates back to tile 0..1 coordinates."""
    w = tile.width
    big = tile.resize((round(w * zoom),) * 2, Image.BILINEAR)
    s = round(shift * big.width)
    # the box keeps the field's size; the part beyond the tile is white glass on screen
    canvas = Image.new("RGB", (big.width + s,) * 2, (255, 255, 255))
    canvas.paste(big, (0, 0))
    crop = canvas.crop((s, s, s + big.width, s + big.width))
    return crop, (lambda pts: [[x + shift, y + shift] for x, y in pts])


def inside(outline, lo, hi):
    p = np.asarray(outline)
    return bool(p.min() >= lo and p.max() <= hi)


def run(finder_name="classical"):
    find = CF.FINDERS[finder_name]
    names = images()
    rows = []
    for k, name in enumerate(names, 1):
        tile = Image.open(GT.DATA / "mhist/images" / name).convert("RGB")
        raw = find(tile)
        for vname, zoom, shift in VARIANTS:
            im, back = snip(tile, zoom, shift)
            got = [{**g, "outline": back(g["outline"])} for g in find(im)]
            ref = raw
            if shift:   # only crypts fully inside the area both images show can be compared
                ref = [g for g in raw if inside(g["outline"], shift + 0.01, 0.99)]
                got = [g for g in got if inside(g["outline"], shift + 0.01, 0.99)]
            pairs, _ = GT.match(ref, got)
            same = [GT.main_label(ref[i]["tags"]) == GT.main_label(got[j]["tags"]) for i, j, _ in pairs]
            rows.append({"image": name, "variant": vname, "raw": len(ref), "snip": len(got), "matched": len(pairs),
                         "label_same": sum(same), "iou": float(np.mean([v for _, _, v in pairs])) if pairs else np.nan})
        print(f"{k}/{len(names)} {name}", flush=True)
    d = pd.DataFrame(rows)
    res = {"finder": finder_name, "images": len(names), "variants": {}}
    for v, g in d.groupby("variant", sort=False):
        m, a, b = g.matched.sum(), g.raw.sum(), g.snip.sum()
        res["variants"][v] = {"crypts_raw": int(a), "crypts_snip": int(b), "matched": int(m),
                              "f1_same_crypts": round(2 * m / (a + b), 3) if a + b else None,
                              "label_same_on_matched": round(float(g.label_same.sum() / m), 3) if m else None,
                              "mean_iou_matched": round(float(g.iou.mean()), 3),
                              "images_identical_set": round(float(((g.matched == g.raw) & (g.matched == g.snip)).mean()), 3)}
    return res


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "classical"
    res = run(name)
    out = Path(__file__).resolve().parent.parent / "docs/results" / f"crypt_consistency_{name}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
