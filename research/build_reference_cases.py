#!/usr/bin/env python3
"""Reference cases for the app's "Similar cases" view: fields from labs the models never trained on, each with the
app's own ConvNeXt feature (the one it already uses for the unlike-training-data check) and a thumbnail.

  HP : UniToPatho HP crops from its TRAIN split (0.4415 um/px; another lab; label from UniToPatho's pathologists).
  SSL: SPIDER-colorectal fields its pathologists labelled sessile serrated lesion (~0.48 um/px; many slides; a lab
       the models never trained on), kept only where the app also calls SSL (at most --per-slide per slide), plus the
       HunCRC slide 135 fields the app calls SSL (that slide's lesion area is not marked; much of it is normal or
       unusable, so only a few fields qualify). Be honest that the app's own call is part of the selection.

Every field is cut at the same physical size as the app's field (--field-um, default the app's 448 um) and goes
through the panel exactly as a snipped field does. Quiet by default: below-normal priority, 2 CPU threads, small
batches with pauses. GPU only with --gpu and only when on mains power.

  python research/build_reference_cases.py --field-um 448 [--gpu]
writes data/reference_cases/cases.npz and data/reference_cases/thumbs/<i>.jpg (data root: $DEMHIST_DATA or <repo>/data)
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--field-um", type=float, default=448)
ap.add_argument("--n-hp", type=int, default=240)
ap.add_argument("--n-ssl", type=int, default=120)
ap.add_argument("--per-slide", type=int, default=6)
ap.add_argument("--gpu", action="store_true")
args = ap.parse_args()
if not args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # "" is dropped on Windows and leaves the GPU visible

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DEMHIST_DATA", ROOT / "data"))
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT / "research"))
import engine as E  # noqa: E402

torch.set_num_threads(2)
try:  # below-normal priority so the laptop stays quiet and responsive
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
except Exception:
    pass
if args.gpu:
    import subprocess
    on_mains = subprocess.run(["powershell", "-c", "(Get-CimInstance Win32_Battery).BatteryStatus"], capture_output=True, text=True).stdout.strip() == "2"
    if not on_mains:
        sys.exit("not on mains power: run without --gpu, or plug in")

OUT = DATA / "reference_cases"
(OUT / "thumbs").mkdir(parents=True, exist_ok=True)
FIELD = args.field_um
rng = random.Random(0)


class Panel:  # just enough of backend.engine.Engine to score fields and get their features the app's way
    device = "cuda" if torch.cuda.is_available() else "cpu"


P = Panel()
P.panel = E.Engine._load_panel(P)
assert P.panel, "panel weights not found"


def score_fields(images, batch=8):
    """(feature, SSL score) per field: the app's ConvNeXt feature, and its displayed score (cut-off at 50%)."""
    feats, scores = [], []
    cut = float(P.panel["shown_cut_off"])
    for i in range(0, len(images), batch):
        logits, f = E.Engine._panel_logits(P, images[i:i + batch], batch_size=batch)
        p = E.Engine._shown(P, E.Engine._panel_score(P, logits))
        feats.append(f.float().cpu().numpy())
        scores += [0.5 * q / cut if q <= cut else 0.5 + 0.5 * (q - cut) / (1 - cut) for q in np.atleast_1d(p)]
        time.sleep(0.3)  # quiet: never run flat out
    return np.concatenate(feats), np.array(scores)


def centre(im, side):
    w, h = im.size
    side = min(side, w, h)
    return im.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))


# ---- HP: UniToPatho train split ----
U = DATA / "unitopatho/unitopath-public/800"
tr = pd.read_csv(U / "train.csv")
hp_ids = sorted(tr[tr.top_label_name == "HP"].image_id)
hp_ids = rng.sample(hp_ids, min(args.n_hp, len(hp_ids)))
side_u = round(FIELD / 0.4415)
hp_imgs = [centre(Image.open(U / "HP" / i).convert("RGB"), side_u) for i in hp_ids]
print(f"HP: {len(hp_imgs)} UniToPatho train fields of {FIELD:.0f} um (crop {side_u} px; crops are 800 um at most)", flush=True)
hp_f, hp_s = score_fields(hp_imgs)

# ---- SSL: HunCRC slide 135, fields the app calls SSL ----
H, PATCH_UM, PATCH_PX, STEP = DATA / "external/huncrc", 248, 512, 2048
lab = next((H / "slide135").rglob("*_labels.csv"))
at = {(r.topleft_x, r.topleft_y): lab.parent.parent / r.fname for r in pd.read_csv(lab).itertuples()}
n = math.ceil(FIELD / PATCH_UM)
crop = round(FIELD / PATCH_UM * PATCH_PX)
blocks = [(x, y) for (x, y) in at if all((x + i * STEP, y + j * STEP) in at for j in range(n) for i in range(n))]
blocks = [b for b in blocks if (b[0] // STEP) % n == 0 and (b[1] // STEP) % n == 0] or blocks  # non-overlapping first


def block_img(x, y):
    im = Image.new("RGB", (n * PATCH_PX, n * PATCH_PX))
    for j in range(n):
        for i in range(n):
            im.paste(Image.open(at[(x + i * STEP, y + j * STEP)]).convert("RGB"), (i * PATCH_PX, j * PATCH_PX))
    return centre(im, crop)


ssl_imgs = [block_img(*b) for b in blocks]
print(f"SSL: scoring {len(ssl_imgs)} HunCRC slide 135 fields of {FIELD:.0f} um", flush=True)
ssl_f, ssl_s = score_fields(ssl_imgs)
keep = [i for i in np.argsort(-ssl_s) if ssl_s[i] > 0.5][:args.n_ssl]
print(f"SSL: {int((ssl_s > 0.5).sum())} of {len(ssl_s)} fields called SSL by the app; keeping the top {len(keep)}", flush=True)

# ---- SSL: SPIDER fields labelled SSL (any split: the models never trained on SPIDER) ----
import json  # noqa: E402
RES = Path(os.environ.get("DEMHIST_RESULTS", ROOT / "docs/results"))
rows = [json.loads(l) for l in (RES / "spider_panel_rows.jsonl").read_text().splitlines()]
recs = {r["image_name"]: r for r in json.loads((DATA / "spider/hp_ssl.json").read_text())}
cand = [r for r in rows if r["truth"] == "SSL" and r.get("call") == "SSL" and r["image"] in recs]
by = {}
for r in sorted(cand, key=lambda r: -r["p_ssl"]):
    by.setdefault(r["slide"], []).append(r)
picked = [r for rs in by.values() for r in rs[:args.per_slide]]
side_s = round(FIELD / 0.48)


def spider_field(rec):
    im = Image.new("RGB", (1120, 1120), (255, 255, 255))
    for i in range(5):
        for j in range(5):
            nm = rec["image_name"] if (i, j) == (2, 2) else rec["context_info"].get(f"{i}_{j}")
            if nm:
                im.paste(Image.open(DATA / "spider/images" / nm).convert("RGB"), (j * 224, i * 224))
    return centre(im, side_s)


sp_imgs = [spider_field(recs[r["image"]]) for r in picked]
print(f"SSL: scoring {len(sp_imgs)} SPIDER SSL-labelled fields from {len(by)} slides", flush=True)
sp_f, sp_s = score_fields(sp_imgs)
sp_keep = [i for i in range(len(sp_imgs)) if sp_s[i] > 0.5]
print(f"SSL: {len(sp_keep)} of them still called SSL by today's app; keeping those", flush=True)

# ---- save ----
feats = np.concatenate([hp_f, ssl_f[keep], sp_f[sp_keep]]).astype(np.float16)
labels = ["HP"] * len(hp_imgs) + ["SSL"] * (len(keep) + len(sp_keep))
sources = (["UniToPatho (train split)"] * len(hp_imgs) + ["HunCRC slide 135"] * len(keep)
           + [f"SPIDER {picked[i]['slide']}" for i in sp_keep])
ids = list(hp_ids) + [f"135:{blocks[i][0]}:{blocks[i][1]}" for i in keep] + [picked[i]["image"] for i in sp_keep]
scores = np.concatenate([hp_s, ssl_s[keep], sp_s[sp_keep]]).astype(np.float32)
for k, im in enumerate(hp_imgs + [ssl_imgs[i] for i in keep] + [sp_imgs[i] for i in sp_keep]):
    im.resize((256, 256), Image.LANCZOS).save(OUT / "thumbs" / f"{k}.jpg", quality=88)
np.savez_compressed(OUT / "cases.npz", feats=feats, labels=np.array(labels), sources=np.array(sources), ids=np.array(ids),
                    scores=scores, field_um=FIELD)
print(f"saved {len(labels)} reference cases ({labels.count('HP')} HP, {labels.count('SSL')} SSL) to {OUT}", flush=True)
