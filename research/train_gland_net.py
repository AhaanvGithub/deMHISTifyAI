#!/usr/bin/env python3
"""Train the gland outliner (backend/gland_net.py) on GlaS (165 H&E colon images, 0.62 um/px, expert gland outlines;
data/glas/Warwick_QU_Dataset), shrunk to MHIST's ~2 um/px so the network sees what an MHIST tile shows.

    python research/train_gland_net.py fold0 | fold1 | final  [--enc base]

fold0 / fold1: trained on half of the GlaS patients, so every GlaS image can be scored by a model that never saw
its patient (research/score_crypt_finders.py). final: all 165 images -> data/models/gland_net.pt (used on MHIST).

Training views: random 1/0.85..1/1.18 scale around 2 um/px (MHIST's exact scale is ~1.8-2.2 um/px), flips and
rotations, stain variation in haematoxylin/eosin space, brightness/contrast, 256 px crops at 1 um/px.
Targets: stroma / gland / gland border (2 um wide, so touching glands come apart).
Stops (and keeps the last checkpoint) if the laptop goes on battery.
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import crypt_gt as GT  # noqa: E402
import gland_net as GN  # noqa: E402

GLAS = GT.DATA / "glas" / "Warwick_QU_Dataset"
MPP_GLAS, MPP_MHIST = 0.62, 2.0
CROP = 256
EPOCHS = 80


def glas_table() -> pd.DataFrame:
    g = pd.read_csv(GLAS / "Grade.csv")
    g.columns = ["name", "patient", "grade", "subtype"]
    g["grade"] = g.grade.str.strip()
    g["subtype"] = g.subtype.str.strip()
    # two patient-disjoint halves with similar benign / malignant make-up
    pats = g.groupby("patient").grade.agg(lambda s: (s == "benign").mean()).sort_values()
    half = {p: k % 2 for k, p in enumerate(pats.index)}
    g["fold"] = g.patient.map(half)
    return g


def load(name):
    im = np.asarray(Image.open(GLAS / f"{name}.bmp").convert("RGB"))
    an = np.asarray(Image.open(GLAS / f"{name}_anno.bmp"))
    return im, an


def on_battery() -> bool:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        return r not in ("", "2")       # 2 = on mains power
    except Exception:
        return False


def view(im, an, rng, train=True):
    """One training view: image at ~2 um/px enlarged to 1 um/px (as for MHIST), labels at 1 um/px, 256 px crop."""
    from skimage import morphology
    from skimage.segmentation import find_boundaries
    s = rng.uniform(0.85, 1.18) if train else 1.0
    h, w = an.shape
    lo = (max(8, round(w * MPP_GLAS / MPP_MHIST * s)), max(8, round(h * MPP_GLAS / MPP_MHIST * s)))
    small = Image.fromarray(im).resize(lo, Image.LANCZOS)
    big = small.resize((lo[0] * GN.UP, lo[1] * GN.UP), Image.BICUBIC)
    lab = np.asarray(Image.fromarray(an.astype(np.int32)).resize(big.size, Image.NEAREST))
    a = np.asarray(big, np.float32) / 255
    if train:
        a = stain_jitter(a, rng)
    border = find_boundaries(lab, mode="inner") & (lab > 0)
    border = morphology.dilation(border, morphology.disk(1)) & (lab > 0)
    y = np.where(lab > 0, 1, 0)
    y[border] = 2
    # pad with glass to at least CROP, then crop
    H, W = y.shape
    ph, pw = max(0, CROP - H), max(0, CROP - W)
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw), (0, 0)), constant_values=0.94)
        y = np.pad(y, ((0, ph), (0, pw)))
    H, W = y.shape
    r0, c0 = rng.integers(0, H - CROP + 1), rng.integers(0, W - CROP + 1)
    a, y = a[r0:r0 + CROP, c0:c0 + CROP], y[r0:r0 + CROP, c0:c0 + CROP]
    if train:
        k = rng.integers(4)
        a, y = np.rot90(a, k), np.rot90(y, k)
        if rng.random() < 0.5:
            a, y = a[:, ::-1], y[:, ::-1]
    a = (a - GN.MEAN) / GN.STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)), np.ascontiguousarray(y)


def stain_jitter(a, rng):
    from skimage.color import hed2rgb, rgb2hed
    hed = rgb2hed(np.clip(a, 1e-4, 1))
    hed = hed * rng.uniform(0.8, 1.2, 3) + rng.uniform(-0.03, 0.03, 3)
    out = np.clip(hed2rgb(hed), 0, 1)
    out = (out - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.06, 0.06)
    return np.clip(out, 0, 1).astype(np.float32)


ENCODERS = {"tiny": "convnext_tiny.fb_in22k_ft_in1k", "base": "convnext_base.fb_in22k_ft_in1k"}


def train(which: str, enc: str = "tiny"):
    import torch
    import torch.nn.functional as F
    g = glas_table()
    tr = g if which == "final" else g[g.fold != int(which[-1])]
    data = [load(n) for n in tr.name]
    tag = "" if enc == "tiny" else f"_{enc}"
    out = GN.WEIGHTS.with_name(f"gland_net{tag}.pt" if which == "final" else f"gland_net{tag}_{which}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = GN.build(pretrained=True, encoder=ENCODERS[enc]).cuda()
    enc = [p for n, p in net.named_parameters() if n.startswith("enc.")]
    dec = [p for n, p in net.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": 1e-4}, {"params": dec, "lr": 5e-4}], weight_decay=1e-4)
    steps_per_epoch = (len(data) * 4) // 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 5e-4], total_steps=EPOCHS * steps_per_epoch, pct_start=0.1)
    cw = torch.tensor([1.0, 1.0, 3.0]).cuda()
    print(f"{which}: {len(data)} GlaS images, {EPOCHS} epochs x {steps_per_epoch} steps -> {out}", flush=True)
    t0 = time.time()
    for ep in range(EPOCHS):
        if ep % 5 == 0 and on_battery():
            torch.save(net.state_dict(), out)
            print("ON BATTERY - stopped, checkpoint saved", flush=True)
            sys.exit(2)
        net.train()
        order = rng.permutation(np.repeat(np.arange(len(data)), 4))
        tot = 0
        for b in range(steps_per_epoch):
            xs, ys = zip(*[view(*data[i], rng) for i in order[b * 8:(b + 1) * 8]])
            x = torch.from_numpy(np.stack(xs)).cuda()
            y = torch.from_numpy(np.stack(ys)).long().cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = net(x)
            logit = logit.float()
            p = torch.softmax(logit, 1)
            fg, tfg = p[:, 1] + p[:, 2], (y > 0).float()
            dice = 1 - (2 * (fg * tfg).sum() + 1) / (fg.sum() + tfg.sum() + 1)
            loss = F.cross_entropy(logit, y, weight=cw) + dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"  epoch {ep + 1}/{EPOCHS}  loss {tot / steps_per_epoch:.3f}  {time.time() - t0:.0f}s", flush=True)
    torch.save(net.state_dict(), out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    enc = args[args.index("--enc") + 1] if "--enc" in args else "tiny"
    for w in [a for a in args if a.startswith(("fold", "final"))]:
        train(w, enc)
