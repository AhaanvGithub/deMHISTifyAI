#!/usr/bin/env python3
"""Do image adjustments or pathologist-style viewing help the deployed fine-tuned judges?

Scores the 4 fine-tuned CONCH versions (raw image, as the app's panel does) under:
  A. orientation averaging: mean over the 8 flips/rotations (the model was trained with them)
  B. single adjustments: contrast / brightness / gamma / saturation at several strengths, and the
     average over a set of mild ones (test-time augmentation)
  C. multi-scale: the whole field plus zoomed-in crops (centre + 4 quadrants, each resized to 224),
     combined with a weight w on the zoomed views (like switching objectives)
Every option is scored on the validation slice of MHIST train; the best few are then scored once
on the 977 test images. Writes docs/results/image_tta_sweep.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research"))
from engine import Engine  # noqa: E402  (first, so OpenBLAS is single-threaded before numpy is used)
from select_prompts import load_labels  # noqa: E402
from train_judges import ece  # noqa: E402

OUT = ROOT / "docs" / "results" / "image_tta_sweep.json"

DIHEDRAL = [lambda im, k=k, f=f: (im.transpose(Image.FLIP_LEFT_RIGHT) if f else im).rotate(90 * k, expand=True)
            for f in (0, 1) for k in range(4)]


def gamma(im, g):
    lut = [round(255 * (i / 255) ** g) for i in range(256)] * 3
    return im.point(lut)


ADJUST = {
    **{f"contrast_{v}": (lambda im, v=v: ImageEnhance.Contrast(im).enhance(v)) for v in (0.8, 0.9, 1.1, 1.2, 1.35)},
    **{f"brightness_{v}": (lambda im, v=v: ImageEnhance.Brightness(im).enhance(v)) for v in (0.85, 0.93, 1.07, 1.15)},
    **{f"gamma_{v}": (lambda im, v=v: gamma(im, v)) for v in (0.8, 0.9, 1.1, 1.25)},
    **{f"saturation_{v}": (lambda im, v=v: ImageEnhance.Color(im).enhance(v)) for v in (0.8, 1.2)},
}
MILD = ["contrast_0.9", "contrast_1.1", "brightness_0.93", "brightness_1.07", "gamma_0.9", "gamma_1.1"]


def crops(im):
    w, h = im.size
    boxes = [(w // 4, h // 4, 3 * w // 4, 3 * h // 4), (0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
             (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
    return [im.crop(b) for b in boxes]


def metrics(p, y, votes):
    return {"auc": round(float(roc_auc_score(y, p)), 4), "bal_acc": round(float(balanced_accuracy_score(y, p > .5)), 4),
            "ece": round(ece(p, y), 4), "spearman_vs_votes": round(float(spearmanr(p, votes)[0]), 4)}


def main():
    lab = load_labels(ROOT / "data/mhist/annotations.csv")
    names = np.array(sorted(lab["image"]))
    lab = lab.set_index("image").loc[names]
    y = (lab["label"] == "SSA").to_numpy().astype(int)
    votes = lab["ssa_votes"].to_numpy(float)
    train_all = (lab["split"] == "train").to_numpy()
    va = train_all & (np.random.default_rng(0).random(len(lab)) < 0.2)  # same slice as training
    ts = (lab["split"] == "test").to_numpy()
    eng = Engine()

    def score_views(idx, make_views):
        """mean P(SSA) over the 4 fine-tuned versions for each view: [n_images, n_views]"""
        out = []
        for n in names[idx]:
            im = Image.open(ROOT / "data/mhist/images" / n).convert("RGB")
            views = make_views(im)
            out.append(eng.encode(views)[1].mean(axis=1))
        return np.array(out)

    def evaluate(idx):
        t0 = time.time()
        res = {}
        base = score_views(idx, lambda im: [im])[:, 0]
        res["original"] = base
        dih = score_views(idx, lambda im: [f(im) for f in DIHEDRAL])
        res["A_orientations_x8"] = dih.mean(1)
        adj = score_views(idx, lambda im: [ADJUST[k](im) for k in ADJUST])
        for j, k in enumerate(ADJUST):
            res[f"B_{k}"] = adj[:, j]
        mild = [list(ADJUST).index(k) for k in MILD]
        res["B_mild_average"] = (adj[:, mild].sum(1) + base) / (len(mild) + 1)
        zoom = score_views(idx, crops).mean(1)
        for w in (0.25, 0.5):
            res[f"C_multiscale_w{w}"] = (1 - w) * base + w * zoom
            res[f"C_multiscale_w{w}_plus_orient"] = (1 - w) * res["A_orientations_x8"] + w * zoom
        print(f"  scored {idx.sum()} images in {time.time() - t0:.0f}s", flush=True)
        return res

    print("validation...", flush=True)
    val = evaluate(va)
    val_m = {k: metrics(p, y[va], votes[va]) for k, p in val.items()}
    ranked = sorted(val_m, key=lambda k: -val_m[k]["auc"])
    for k in ranked[:12]:
        print(f"  val {k:34} AUC {val_m[k]['auc']:.4f} bal {val_m[k]['bal_acc']:.3f} rho {val_m[k]['spearman_vs_votes']:.3f}", flush=True)
    finalists = ["original"] + [k for k in ranked if k != "original"][:4]

    print("test (finalists only)...", flush=True)
    test = evaluate(ts)
    test_m = {k: metrics(test[k], y[ts], votes[ts]) for k in finalists}
    for k in finalists:
        print(f"  TEST {k:34} AUC {test_m[k]['auc']:.4f} bal {test_m[k]['bal_acc']:.3f} ECE {test_m[k]['ece']:.3f} "
              f"rho {test_m[k]['spearman_vs_votes']:.3f}", flush=True)
    OUT.write_text(json.dumps({"what": "test-time image adjustments / orientation / multi-scale on the fine-tuned judges; "
                                       "ranked on validation, finalists scored once on test",
                               "validation": val_m, "finalists": finalists, "test": test_m}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
